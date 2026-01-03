import os
import torch
import numpy as np
from model.model import GeometryAwareTransformer
from utils.io_utils import save_features_to_npz
from utils.downsampling import fps_with_cache as fps

def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.PREDICTED_DATA_DIR, exist_ok=True)

    test_files = [os.path.join(config.TEST_DATA_DIR, f)
                  for f in os.listdir(config.TEST_DATA_DIR)
                  if f.endswith('.npz') and not f.startswith('T1') and not f.startswith('T2') and not f.startswith('T3')]

    max_points = getattr(config, 'MAX_POINTS', 4096)

    model = GeometryAwareTransformer(config).to(device)
    model_path = "weights/best_finetune.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    alpha = getattr(config, 'CURVATURE_WEIGHT_ALPHA', 0.3)  # 建议 0.2~0.5
    thresh = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
    curv_gate = getattr(config, 'CURVATURE_GATE', 0)     # 低曲率直接过滤掉

    for file_path in test_files:
        data = np.load(file_path)
        features = data['features'].astype(np.float32)
        points = features[..., :3]
        curv_np = data['curvature'].astype(np.float32)  # [N,1]
        pd_np = data['principal_dir'].astype(np.float32)  # [N,3]
        den_np = data['local_density'].astype(np.float32)            # [N,1]
        nor_np = data['normals'].astype(np.float32)            # [N,3]
        lin_np = data['linearity'].astype(np.float32)          # [N,1]
        
        N = features.shape[0]
        idxs = fps(points, max_points)
        features = features[idxs]
        points = points[idxs]
        curv_np = curv_np[idxs]
        den_np = den_np[idxs]
        nor_np = nor_np[idxs]
        lin_np = lin_np[idxs]
        pd_np = pd_np[idxs]
        
        
        preds_binary = np.zeros((min(N, max_points), 1), dtype=np.float32)
        
        f = torch.from_numpy(features[np.newaxis, ...]).to(device)
        pd = torch.from_numpy(pd_np[np.newaxis, ...]).to(device)
        curv = torch.from_numpy(curv_np[np.newaxis, ...]).to(device)  # [1,M,1]
        den = torch.from_numpy(den_np[np.newaxis, ...]).to(device)
        nor = torch.from_numpy(nor_np[np.newaxis, ...]).to(device)
        lin = torch.from_numpy(lin_np[np.newaxis, ...]).to(device)

        with torch.no_grad():
            logits = model(f, pd, curv, den, nor, lin, task='class').squeeze(0).squeeze(-1)

            # ---- 曲率归一化 ----
            curv_vec = curv.squeeze(0).squeeze(-1)  # [M]
            cmin = torch.min(curv_vec)
            cmax = torch.max(curv_vec)
            curv_norm = (curv_vec - cmin) / (cmax - cmin + 1e-8)

            # ---- logits 加权 ----
            logits_adj = logits + alpha * curv_norm

            # ---- sigmoid ----
            probs = torch.sigmoid(logits_adj)

            # ---- 低曲率过滤 ----
            probs[curv_vec < curv_gate] = 0.0

            # ---- 最终阈值 ----
            preds_binary = (probs > thresh).float().cpu().numpy()[:, None]

        # 保存
        save_path = os.path.join(
            config.PREDICTED_DATA_DIR,
            os.path.basename(file_path).replace('.npz', '_pred.npz'),
        )
        save_features_to_npz(np.concatenate([points, preds_binary], axis=1), save_path)
        print(f"Saved predictions to: {save_path}")

    print("Inference complete.")
