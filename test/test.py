import os
import torch
import numpy as np
from model.model import GeometryAwareTransformer
from utils.io_utils import save_features_to_npz

def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.PREDICTED_DATA_DIR, exist_ok=True)

    test_files = [os.path.join(config.TEST_DATA_DIR, f)
                  for f in os.listdir(config.TEST_DATA_DIR)
                  if f.endswith('.npz')]

    max_points = getattr(config, 'MAX_POINTS', None)

    model = GeometryAwareTransformer(config).to(device)
    model_path = "weights/best_finetune.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    alpha = getattr(config, 'CURVATURE_WEIGHT_ALPHA', 0.3)  # 建议 0.2~0.5
    thresh = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
    curv_gate = getattr(config, 'CURVATURE_GATE', 0.0008)     # 低曲率直接过滤掉

    for file_path in test_files:
        data = np.load(file_path)
        features = data['features'].astype(np.float32)
        points = features[..., :3]
        curvature_np = data['curvature'].astype(np.float32)  # [N,1]
        N = features.shape[0]

        preds_binary = np.zeros((N, 1), dtype=np.float32)

        def run_chunk(feats, pd_np, curv_np, den_np, nor_np, lin_np, out_slice):
            """内部小函数：支持完整 GPU + 曲率归一化 + 安全 try"""
            f = torch.from_numpy(feats[None]).to(device)
            pd = torch.from_numpy(pd_np[None]).to(device)
            curv = torch.from_numpy(curv_np[None]).to(device)  # [1,M,1]
            den = torch.from_numpy(den_np[None]).to(device)
            nor = torch.from_numpy(nor_np[None]).to(device)
            lin = torch.from_numpy(lin_np[None]).to(device)

            with torch.no_grad():
                try:
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
                    out = (probs > thresh).float().cpu().numpy()[:, None]
                    preds_binary[out_slice] = out

                except Exception as e:
                    print("Chunk failed:", e)
                    preds_binary[out_slice] = 0.0

        # === No chunk ===
        if max_points is None or N <= max_points:
            run_chunk(
                features,
                data['principal_dir'].astype(np.float32),
                curvature_np,
                data['local_density'].astype(np.float32),
                data['normals'].astype(np.float32),
                data['linearity'].astype(np.float32),
                slice(0, N),
            )
        else:
            num_chunks = (N + max_points - 1) // max_points
            for c in range(num_chunks):
                s, e = c * max_points, min(N, (c + 1) * max_points)
                run_chunk(
                    features[s:e],
                    data['principal_dir'][s:e].astype(np.float32),
                    curvature_np[s:e],
                    data['local_density'][s:e].astype(np.float32),
                    data['normals'][s:e].astype(np.float32),
                    data['linearity'][s:e].astype(np.float32),
                    slice(s, e),
                )

        # 保存
        save_path = os.path.join(
            config.PREDICTED_DATA_DIR,
            os.path.basename(file_path).replace('.npz', '_pred.npz'),
        )
        save_features_to_npz(np.concatenate([points, preds_binary], axis=1), save_path)
        print(f"Saved predictions to: {save_path}")

    print("Inference complete.")
