import os
import torch
import numpy as np
import pandas as pd
from model.model import GeometryAwareTransformer
from utils.io_utils import load_features_from_csv
from utils.downsampling import fps_with_cache as fps
from utils.fold_data_split import collect_files_by_fold

def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.PREDICTED_DATA_DIR, exist_ok=True)

    use_fold_split = bool(getattr(config, 'USE_FOLD_SPLIT', False))
    if use_fold_split:
        fold_id = getattr(config, 'FOLD_ID', 'fold_1')
        strict = bool(getattr(config, 'FOLD_SPLIT_STRICT', True))
        labeled_dir = getattr(config, 'LABELED_DATA_DIR', config.TEST_DATA_DIR)
        test_files = collect_files_by_fold(
            mode='test',
            fold_id=fold_id,
            data_dir=labeled_dir,
            strict=strict,
            print_stats=True,
        )
    else:
        test_files = [os.path.join(config.TEST_DATA_DIR, f)
                      for f in os.listdir(config.TEST_DATA_DIR)
                      if f.endswith('.csv') and '_pred' not in f]

    max_points = getattr(config, 'MAX_POINTS', 4096)

    model = GeometryAwareTransformer(config).to(device)
    model_path = "weights/best_finetune.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    alpha = getattr(config, 'CURVATURE_WEIGHT_ALPHA', 0.5)  # 建议 0.2~0.5
    thresh = getattr(config, 'PREDICTION_THRESHOLD', 0.7)
    curv_gate = getattr(config, 'CURVATURE_GATE', 0)     # 低曲率直接过滤掉

    for file_path in test_files:
        sample = load_features_from_csv(file_path)
        features = sample['features'].astype(np.float32)
        points = sample['coordinate'].astype(np.float32)
        curv_np = sample['curvature'].astype(np.float32)  # [N,1]
        pd_np = sample['principal_dir'].astype(np.float32)  # [N,3]
        den_np = sample['local_density'].astype(np.float32)            # [N,1]
        nor_np = sample['normals'].astype(np.float32)            # [N,3]
        lin_np = sample['linearity'].astype(np.float32)          # [N,1]
        
        N = features.shape[0]

        # 如果点数超过最大限制，进行FPS下采样
        if N > max_points:
            idxs = fps(points, max_points)
            features = features[idxs]
            points = points[idxs]
            curv_np = curv_np[idxs]
            den_np = den_np[idxs]
            nor_np = nor_np[idxs]
            lin_np = lin_np[idxs]
            pd_np = pd_np[idxs]

        preds_binary = np.zeros((features.shape[0], 1), dtype=np.float32)
        
        feat_t = torch.from_numpy(features[np.newaxis, ...]).to(device)
        coord_t = torch.from_numpy(points[np.newaxis, ...]).to(device)
        principal_t = torch.from_numpy(pd_np[np.newaxis, ...]).to(device)
        curv_t = torch.from_numpy(curv_np[np.newaxis, ...]).to(device)  # [1,M,1]
        den_t = torch.from_numpy(den_np[np.newaxis, ...]).to(device)
        nor_t = torch.from_numpy(nor_np[np.newaxis, ...]).to(device)
        lin_t = torch.from_numpy(lin_np[np.newaxis, ...]).to(device)

        with torch.no_grad():
            logits = model(
                feat_t,
                coord_t,
                principal_t,
                curv_t,
                den_t,
                nor_t,
                lin_t,
                task='class'
            ).squeeze(0).squeeze(-1)

            # ---- 曲率归一化 ----
            curv_vec = curv_t.squeeze(0).squeeze(-1)  # [M]
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

        # 保存为CSV格式
        save_path = os.path.join(
            config.PREDICTED_DATA_DIR,
            os.path.basename(file_path).replace('.csv', '_pred.csv'),
        )

        # 创建包含预测结果的DataFrame
        df_pred = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'prediction': preds_binary[:, 0]
        })
        df_pred.to_csv(save_path, index=False)
        print(f"Saved predictions to: {save_path}")

    print("Inference complete.")
