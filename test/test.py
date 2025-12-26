import os
import torch
import numpy as np
import pandas as pd
from model.model import GeometryAwareTransformer


def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.PREDICTED_DATA_DIR, exist_ok=True)

    # 读取CSV测试文件
    test_files = [os.path.join(config.TEST_DATA_DIR, f)
                  for f in os.listdir(config.TEST_DATA_DIR)
                  if f.endswith('.csv')]

    max_points = getattr(config, 'MAX_POINTS', None)

    model = GeometryAwareTransformer(config).to(device)
    model_path = os.path.join(config.WEIGHTS_SAVE_DIR, "best_finetune.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    alpha = getattr(config, 'CURVATURE_WEIGHT_ALPHA', 0.3)
    thresh = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
    curv_gate = getattr(config, 'CURVATURE_GATE', 0.0008)

    for file_path in test_files:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取特征
        features = df[['x', 'y', 'z', 'nx', 'ny', 'nz', 'curvature', 'density']].values.astype(np.float32)
        points = features[:, :3]
        curvature_np = df[['curvature']].values.astype(np.float32)
        normals_np = df[['nx', 'ny', 'nz']].values.astype(np.float32)
        density_np = df[['density']].values.astype(np.float32)

        # 主方向和线性度（可选）
        if 'principal_dir_x' in df.columns:
            principal_dir_np = df[['principal_dir_x', 'principal_dir_y', 'principal_dir_z']].values.astype(np.float32)
        else:
            principal_dir_np = np.zeros_like(normals_np)

        if 'linearity' in df.columns:
            linearity_np = df[['linearity']].values.astype(np.float32)
        else:
            linearity_np = np.zeros((features.shape[0], 1), dtype=np.float32)

        N = features.shape[0]
        preds_binary = np.zeros((N, 1), dtype=np.float32)

        def run_chunk(feats, pd_np, curv_np, den_np, nor_np, lin_np, out_slice):
            """内部函数：GPU推理 + 曲率归一化"""
            f = torch.from_numpy(feats[None]).to(device)
            pd = torch.from_numpy(pd_np[None]).to(device)
            curv = torch.from_numpy(curv_np[None]).to(device)
            den = torch.from_numpy(den_np[None]).to(device)
            nor = torch.from_numpy(nor_np[None]).to(device)
            lin = torch.from_numpy(lin_np[None]).to(device)

            with torch.no_grad():
                try:
                    logits = model(f, pd, curv, den, nor, lin, task='class').squeeze(0).squeeze(-1)

                    # 曲率归一化
                    curv_vec = curv.squeeze(0).squeeze(-1)
                    cmin = torch.min(curv_vec)
                    cmax = torch.max(curv_vec)
                    curv_norm = (curv_vec - cmin) / (cmax - cmin + 1e-8)

                    # logits加权
                    logits_adj = logits + alpha * curv_norm

                    # sigmoid
                    probs = torch.sigmoid(logits_adj)

                    # 低曲率过滤
                    probs[curv_vec < curv_gate] = 0.0

                    # 阈值化
                    out = (probs > thresh).float().cpu().numpy()[:, None]
                    preds_binary[out_slice] = out

                except Exception as e:
                    print("Chunk failed:", e)
                    preds_binary[out_slice] = 0.0

        # 推理（是否分块）
        if max_points is None or N <= max_points:
            run_chunk(
                features,
                principal_dir_np,
                curvature_np,
                density_np,
                normals_np,
                linearity_np,
                slice(0, N),
            )
        else:
            num_chunks = (N + max_points - 1) // max_points
            for c in range(num_chunks):
                s, e = c * max_points, min(N, (c + 1) * max_points)
                run_chunk(
                    features[s:e],
                    principal_dir_np[s:e],
                    curvature_np[s:e],
                    density_np[s:e],
                    normals_np[s:e],
                    linearity_np[s:e],
                    slice(s, e),
                )

        # 保存预测结果为CSV
        result_df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'prediction': preds_binary.flatten().astype(int)
        })

        save_path = os.path.join(
            config.PREDICTED_DATA_DIR,
            os.path.basename(file_path).replace('.csv', '_pred.csv'),
        )
        result_df.to_csv(save_path, index=False, float_format='%.6f')
        print(f"Saved predictions to: {save_path}")

    print("Inference complete.")
