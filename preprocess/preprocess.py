# weld_seg_project/preprocess/preprocess.py 预处理主函数
import os
import numpy as np
from .geometric_feature import GeometricFeatureCalculator
from utils.io_utils import read_all_ply_from_dir, read_ply_with_scalar_seam


class PointCloudPreprocessor:
    def __init__(self, config):
        self.config = config
        self.feature_calculator = GeometricFeatureCalculator(
            k_neighbors=config.K_NEIGHBORS,
            radius_ratio=config.RADIUS_RATIO
        )
        self.mu = None
        self.sigma = None

    def _compute_raw_features(self, point_cloud_np):
        """计算未归一化的8维特征和额外几何信息"""
        geom = self.feature_calculator.calculate(point_cloud_np)
        
        # [x, y, z, nx, ny, nz, kappa, rho]
        feature_tensor = np.concatenate([
            point_cloud_np,
            geom['normals'],
            geom['curvature'],
            geom['local_density']
        ], axis=1)
        
        # 返回特征和额外信息（用于模型和训练）
        extra_info = {
            'principal_dir': geom['principal_dir'],
            'curvature': geom['curvature'],
            'linearity': geom['linearity'],
            # also expose normals and local_density so downstream code can load them directly
            'normals': geom['normals'],
            'local_density': geom['local_density']
        }
        return feature_tensor, extra_info

    def fit(self, dataset_dir):
        """使用训练数据计算归一化参数"""
        print("Fitting preprocessor...")
        point_clouds, _ = read_all_ply_from_dir(dataset_dir)
        
        all_features = []
        for pc in point_clouds:
            features, _ = self._compute_raw_features(pc)
            all_features.append(features)
            
        all_features = np.vstack(all_features)
        self.mu = np.mean(all_features, axis=0)
        self.sigma = np.std(all_features, axis=0)
        self.sigma[self.sigma == 0] = 1.0
        print("Preprocessor fitted.")

    def transform(self, point_cloud_np):
        """对单个点云计算并归一化特征"""
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Preprocessor has not been fitted. Call 'fit' first.")
            
        raw_features, extra_info = self._compute_raw_features(point_cloud_np)
        normalized_features = (raw_features - self.mu) / self.sigma
        return normalized_features, extra_info

    def _try_load_labels(self, file_path, expected_n):
        """尝试从带标注PLY读取标签（scalar_seam/seam/label），失败则返回None。"""
        try:
            info = read_ply_with_scalar_seam(file_path)
            labels = info.get('scalar_seam', None)
            if labels is None:
                return None
            labels = np.asarray(labels)
            if labels.ndim > 1:
                labels = labels.reshape(-1)
            if labels.shape[0] != expected_n:
                print(f"[warn] label length mismatch for {file_path}: {labels.shape[0]} vs {expected_n}, ignore labels")
                return None
            # 统一为0/1
            labels = (labels > 0.5).astype(np.float32)
            return labels
        except Exception:
            return None

    def process_and_save_dataset(self, input_dir, output_dir):
        """处理整个数据集并保存（自动兼容带label的PLY）"""
        os.makedirs(output_dir, exist_ok=True)
        point_clouds, file_paths = read_all_ply_from_dir(input_dir)

        # 可选：打散点顺序，避免“正样本都在前面”带来的顺序偏置
        shuffle_points = bool(getattr(self.config, 'SHUFFLE_POINTS_BEFORE_SAVE', True))
        shuffle_seed = int(getattr(self.config, 'SHUFFLE_SEED', 42))
        
        for pc, file_path in zip(point_clouds, file_paths):
            try:
                features, extra_info = self.transform(pc)

                labels = self._try_load_labels(file_path, expected_n=features.shape[0])

                if shuffle_points:
                    rng = np.random.default_rng(shuffle_seed)
                    perm = rng.permutation(features.shape[0])
                    features = features[perm]
                    extra_info['linearity'] = extra_info['linearity'][perm]
                    extra_info['principal_dir'] = extra_info['principal_dir'][perm]
                    if labels is not None:
                        labels = labels[perm]

                base_name = os.path.basename(file_path).replace('.ply', '.csv')
                if labels is not None and '_label_' not in base_name:
                    stem = base_name[:-4]
                    base_name = f"{stem}_label_auto.csv"

                save_path = os.path.join(output_dir, base_name)

                import pandas as pd
                df = pd.DataFrame(features, columns=['x', 'y', 'z', 'nx', 'ny', 'nz', 'curvature', 'density'])
                df['linearity'] = extra_info['linearity'].flatten()
                df['principal_dir_x'] = extra_info['principal_dir'][:, 0]
                df['principal_dir_y'] = extra_info['principal_dir'][:, 1]
                df['principal_dir_z'] = extra_info['principal_dir'][:, 2]

                if labels is not None:
                    df['label'] = labels.astype(np.float32)

                df.to_csv(save_path, index=False, float_format='%.6f')  
                print(f"Processed and saved: {save_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
