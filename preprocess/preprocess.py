# weld_seg_project/preprocess/preprocess.py 预训练主函数
import os
import numpy as np
from .geometric_feature import GeometricFeatureCalculator
from utils.io_utils import read_all_ply_from_dir

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
        # print(geom['local_density'])
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

    def process_and_save_dataset(self, input_dir, output_dir):
        """处理整个数据集并保存"""
        os.makedirs(output_dir, exist_ok=True)
        point_clouds, file_paths = read_all_ply_from_dir(input_dir)
        
        for pc, file_path in zip(point_clouds, file_paths):
            try:
                features, extra_info = self.transform(pc)
                base_name = os.path.basename(file_path).replace('.ply', '.csv')
                save_path = os.path.join(output_dir, base_name)
                # 构建DataFrame，列名清晰
                import pandas as pd
                df = pd.DataFrame(features, columns=['x','y','z','nx','ny','nz','curvature','density'])
                # 添加额外信息
                df['linearity'] = extra_info['linearity'].flatten()
                df['principal_dir_x'] = extra_info['principal_dir'][:, 0]
                df['principal_dir_y'] = extra_info['principal_dir'][:, 1]
                df['principal_dir_z'] = extra_info['principal_dir'][:, 2]
                df.to_csv(save_path, index=False, float_format='%.6f')  
                print(f"Processed and saved: {save_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")