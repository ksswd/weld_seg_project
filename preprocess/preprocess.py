# weld_seg_project/preprocess/preprocess.py 预训练主函数
import os
import numpy as np
import open3d as o3d
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
        seam_label = None
        # Support multiple input types:
        # - Open3D PointCloud (xyz only)
        # - dict from utils.io_utils.read_ply_with_scalar_seam: {'xyz':..., 'scalar_seam':...}
        if isinstance(point_cloud_np, dict):
            xyz = np.asarray(point_cloud_np.get("xyz"), dtype=np.float64)
            seam_label = point_cloud_np.get("scalar_seam", None)
        elif isinstance(point_cloud_np, o3d.geometry.PointCloud):
            xyz = np.asarray(point_cloud_np.points, dtype=np.float64)
        else:
            xyz = np.asarray(point_cloud_np, dtype=np.float64)
        # xyz = point_cloud_np[:, :3].copy()
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True) # 坐标局部中心化

        geom = self.feature_calculator.calculate(xyz)
  
        # [x, y, z, nx, ny, nz, kappa, rho]
        feature_tensor = np.concatenate([
            xyz,
            geom['normals'],
            geom['curvature'],
            geom['local_density']
        ], axis=1)
        
        # 返回特征和额外信息（用于模型和训练）
        extra_info = {
            'principal_dir': geom['principal_dir'],
            'curvature': geom['curvature'],
            'linearity': geom['linearity'],
            'normals': geom['normals'],
            'local_density': geom['local_density']
        }

        # seam label (0/1) from labeled PLYs
        if seam_label is not None:
            seam_label = np.asarray(seam_label).reshape(-1)
            # ensure length matches points
            if seam_label.shape[0] == xyz.shape[0]:
                extra_info['label'] = seam_label.astype(np.float32)
        return feature_tensor, extra_info

    def fit(self, dataset_dir):
        """使用训练数据计算归一化参数"""
        print("Fitting preprocessor...")
        point_clouds, _ = read_all_ply_from_dir(dataset_dir)

        all_features = []
        for pc in point_clouds:
            features, _ = self._compute_raw_features(pc)
            all_features.append(features[:, 3:]) # 不处理xyz

        all_features = np.vstack(all_features)
        self.mu = np.mean(all_features, axis=0)
        self.sigma = np.std(all_features, axis=0)
        self.sigma[self.sigma == 0] = 1.0
        print("Preprocessor fitted.")
        print(f"  mu: {self.mu}")
        print(f"  sigma: {self.sigma}")

    def save_params(self, save_path="normalization_params.npz"):
        """保存归一化参数"""
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Preprocessor has not been fitted. Call 'fit' first.")
        np.savez(save_path, mu=self.mu, sigma=self.sigma)
        print(f"Saved normalization parameters to: {save_path}")

    def load_params(self, load_path="normalization_params.npz"):
        """加载归一化参数"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Normalization params not found: {load_path}")
        params = np.load(load_path)
        self.mu = params['mu']
        self.sigma = params['sigma']
        print(f"Loaded normalization parameters from: {load_path}")
        print(f"  mu: {self.mu}")
        print(f"  sigma: {self.sigma}")

    def transform(self, point_cloud_np):
        """对单个点云计算并归一化特征"""
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Preprocessor has not been fitted. Call 'fit' first.")
            
        raw_features, extra_info = self._compute_raw_features(point_cloud_np)
        features = raw_features.copy()
        features[:, 3:] = (features[:, 3:] - self.mu) / self.sigma # 归一化非坐标部分
        return features, extra_info

    def process_and_save_dataset(self, input_dir, output_dir):
        """
        将点云预处理结果保存为 CSV（信息等价于 NPZ）
        - 同时保存 normalized features（模型输入）
        - 和 raw geometric features（分析 / 可复现）
        """
        os.makedirs(output_dir, exist_ok=True)
        point_clouds, file_paths = read_all_ply_from_dir(input_dir)

        import pandas as pd

        for pc, file_path in zip(point_clouds, file_paths):
            try:
                features, extra_info = self.transform(pc)

                base_name = os.path.basename(file_path).replace('.ply', '.csv')
                save_path = os.path.join(output_dir, base_name)

                df = pd.DataFrame(features, columns=[
                    'x', 'y', 'z',
                    'nx_norm', 'ny_norm', 'nz_norm',
                    'curvature_norm', 'density_norm'
                ])

                # raw normals
                df['nx_raw'] = extra_info['normals'][:, 0]
                df['ny_raw'] = extra_info['normals'][:, 1]
                df['nz_raw'] = extra_info['normals'][:, 2]

                # raw curvature & density
                df['curvature_raw'] = extra_info['curvature'].reshape(-1)
                df['local_density_raw'] = extra_info['local_density'].reshape(-1)

                # linearity
                df['linearity'] = extra_info['linearity'].reshape(-1)

                # principal direction
                df['principal_dir_x'] = extra_info['principal_dir'][:, 0]
                df['principal_dir_y'] = extra_info['principal_dir'][:, 1]
                df['principal_dir_z'] = extra_info['principal_dir'][:, 2]

                # label
                if 'label' in extra_info:
                    df['label'] = extra_info['label']
                
                df.to_csv(save_path, index=False, float_format='%.6f')
                print(f"Processed and saved CSV: {save_path}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    # def process_and_save_dataset(self, input_dir, output_dir):
    #     """处理整个数据集并保存"""
    #     os.makedirs(output_dir, exist_ok=True)
    #     point_clouds, file_paths = read_all_ply_from_dir(input_dir)
        
    #     for pc, file_path in zip(point_clouds, file_paths):
    #         try:
    #             features, extra_info = self.transform(pc)
    #             base_name = os.path.basename(file_path).replace('.ply', '.npz')
    #             save_path = os.path.join(output_dir, base_name)
                
    #             # 保存特征和额外信息
    #             data_to_save = {'features': features}
    #             # extra_info now contains 'normals' and 'local_density' as well as curvature/linearity/principal_dir
    #             data_to_save.update(extra_info)
    #             # print(data_to_save['local_density'])
    #             np.savez_compressed(save_path, **data_to_save)
                
    #             print(f"Processed and saved: {save_path}")
    #         except Exception as e:
    #             print(f"Failed to process {file_path}: {e}")