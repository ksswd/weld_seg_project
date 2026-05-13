# weld_seg_project/preprocess/preprocess.py 预处理主函数
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometric_feature import GeometricFeatureCalculator
from utils.io_utils import read_all_ply_from_dir, read_ply_with_scalar_seam


class PointCloudPreprocessor:
    def __init__(self, config):
        self.config = config
        self.feature_calculator = GeometricFeatureCalculator(
            k_neighbors=config.K_NEIGHBORS,
            radius_ratio=config.RADIUS_RATIO,
        )
        self.mu = None
        self.sigma = None

    def _compute_raw_features(self, point_cloud_np):
        geom = self.feature_calculator.calculate(point_cloud_np)
        return {
            "x": point_cloud_np[:, 0],
            "y": point_cloud_np[:, 1],
            "z": point_cloud_np[:, 2],
            "nx": geom["normals"][:, 0],
            "ny": geom["normals"][:, 1],
            "nz": geom["normals"][:, 2],
            "curvature": geom["curvature"].reshape(-1),
            "density": geom["local_density"].reshape(-1),
            "linearity": geom["linearity"].reshape(-1),
            "principal_dir_x": geom["principal_dir"][:, 0],
            "principal_dir_y": geom["principal_dir"][:, 1],
            "principal_dir_z": geom["principal_dir"][:, 2],
        }

    def _to_feature_matrix(self, raw_data):
        return np.column_stack(
            [
                raw_data["x"],
                raw_data["y"],
                raw_data["z"],
                raw_data["nx"],
                raw_data["ny"],
                raw_data["nz"],
                raw_data["curvature"],
                raw_data["density"],
            ]
        )

    def _fit_standardizer(self, feature_matrices):
        all_features = np.vstack(feature_matrices)
        self.mu = np.mean(all_features, axis=0)
        self.sigma = np.std(all_features, axis=0)
        self.sigma[self.sigma == 0] = 1.0

    def _standardize(self, feature_matrix):
        if not bool(getattr(self.config, "PREPROCESS_STANDARDIZE", False)):
            return feature_matrix
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        return (feature_matrix - self.mu) / self.sigma

    def _try_load_labels(self, file_path, expected_n):
        try:
            info = read_ply_with_scalar_seam(file_path)
            labels = info.get("scalar_label", None)
            if labels is None:
                return None
            labels = np.asarray(labels)
            if labels.ndim > 1:
                labels = labels.reshape(-1)
            if labels.shape[0] != expected_n:
                print(f"[warn] label length mismatch for {file_path}: {labels.shape[0]} vs {expected_n}, ignore labels")
                return None
            return (labels > 0.5).astype(np.float32)
        except Exception:
            return None

    def _build_soft_labels(self, xyz, hard, radius):
        hard = np.asarray(hard, dtype=np.float32).reshape(-1)
        soft = hard.copy()

        pos_idx = np.where(hard >= 0.5)[0]
        neg_idx = np.where(hard < 0.5)[0]
        if len(pos_idx) == 0:
            return soft

        tree = cKDTree(xyz[pos_idx])
        dists, _ = tree.query(xyz[neg_idx], k=1, workers=-1)
        cand = 1.0 - (dists / max(radius, 1e-8))
        cand = np.clip(cand, 0.0, 1.0)
        cand[dists > radius] = 0.0

        soft[neg_idx] = cand.astype(np.float32)
        soft[pos_idx] = 1.0
        return soft

    def process_and_save_dataset(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        point_clouds, file_paths = read_all_ply_from_dir(input_dir)

        shuffle_points = bool(getattr(self.config, "SHUFFLE_POINTS_BEFORE_SAVE", True))
        shuffle_seed = int(getattr(self.config, "SHUFFLE_SEED", 42))
        generate_soft_label = bool(getattr(self.config, "GENERATE_SOFT_LABEL_IN_PREPROCESS", True))
        standardize = bool(getattr(self.config, "PREPROCESS_STANDARDIZE", False))

        cached = []
        feature_matrices = []
        for pc, file_path in zip(point_clouds, file_paths):
            raw_data = self._compute_raw_features(pc)
            feature_matrix = self._to_feature_matrix(raw_data)
            cached.append((pc, file_path, raw_data, feature_matrix))
            if standardize:
                feature_matrices.append(feature_matrix)

        if standardize and len(feature_matrices) > 0:
            self._fit_standardizer(feature_matrices)

        for pc, file_path, raw_data, feature_matrix in cached:
            try:
                features = self._standardize(feature_matrix)
                labels = self._try_load_labels(file_path, expected_n=features.shape[0])

                if shuffle_points:
                    rng = np.random.default_rng(shuffle_seed)
                    perm = rng.permutation(features.shape[0])
                    features = features[perm]
                    raw_data = {k: v[perm] for k, v in raw_data.items()}
                    if labels is not None:
                        labels = labels[perm]

                base_name = os.path.basename(file_path).replace(".ply", ".csv")
                if labels is not None and "_label_" not in base_name:
                    stem = base_name[:-4]
                    base_name = f"{stem}_label_auto.csv"

                save_path = os.path.join(output_dir, base_name)
                df = pd.DataFrame(features, columns=["x", "y", "z", "nx", "ny", "nz", "curvature", "density"])
                df["linearity"] = raw_data["linearity"].flatten()
                df["principal_dir_x"] = raw_data["principal_dir_x"]
                df["principal_dir_y"] = raw_data["principal_dir_y"]
                df["principal_dir_z"] = raw_data["principal_dir_z"]

                if labels is not None:
                    df["label"] = labels.astype(np.float32)
                    if generate_soft_label:
                        soft_radius = float(getattr(self.config, "SOFT_LABEL_RADIUS", 0.2))
                        df["label_soft"] = self._build_soft_labels(xyz=pc, hard=labels, radius=soft_radius)

                df.to_csv(save_path, index=False, float_format="%.6f")
                print(f"Processed and saved: {save_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
