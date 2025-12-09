# weld_seg_project/preprocess/geometric_feature.py 几何特征计算，法向/曲率/局部密度/线性度/主方向
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

class GeometricFeatureCalculator:
    def __init__(self, k_neighbors=20, radius_ratio=2.0):
        self.k_neighbors = k_neighbors
        self.radius_ratio = radius_ratio

    def compute_normals_and_curvature(self, pcd):
        """计算法向量和曲率"""
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.k_neighbors))
        pcd.orient_normals_consistent_tangent_plane(100)
        normals = np.asarray(pcd.normals)

        points = np.asarray(pcd.points)
        kdtree = KDTree(points)
        curvature = np.zeros((points.shape[0], 1))

        for i in range(points.shape[0]):
            _, idx = kdtree.query(points[i], k=self.k_neighbors + 1)
            neighbors = points[idx[1:]]
            cov = np.cov(neighbors, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)
            curvature[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-10)

        return normals, curvature

    def compute_local_density(self, pcd):
        """计算局部密度"""
        points = np.asarray(pcd.points)
        kdtree = KDTree(points)
        distances, _ = kdtree.query(points, k=2)
        avg_dist = np.mean(distances[:, 1])
        radius = avg_dist * self.radius_ratio

        density = np.zeros((points.shape[0], 1))
        for i in range(points.shape[0]):
            idx = kdtree.query_ball_point(points[i], radius)
            density[i] = len(idx) - 1

        density = (density - density.min()) / (density.max() - density.min() + 1e-10)
        return density

    def compute_linearity(self, pcd):
        """计算线性度"""
        points = np.asarray(pcd.points)
        kdtree = KDTree(points)
        linearity = np.zeros((points.shape[0], 1))

        for i in range(points.shape[0]):
            _, idx = kdtree.query(points[i], k=self.k_neighbors + 1)
            neighbors = points[idx[1:]]
            cov = np.cov(neighbors, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1] # 降序排列
            linearity[i] = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-10)

        return linearity

    def get_principal_direction(self, pcd):
        """计算主方向"""
        points = np.asarray(pcd.points)
        kdtree = KDTree(points)
        principal_dir = np.zeros((points.shape[0], 3))

        for i in range(points.shape[0]):
            _, idx = kdtree.query(points[i], k=self.k_neighbors + 1)
            neighbors = points[idx[1:]]
            cov = np.cov(neighbors, rowvar=False)
            eigen_vals, eigen_vecs = np.linalg.eig(cov)
            max_idx = np.argmax(eigen_vals)
            principal_dir[i] = eigen_vecs[:, max_idx]
            if principal_dir[i, 0] < 0:
                principal_dir[i] *= -1
        return principal_dir

    def calculate(self, point_cloud_np):
        """计算所有几何特征"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

        normals, curvature = self.compute_normals_and_curvature(pcd)
        local_density = self.compute_local_density(pcd)
        # print(local_density)
        linearity = self.compute_linearity(pcd)
        principal_dir = self.get_principal_direction(pcd)

        return {
            'normals': normals,
            'curvature': curvature,
            'local_density': local_density,
            'linearity': linearity,
            'principal_dir': principal_dir
        }