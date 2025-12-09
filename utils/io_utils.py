# weld_seg_project/utils/io_utils.py 输入输出相关
import os
import numpy as np
import open3d as o3d
from glob import glob

def read_ply_file(file_path):
    """读取单个PLY文件并返回点云坐标"""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def read_all_ply_from_dir(dir_path):
    """读取目录中所有PLY文件"""
    ply_files = glob(os.path.join(dir_path, "*.ply"))
    point_clouds = []
    for file in ply_files:
        pc = read_ply_file(file)
        point_clouds.append(pc)
    return point_clouds, ply_files

def save_features_to_npz(features, file_path):
    """将预处理后的特征保存为NPZ文件"""
    np.savez_compressed(file_path, features=features)

def load_features_from_npz(file_path):
    """从NPZ文件加载特征"""
    data = np.load(file_path)
    return data['features']