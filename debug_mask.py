# debug_mask_scan.py
import numpy as np
import torch
import os
from utils.config import Config
from train.mask_strategy import HighCurvatureMasker

import open3d as o3d
import numpy as np
import torch
import os

def visualize_mask_with_open3d(npz_file, mask, window_name="Mask Visualization"):
    """
    使用Open3D可视化点云，Mask掉的点标红，保留的点标蓝
    Args:
        npz_file: npz文件路径（包含features字段，前3列为xyz坐标）
        mask: torch张量，mask掩码（1=保留，0=Mask掉），形状为(1, N)或(N,)
        window_name: 可视化窗口名称
    """
    # 加载npz文件
    data = np.load(npz_file)
    
    # 提取点云坐标（features前3列）
    features = data['features']
    points = features[:, :3].astype(np.float32)  # 仅取xyz坐标
    
    # 处理mask张量
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # 展平mask（兼容(1, N)或(N,)形状）
    mask = mask.flatten()
    
    # 校验mask和点云数量匹配
    if len(mask) != len(points):
        raise ValueError(f"Mask数量({len(mask)})与点云数量({len(points)})不匹配！")
    
    # 初始化Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点的颜色：保留的点(蓝)，Mask的点(红)
    colors = np.zeros((len(points), 3))  # 初始化全黑
    colors[mask == 1] = [0, 0, 1]       # 保留的点：蓝色
    colors[mask == 0] = [1, 0, 0]       # Mask的点：红色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    vis.add_geometry(pcd)
    
    # 设置视角（自动适配点云范围）
    view_control = vis.get_view_control()
    view_control.set_front([0.5, 0.5, 1.0])  # 相机前视方向
    view_control.set_lookat(pcd.get_center())  # 看向点云中心
    view_control.set_up([0, 1, 0])  # 相机上方向
    view_control.set_zoom(0.8)
    
    # 渲染并显示
    vis.poll_events()
    vis.update_renderer()
    
    
    # 等待用户关闭窗口
    print("按ESC键关闭可视化窗口...")
    vis.run()
    vis.destroy_window()

# files = sorted([os.path.join(Config.PROCESSED_DATA_DIR, f) for f in os.listdir(Config.PROCESSED_DATA_DIR) if f.endswith('.npz')])
f = os.path.join('data/lap1_aug_1_pred.npz')
masker = HighCurvatureMasker(mask_ratio=getattr(Config, 'MASK_RATIO', 0.1))
d = np.load(f)
# print(d['features'].shape)
cur = (d['features'].astype(np.float32)[:, 2:3]) # 假设curvature在features的第6列
print(cur.shape)
m = masker.generate_mask(cur)
if m.dim()==3 and m.size(-1)==1:
    m = m.squeeze(-1)
visualize_mask_with_open3d(
        npz_file=f,
        mask=m,
        window_name=f"File: {os.path.basename(f)}"
    )


