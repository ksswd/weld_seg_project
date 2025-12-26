import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_npz_point_cloud_with_labels(npz_path, point_key="features"):
    """
    读取.npz文件中的点云数据及其标签。

    :param npz_path: .npz文件路径。
    :param point_key: 包含点云数据和标签的键名。
    :return: 一个元组 (point_cloud, labels)，其中：
             - point_cloud 是 (N, 3) 的点云坐标数组。
             - labels 是 (N,) 的标签数组 (0 或 1)。
    """
    npz_data = np.load(npz_path, allow_pickle=True)
    
    print("npz文件包含的键名：", list(npz_data.keys()))
    
    try:
        # 数据是 (N, 4)，包含 x, y, z, label
        data = npz_data[point_key]
    except KeyError:
        raise KeyError(f"未找到键名'{point_key}'，请从上述键名中选择正确的键。")
    
    data = np.array(data, dtype=np.float32)

    # if data.ndim != 2 or data.shape[1] != 4:
    #     raise ValueError(f"数据格式不正确！期望形状为 (N, 4)，但当前为 {data.shape}。")
    
    # 分离坐标和标签
    point_cloud = data[:, :3]
    labels = data[:, 3].astype(np.int32) # 确保标签是整数类型

    print(f"点云加载完成：共{point_cloud.shape[0]}个点。")
    print(f"标签统计：类别 0 有 {np.sum(labels == 0)} 个点，类别 1 有 {np.sum(labels == 1)} 个点。")
    
    return point_cloud, labels

def visualize_colored_point_cloud(point_cloud, labels, method="open3d"):
    """
    根据标签可视化彩色点云。

    :param point_cloud: (N, 3) 点云坐标数组。
    :param labels: (N,) 标签数组 (0 或 1)。
    :param method: "open3d" 或 "matplotlib"。
    """
    # 定义颜色映射 (示例：0 -> 蓝色, 1 -> 红色)
    # colors数组将是 (N, 3)，每个点对应一个RGB颜色
    colors = np.zeros((point_cloud.shape[0], 3))
    colors[labels == 0] = [0, 0, 1]      # 标签为0的点设置为蓝色 (B, G, R)
    colors[labels == 1] = [1, 0, 0]      # 标签为1的点设置为红色

    if method == "open3d":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors) # 应用颜色
        
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        print("\nOpen3D可视化窗口已打开。")
        print("交互操作：")
        print("  - 左键拖动：旋转视角")
        print("  - 滚轮：缩放")
        print("  - 右键拖动：平移视角")
        print("  - 按 'R' 键：重置视角")
        print("  - 按 'Q' 键：关闭窗口")
        
        o3d.visualization.draw_geometries([pcd], 
                                          window_name="Colored Point Cloud Visualization")

    elif method == "matplotlib":
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Matplotlib的scatter可以一次性处理，并根据labels使用cmap着色
        # 我们用已创建的colors数组来精确控制颜色
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                   c=colors, s=2, edgecolors="none")
        
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Colored Point Cloud by Label (0: Blue, 1: Red)")
        
        plt.show()
    else:
        raise ValueError("method仅支持'open3d'或'matplotlib'")

# -------------------------- 主函数调用 --------------------------
if __name__ == "__main__":
    # 1. 替换为你的.npz文件路径
    npz_file_path = "data/wider_aug_1_pred.npz"  # 示例："data/labeled_pcd.npz"
    
    # 2. 读取点云和标签
    # 注意：这里的键名是 "features"
    point_cloud, labels = read_npz_point_cloud_with_labels(npz_file_path, point_key="features")
    
    # 3. 可视化彩色点云
    # 推荐使用 open3d 获得更好的交互体验
    visualize_colored_point_cloud(point_cloud, labels, method="open3d")
    
    # 如果没有安装open3d，可以使用matplotlib
    # visualize_colored_point_cloud(point_cloud, labels, method="matplotlib")