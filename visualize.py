# visualize_predictions.py

import os
import numpy as np

prediction_file_folder = 'data/predictions/new'  # 预测文件夹路径

# --- 加载和可视化 ---
try:
    # 加载预测文件
    for name in os.listdir(prediction_file_folder):
        if name.endswith('_pred.npz'):
            prediction_file_path = os.path.join(prediction_file_folder, name)
            print(f"正在加载预测文件: {prediction_file_path}")
            pred_data = np.load(prediction_file_path)
            pred_array = pred_data['features'] # 形状为 (N, 4)
             # 可视化
            print(f"--- 可视化预测结果: {name} ---")
            print(f"总点数: {len(pred_array)}")
            print(f"预测为焊缝的点数: {len(pred_array[pred_array[:, 3] == 1])}")
            print(f"预测为非焊缝的点数: {len(pred_array[pred_array[:, 3] == 0])}")
    else:
        raise FileNotFoundError("未找到以 '_pred.npz' 结尾的预测文件。请确保已运行测试脚本。")

except FileNotFoundError:
    print(f"错误: 预测文件未找到于 '{prediction_file_path}'。")
    print("请确保已成功运行 'python main.py --mode test'。")
except Exception as e:
    print(f"发生未知错误: {e}")