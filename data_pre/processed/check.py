import numpy as np

# !!! 将 'your_file_name.npz' 替换为你 data/processed/ 目录下的一个真实文件名 !!!
file_path = 'data/processed/000.npz' 

try:
    data = np.load(file_path)
    print(f"成功加载文件: {file_path}")
    print("\n--- 文件中包含的键 (Keys) ---")
    print(data.files) # 这会打印出文件里所有的键

    # 检查 'local_density' 是否存在
    if 'local_density' in data.files:
        print("\n✅ 'local_density' 键存在!")
        # 打印其形状以确认
        print(f"   'local_density' 的形状: {data['local_density'].shape}")
    else:
        print("\n❌ 'local_density' 键不存在!")

except FileNotFoundError:
    print(f"错误: 文件未找到于 '{file_path}'。请确保路径和文件名正确。")
except Exception as e:
    print(f"发生未知错误: {e}")