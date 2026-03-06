#!/usr/bin/env python3
"""
使用OpenCV的CLAHE方法增强点云特征质量

CLAHE (Contrast Limited Adaptive Histogram Equalization) 可以增强点云特征的对比度，
特别适用于曲率、密度等几何特征，使细节更明显。

Usage:
    python scripts/clahe_enhance.py data/processed_csv/lap1_aug0.csv --features curvature_raw density_raw
    python scripts/clahe_enhance.py data/processed_csv --out data/enhanced --features curvature_raw

    # PLY输入/输出（增强顶点属性，如RGB或intensity）
    python scripts/clahe_enhance.py data/predictions/recon_vis/lap1_aug0_recon_vis_mask.ply --out data/enhanced_ply
    python scripts/clahe_enhance.py input.ply --features intensity --out out_dir
"""
import argparse
import os
import sys

import numpy as np
import cv2

# PLY I/O via plyfile (avoid open3d)
try:
    from plyfile import PlyData, PlyElement
except Exception:
    PlyData = None
    PlyElement = None


def project_to_2d(points, feature_values, grid_size=512):
    """
    将3D点云投影到2D网格，用于CLAHE处理
    
    Args:
        points: (N, 3) 点云坐标
        feature_values: (N,) 要增强的特征值
        grid_size: 2D网格大小
    
    Returns:
        grid: (grid_size, grid_size) 2D特征图
        point_to_grid: (N, 2) 每个点在网格中的位置
    """
    # 归一化点云到[0, 1]范围
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    points_range = points_max - points_min
    points_range = np.where(points_range < 1e-8, 1.0, points_range)
    points_norm = (points - points_min) / points_range
    
    # 投影到XY平面（也可以选择其他投影方式）
    grid_coords = (points_norm[:, :2] * (grid_size - 1)).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # 创建2D网格，使用最近邻或平均
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for i in range(len(points)):
        x, y = grid_coords[i]
        grid[y, x] += feature_values[i]
        count[y, x] += 1
    
    # 计算平均值
    mask = count > 0
    grid[mask] = grid[mask] / count[mask]
    
    return grid, grid_coords


def apply_clahe_to_feature(feature_values, points, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对点云特征应用CLAHE增强
    
    Args:
        feature_values: (N,) 特征值数组
        points: (N, 3) 点云坐标
        clip_limit: CLAHE的对比度限制
        tile_grid_size: CLAHE的网格大小
    
    Returns:
        enhanced_values: (N,) 增强后的特征值
    """
    # 归一化特征值到[0, 255]用于CLAHE
    f_min = feature_values.min()
    f_max = feature_values.max()
    f_range = f_max - f_min
    if f_range < 1e-8:
        return feature_values  # 如果特征值没有变化，直接返回
    
    # 归一化到[0, 1]
    feature_norm = (feature_values - f_min) / f_range
    
    # 方法1: 投影到2D网格后应用CLAHE（推荐，保持空间关系）
    grid_size = 512
    grid, grid_coords = project_to_2d(points, feature_norm, grid_size)
    
    # 转换为uint8
    grid_uint8 = (grid * 255).astype(np.uint8)
    
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    grid_enhanced = clahe.apply(grid_uint8)
    
    # 转换回float并映射回点云
    grid_enhanced_float = grid_enhanced.astype(np.float32) / 255.0
    
    # 将增强后的值映射回每个点
    enhanced_values = np.zeros_like(feature_values)
    for i in range(len(points)):
        x, y = grid_coords[i]
        enhanced_values[i] = grid_enhanced_float[y, x]
    
    # 反归一化到原始范围
    enhanced_values = enhanced_values * f_range + f_min
    
    return enhanced_values


def read_csv_simple(csv_path):
    """简单的CSV读取，不依赖pandas"""
    data = {}
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip()
        header = [h.strip() for h in header_line.split(',')]
        rows = []
        for line in f:
            if line.strip():
                values = [v.strip() for v in line.strip().split(',')]
                rows.append(values)
    
    # 转换为numpy数组
    n_rows = len(rows)
    for i, col in enumerate(header):
        try:
            col_data = []
            for row in rows:
                if i < len(row) and row[i]:
                    try:
                        col_data.append(float(row[i]))
                    except ValueError:
                        col_data.append(0.0)
                else:
                    col_data.append(0.0)
            data[col] = np.array(col_data, dtype=np.float32)
        except Exception as e:
            # 如果转换失败，使用字符串
            data[col] = np.array([row[i] if i < len(row) else '' for row in rows])
    
    return data, header


def enhance_csv_features(csv_path, out_path, features_to_enhance, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    增强CSV文件中的点云特征
    
    Args:
        csv_path: 输入CSV文件路径
        out_path: 输出CSV文件路径
        features_to_enhance: 要增强的特征列表，如 ['curvature_raw', 'local_density_raw']
        clip_limit: CLAHE对比度限制
        tile_grid_size: CLAHE网格大小
    """
    print(f"Processing: {csv_path}")
    
    # 读取CSV（使用numpy方式，避免pandas依赖）
    data_dict, header = read_csv_simple(csv_path)
    
    # 检查是否有pandas可用（但通常不可用）
    if False:  # 暂时禁用pandas路径
        print(f"  Loaded {len(df)} points")
        # 加载点云数据
        if load_features_from_csv:
            sample = load_features_from_csv(csv_path)
            points = sample['coordinate']
        else:
            points = np.column_stack([df['x'].values, df['y'].values, df['z'].values])
        
        # 创建输出DataFrame的副本
        df_out = df.copy()
        
        # 对每个特征应用CLAHE
        for feature_name in features_to_enhance:
            if feature_name not in df.columns:
                print(f"  Warning: Feature '{feature_name}' not found, skipping")
                continue
            
            print(f"  Enhancing {feature_name}...")
            feature_values = df[feature_name].values.astype(np.float32)
            
            # 应用CLAHE
            enhanced_values = apply_clahe_to_feature(
                feature_values, 
                points, 
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size
            )
            
            # 更新DataFrame
            df_out[feature_name] = enhanced_values
            
            # 打印统计信息
            print(f"    Original range: [{feature_values.min():.6f}, {feature_values.max():.6f}]")
            print(f"    Enhanced range: [{enhanced_values.min():.6f}, {enhanced_values.max():.6f}]")
            print(f"    Original std: {feature_values.std():.6f}")
            print(f"    Enhanced std: {enhanced_values.std():.6f}")
        
        # 保存增强后的CSV
        df_out.to_csv(out_path, index=False)
        print(f"  Saved enhanced CSV: {out_path}")
    else:
        # 使用numpy方式
        print(f"  Loaded {len(data_dict[header[0]])} points")
        points = np.column_stack([data_dict['x'], data_dict['y'], data_dict['z']])
        
        # 对每个特征应用CLAHE
        for feature_name in features_to_enhance:
            if feature_name not in data_dict:
                print(f"  Warning: Feature '{feature_name}' not found, skipping")
                continue
            
            print(f"  Enhancing {feature_name}...")
            feature_values = data_dict[feature_name].astype(np.float32)
            
            # 应用CLAHE
            enhanced_values = apply_clahe_to_feature(
                feature_values, 
                points, 
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size
            )
            
            # 更新数据
            data_dict[feature_name] = enhanced_values
            
            # 打印统计信息
            print(f"    Original range: [{feature_values.min():.6f}, {feature_values.max():.6f}]")
            print(f"    Enhanced range: [{enhanced_values.min():.6f}, {enhanced_values.max():.6f}]")
            print(f"    Original std: {feature_values.std():.6f}")
            print(f"    Enhanced std: {enhanced_values.std():.6f}")
        
        # 保存增强后的CSV
        with open(out_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for i in range(len(data_dict[header[0]])):
                row = [str(data_dict[col][i]) for col in header]
                f.write(','.join(row) + '\n')
        print(f"  Saved enhanced CSV: {out_path}")


def enhance_ply_vertices(
    ply_path: str,
    out_path: str,
    features_to_enhance,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
    text: bool = False,
):
    """
    对PLY点云顶点属性做CLAHE增强（不依赖open3d）。

    常见属性：
      - RGB: red/green/blue (通常是uint8)
      - intensity (float或uint16等)
    """
    if PlyData is None or PlyElement is None:
        raise ImportError("plyfile not available. Please install: pip install plyfile")

    print(f"Processing: {ply_path}")
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError("PLY must contain 'vertex' element.")

    v = ply["vertex"].data
    names = v.dtype.names or ()
    for req in ("x", "y", "z"):
        if req not in names:
            raise ValueError("PLY vertex must contain x,y,z properties.")

    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    n = points.shape[0]
    print(f"  Loaded {n} points")

    # Copy vertex data to preserve all properties
    v_out = np.empty(v.shape, dtype=v.dtype)
    v_out[:] = v

    for feature_name in features_to_enhance:
        if feature_name not in names:
            print(f"  Warning: Feature '{feature_name}' not found in PLY vertex properties, skipping")
            continue

        print(f"  Enhancing {feature_name}...")
        vals = v[feature_name].astype(np.float32)

        enhanced = apply_clahe_to_feature(
            vals,
            points,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

        # Cast back to original dtype
        dtype = v.dtype.fields[feature_name][0]
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            enhanced_cast = np.clip(np.rint(enhanced), info.min, info.max).astype(dtype)
        else:
            enhanced_cast = enhanced.astype(dtype)

        v_out[feature_name] = enhanced_cast

        print(f"    Original range: [{vals.min():.6f}, {vals.max():.6f}]")
        print(f"    Enhanced range: [{enhanced.min():.6f}, {enhanced.max():.6f}]")
        print(f"    Original std: {vals.std():.6f}")
        print(f"    Enhanced std: {enhanced.std():.6f}")

    # Rebuild PLY, keep other elements unchanged
    elements = []
    for el in ply.elements:
        if el.name == "vertex":
            elements.append(PlyElement.describe(v_out, "vertex"))
        else:
            elements.append(el)
    out_ply = PlyData(elements, text=bool(text))
    # preserve comments/obj_info if present
    out_ply.comments = list(getattr(ply, "comments", []))
    out_ply.obj_info = list(getattr(ply, "obj_info", []))
    out_ply.write(out_path)
    print(f"  Saved enhanced PLY: {out_path}")


def _default_ply_features(ply_path: str):
    """Infer default features to enhance for a PLY file."""
    if PlyData is None:
        raise ImportError("plyfile not available. Please install: pip install plyfile")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = set(v.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        return ["red", "green", "blue"]
    if "intensity" in names:
        return ["intensity"]
    raise ValueError("No default PLY feature found. Please pass --features <prop...> (e.g., red green blue).")


def main():
    parser = argparse.ArgumentParser(description='Enhance point cloud features using CLAHE')
    parser.add_argument('input', help='Input CSV/PLY file or directory')
    parser.add_argument('--out', help='Output directory (default: same as input)')
    parser.add_argument('--features', nargs='+', 
                       default=None,
                       help=('Features/properties to enhance. '
                             'CSV example: curvature_raw local_density_raw linearity. '
                             'PLY example: red green blue or intensity. '
                             'If omitted: CSV uses curvature_raw/local_density_raw/linearity; '
                             'PLY uses red/green/blue if present else intensity if present.'))
    parser.add_argument('--clip_limit', type=float, default=2.0,
                       help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[8, 8],
                       help='CLAHE tile grid size (default: 8 8)')
    parser.add_argument('--suffix', default='_clahe',
                       help='Suffix for output files (default: _clahe)')
    parser.add_argument('--ply_text', action='store_true',
                        help='Write PLY as ASCII (text) instead of binary (default: binary)')
    args = parser.parse_args()
    
    # 确定输入文件
    input_paths = []
    if os.path.isdir(args.input):
        for f in sorted(os.listdir(args.input)):
            if (f.endswith('.csv') or f.endswith('.ply')) and '_pred' not in f and '_clahe' not in f:
                input_paths.append(os.path.join(args.input, f))
    elif os.path.isfile(args.input):
        input_paths = [args.input]
    else:
        raise SystemExit(f"Input path not found: {args.input}")
    
    if not input_paths:
        raise SystemExit("No CSV/PLY files found")
    
    # 确定输出目录
    if args.out:
        out_dir = args.out
    elif os.path.isdir(args.input):
        out_dir = args.input
    else:
        out_dir = os.path.dirname(args.input)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Found {len(input_paths)} file(s) to process")
    print(f"CLAHE parameters: clip_limit={args.clip_limit}, tile_size={args.tile_size}")
    print()
    
    # 处理每个文件
    for in_path in input_paths:
        ext = os.path.splitext(in_path)[1].lower()
        base_name = os.path.basename(in_path).replace(ext, '')
        out_path = os.path.join(out_dir, f"{base_name}{args.suffix}{ext}")
        
        try:
            if ext == ".csv":
                features = args.features if args.features is not None else ["curvature_raw", "local_density_raw", "linearity"]
                print(f"Features to enhance: {features}")
                enhance_csv_features(
                    in_path,
                    out_path,
                    features,
                    clip_limit=args.clip_limit,
                    tile_grid_size=tuple(args.tile_size),
                )
            elif ext == ".ply":
                features = args.features if args.features is not None else _default_ply_features(in_path)
                print(f"Features to enhance: {features}")
                enhance_ply_vertices(
                    in_path,
                    out_path,
                    features,
                    clip_limit=args.clip_limit,
                    tile_grid_size=tuple(args.tile_size),
                    text=args.ply_text,
                )
            else:
                print(f"Skipping unsupported file type: {in_path}")
        except Exception as e:
            print(f"Error processing {in_path}: {e}")
            continue
    
    print("\nEnhancement complete!")


if __name__ == "__main__":
    main()
