import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


# ===============================
# 数据增强：
# python augmentation.py \
#   --input_dir data/raw_ply \
#   --output_dir data/aug_ply \
#   --num_aug 5 \
#   --scale 缩放机制
# ===============================


# ===============================
# 工具函数
# ===============================

def normalize_normal(normals):
    norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals / norm


def random_rotation_z(xyz, normals, angle_range=(-10, 10)):
    angle = np.deg2rad(np.random.uniform(*angle_range))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    xyz = xyz @ R.T
    normals = normals @ R.T
    return xyz, normals


def random_translation(xyz, t_range=1.0):
    t = np.random.uniform(-t_range, t_range, size=(1, 3))
    return xyz + t


def random_scaling(xyz, scale_range=(0.98, 1.02)):
    scale = np.random.uniform(*scale_range)
    return xyz * scale


def gaussian_noise(xyz, sigma=0.1):
    return xyz + np.random.normal(0, sigma, size=xyz.shape)


def normal_jitter(normals, sigma=0.02):
    normals = normals + np.random.normal(0, sigma, size=normals.shape)
    return normalize_normal(normals)


def random_point_dropout(xyz, normals, rgb=None, seam=None, drop_ratio=0.1):
    """
    无 label：完全随机丢弃
    有 label：优先丢弃非焊缝点
    """
    N = len(xyz)
    keep_mask = np.ones(N, dtype=bool)

    if seam is None:
        drop_idx = np.random.choice(
            N, int(N * drop_ratio), replace=False
        )
    else:
        non_seam_idx = np.where(seam == 0)[0]
        drop_num = int(len(non_seam_idx) * drop_ratio)
        drop_idx = np.random.choice(non_seam_idx, drop_num, replace=False)

    keep_mask[drop_idx] = False

    xyz = xyz[keep_mask]
    normals = normals[keep_mask]
    if rgb is not None:
        rgb = rgb[keep_mask]
    if seam is not None:
        seam = seam[keep_mask]

    return xyz, normals, rgb, seam


def random_patch_crop(xyz, normals, rgb=None, seam=None, min_seam_ratio=0.1):
    """
    无 label：直接随机裁剪
    有 label：保证焊缝点比例
    """
    center = xyz[np.random.randint(len(xyz))]
    radius = np.random.uniform(5.0, 15.0)

    dist = np.linalg.norm(xyz - center, axis=1)
    mask = dist < radius

    if mask.sum() < 50:
        return xyz, normals, rgb, seam

    if seam is not None:
        seam_ratio = seam[mask].sum() / mask.sum()
        if seam_ratio < min_seam_ratio:
            return xyz, normals, rgb, seam

    xyz = xyz[mask]
    normals = normals[mask]
    if rgb is not None:
        rgb = rgb[mask]
    if seam is not None:
        seam = seam[mask]

    return xyz, normals, rgb, seam


# ===============================
# 主增强接口（统一入口）
# ===============================

def augment_point_cloud(
    xyz,
    normals,
    rgb=None,
    seam=None,
    use_scaling=False
):
    """
    支持：
    - 有 label
    - 无 label
    """

    # 1. 受限旋转
    xyz, normals = random_rotation_z(xyz, normals)

    # 2. 平移
    xyz = random_translation(xyz)

    # 3. 缩放（可选）
    if use_scaling:
        xyz = random_scaling(xyz)

    # 4. 高斯噪声
    xyz = gaussian_noise(xyz, sigma=0.1)

    # 5. 法向扰动
    normals = normal_jitter(normals)

    # 6. 点丢弃
    xyz, normals, rgb, seam = random_point_dropout(
        xyz, normals, rgb, seam, drop_ratio=0.1
    )

    # 7. Patch 裁剪
    xyz, normals, rgb, seam = random_patch_crop(
        xyz, normals, rgb, seam
    )

    return xyz, normals, rgb, seam


def load_ply(path, has_label):
    ply = PlyData.read(path)
    v = ply['vertex'].data

    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    normals = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float32)

    rgb = None
    if {'red', 'green', 'blue'}.issubset(v.dtype.names):
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

    seam = None
    if has_label and 'scalar_seam' in v.dtype.names:
        seam = v['scalar_seam'].astype(np.float32)

    return xyz, normals, rgb, seam


def save_ply(path, xyz, normals, rgb=None, seam=None):
    data = []
    for i in range(len(xyz)):
        row = [xyz[i, 0], xyz[i, 1], xyz[i, 2]]

        if rgb is not None:
            row += list(rgb[i])

        row += [normals[i, 0], normals[i, 1], normals[i, 2]]

        if seam is not None:
            row.append(seam[i])

        data.append(tuple(row))

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    if rgb is not None:
        dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    dtype += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

    if seam is not None:
        dtype += [('scalar_seam', 'f4')]

    vertex = np.array(data, dtype=dtype)
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(path)


# ===============================
# main (batch + N augment)
# ===============================

def main():
    parser = argparse.ArgumentParser("Batch PLY Augmentation (N times)")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_aug', type=int, default=1,
                        help='number of augmented samples per file')
    parser.add_argument('--scale', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ply_files = sorted([
        f for f in os.listdir(args.input_dir) if f.endswith('.ply')
    ])

    print(f"[INFO] Found {len(ply_files)} ply files")
    print(f"[INFO] Each file will generate {args.num_aug} samples")

    for fname in ply_files:
        in_path = os.path.join(args.input_dir, fname)
        has_label = 'label' in fname.lower()

        print(f"\n[PROCESS] {fname} | has_label={has_label}")

        # 只读一次原始数据（很重要）
        xyz0, normals0, rgb0, seam0 = load_ply(in_path, has_label)

        for i in range(args.num_aug):
            xyz = xyz0.copy()
            normals = normals0.copy()
            rgb = None if rgb0 is None else rgb0.copy()
            seam = None if seam0 is None else seam0.copy()

            xyz, normals, rgb, seam = augment_point_cloud(
                xyz,
                normals,
                rgb=rgb,
                seam=seam,
                use_scaling=args.scale
            )

            base, ext = os.path.splitext(fname)
            out_name = f"{base}_aug{i}{ext}"
            out_path = os.path.join(args.output_dir, out_name)

            save_ply(out_path, xyz, normals, rgb, seam)

            print(f"  └─ saved: {out_name}")

    print("\n[INFO] Batch augmentation finished.")


if __name__ == '__main__':
    main()
