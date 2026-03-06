#!/usr/bin/env python3
"""
Visualize reconstruction results from CSV files.

For each reconstruction CSV file, this script creates multiple visualizations:
1. Point cloud colored by reconstruction error for each channel (curvature, x, y, z)
2. Comparison between GT and predicted values
3. Mask vs non-mask region analysis

The reconstruction targets are:
  - curvature_raw: curvature value (log-space normalized during training)
  - x, y, z: 3D coordinates (min-max normalized during training)

Usage:
    python scripts/visualize_recon.py data/predictions/recon_vis/lap1_aug0_recon_vis.csv
    python scripts/visualize_recon.py data/predictions/recon_vis --out data/vis_recon
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

def create_error_colored_pcd(points, errors, title, cmap='viridis', vmin=None, vmax=None):
    """Create point cloud colored by error values with memory-safe processing."""
    try:
        # Validate inputs
        if len(points) != len(errors):
            raise ValueError(f"Points ({len(points)}) and errors ({len(errors)}) length mismatch")

        if len(points) == 0:
            raise ValueError("Empty point cloud")

        print(f"    Creating point cloud with {len(points)} points...")

        # Validate points shape
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3) array, got shape {points.shape}")

        # Validate errors shape
        if errors.ndim != 1:
            raise ValueError(f"Errors must be 1D array, got shape {errors.shape}")

        # For very large point clouds, use simpler processing
        if len(points) > 10000:
            print("    Large point cloud detected, using memory-efficient processing")

        # Check for NaN/Inf values
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print("    Warning: NaN/Inf detected in points, cleaning...")
            valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
            points = points[valid_mask]
            errors = errors[valid_mask]
            if len(points) == 0:
                raise ValueError("All points were invalid after cleaning")

        if np.any(np.isnan(errors)) or np.any(np.isinf(errors)):
            print("    Warning: NaN/Inf detected in errors, cleaning...")
            valid_mask = ~(np.isnan(errors) | np.isinf(errors))
            points = points[valid_mask]
            errors = errors[valid_mask]
            if len(points) == 0:
                raise ValueError("All errors were invalid after cleaning")

        # Calculate percentiles safely
        if vmin is None:
            vmin = float(np.nanpercentile(errors, 5))  # Robust min (5th percentile)
        if vmax is None:
            vmax = float(np.nanpercentile(errors, 95))  # Robust max (95th percentile)

        # Ensure valid range
        if vmax <= vmin:
            vmax = vmin + 1e-8

        # Normalize errors to [0, 1] for colormap
        errors_norm = np.clip((errors - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Apply colormap safely
        try:
            cmap_obj = plt.get_cmap(cmap)
            colors = cmap_obj(errors_norm)[:, :3]  # RGB values
            # Ensure colors are in [0, 1] range
            colors = np.clip(colors, 0, 1)
        except Exception as e:
            print(f"    Error applying colormap: {e}")
            # Fallback to grayscale
            colors = np.stack([errors_norm, errors_norm, errors_norm], axis=1)

        # Memory-efficient conversion for large point clouds
        try:
            # Ensure contiguous arrays
            points = np.ascontiguousarray(points, dtype=np.float64)
            colors = np.ascontiguousarray(colors, dtype=np.float64)

            # Validate array shapes before creating point cloud
            assert points.shape[1] == 3, f"Points must have 3 columns, got {points.shape[1]}"
            assert colors.shape[1] == 3, f"Colors must have 3 columns, got {colors.shape[1]}"
            assert points.shape[0] == colors.shape[0], f"Points and colors must have same length"

            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        except Exception as e:
            print(f"    Error creating Open3D point cloud: {e}")
            print(f"    Points shape: {points.shape}, dtype: {points.dtype}")
            print(f"    Colors shape: {colors.shape}, dtype: {colors.dtype}")
            import gc
            gc.collect()
            raise

        print(f"    Point cloud created successfully")
        return pcd, vmin, vmax

    except Exception as e:
        print(f"    Error creating point cloud: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_comparison_plot(df, channel_name, save_path=None):
    """Create comparison plot between GT and predicted values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{channel_name.upper()} Reconstruction Analysis', fontsize=16)

    # Channel-specific column names
    gt_col = f'gt_{channel_name}'
    pred_col = f'pred_{channel_name}'
    err_col = f'abs_err_{channel_name}'

    # 1. Scatter plot: GT vs Predicted
    axes[0,0].scatter(df[gt_col], df[pred_col], alpha=0.6, s=1)
    min_val = min(df[gt_col].min(), df[pred_col].min())
    max_val = max(df[gt_col].max(), df[pred_col].max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0,0].set_xlabel('Ground Truth')
    axes[0,0].set_ylabel('Prediction')
    axes[0,0].set_title('GT vs Prediction')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Error distribution
    axes[0,1].hist(df[err_col], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(df[err_col].mean(), color='red', linestyle='--',
                      label=f'Mean: {df[err_col].mean():.4f}')
    axes[0,1].axvline(df[err_col].median(), color='orange', linestyle='--',
                      label=f'Median: {df[err_col].median():.4f}')
    axes[0,1].set_xlabel('Absolute Error')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Error Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Mask vs Non-mask error comparison
    mask_errors = df[df['mask'] == 1][err_col]
    nonmask_errors = df[df['mask'] == 0][err_col]

    axes[1,0].hist([mask_errors, nonmask_errors], bins=30, alpha=0.7,
                   label=['Masked', 'Non-masked'], edgecolor='black')
    axes[1,0].axvline(mask_errors.mean(), color='blue', linestyle='--',
                      label=f'Masked Mean: {mask_errors.mean():.4f}')
    axes[1,0].axvline(nonmask_errors.mean(), color='orange', linestyle='--',
                      label=f'Non-masked Mean: {nonmask_errors.mean():.4f}')
    axes[1,0].set_xlabel('Absolute Error')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Mask vs Non-mask Error')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Error vs GT value (check if error correlates with magnitude)
    axes[1,1].scatter(df[gt_col], df[err_col], alpha=0.6, s=1)
    axes[1,1].set_xlabel('Ground Truth Value')
    axes[1,1].set_ylabel('Absolute Error')
    axes[1,1].set_title('Error vs GT Magnitude')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved comparison plot: {save_path}')
        plt.close()
    else:
        plt.show()

def visualize_recon_csv(csv_path, out_dir=None, max_points=0, subsample_large=False, skip_ply=False):
    """Main function to visualize reconstruction results.
    
    Args:
        csv_path: Path to reconstruction CSV file
        out_dir: Output directory for visualizations
        max_points: Maximum points to process (0 = no limit)
        subsample_large: Automatically subsample large point clouds
        skip_ply: Skip PLY file generation (only create analysis plots)
    """
    try:
        print(f"Loading CSV: {csv_path}")
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

        base_name = os.path.basename(csv_path).replace('.csv', '')

        if out_dir is None:
            out_dir = os.path.dirname(csv_path)
        os.makedirs(out_dir, exist_ok=True)

        # Extract point coordinates with validation
        required_cols = ['x', 'y', 'z']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        points = df[['x', 'y', 'z']].values.astype(np.float64)
        print(f"Point coordinates shape: {points.shape}")

        # Check for invalid coordinates
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print("Warning: NaN/Inf detected in coordinates, will be cleaned during processing")

        original_count = len(df)
        print(f"Processing {base_name}: {original_count} points")

        # Handle large point clouds
        effective_max_points = max_points if max_points > 0 else (5000 if subsample_large else 0)

        if effective_max_points > 0 and len(df) > effective_max_points:
            print(f"Subsampling from {len(df)} to {effective_max_points} points")
            np.random.seed(42)  # For reproducible subsampling
            indices = np.random.choice(len(df), size=effective_max_points, replace=False)
            df = df.iloc[indices].reset_index(drop=True)
            points = points[indices]
            print(f"After subsampling: {len(df)} points")

        # Memory check for large point clouds
        if len(df) > 5000:
            print(f"Warning: Large point cloud ({len(df)} points) detected.")
            print("This may cause memory issues. Consider using --subsample_large or --max_points")
            print("You can also try using a smaller batch size or downsampling.")

    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        raise

    # Define channels to visualize (curvature + xyz coordinates)
    channels = ['curvature_raw', 'x', 'y', 'z']

    # Create visualizations for each channel
    for channel in channels:
        err_col = f'abs_err_{channel}'
        if err_col not in df.columns:
            print(f"Warning: {err_col} not found, skipping {channel}")
            continue

        print(f"  Visualizing {channel}...")

        # 1. Create error-colored point cloud
        # Use a copy of points to avoid modifying the original
        channel_points = points.copy()
        errors = df[err_col].values.astype(np.float64)
        print(f"    Processing {len(errors)} error values for {channel}")

        # Ensure points and errors are aligned
        if len(channel_points) != len(errors):
            min_len = min(len(channel_points), len(errors))
            channel_points = channel_points[:min_len]
            errors = errors[:min_len]
            print(f"    Warning: Aligned to {min_len} points")

        # Create PLY file (skip if requested or if it fails)
        if not skip_ply:
            try:
                pcd, vmin, vmax = create_error_colored_pcd(channel_points, errors, f'{channel} Error')
                
                ply_path = os.path.join(out_dir, f'{base_name}_{channel}_error.ply')
                print(f"    Saving PLY to: {ply_path}")
                try:
                    success = o3d.io.write_point_cloud(ply_path, pcd)
                    if success:
                        print(f"    Saved error-colored PLY: {ply_path}")
                        print(f"    Error range: [{vmin:.6f}, {vmax:.6f}]")
                    else:
                        print(f"    Failed to save PLY: {ply_path}")
                except Exception as e:
                    print(f"    Error saving PLY file: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"    Failed to create point cloud for {channel}: {e}")
                print(f"    Skipping PLY file for {channel}, continuing with analysis plot...")
        else:
            print(f"    Skipping PLY generation (--skip_ply flag)")

        # 2. Create comparison plot
        try:
            plot_path = os.path.join(out_dir, f'{base_name}_{channel}_analysis.png')
            create_comparison_plot(df, channel, plot_path)
        except Exception as e:
            print(f"    Error creating analysis plot for {channel}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Create mask visualization (skip if --skip_ply flag is set)
    if not skip_ply:
        try:
            print("  Creating mask visualization...")
            # Ensure points are valid
            valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
            if valid_mask.sum() == 0:
                print("  Warning: No valid points for mask visualization, skipping")
            else:
                points_valid = points[valid_mask]
                mask_valid = df['mask'].values[valid_mask]
                
                mask_colors = np.full((len(points_valid), 3), [0.7, 0.7, 0.7], dtype=np.float64)  # Gray for non-masked
                mask_colors[mask_valid == 1] = [1.0, 0.0, 0.0]  # Red for masked

                mask_pcd = o3d.geometry.PointCloud()
                mask_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points_valid, dtype=np.float64))
                mask_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(mask_colors, dtype=np.float64))

                mask_ply_path = os.path.join(out_dir, f'{base_name}_mask.ply')
                success = o3d.io.write_point_cloud(mask_ply_path, mask_pcd)
                if success:
                    print(f"  Saved mask visualization: {mask_ply_path}")
                else:
                    print(f"  Failed to save mask PLY: {mask_ply_path}")
        except Exception as e:
            print(f"  Error creating mask visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping mask visualization (--skip_ply flag)")

    # 4. Print summary statistics
    print(f"\nSummary for {base_name}:")
    print(f"  Total points: {len(df)}")
    print(f"  Masked points: {df['mask'].sum()} ({df['mask'].mean()*100:.1f}%)")

    for channel in channels:
        err_col = f'abs_err_{channel}'
        if err_col in df.columns:
            mask_errors = df[df['mask'] == 1][err_col]
            nonmask_errors = df[df['mask'] == 0][err_col]

            print(f"\n  {channel.upper()} Errors:")
            print(f"    Overall - Mean: {df[err_col].mean():.6f}, Median: {df[err_col].median():.6f}")
            print(f"    Masked - Mean: {mask_errors.mean():.6f}, Median: {mask_errors.median():.6f}")
            print(f"    Non-masked - Mean: {nonmask_errors.mean():.6f}, Median: {nonmask_errors.median():.6f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize reconstruction results')
    parser.add_argument('path', help='CSV file or directory with CSV files')
    parser.add_argument('--out', help='Output directory for visualizations', default=None)
    parser.add_argument('--max_points', type=int, default=0,
                       help='Maximum points to process (0 = no limit, will subsample large clouds)')
    parser.add_argument('--subsample_large', action='store_true',
                       help='Automatically subsample large point clouds (>5000 points) to avoid memory issues')
    parser.add_argument('--skip_ply', action='store_true',
                       help='Skip PLY file generation (only create analysis plots). Useful if Open3D causes segfaults.')
    args = parser.parse_args()

    # Find CSV files
    paths = []
    if os.path.isdir(args.path):
        for f in sorted(os.listdir(args.path)):
            if f.endswith('_recon_vis.csv'):
                paths.append(os.path.join(args.path, f))
    elif os.path.isfile(args.path) and args.path.endswith('.csv'):
        paths = [args.path]
    else:
        raise SystemExit('Please provide a CSV file or directory containing CSV files')

    if not paths:
        raise SystemExit('No CSV files found')

    out_dir = args.out
    if out_dir is None and os.path.isdir(args.path):
        out_dir = args.path

    print(f"Found {len(paths)} CSV file(s) to process")

    for csv_path in paths:
        try:
            visualize_recon_csv(csv_path, out_dir, args.max_points, args.subsample_large, args.skip_ply)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nVisualization complete!")
    print("\nGenerated files:")
    print("- *_curvature_raw_error.ply: Point cloud colored by curvature reconstruction error")
    print("- *_x_error.ply, *_y_error.ply, *_z_error.ply: Point clouds colored by coordinate errors")
    print("- *_curvature_raw_analysis.png, *_x_analysis.png, etc.: Statistical analysis plots")
    print("- *_mask.ply: Mask visualization (red=masked, gray=non-masked)")

if __name__ == "__main__":
    main()
