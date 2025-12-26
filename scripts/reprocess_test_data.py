#!/usr/bin/env python
"""
Reprocess test data using normalization parameters from training data.
This ensures test data is normalized consistently with training data.
"""
import sys
sys.path.insert(0, '.')

from utils.config import Config
from preprocess.preprocess import PointCloudPreprocessor

def reprocess_test_data():
    config = Config()

    print("=" * 80)
    print("Reprocessing Test Data with Training Normalization")
    print("=" * 80)

    # Create preprocessor
    preprocessor = PointCloudPreprocessor(config)

    # Load normalization params from training data
    print("\n1. Loading normalization parameters from training...")
    try:
        preprocessor.load_params("normalization_params.npz")
    except FileNotFoundError:
        print("ERROR: normalization_params.npz not found!")
        print("Please run preprocessing on training data first:")
        print("  python main.py --mode preprocess")
        return

    # Process test data directory
    import os
    test_raw_dir = "data/test/new"  # Raw test data (PLY files)
    test_processed_dir = "data/test/processed"  # Processed test data (NPZ files)

    if not os.path.exists(test_raw_dir):
        print(f"\nERROR: Test data directory not found: {test_raw_dir}")
        print("Expected directory structure:")
        print("  data/test/new/           <- Raw PLY files")
        print("  data/test/processed/     <- Will be created for NPZ files")
        return

    print(f"\n2. Processing test data...")
    print(f"   Input dir:  {test_raw_dir}")
    print(f"   Output dir: {test_processed_dir}")

    # Check if there are actually npz files (already processed)
    npz_files = [f for f in os.listdir(test_raw_dir) if f.endswith('.npz')]

    if npz_files:
        print(f"\n⚠️  Found {len(npz_files)} .npz files in {test_raw_dir}")
        print("These appear to be already-processed files with WRONG normalization.")
        print("\nOptions:")
        print("  1. If you have original .ply files, move them to data/test/raw/")
        print("  2. Or manually fix by re-preprocessing from original source")

        # Let's still try to reprocess if there are PLY files
        ply_files = [f for f in os.listdir(test_raw_dir) if f.endswith('.ply')]
        if not ply_files:
            print("\nNo .ply files found. Cannot reprocess.")
            print("Please provide original point cloud files in .ply format.")
            return
        else:
            print(f"\nFound {len(ply_files)} .ply files. Processing those...")

    # Process and save with correct normalization
    preprocessor.process_and_save_dataset(test_raw_dir, test_processed_dir)

    print("\n" + "=" * 80)
    print("✓ Test data reprocessed successfully!")
    print(f"✓ Processed files saved to: {test_processed_dir}")
    print("\nNext steps:")
    print("  1. Update config.py to use TEST_DATA_DIR = 'data/test/processed'")
    print("  2. Run inference: python main.py --mode test")
    print("=" * 80)

if __name__ == "__main__":
    reprocess_test_data()
