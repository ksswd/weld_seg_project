# weld_seg_project/main.py
from utils.config import Config
from preprocess.preprocess import PointCloudPreprocessor
from train.pretrain import run_pretrain
from train.finetune import run_finetune
from test.test import test_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Weld Segmentation Project")
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'test'], 
                        help="Mode to run: 'preprocess', 'train', or 'test'")
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == 'preprocess':
        print("--- Starting Preprocessing ---")
        preprocessor = PointCloudPreprocessor(config)
        # Fit on training data
        preprocessor.fit(config.RAW_DATA_DIR)
        # Process and save the entire dataset (train/val/test combined here)
        preprocessor.process_and_save_dataset(config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
        print("--- Preprocessing Complete ---")
        
    elif args.mode == 'train':
        print("--- Starting Training ---")
        run_pretrain(config)
        run_finetune(config)
        print("--- Training Complete ---")
        
    elif args.mode == 'test':
        print("--- Starting Inference ---")
        test_model(config)
        print("--- Inference Complete ---")

if __name__ == "__main__":
    main()