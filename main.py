# weld_seg_project/main.py
import os
import torch
import argparse
import sys
import random
import numpy as np
from utils.config import Config
from preprocess.preprocess import PointCloudPreprocessor
from train.pretrain import run_pretrain
from train.finetune import run_finetune
from test.test import test_model
from test.full_eval import run_full_eval


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Weld Segmentation Project")
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'test', 'pretrain', 'finetune', 'full_eval'], 
                        help="Mode to run: 'preprocess', 'train(pretrain + finetune)', 'test', 'pretrain', 'finetune', or 'full_eval'")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--protocol_seed', type=int, default=20260823,
                        help='fixed seed for the common benchmark point permutation')
    parser.add_argument('--fold_id', type=str, default=None, choices=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6'],
                        help="6-fold group split id")
    parser.add_argument('--weights', type=str, default=None, help='checkpoint path for full_eval')
    parser.add_argument('--data_dir', type=str, default=None, help='data dir for full_eval')
    parser.add_argument('--out_dir', type=str, default=None, help='output dir for full_eval')
    parser.add_argument('--raw_data_dir', type=str, default=None, help='raw ply data dir for preprocess')
    parser.add_argument('--processed_data_dir', type=str, default=None, help='processed csv data dir')
    parser.add_argument('--pretrain_data_dir', type=str, default=None, help='pretrain data dir')
    parser.add_argument('--labeled_data_dir', type=str, default=None, help='labeled data dir for finetune')
    parser.add_argument('--weights_save_dir', type=str, default=None, help='directory to save checkpoints')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='pretrained checkpoint path for finetune')
    parser.add_argument('--test_weights', type=str, default=None, help='checkpoint path for test')
    parser.add_argument('--max_points', type=int, default=1024, help='chunk size for full_eval')
    parser.add_argument('--passes', type=int, default=1, help='coverage passes for full_eval')
    parser.add_argument('--simple_thr', type=float, default=0.53, help='simple export threshold for full_eval')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = Config()
    if args.fold_id is not None:
        config.FOLD_ID = args.fold_id
    if args.raw_data_dir is not None:
        config.RAW_DATA_DIR = args.raw_data_dir
    if args.processed_data_dir is not None:
        config.PROCESSED_DATA_DIR = args.processed_data_dir
    if args.pretrain_data_dir is not None:
        config.PRETRAIN_DATA_DIR = args.pretrain_data_dir
    if args.labeled_data_dir is not None:
        config.LABELED_DATA_DIR = args.labeled_data_dir
    if args.weights_save_dir is not None:
        config.WEIGHTS_SAVE_DIR = args.weights_save_dir
    if args.pretrained_weights is not None:
        config.PRETRAINED_WEIGHTS = args.pretrained_weights
    if args.test_weights is not None:
        config.TEST_WEIGHTS = args.test_weights

    # 按 mode 打印“最终生效配置”（命令行优先，未传则回退到 config 默认）
    print("Running with effective configuration:")
    print(f"mode: {args.mode}")
    print(f"seed: {args.seed}")
    print(f"protocol_seed: {args.protocol_seed}")
    print(f"fold_id: {config.FOLD_ID}")

    print(f"USE_PRETRAINED_WEIGHTS: {getattr(config, 'USE_PRETRAINED_WEIGHTS', None)}")
    print(f"USE_GEOMETRY_BIAS: {getattr(config, 'USE_GEOMETRY_BIAS', None)}")
    print(f"USE_COORDINATE_IN_ATTENTION: {getattr(config, 'USE_COORDINATE_IN_ATTENTION', None)}")
    print(f"USE_PRINCIPAL_DIR_IN_ATTENTION: {getattr(config, 'USE_PRINCIPAL_DIR_IN_ATTENTION', None)}")
    print(f"USE_CURVATURE_IN_ATTENTION: {getattr(config, 'USE_CURVATURE_IN_ATTENTION', None)}")
    print(f"USE_DENSITY_IN_ATTENTION: {getattr(config, 'USE_DENSITY_IN_ATTENTION', None)}")
    print(f"USE_NORMALS_IN_ATTENTION: {getattr(config, 'USE_NORMALS_IN_ATTENTION', None)}")
    print(f"USE_LINEARITY_IN_ATTENTION: {getattr(config, 'USE_LINEARITY_IN_ATTENTION', None)}")
    if args.mode == 'preprocess':
        print(f"RAW_DATA_DIR: {getattr(config, 'RAW_DATA_DIR', None)}")
        print(f"PROCESSED_DATA_DIR: {getattr(config, 'PROCESSED_DATA_DIR', None)}")
    elif args.mode == 'pretrain':
        print(f"PRETRAIN_DATA_DIR: {getattr(config, 'PRETRAIN_DATA_DIR', None)}")
        print(f"WEIGHTS_SAVE_DIR: {getattr(config, 'WEIGHTS_SAVE_DIR', None)}")
    elif args.mode == 'finetune':
        print(f"LABELED_DATA_DIR: {getattr(config, 'LABELED_DATA_DIR', None)}")
        print(f"PRETRAINED_WEIGHTS: {getattr(config, 'PRETRAINED_WEIGHTS', None)}")
        print(f"WEIGHTS_SAVE_DIR: {getattr(config, 'WEIGHTS_SAVE_DIR', None)}")
        print(f"TEST_WEIGHTS: {getattr(config, 'TEST_WEIGHTS', None)}")
    elif args.mode == 'test':
        print(f"TEST_WEIGHTS: {getattr(config, 'TEST_WEIGHTS', None)}")
    elif args.mode == 'full_eval':
        print(f"weights(full_eval): {args.weights}")
        print(f"data_dir(full_eval): {args.data_dir}")
        print(f"out_dir(full_eval): {args.out_dir}")
        print(f"max_points: {args.max_points}")
        print(f"passes: {args.passes}")
        print(f"simple_thr: {args.simple_thr}")

    if args.mode == 'preprocess':
        print("--- Starting Preprocessing ---")
        if not config.RAW_DATA_DIR or not config.PROCESSED_DATA_DIR:
            raise ValueError("preprocess mode requires RAW_DATA_DIR and PROCESSED_DATA_DIR from run script args/env")
        preprocessor = PointCloudPreprocessor(config)
        preprocessor.process_and_save_dataset(config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
        print("--- Preprocessing Complete ---")
    
    elif args.mode == 'finetune':
        print("--- Starting Fine-tuning ---")
        run_finetune(config)
        print("--- Fine-tuning Complete ---")
    
    elif args.mode == 'pretrain':
        print("--- Starting Pre-training ---")
        run_pretrain(config)
        print("--- Pre-training Complete ---")
    
    elif args.mode == 'test':
        print("--- Starting Inference ---")
        test_model(config)
        print("--- Inference Complete ---")

    elif args.mode == 'full_eval':
        print("--- Starting Full Evaluation ---")
        if args.weights is None or args.data_dir is None or args.out_dir is None:
            raise ValueError("full_eval mode requires --weights --data_dir --out_dir from run script")
        out_dir = os.path.join(args.out_dir, config.FOLD_ID) if getattr(config, 'USE_FOLD_SPLIT', False) else args.out_dir
        eval_args = argparse.Namespace(
            weights=args.weights,
            data_dir=args.data_dir,
            val_ratio=1.0,
            max_points=args.max_points,
            passes=args.passes,
            seed=args.seed,
            protocol_seed=args.protocol_seed,
            out_dir=out_dir,
            export_simple_pred=True,
            simple_thr=args.simple_thr,
            use_fold_split=bool(getattr(config, 'USE_FOLD_SPLIT', False)),
            fold_id=config.FOLD_ID,
        )
        run_full_eval(eval_args)
        print("--- Full Evaluation Complete ---")
if __name__ == "__main__":
    main()
