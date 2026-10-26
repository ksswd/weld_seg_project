#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

# python postprocess/postprocess_eval.py \
#   --input_dir data/new/full_eval0/fold_6 \
#   --pattern "*_fullpred.csv" \
#   --fixed_thr 0.80

# python postprocess/postprocess_eval.py \
#   --input_dir data/new/full_eval0/fold_6 \
#   --pattern "*_fullpred.csv" \
#   --fixed_thr 0.80 \
#   --enable_remove_small_components \
#   --enable_bridge \
#   --enable_fill_holes \
#   --enable_remove_sparse \
#   --save_per_file_csv

python postprocess/grid_search_postprocess.py \
  --input_dir data/new/full_eval0/fold_6 \
  --fixed_thr 0.80 \
  --rank_metric hard_miou_gain \
  --topk 20