#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
set -e

INPUT_ROOT="data/dataset_processed"
OUTPUT_ROOT="data/dataset_processed_fixed"
FIX_IDS="17 18 19"

python scripts/csv_fix.py \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --fix_ids ${FIX_IDS} \
  --replace_original_columns \
  --keep_new_columns \
  --vis
