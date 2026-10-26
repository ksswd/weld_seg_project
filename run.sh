#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:2

set -e

# 不使用fold划分，直接使用已划分目录
export USE_FOLD_SPLIT=False

SEED="114514"
MAX_POINTS="4096"
PASSES="1"
SIMPLE_THR="0.5"
SOFT_THR="0.5"

DATA_ROOT="data/dataset_processed_fixed"
PRETRAIN_DIR="${DATA_ROOT}"
FINETUNE_DIR="${DATA_ROOT}"

WEIGHTS_SAVE_DIR="results/final/dataset_split_run/weights"
  PRETRAINED_WEIGHTS="${WEIGHTS_SAVE_DIR}/best_pretrain.pth"
  TEST_WEIGHTS="${WEIGHTS_SAVE_DIR}/best_finetune.pth"    
OUT_DIR="results/final/dataset_split_run/full_eval"

mkdir -p "${WEIGHTS_SAVE_DIR}" "${OUT_DIR}"

  echo "=============================================="
echo "[RUN] pre-split dataset full pipeline (no fold)"
  echo "=============================================="

# python main.py --mode pretrain \
#   --seed "${SEED}" \
#   --processed_data_dir "${DATA_ROOT}" \
#   --pretrain_data_dir "${PRETRAIN_DIR}" \
#     --weights_save_dir "${WEIGHTS_SAVE_DIR}"

python main.py --mode finetune \
  --seed "${SEED}" \
  --processed_data_dir "${DATA_ROOT}" \
  --labeled_data_dir "${FINETUNE_DIR}" \
    --weights_save_dir "${WEIGHTS_SAVE_DIR}" \
    --pretrained_weights "${PRETRAINED_WEIGHTS}" \
    --test_weights "${TEST_WEIGHTS}"

python main.py --mode full_eval \
  --weights "${TEST_WEIGHTS}" \
  --data_dir "${FINETUNE_DIR}/test" \
  --out_dir "${OUT_DIR}" \
  --max_points "${MAX_POINTS}" \
  --passes "${PASSES}" \
  --simple_thr "${SIMPLE_THR}" \
  --seed "${SEED}" \
  --processed_data_dir "${DATA_ROOT}" \
  --weights_save_dir "${WEIGHTS_SAVE_DIR}"

# python scripts/visualize_fullpred.py "${OUT_DIR}" --out "${OUT_DIR}/vis" --thr "${SIMPLE_THR}" --show_bg
# python test/per_case_metrics.py --input_dir "${OUT_DIR}" --fixed_thr "${SIMPLE_THR}" --soft_thr "${SOFT_THR}"
# python test/plot_case_metrics.py --input_dir "${OUT_DIR}" --thr "${SIMPLE_THR}" --out_png "${OUT_DIR}/case_metrics_bars_percent_mm.png"
