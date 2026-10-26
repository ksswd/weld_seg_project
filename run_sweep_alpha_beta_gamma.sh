#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-ablation-abg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:2

set -e

export USE_FOLD_SPLIT=False

SEED="114514"
DATA_ROOT="data/dataset_processed"
FINETUNE_DIR="${DATA_ROOT}"
MAX_POINTS="4096"
PASSES="1"
SIMPLE_THR="0.5"

# ===== 参数组列表：每行一组 "ALPHA0 BETA0 GAMMA" =====
ABC_LIST=(
  "0 0.5 1"
  "2 0 1"
  "2 0.5 0"
  # "2 1 0.5"
)
# ================================================

EXP_ROOT="results/final/sweep_alpha_beta_gamma_manual"
mkdir -p "${EXP_ROOT}"

echo "=============================================="
echo "[RUN] grouped ALPHA0/BETA0/GAMMA sweep"
echo "=============================================="

for ABC in "${ABC_LIST[@]}"; do
  read -r ALPHA0 BETA0 GAMMA <<< "${ABC}"

  TAG="a${ALPHA0}_b${BETA0}_g${GAMMA}"
  WEIGHTS_SAVE_DIR="${EXP_ROOT}/${TAG}/weights"
  TEST_WEIGHTS="${WEIGHTS_SAVE_DIR}/best_finetune.pth"
  OUT_DIR="${EXP_ROOT}/${TAG}/full_eval"

  mkdir -p "${WEIGHTS_SAVE_DIR}" "${OUT_DIR}"

  echo "----------------------------------------------"
  echo "[EXP] ${TAG}"
  echo "ALPHA0=${ALPHA0}, BETA0=${BETA0}, GAMMA=${GAMMA}"
  echo "----------------------------------------------"

  ALPHA0="${ALPHA0}" \
  BETA0="${BETA0}" \
  GAMMA="${GAMMA}" \
  USE_PRETRAINED_WEIGHTS=False \
  python main.py --mode finetune \
    --seed "${SEED}" \
    --processed_data_dir "${DATA_ROOT}" \
    --labeled_data_dir "${FINETUNE_DIR}" \
    --weights_save_dir "${WEIGHTS_SAVE_DIR}" \
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

done

echo "All grouped experiments finished."