#!/bin/bash
#SBATCH -o pointconv_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J pointconv-weld-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

cd /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/pointconv_pytorch

SEEDS=(444 523 1944 8120)
MAX_POINTS=4096
EPOCHS=200
DATA_ROOT="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset_1-17"
RESULT_ROOT="log/seed_points_sweep"
REPORT_TOOL="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/baseline_report.py"

for SEED in "${SEEDS[@]}"; do
  EXP_NAME="pointconv_seed${SEED}_pts${MAX_POINTS}"
  LOG_DIR="${RESULT_ROOT}/${EXP_NAME}"
  PRED_DIR="${LOG_DIR}/pred_xyz_label_test"
  MODEL_PATH="${LOG_DIR}/checkpoints/best_model.pth"

  mkdir -p "${LOG_DIR}"
  export PYTHONHASHSEED="${SEED}"

python train_weld_semseg.py \
    --gpu 0 \
    --batch_size 8 \
    --epoch "${EPOCHS}" \
    --npoint "${MAX_POINTS}" \
    --num_workers 4 \
    --root "${DATA_ROOT}" \
    --log_dir "${LOG_DIR}"

python train_weld_semseg.py \
    --eval_only \
    --root "${DATA_ROOT}" \
    --log_dir "${LOG_DIR}" \
    --checkpoint "${MODEL_PATH}" \
    --npoint "${MAX_POINTS}" \
    --save_pred \
    --time_inference \
    --time_warmup 0 \
    --timing_csv "${LOG_DIR}/inference_timing.csv"

  python "${REPORT_TOOL}" \
    --method pointconv \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --max_points "${MAX_POINTS}" \
    --epochs "${EPOCHS}" \
    --model_path "${MODEL_PATH}" \
    --pred_dir "${PRED_DIR}" \
    --out_dir "${LOG_DIR}" \
    --log_path "${LOG_DIR}/log.txt"
done

python "${REPORT_TOOL}" --aggregate --root_dir "${RESULT_ROOT}" --method pointconv
