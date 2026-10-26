#!/bin/bash
#SBATCH -o pointmlp_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J pointmlp-weld-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

cd /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/pointMLP-pytorch/part_segmentation

SEEDS=(444 523 1944 8120)
MAX_POINTS=4096
EPOCHS=200
DATA_ROOT=/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset_processed_1-17
RESULT_ROOT="outputs/seed_points_sweep"
REPORT_TOOL="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/baseline_report.py"

for SEED in "${SEEDS[@]}"; do
  EXP_NAME="pointmlp_seed${SEED}_pts${MAX_POINTS}"
  PRED_DIR="${RESULT_ROOT}/${EXP_NAME}/pred_xyz_label_test"
  FULLPRED_DIR="${RESULT_ROOT}/${EXP_NAME}/fullpred_test"
  MODEL_PATH="checkpoints/${EXP_NAME}/models/model_best.pth"
  mkdir -p "${PRED_DIR}" "${FULLPRED_DIR}"

  python main.py \
    --dataset weld \
    --data_root "${DATA_ROOT}" \
    --split train \
    --model pointMLP \
    --exp_name "${EXP_NAME}" \
    --batch_size 16 \
    --test_batch_size 16 \
    --epochs "${EPOCHS}" \
    --num_points "${MAX_POINTS}" \
    --workers 6 \
    --num_part 2 \
    --num_classes 1 \
    --manual_seed "${SEED}"

python main.py \
  --dataset weld \
    --data_root "${DATA_ROOT}" \
  --split test \
  --model pointMLP \
    --exp_name "${EXP_NAME}" \
  --eval_only \
  --model_type insiou \
  --test_batch_size 16 \
    --num_points "${MAX_POINTS}" \
  --workers 6 \
  --num_part 2 \
  --num_classes 1 \
    --manual_seed "${SEED}" \
    --pred_xyz_label_dir "${PRED_DIR}" \
    --fullpred_csv_dir "${FULLPRED_DIR}" \
    --timing_csv "${RESULT_ROOT}/${EXP_NAME}/inference_timing.csv"

  python "${REPORT_TOOL}" \
    --method pointmlp \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --max_points "${MAX_POINTS}" \
    --epochs "${EPOCHS}" \
    --model_path "${MODEL_PATH}" \
    --pred_dir "${FULLPRED_DIR}" \
    --out_dir "${RESULT_ROOT}/${EXP_NAME}" \
    --log_path "outputs/${EXP_NAME}/run.log"
done

python "${REPORT_TOOL}" --aggregate --root_dir "${RESULT_ROOT}" --method pointmlp
