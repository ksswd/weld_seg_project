#!/bin/bash
#SBATCH -o dgcnn_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J dgcnn-weld-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

cd /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/dgcnn.pytorch

SEEDS=(444 523 1944 8120)
MAX_POINTS=4096
EPOCHS=200
DATA_ROOT="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset_1-17"
RESULT_ROOT="outputs/seed_points_sweep"
REPORT_TOOL="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/baseline_report.py"

for SEED in "${SEEDS[@]}"; do
  EXP_NAME="dgcnn_seed${SEED}_pts${MAX_POINTS}"
  TRAIN_EXP="${EXP_NAME}_train"
  EVAL_DIR="${RESULT_ROOT}/${EXP_NAME}"
  MODEL_PATH="outputs/${TRAIN_EXP}/models/best_model.pth"
  PRED_TXT_DIR="${EVAL_DIR}/pred_xyz_label_test"
  FULLPRED_DIR="${EVAL_DIR}/fullpred_test"

  mkdir -p "${EVAL_DIR}" "${PRED_TXT_DIR}" "${FULLPRED_DIR}"

  python main_semseg_weld.py \
    --exp_name "${TRAIN_EXP}" \
    --data_root "${DATA_ROOT}" \
    --fold dataset \
    --batch_size 16 \
    --test_batch_size 16 \
    --epochs "${EPOCHS}" \
    --lr 0.0001 \
    --num_points "${MAX_POINTS}" \
    --seed "${SEED}" \
    --k 20

python main_semseg_weld.py \
  --eval \
    --exp_name "${EXP_NAME}" \
    --data_root "${DATA_ROOT}" \
  --fold dataset \
  --split test \
    --model_path "${MODEL_PATH}" \
    --num_points "${MAX_POINTS}" \
    --seed "${SEED}" \
    --save_all_pred_xyz_label_dir "${PRED_TXT_DIR}" \
    --save_all_fullpred_dir "${FULLPRED_DIR}" \
  --benchmark_infer_time \
  --timing_csv "${EVAL_DIR}/inference_timing.csv"

  python "${REPORT_TOOL}" \
    --method dgcnn \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --max_points "${MAX_POINTS}" \
    --epochs "${EPOCHS}" \
    --model_path "${MODEL_PATH}" \
    --pred_dir "${FULLPRED_DIR}" \
    --out_dir "${EVAL_DIR}" \
    --log_path "outputs/${EXP_NAME}/run.log"
done

python "${REPORT_TOOL}" --aggregate --root_dir "${RESULT_ROOT}" --method dgcnn
