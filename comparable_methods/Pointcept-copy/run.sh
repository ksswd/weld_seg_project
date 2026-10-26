#!/bin/bash
#SBATCH -o pointcept_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J pointcept-weld-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -euo pipefail

cd /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/Pointcept-copy

SEEDS=(444 523 1944 8120)
MAX_POINTS=4096
EPOCHS=200
DATASET_NAME="weld_1_17"
CONFIG_NAME="semseg-pt-v3m1-0-base"
GPU_NUM=1
RESULT_ROOT="exp/${DATASET_NAME}/seed_points_sweep"
REPORT_TOOL="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/baseline_report.py"

for SEED in "${SEEDS[@]}"; do
  EXP_NAME="pointcept_seed${SEED}_pts${MAX_POINTS}"
  RUN_NAME="seed_points_sweep/${EXP_NAME}"
  OUT_DIR="${RESULT_ROOT}/${EXP_NAME}"
  mkdir -p "${OUT_DIR}"
  export PYTHONHASHSEED="${SEED}"

  bash scripts/train.sh     -g "${GPU_NUM}"     -d "${DATASET_NAME}"     -c "${CONFIG_NAME}"     -n "${RUN_NAME}"     --options seed="${SEED}" data.train.transform.0.num_points="${MAX_POINTS}" data.val.transform.0.num_points="${MAX_POINTS}" epoch="${EPOCHS}"

  bash scripts/test.sh     -g "${GPU_NUM}"     -d "${DATASET_NAME}"     -c "${CONFIG_NAME}"     -n "${RUN_NAME}"     -w model_best

  PRED_DIR="${OUT_DIR}/result/submit"
  MODEL_PATH="${OUT_DIR}/model/model_best.pth"
  python "${REPORT_TOOL}"     --method pointcept     --exp_name "${EXP_NAME}"     --seed "${SEED}"     --max_points "${MAX_POINTS}"     --epochs "${EPOCHS}"     --model_path "${MODEL_PATH}"     --pred_dir "${PRED_DIR}"     --out_dir "${OUT_DIR}/unified_eval"     --log_path "${OUT_DIR}/test.log"
done

python "${REPORT_TOOL}" --aggregate --root_dir "${RESULT_ROOT}" --method pointcept
