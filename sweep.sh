#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-sweep
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:2

set -e

export USE_FOLD_SPLIT=False

SEEDS=(444 523 1944 8120)
MAX_POINTS_LIST=(4096)
PASSES="1"
SIMPLE_THR="0.5"
SOFT_THR="0.5"
DATA_ROOT="data/dataset_processed_1-17"
PRETRAIN_DIR="${DATA_ROOT}"
FINETUNE_DIR="${DATA_ROOT}"
BASE_PRETRAINED_WEIGHTS="results/final/dataset_split_run/weights/best_pretrain.pth"
RESULT_ROOT="results/final/seed_points_sweep"

run_one() {
  local exp_name="$1"
  local use_pretrained="$2"
  local use_geometry_bias="$3"
  local use_curvature="$4"
  local use_density="$5"
  local use_linearity="$6"
  local use_principal_dir="$7"
  local use_side_gate="$8"
  local seed="$9"
  local max_points="${10}"

  local run_name="${exp_name}_seed${seed}_pts${max_points}"
  local WEIGHTS_SAVE_DIR="${RESULT_ROOT}/${run_name}/weights"
  local TEST_WEIGHTS="${WEIGHTS_SAVE_DIR}/best_finetune.pth"
  local OUT_DIR="${RESULT_ROOT}/${run_name}/full_eval"

  mkdir -p "${WEIGHTS_SAVE_DIR}" "${OUT_DIR}"

  echo "=============================================="
  echo "[RUN] ${run_name}"
  echo "seed=${seed}, max_points=${max_points}"
  echo "use_pretrained=${use_pretrained}, geometry_bias=${use_geometry_bias}, side_gate=${use_side_gate}, curvature=${use_curvature}, density=${use_density}, linearity=${use_linearity}, principal_dir=${use_principal_dir}"
  echo "weights_save_dir=${WEIGHTS_SAVE_DIR}"
  echo "out_dir=${OUT_DIR}"
  echo "=============================================="

  # USE_FOLD_SPLIT="false"   USE_PRETRAINED_WEIGHTS="${use_pretrained}"   USE_GEOMETRY_BIAS="${use_geometry_bias}"   USE_SIDE_GATE="${use_side_gate}"   USE_CURVATURE_IN_ATTENTION="${use_curvature}"   USE_DENSITY_IN_ATTENTION="${use_density}"   USE_LINEARITY_IN_ATTENTION="${use_linearity}"   USE_PRINCIPAL_DIR_IN_ATTENTION="${use_principal_dir}"   python main.py --mode finetune     --seed "${seed}"     --processed_data_dir "${DATA_ROOT}"     --labeled_data_dir "${FINETUNE_DIR}"     --weights_save_dir "${WEIGHTS_SAVE_DIR}"     --pretrained_weights "${BASE_PRETRAINED_WEIGHTS}"     --test_weights "${TEST_WEIGHTS}"

  USE_FOLD_SPLIT="false"   USE_PRETRAINED_WEIGHTS="${use_pretrained}"   USE_GEOMETRY_BIAS="${use_geometry_bias}"   USE_SIDE_GATE="${use_side_gate}"   USE_CURVATURE_IN_ATTENTION="${use_curvature}"   USE_DENSITY_IN_ATTENTION="${use_density}"   USE_LINEARITY_IN_ATTENTION="${use_linearity}"   USE_PRINCIPAL_DIR_IN_ATTENTION="${use_principal_dir}"   python main.py --mode full_eval     --weights "${TEST_WEIGHTS}"     --data_dir "${FINETUNE_DIR}/test"     --out_dir "${OUT_DIR}"     --max_points "${max_points}"     --passes "${PASSES}"     --simple_thr "${SIMPLE_THR}"     --seed "${seed}"     --processed_data_dir "${DATA_ROOT}"     --weights_save_dir "${WEIGHTS_SAVE_DIR}"
}

run_all_for_setting() {
  local exp_name="$1"
  local use_pretrained="$2"
  local use_geometry_bias="$3"
  local use_curvature="$4"
  local use_density="$5"
  local use_linearity="$6"
  local use_principal_dir="$7"
  local use_side_gate="$8"

  for seed in "${SEEDS[@]}"; do
    for max_points in "${MAX_POINTS_LIST[@]}"; do
      run_one         "${exp_name}"         "${use_pretrained}"         "${use_geometry_bias}"         "${use_curvature}"         "${use_density}"         "${use_linearity}"         "${use_principal_dir}"         "${use_side_gate}"         "${seed}"         "${max_points}"
    done
  done
}

# 2x2 消融：几何注意力偏置 × 边侧门控
# 参数顺序：exp_name use_pretrained use_geometry_bias use_curvature use_density use_linearity use_principal_dir use_side_gate
# 如果只想跑某一个采样点数或随机种子，修改上面的 SEEDS 和 MAX_POINTS_LIST 即可。
run_all_for_setting "ablation_geom_on_gate_on" "false" "true" "true" "true" "true" "true" "true"
# run_all_for_setting "ablation_geom_off_gate_on" "false" "false" "true" "true" "true" "true" "true"
# run_all_for_setting "ablation_geom_on_gate_off" "false" "true" "true" "true" "true" "true" "false"
# run_all_for_setting "ablation_geom_off_gate_off" "false" "false" "true" "true" "true" "true" "false"
