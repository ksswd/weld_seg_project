#!/bin/bash
#SBATCH -o inference_benchmark_%j.out
#SBATCH -e inference_benchmark_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-inference-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -euo pipefail

trap 'status=$?; echo "[ERROR] line=${LINENO} exit=${status} command=${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "[START] $(date '+%F %T') host=$(hostname) job_id=${SLURM_JOB_ID:-manual}"
echo "[START] submitted_from=${SLURM_SUBMIT_DIR:-unknown} script=${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Under sbatch, BASH_SOURCE points to /var/spool/slurmd/.../slurm_script.
# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked and is the
# correct default project root.  Manual bash execution still uses SCRIPT_DIR.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
if [[ ! -f "${PROJECT_ROOT}/main.py" || ! -d "${PROJECT_ROOT}/comparable_methods" ]]; then
  echo "[ERROR] PROJECT_ROOT is not the project root: ${PROJECT_ROOT}" >&2
  echo "[ERROR] Upload this script into the repository root, or submit with:" >&2
  echo "        PROJECT_ROOT=/absolute/path/to/weld_seg_project sbatch run_inference_timing.sh" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"
echo "[START] project_root=${PROJECT_ROOT}"
echo "[START] methods=${METHODS:-ours dgcnn pointnet2 pointmlp pointconv pointcept}"
echo "[START] conda=$(command -v conda || echo NOT_FOUND)"

SEEDS=(${SEEDS:-444 523 1944 8120})
MAX_POINTS="${MAX_POINTS:-4096}"
TEST_CSV_ROOT="${TEST_CSV_ROOT:-data/dataset_processed_1-17/test}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-data/dataset_1-17}"
METHODS="${METHODS:-ours dgcnn pointnet2 pointmlp pointconv pointcept}"

has_method() { [[ " ${METHODS} " == *" $1 "* ]]; }
run_env() {
  local env_name="$1"
  shift
  conda run --no-capture-output -n "${env_name}" "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] required file does not exist: ${PROJECT_ROOT}/${path}" >&2
    return 1
  fi
}

if has_method ours; then
  for seed in "${SEEDS[@]}"; do
    run_name="ablation_geom_on_gate_on_seed${seed}_pts${MAX_POINTS}"
    weights="results/final/seed_points_sweep/${run_name}/weights/best_finetune.pth"
    out="results/unified_inference/ours/${run_name}"
    require_file "${weights}"
    mkdir -p "${out}"
    USE_FOLD_SPLIT=false run_env weld_seg python main.py --mode full_eval \
      --weights "${weights}" --data_dir "${TEST_CSV_ROOT}" --out_dir "${out}" \
      --max_points "${MAX_POINTS}" --passes 1 --simple_thr 0.5 --seed "${seed}"
  done
fi

if has_method dgcnn; then
  pushd comparable_methods/dgcnn.pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="dgcnn_seed${seed}_pts${MAX_POINTS}"
    model="outputs/${exp}_train/models/best_model.pth"
    out="outputs/unified_inference/${exp}"
    require_file "${model}"
    mkdir -p "${out}/fullpred_test" "${out}/pred_xyz_label_test"
    run_env weld_seg python main_semseg_weld.py --eval --exp_name "${exp}" \
      --data_root "${PROJECT_ROOT}/${RAW_DATA_ROOT}" --fold dataset --split test \
      --model_path "${model}" --num_points "${MAX_POINTS}" --seed "${seed}" \
      --save_all_pred_xyz_label_dir "${out}/pred_xyz_label_test" \
      --save_all_fullpred_dir "${out}/fullpred_test" --benchmark_infer_time \
      --timing_csv "${out}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointnet2; then
  pushd comparable_methods/Pointnet_Pointnet2_pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointnet2_seed${seed}_pts${MAX_POINTS}"
    log_dir="log/seed_points_sweep/${exp}"
    out="fullpred/unified_inference/${exp}"
    require_file "${log_dir}/checkpoints/best_model.pth"
    mkdir -p "${out}/fullpred_test"
    run_env pointcept python tools/export_fullpred_weld.py --root data/weld_from_ply \
      --split test --model pointnet2_sem_seg --log_dir "${log_dir}" \
      --npoint "${MAX_POINTS}" --gpu 0 --out_dir "${out}/fullpred_test" \
      --time_csv "${out}/inference_timing.csv" --seed "${seed}"
  done
  popd >/dev/null
fi

if has_method pointmlp; then
  pushd comparable_methods/pointMLP-pytorch/part_segmentation >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointmlp_seed${seed}_pts${MAX_POINTS}"
    out="outputs/unified_inference/${exp}"
    require_file "checkpoints/pointMLP_${exp}/best_insiou_model.pth"
    mkdir -p "${out}/fullpred_test" "${out}/pred_xyz_label_test"
    run_env pointmlp python main.py --dataset weld --data_root "${PROJECT_ROOT}/data/dataset_processed_1-17" \
      --split test --model pointMLP --exp_name "${exp}" --eval_only --model_type insiou \
      --test_batch_size 1 --num_points "${MAX_POINTS}" --workers 0 --num_part 2 --num_classes 1 \
      --manual_seed "${seed}" --pred_xyz_label_dir "${out}/pred_xyz_label_test" \
      --fullpred_csv_dir "${out}/fullpred_test" --timing_csv "${out}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointconv; then
  pushd comparable_methods/pointconv_pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointconv_seed${seed}_pts${MAX_POINTS}"
    log_dir="log/seed_points_sweep/${exp}"
    require_file "${log_dir}/checkpoints/best_model.pth"
    run_env pointcept python train_weld_semseg.py --eval_only --root "${PROJECT_ROOT}/${RAW_DATA_ROOT}" \
      --log_dir "${log_dir}" --checkpoint "${log_dir}/checkpoints/best_model.pth" \
      --batch_size 1 --npoint "${MAX_POINTS}" --num_workers 0 --save_pred \
      --time_inference --time_warmup 0 --timing_csv "${log_dir}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointcept; then
  pushd comparable_methods/Pointcept-copy >/dev/null
  require_file exp/weld_1_17/weld_newdata_200epochs/model/model_best.pth
  # Pointcept currently has one final checkpoint rather than four seed checkpoints.
  # Remove/rename an old result cache before this run if its tester reports "loaded pred";
  # cached predictions execute no model forward and therefore have zero timing value.
  run_env pointcept bash scripts/test.sh -g 1 -d weld_1_17 \
    -c semseg-pt-v3m1-0-base -n weld_newdata_200epochs -w model_best
  popd >/dev/null
  run_env pointcept python scripts/export_pointcept_predictions.py \
    --data-root comparable_methods/Pointcept-copy/data/weld_1_17 \
    --split test \
    --result-dir comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/result \
    --out-dir comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/fullpred_test
fi

python scripts/compare_all_methods.py \
  --method ours=results/unified_inference/ours \
  --method dgcnn=comparable_methods/dgcnn.pytorch/outputs/unified_inference \
  --method pointnet2=comparable_methods/Pointnet_Pointnet2_pytorch/fullpred/unified_inference \
  --method pointmlp=comparable_methods/pointMLP-pytorch/part_segmentation/outputs/unified_inference \
  --method pointconv=comparable_methods/pointconv_pytorch/log/seed_points_sweep \
  --method pointcept=comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs \
  --chunk-size "${MAX_POINTS}" --threshold 0.5 --out-dir results/unified_benchmark
