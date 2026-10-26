#!/bin/bash
#SBATCH -o final_benchmark_%j.out
#SBATCH -e final_benchmark_%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-final-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -euo pipefail
trap 's=$?; echo "[ERROR] line=${LINENO} exit=${s} command=${BASH_COMMAND}" >&2; exit "${s}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
if [[ ! -f "${PROJECT_ROOT}/main.py" || ! -d "${PROJECT_ROOT}/comparable_methods" ]]; then
  echo "[ERROR] invalid PROJECT_ROOT=${PROJECT_ROOT}" >&2
  echo "Submit from the repository root, or use PROJECT_ROOT=/absolute/path sbatch run_final_benchmark.sh" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"

SEEDS=(${SEEDS:-444 523 1944 8120})
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
PROTOCOL_SEED="${PROTOCOL_SEED:-20260823}"
METHODS="${METHODS:-ours dgcnn pointnet2 pointmlp pointconv pointcept}"
FINAL_ROOT="${FINAL_ROOT:-results/final_benchmark_${SLURM_JOB_ID:-manual}}"
TEST_CSV_ROOT="${TEST_CSV_ROOT:-data/dataset_processed_1-17/test}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-data/dataset_1-17}"

has_method() { [[ " ${METHODS} " == *" $1 "* ]]; }
run_env() { local env_name="$1"; shift; conda run --no-capture-output -n "${env_name}" "$@"; }
require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing checkpoint: $1" >&2; return 1; }; }

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[PROTOCOL] methods=${METHODS} seeds=${SEEDS[*]} chunk=${CHUNK_SIZE} protocol_seed=${PROTOCOL_SEED}"
echo "[OUTPUT] ${PROJECT_ROOT}/${FINAL_ROOT}"
mkdir -p "${FINAL_ROOT}/predictions" "${FINAL_ROOT}/summary"

if has_method ours; then
  for seed in "${SEEDS[@]}"; do
    run_name="ablation_geom_on_gate_on_seed${seed}_pts${CHUNK_SIZE}"
    weights="results/final/seed_points_sweep/${run_name}/weights/best_finetune.pth"
    out="${FINAL_ROOT}/predictions/ours/seed${seed}"
    require_file "${weights}"; mkdir -p "${out}"
    USE_FOLD_SPLIT=false run_env weld_seg python main.py --mode full_eval \
      --weights "${weights}" --data_dir "${TEST_CSV_ROOT}" --out_dir "${out}" \
      --max_points "${CHUNK_SIZE}" --passes 1 --simple_thr 0.5 --seed "${seed}" \
      --protocol_seed "${PROTOCOL_SEED}"
  done
fi

if has_method dgcnn; then
  pushd comparable_methods/dgcnn.pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="dgcnn_seed${seed}_pts${CHUNK_SIZE}"
    model="outputs/${exp}_train/models/best_model.pth"
    out="${PROJECT_ROOT}/${FINAL_ROOT}/predictions/dgcnn/seed${seed}"
    require_file "${model}"; mkdir -p "${out}/fullpred_test" "${out}/pred_xyz_label_test"
    run_env weld_seg python main_semseg_weld.py --eval --exp_name "final_${exp}" \
      --data_root "${PROJECT_ROOT}/${RAW_DATA_ROOT}" --fold dataset --split test \
      --model_path "${model}" --num_points "${CHUNK_SIZE}" --seed "${seed}" \
      --protocol_seed "${PROTOCOL_SEED}" --save_all_pred_xyz_label_dir "${out}/pred_xyz_label_test" \
      --save_all_fullpred_dir "${out}/fullpred_test" --benchmark_infer_time \
      --timing_csv "${out}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointnet2; then
  pushd comparable_methods/Pointnet_Pointnet2_pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointnet2_seed${seed}_pts${CHUNK_SIZE}"
    log_dir="log/seed_points_sweep/${exp}"
    out="${PROJECT_ROOT}/${FINAL_ROOT}/predictions/pointnet2/seed${seed}"
    require_file "${log_dir}/checkpoints/best_model.pth"; mkdir -p "${out}/fullpred_test"
    run_env pointcept python tools/export_fullpred_weld.py --root data/weld_from_ply --split test \
      --model pointnet2_sem_seg --log_dir "${log_dir}" --npoint "${CHUNK_SIZE}" --gpu 0 \
      --out_dir "${out}/fullpred_test" --time_csv "${out}/inference_timing.csv" \
      --seed "${seed}" --protocol-seed "${PROTOCOL_SEED}"
  done
  popd >/dev/null
fi

if has_method pointmlp; then
  pushd comparable_methods/pointMLP-pytorch/part_segmentation >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointmlp_seed${seed}_pts${CHUNK_SIZE}"
    out="${PROJECT_ROOT}/${FINAL_ROOT}/predictions/pointmlp/seed${seed}"
    require_file "checkpoints/pointMLP_${exp}/best_insiou_model.pth"
    mkdir -p "${out}/fullpred_test" "${out}/pred_xyz_label_test"
    run_env pointmlp python main.py --dataset weld --data_root "${PROJECT_ROOT}/data/dataset_processed_1-17" \
      --split test --model pointMLP --exp_name "${exp}" --eval_only --model_type insiou \
      --test_batch_size 1 --num_points "${CHUNK_SIZE}" --workers 0 --num_part 2 --num_classes 1 \
      --manual_seed "${seed}" --protocol_seed "${PROTOCOL_SEED}" \
      --pred_xyz_label_dir "${out}/pred_xyz_label_test" --fullpred_csv_dir "${out}/fullpred_test" \
      --timing_csv "${out}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointconv; then
  pushd comparable_methods/pointconv_pytorch >/dev/null
  for seed in "${SEEDS[@]}"; do
    exp="pointconv_seed${seed}_pts${CHUNK_SIZE}"
    checkpoint="log/seed_points_sweep/${exp}/checkpoints/best_model.pth"
    out="${PROJECT_ROOT}/${FINAL_ROOT}/predictions/pointconv/seed${seed}"
    require_file "${checkpoint}"
    run_env pointcept python train_weld_semseg.py --eval_only --root "${PROJECT_ROOT}/${RAW_DATA_ROOT}" \
      --log_dir "${out}" --checkpoint "${checkpoint}" --batch_size 1 --npoint "${CHUNK_SIZE}" \
      --num_workers 0 --save_pred --time_inference --time_warmup 0 \
      --protocol_seed "${PROTOCOL_SEED}" --timing_csv "${out}/inference_timing.csv"
  done
  popd >/dev/null
fi

if has_method pointcept; then
  pushd comparable_methods/Pointcept-copy >/dev/null
  require_file exp/weld_1_17/weld_newdata_200epochs/model/model_best.pth
  result_dir="exp/weld_1_17/weld_newdata_200epochs/result"
  if [[ -d "${result_dir}" ]]; then
    mv "${result_dir}" "${result_dir}.before_final_$(date +%Y%m%d_%H%M%S)"
  fi
  run_env pointcept bash scripts/test.sh -g 1 -d weld_1_17 \
    -c semseg-pt-v3m1-0-base -n weld_newdata_200epochs -w model_best
  popd >/dev/null
  out="${FINAL_ROOT}/predictions/pointcept/native_protocol"
  mkdir -p "${out}/fullpred_test"
  run_env pointcept python scripts/export_pointcept_predictions.py \
    --data-root comparable_methods/Pointcept-copy/data/weld_1_17 --split test \
    --result-dir comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/result \
    --out-dir "${out}/fullpred_test"
  cp comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/result/inference_timing.csv \
    "${out}/inference_timing.csv"
fi

compare_args=(
  --chunk-size "${CHUNK_SIZE}" --protocol-seed "${PROTOCOL_SEED}" --threshold 0.5
  --out-dir "${FINAL_ROOT}/summary" --strict-protocol
)
for method in ours dgcnn pointnet2 pointmlp pointconv pointcept; do
  if has_method "${method}"; then
    compare_args+=(--method "${method}=${FINAL_ROOT}/predictions/${method}")
  fi
done
python scripts/compare_all_methods.py "${compare_args[@]}"

echo "[DONE] $(date '+%F %T')"
echo "[RESULT] ${PROJECT_ROOT}/${FINAL_ROOT}/summary/benchmark_summary.csv"
echo "[AUDIT]  ${PROJECT_ROOT}/${FINAL_ROOT}/summary/benchmark_audit.json"
