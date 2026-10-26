#!/bin/bash
#SBATCH -o traditional_baselines_%j.out
#SBATCH -e traditional_baselines_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-traditional-baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6

set -euo pipefail
trap 'status=$?; echo "[ERROR] line=${LINENO} exit=${status} command=${BASH_COMMAND}" >&2; exit "${status}"' ERR

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-data/dataset_processed_1-17}"
OUT_DIR="${OUT_DIR:-results/traditional_baselines}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[CONFIG] project=${PROJECT_ROOT} data=${DATA_ROOT} out=${OUT_DIR} save_predictions=${SAVE_PREDICTIONS}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --out-dir "${OUT_DIR}"
  --threshold-steps 201
  --tune-max-points 500000
  --ransac-iterations 120
  --ransac-sample-cap 20000
  --seed 42
)
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  ARGS+=(--save-predictions)
fi

conda run --no-capture-output -n weld_seg \
  python comparable_methods/mathmatic/run_traditional_baselines.py "${ARGS[@]}"

echo "[DONE] summary=${OUT_DIR}/traditional_baselines_summary.csv"
