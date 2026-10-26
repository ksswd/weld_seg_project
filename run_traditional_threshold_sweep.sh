#!/bin/bash
#SBATCH -o traditional_threshold_sweep_%j.out
#SBATCH -e traditional_threshold_sweep_%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-threshold-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

set -euo pipefail
trap 'status=$?; echo "[ERROR] line=${LINENO} exit=${status} command=${BASH_COMMAND}" >&2; exit "${status}"' ERR

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

PROCESSED_ROOT="${PROCESSED_ROOT:-data/dataset_processed_1-17}"
OUT_DIR="${OUT_DIR:-results/traditional_threshold_sweep_${SLURM_JOB_ID:-manual}}"
RANSAC_SUMMARY="${RANSAC_SUMMARY:-results/traditional_baselines_v2_78810/traditional_baselines_v2_summary.csv}"
LIMIT_FILES="${LIMIT_FILES:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

if [[ ! -f "comparable_methods/mathmatic/sweep_traditional_thresholds.py" ]]; then
  echo "[ERROR] PROJECT_ROOT is not the repository root: ${PROJECT_ROOT}" >&2
  exit 2
fi

ARGS=(
  --processed-root "${PROCESSED_ROOT}"
  --out-dir "${OUT_DIR}"
  --limit-files "${LIMIT_FILES}"
)
if [[ -n "${RANSAC_SUMMARY}" ]]; then
  ARGS+=(--ransac-summary "${RANSAC_SUMMARY}")
fi

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[PROTOCOL] exact_threshold_sweep=validation evaluate_once=test"
echo "[CONFIG] processed=${PROCESSED_ROOT} out=${OUT_DIR} ransac_summary=${RANSAC_SUMMARY}"

conda run --no-capture-output -n weld_seg \
  python comparable_methods/mathmatic/sweep_traditional_thresholds.py "${ARGS[@]}"

echo "[DONE] summary=${OUT_DIR}/threshold_sweep_summary.csv"
echo "[DONE] combined_table=${OUT_DIR}/math_baselines_table.csv"
