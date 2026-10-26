#!/bin/bash
#SBATCH -o traditional_baselines_v2_%j.out
#SBATCH -e traditional_baselines_v2_%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-math-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

set -euo pipefail
trap 'status=$?; echo "[ERROR] line=${LINENO} exit=${status} command=${BASH_COMMAND}" >&2; exit "${status}"' ERR

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

PROCESSED_ROOT="${PROCESSED_ROOT:-data/dataset_processed_1-17}"
RAW_ROOT="${RAW_ROOT:-data/dataset_1-17}"
OUT_DIR="${OUT_DIR:-results/traditional_baselines_v2_${SLURM_JOB_ID:-manual}}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
LIMIT_FILES="${LIMIT_FILES:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

if [[ ! -f "comparable_methods/mathmatic/run_traditional_baselines_v2.py" ]]; then
  echo "[ERROR] PROJECT_ROOT is not the repository root: ${PROJECT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${PROCESSED_ROOT}/validation" || ! -d "${PROCESSED_ROOT}/test" ]]; then
  echo "[ERROR] processed dataset is incomplete: ${PROCESSED_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${RAW_ROOT}/validation" || ! -d "${RAW_ROOT}/test" ]]; then
  echo "[ERROR] raw dataset is incomplete: ${RAW_ROOT}" >&2
  exit 2
fi

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[PROTOCOL] tune=validation evaluate=test global_confusion_matrix=true"
echo "[CONFIG] project=${PROJECT_ROOT} processed=${PROCESSED_ROOT} raw=${RAW_ROOT} out=${OUT_DIR}"

ARGS=(
  --processed-root "${PROCESSED_ROOT}"
  --raw-root "${RAW_ROOT}"
  --out-dir "${OUT_DIR}"
  --threshold-steps 160
  --tune-max-points 600000
  --ransac-iterations 160
  --ransac-sample-cap 30000
  --ransac-fit-tolerance-mult 1.5
  --seed 42
  --limit-files "${LIMIT_FILES}"
)
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  ARGS+=(--save-predictions)
fi

conda run --no-capture-output -n weld_seg \
  python comparable_methods/mathmatic/run_traditional_baselines_v2.py "${ARGS[@]}"

echo "[DONE] summary=${OUT_DIR}/traditional_baselines_v2_summary.csv"
