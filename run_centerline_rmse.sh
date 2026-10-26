#!/bin/bash
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --job-name=weld-centerline-rmse
#SBATCH --output=centerline_rmse_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
trap 'status=$?; echo "[ERROR] line=${LINENO} exit=${status} command=${BASH_COMMAND}" >&2; exit "${status}"' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${SUBMIT_DIR}}"
cd "${PROJECT_ROOT}"

if [[ ! -f "scripts/compute_centerline_rmse.py" ]]; then
  echo "[ERROR] PROJECT_ROOT is not the repository root: ${PROJECT_ROOT}" >&2
  echo "Submit from the repository root or use:" >&2
  echo "  PROJECT_ROOT=/absolute/path/to/weld_seg_project sbatch run_centerline_rmse.sh" >&2
  exit 2
fi

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[PROJECT] ${PROJECT_ROOT}"

conda run --no-capture-output -n weld_seg \
python scripts/compute_centerline_rmse.py \
  --project-root "${PROJECT_ROOT}" \
  --raw-test-dir data/dataset_1-17/test \
  --processed-test-dir data/dataset_processed_1-17/test \
  --output-dir "results/centerline_rmse_${SLURM_JOB_ID:-local}" \
  "$@"

echo "[DONE] $(date '+%F %T')"
