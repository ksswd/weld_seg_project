#!/bin/bash
#SBATCH -o benchmark_%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-benchmark-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

set -euo pipefail

# This script does not train models. Run sweep.sh and each comparable method's
# run.sh first, then use this script to recompute identical metrics and audit
# whether their timing records are actually comparable.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

CHUNK_SIZE="${CHUNK_SIZE:-4096}"
THRESHOLD="${THRESHOLD:-0.5}"
OUT_DIR="${OUT_DIR:-results/unified_benchmark}"

# Conda environments used by the training/inference implementations.  The
# summary step itself only needs pandas/numpy, but these names are shared with
# the follow-up inference-only benchmark launcher.
OURS_ENV="${OURS_ENV:-weld_seg}"
DGCNN_ENV="${DGCNN_ENV:-weld_seg}"
POINTCEPT_ENV="${POINTCEPT_ENV:-pointcept}"
POINTMLP_ENV="${POINTMLP_ENV:-pointmlp}"
POINTNET2_ENV="${POINTNET2_ENV:-pointcept}"
POINTCONV_ENV="${POINTCONV_ENV:-pointcept}"

python scripts/compare_all_methods.py \
  --chunk-size "${CHUNK_SIZE}" \
  --threshold "${THRESHOLD}" \
  --out-dir "${OUT_DIR}"

echo "Detailed runs: ${OUT_DIR}/benchmark_runs.csv"
echo "Final table:   ${OUT_DIR}/benchmark_summary.csv"
echo "Audit report:  ${OUT_DIR}/benchmark_audit.json"
