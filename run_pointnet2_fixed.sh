#!/bin/bash
#SBATCH -o pointnet2_fixed_%j.out
#SBATCH -e pointnet2_fixed_%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J pointnet2-fixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -euo pipefail
trap 's=$?; echo "[ERROR] line=${LINENO} exit=${s} command=${BASH_COMMAND}" >&2; exit "${s}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
cd "${PROJECT_ROOT}"
[[ -f main.py && -d comparable_methods/Pointnet_Pointnet2_pytorch ]] || {
  echo "[ERROR] submit from the repository root" >&2; exit 2;
}

SEEDS=(${SEEDS:-444 523 1944 8120})
EPOCHS="${EPOCHS:-200}"
NPOINT="${NPOINT:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL_SEED="${PROTOCOL_SEED:-20260823}"
RUN_ROOT="${RUN_ROOT:-results/pointnet2_fixed_${SLURM_JOB_ID:-manual}}"
PN_ROOT="${PROJECT_ROOT}/comparable_methods/Pointnet_Pointnet2_pytorch"
DATA_ROOT="${PN_ROOT}/data/weld_from_ply"

for split in train val test; do
  [[ -d "${DATA_ROOT}/${split}" ]] || { echo "[ERROR] missing ${DATA_ROOT}/${split}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}/predictions" "${RUN_ROOT}/summary"

echo "[START] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[FIX] normalized radius coordinates + class-weighted loss + controlled seeds"
echo "[RUN] seeds=${SEEDS[*]} epochs=${EPOCHS} npoint=${NPOINT} output=${RUN_ROOT}"

pushd "${PN_ROOT}" >/dev/null
for seed in "${SEEDS[@]}"; do
  log_dir="${PROJECT_ROOT}/${RUN_ROOT}/training/seed${seed}"
  out_dir="${PROJECT_ROOT}/${RUN_ROOT}/predictions/seed${seed}"
  mkdir -p "${log_dir}" "${out_dir}/fullpred_test"

  conda run --no-capture-output -n pointcept python train_weld_semseg.py \
    --model pointnet2_sem_seg --root "${DATA_ROOT}" --batch_size "${BATCH_SIZE}" \
    --epoch "${EPOCHS}" --npoint "${NPOINT}" --gpu 0 --num_workers 4 \
    --seed "${seed}" --log_dir "${log_dir}"

  conda run --no-capture-output -n pointcept python tools/export_fullpred_weld.py \
    --root "${DATA_ROOT}" --split test --model pointnet2_sem_seg --log_dir "${log_dir}" \
    --npoint "${NPOINT}" --gpu 0 --out_dir "${out_dir}/fullpred_test" \
    --time_csv "${out_dir}/inference_timing.csv" --seed "${seed}" \
    --protocol-seed "${PROTOCOL_SEED}"
done
popd >/dev/null

python scripts/compare_all_methods.py \
  --method pointnet2_fixed="${RUN_ROOT}/predictions" --chunk-size "${NPOINT}" \
  --protocol-seed "${PROTOCOL_SEED}" --threshold 0.5 \
  --out-dir "${RUN_ROOT}/summary" --strict-protocol --strict-timing

echo "[DONE] ${PROJECT_ROOT}/${RUN_ROOT}/summary/benchmark_summary.csv"
