#!/bin/bash
#SBATCH -o pointnet2_%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J pointnet2-weld-seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

cd /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/Pointnet_Pointnet2_pytorch

SEEDS=(444 523 1944 8120)
MAX_POINTS=4096
EPOCHS=200
RAW_DATA_ROOT="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset_1-17"
PROCESSED_ROOT="data/weld_from_ply"
RESULT_ROOT="log/seed_points_sweep"
EXPORT_ROOT="fullpred/seed_points_sweep"
REPORT_TOOL="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/baseline_report.py"
MODEL_NAME="pointnet2_sem_seg"
BATCH_SIZE=8
GPU_ID=0

python - <<'PY'
import os, glob, json, numpy as np
RAW_DATA_ROOT = "/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset_1-17"
PROCESSED_ROOT = "data/weld_from_ply"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def read_ascii_ply(ply_path):
    with open(ply_path, "r", encoding="utf-8") as f: lines = f.readlines()
    props, end_idx = [], None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("property "): props.append(s.split()[-1])
        if s == "end_header": end_idx = i; break
    data = np.loadtxt(ply_path, skiprows=end_idx + 1)
    if data.ndim == 1: data = data[None, :]
    prop_to_idx = {name: idx for idx, name in enumerate(props)}
    xyz = data[:, [prop_to_idx["x"], prop_to_idx["y"], prop_to_idx["z"]]].astype(np.float32)
    seg = data[:, prop_to_idx["scalar_label"]].astype(np.int64)
    normal = np.zeros_like(xyz, dtype=np.float32)
    color = np.zeros_like(xyz, dtype=np.float32)
    return xyz, normal, color, seg
summary = {"raw_root": os.path.abspath(RAW_DATA_ROOT), "processed_root": os.path.abspath(PROCESSED_ROOT), "files": []}
for src_split, dst_split in SPLIT_MAP.items():
    split_dir = os.path.join(RAW_DATA_ROOT, src_split)
    for ply_path in sorted(glob.glob(os.path.join(split_dir, "*.ply"))):
        sample_name = os.path.splitext(os.path.basename(ply_path))[0]
        out_dir = os.path.join(PROCESSED_ROOT, dst_split, sample_name)
        ensure_dir(out_dir)
        xyz, normal, color, seg = read_ascii_ply(ply_path)
        np.save(os.path.join(out_dir, "coord.npy"), xyz)
        np.save(os.path.join(out_dir, "normal.npy"), normal)
        np.save(os.path.join(out_dir, "color.npy"), color)
        np.save(os.path.join(out_dir, "segment.npy"), seg)
        summary["files"].append({"split": dst_split, "sample_name": sample_name, "ply_path": os.path.abspath(ply_path), "num_points": int(xyz.shape[0])})
ensure_dir(os.path.join(PROCESSED_ROOT, "meta"))
with open(os.path.join(PROCESSED_ROOT, "meta", "conversion_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[DONE] converted to: {PROCESSED_ROOT}")
PY

for SEED in "${SEEDS[@]}"; do
  EXP_NAME="pointnet2_seed${SEED}_pts${MAX_POINTS}"
  LOG_DIR="${RESULT_ROOT}/${EXP_NAME}"
  EXPORT_DIR="${EXPORT_ROOT}/${EXP_NAME}"
  MODEL_PATH="${LOG_DIR}/checkpoints/best_model.pth"
mkdir -p "${LOG_DIR}" "${EXPORT_DIR}"
  export PYTHONHASHSEED="${SEED}"

  python train_weld_semseg.py \
    --model "${MODEL_NAME}" \
    --root "${PROCESSED_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --epoch "${EPOCHS}" \
    --npoint "${MAX_POINTS}" \
    --gpu "${GPU_ID}" \
    --log_dir "${LOG_DIR}"

python train_weld_semseg.py \
  --model "${MODEL_NAME}" \
  --root "${PROCESSED_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --epoch "${EPOCHS}" \
    --npoint "${MAX_POINTS}" \
  --gpu "${GPU_ID}" \
  --log_dir "${LOG_DIR}" \
  --eval_only

python tools/export_fullpred_weld.py \
  --root "${PROCESSED_ROOT}" \
  --split test \
  --model "${MODEL_NAME}" \
  --log_dir "${LOG_DIR}" \
    --npoint "${MAX_POINTS}" \
  --gpu "${GPU_ID}" \
    --out_dir "${EXPORT_DIR}/fullpred_test" \
    --time_csv "${EXPORT_DIR}/inference_timing.csv" \
    --seed "${SEED}"

  python "${REPORT_TOOL}" \
    --method pointnet2 \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --max_points "${MAX_POINTS}" \
    --epochs "${EPOCHS}" \
    --model_path "${MODEL_PATH}" \
    --pred_dir "${EXPORT_DIR}/fullpred_test" \
  --out_dir "${EXPORT_DIR}" \
    --log_path "${LOG_DIR}/log.txt"
done

python "${REPORT_TOOL}" --aggregate --root_dir "${EXPORT_ROOT}" --method pointnet2
