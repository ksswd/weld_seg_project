#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

# 进入脚本所在目录，避免相对路径问题
# cd "$(dirname "$0")"

# ==============================
# 路径配置
# ==============================
INPUT_DIR="data/raw2"
OUT1="data/preds_method1"
OUT2="data/preds_method2"

# ==============================
# 方法1：曲率 + 法向变化阈值
# ==============================
python method1_curvature_normal_threshold.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUT1}"

python visualize_preds.py "${OUT1}" \
  --out "${OUT1}/vis" \
  --show_bg

python full_eval_mathmatic.py \
  --input_dir "${OUT1}" \
  --pattern "*_pred_method1.csv" \
  --pred_col "pred" \
  --out_per_file "summary_method1.csv" \
  --out_global "summary_method1_global.csv"

# ==============================
# 方法2：RANSAC平面剔除 + DBSCAN（调参版）
# 关键：减小 distance_thr 和 eps，抑制大面积误检
# ==============================
python method2_ransac_plane_cluster.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUT2}" \
  --iters 600 \
  --distance_thr 0.10 \
  --eps 0.45 \
  --min_samples 60

python visualize_preds.py "${OUT2}" \
  --out "${OUT2}/vis" \
  --show_bg

python full_eval_mathmatic.py \
  --input_dir "${OUT2}" \
  --pattern "*_pred_method2.csv" \
  --pred_col "pred" \
  --out_per_file "summary_method2.csv" \
  --out_global "summary_method2_global.csv"
