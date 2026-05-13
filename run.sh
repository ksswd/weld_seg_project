#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

set -e

SIMPLE_THR="0.5"
MAX_POINTS="4096"
PASSES="1"
SEED="114514"
SOFT_THR="0.5"

# ==============================
# 路径与评估参数（不使用fold，直接跑指定目录）
# ==============================
DATA_DIR="data/dataset_processed"
OUT_DIR="results/final/dataset_processed_eval"
WEIGHTS="results/final/fold_1/weights/best_finetune.pth"

  mkdir -p "${OUT_DIR}" "${WEIGHTS_SAVE_DIR}"

  # ==============================
  # 按fold循环：训练 + 评估 + 可视化 + 指标
  # ==============================


echo "=============================================="
echo "[RUN] dataset_processed (no fold split)"
echo "=============================================="

  # 训练主流程（按需开启）
  # python main.py --mode pretrain --fold_id "${FOLD_ID}" --seed "${SEED}" \
  #   --processed_data_dir "${PROCESSED_DATA_DIR}" \
  #   --pretrain_data_dir "${PRETRAIN_DATA_DIR}" \
  #   --weights_save_dir "${WEIGHTS_SAVE_DIR}"
  # python main.py --mode finetune --fold_id "${FOLD_ID}" --seed "${SEED}" \
  #   --processed_data_dir "${PROCESSED_DATA_DIR}" \
  #   --labeled_data_dir "${LABELED_DATA_DIR}" \
  #   --weights_save_dir "${WEIGHTS_SAVE_DIR}" \
  #   --pretrained_weights "${PRETRAINED_WEIGHTS}" \
  #   --test_weights "${TEST_WEIGHTS}"

  # 标准评估：输出到 OUT_DIR/FOLD_ID
  python main.py --mode full_eval \
    --fold_id "${FOLD_ID}" \
    --weights "${WEIGHTS}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --max_points "${MAX_POINTS}" \
    --passes "${PASSES}" \
    --simple_thr "${SIMPLE_THR}" \
    --seed "${SEED}" \
    --processed_data_dir "${PROCESSED_DATA_DIR}" \
    --weights_save_dir "${WEIGHTS_SAVE_DIR}"

  # 可视化与指标统计
  # FOLD_OUT_DIR="${OUT_DIR}/${FOLD_ID}"
  # python scripts/visualize_fullpred.py "${FOLD_OUT_DIR}" --out "${FOLD_OUT_DIR}/vis" --thr "${SIMPLE_THR}" --show_bg
  # python test/per_case_metrics.py --input_dir "${FOLD_OUT_DIR}" --fixed_thr "${SIMPLE_THR}" --soft_thr "${SOFT_THR}"
  # python test/plot_case_metrics.py --input_dir "${FOLD_OUT_DIR}" --thr "${SIMPLE_THR}" --out_png "${FOLD_OUT_DIR}/case_metrics_bars_percent_mm.png"
done

# ==============================
# 多fold汇总：对各fold的 summary_metrics_global.csv 求平均
# 输出:
# 1) summary_metrics_global_all_folds.csv
# 2) summary_metrics_global_all_folds_avg.csv
# ==============================
# python - <<'PY'
# import os
# import pandas as pd

# out_dir = "data/new/full_eval3"
# folds = ["fold_6"]

# rows = []
# for fold in folds:
#     p = os.path.join(out_dir, fold, "summary_metrics_global.csv")
#     if not os.path.exists(p):
#         print(f"[WARN] 缺少: {p}")
#         continue
#     df = pd.read_csv(p)
#     if len(df) == 0:
#         print(f"[WARN] 空文件: {p}")
#         continue
#     r = df.iloc[0].copy()
#     r["fold_id"] = fold
#     rows.append(r)

# if not rows:
#     raise SystemExit("[ERROR] 没有找到任何 fold 的 summary_metrics_global.csv")

# df_all = pd.DataFrame(rows)
# all_path = os.path.join(out_dir, "summary_metrics_global_all_folds.csv")
# df_all.to_csv(all_path, index=False)

# num_cols = df_all.select_dtypes(include="number").columns
# avg_df = pd.DataFrame([df_all[num_cols].mean(numeric_only=True)])
# avg_df.insert(0, "n_folds", len(df_all))
# avg_path = os.path.join(out_dir, "summary_metrics_global_all_folds_avg.csv")
# avg_df.to_csv(avg_path, index=False)

# print(f"[DONE] all folds metrics  -> {all_path}")
# print(f"[DONE] mean over folds   -> {avg_path}")

# for k in ["hard_weld_iou", "hard_weld_f1", "hard_weld_precision", "hard_weld_recall", "hard_miou", "hard_oa"]:
#     if k in df_all.columns:
#         print(f"{k}: mean={df_all[k].mean():.6f}, std={df_all[k].std(ddof=1) if len(df_all)>1 else 0.0:.6f}")
# PY
