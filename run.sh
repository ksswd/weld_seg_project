#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan   # 作业提交的指定分区队列为titan
#SBATCH --qos=titan           # 指定作业的QOS
#SBATCH -J weld-seg-job       # 作业在调度系统中的作业名为weld-seg-job
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 


# python main.py --mode preprocess
# python main.py --mode train
# python main.py --mode finetune
# python test/full_eval.py \
#   --weights weights2/best_finetune.pth \
#   --data_dir data/new/processed_csv_labeled \
#   --val_ratio 1.0 \
#   --max_points 1024 \
#   --passes 1 \
#   --seed 42 \
#   --out_dir data/new/full_eval_all
# python scripts/visualize_fullpred.py data/new/full_eval_all --out data/new/full_eval_all/vis --thr 0.53 --show_bg
# python test/per_case_metrics.py \
#   --input_dir data/new/full_eval_all \
#   --fixed_thr 0.53
# python test/plot_case_metrics.py \
#   --input_dir data/new/full_eval_all \
#   --thr 0.53 \
#   --out_png case_metrics_bars_percent_mm.png
python scripts/visualize_soft_labels.py \
  --input_dir data/new/processed_csv_labeled_soft \
  --out_dir data/new/processed_csv_labeled_soft/soft_vis





# python visualize.py
# python scripts/plot_loss_from_log.py job.41541.out --out loss_plot.png --csv losses.csv

# python main.py --mode preprocess > preprocess_log.txt 2>&1
# python main.py --mode train > train_log.txt 2>&1
# python main.py --mode test > test_log.txt 2>&1
# python scripts/clahe_enhance.py data/ascii_ply/lap1.ply --out . --ply_text

# python main.py --mode pretrain > pretrain_log2.txt 2>&1
# python main.py --mode finetune > finetune_log2.txt 2>&1
# python test/recon_test.py --csv=data/processed_csv/lap1_aug0.csv


# python test/recon_test.py \
#     --csv data/processed_csv2/lap_weld_aug0.csv \
#     --weights weights/best_pretrain.pth \
#     --out_dir data/predictions/recon_vis \
#     --mask_ratio 0.7 \
#     --seed 42
# python scripts/visualize_recon.py data/predictions/recon_vis/lap_weld_aug0_recon_vis.csv --subsample_large
# python test_model_comparison.py

# === 批量 recon_test + 汇总 + 可视化 ===
# TEST_DIR="data/new/recon_test/test_files"
# OUT_DIR="data/new/recon_test/output"
# WEIGHTS="weights2/best_pretrain.pth"
# MASK_RATIO=0.3
# SEED=42

# mkdir -p "$OUT_DIR"

# # 1) 批量运行 recon_test（对 test_files 下每个 csv）
# for csv in "$TEST_DIR"/*.csv; do
#   [ -e "$csv" ] || continue
#   echo "[recon] running: $csv"
#   python test/recon_test.py \
#     --csv "$csv" \
#     --weights "$WEIGHTS" \
#     --out_dir "$OUT_DIR" \
#     --mask_ratio "$MASK_RATIO" \
#     --seed "$SEED"
# done

# # 2) 汇总所有 *_recon_metrics.csv 为一个总表 + 统计表
# python -c "import glob,os,pandas as pd; out_dir='$OUT_DIR'; files=sorted(glob.glob(os.path.join(out_dir,'*_recon_metrics.csv'))); assert files, f'No metrics files found in {out_dir}'; frames=[]; \
# [frames.append(pd.read_csv(f).assign(file=os.path.basename(f).replace('_recon_metrics.csv',''))) for f in files]; \
# all_df=pd.concat(frames,ignore_index=True); all_path=os.path.join(out_dir,'recon_metrics_all.csv'); all_df.to_csv(all_path,index=False); \
# summary=all_df.groupby(['subset','channel'],as_index=False).agg(n_files=('file','nunique'), mae_mean=('mae','mean'), mae_std=('mae','std'), rmse_mean=('rmse','mean'), rmse_std=('rmse','std'), corr_mean=('corr','mean'), corr_std=('corr','std')); \
# sum_path=os.path.join(out_dir,'recon_metrics_summary.csv'); summary.to_csv(sum_path,index=False); print('[summary] wrote:',all_path); print('[summary] wrote:',sum_path)"

# # 3) 生成可视化图（每个样本分析图 + 点云误差图）
# python scripts/visualize_recon.py "$OUT_DIR" --out "$OUT_DIR" --subsample_large

# # 4) 生成“批量汇总可视化”图片（masked曲率为主）
# python -c "import os,pandas as pd,matplotlib.pyplot as plt; out_dir='$OUT_DIR'; p=os.path.join(out_dir,'recon_metrics_all.csv'); df=pd.read_csv(p); d=df[(df['subset']=='masked') & (df['channel'].isin(['curvature_target','curvature_raw']))].copy(); \
# if d.empty: d=df[(df['subset']=='masked')].copy(); d=d.sort_values('file'); \
# plt.figure(figsize=(12,5)); plt.subplot(1,2,1); plt.bar(d['file'], d['mae']); plt.xticks(rotation=75, fontsize=7); plt.title('Masked MAE by file'); plt.tight_layout(); \
# plt.subplot(1,2,2); plt.bar(d['file'], d['corr']); plt.xticks(rotation=75, fontsize=7); plt.title('Masked Corr by file'); plt.tight_layout(); \
# fig_path=os.path.join(out_dir,'recon_metrics_masked_overview.png'); plt.savefig(fig_path,dpi=200,bbox_inches='tight'); print('[summary] wrote:',fig_path)"

# python scripts/make_soft_labels.py \
#   --input_dir data/new/processed_csv_labeled \
#   --output_dir data/new/processed_csv_labeled_soft \
#   --radius 0.2