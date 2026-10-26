#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

# python comparable_methods/Pointcept/tools/export_fullpred_weld.py \
#   --pred_dir comparable_methods/Pointcept/weights/weld/weld_finetune/result \
#   --data_root comparable_methods/Pointcept/data/weld \
#   --split test \
#   --out_dir comparable_methods/Pointcept/weights/weld/weld_finetune/result/fullpred

# python scripts/visualize_fullpred.py comparable_methods/Pointcept/weights/weld/weld_finetune/result/fullpred --out comparable_methods/Pointcept/weights/weld/weld_finetune/result/fullpred/vis --thr 0.5 --show_bg

cd comparable_methods/Pointnet_Pointnet2_pytorch
python tools/export_fullpred_weld.py \
  --root data/weld2 \
  --split test \
  --log_dir log/weld_fold1 \
  --out_dir fullpred \
  --npoint 4096 \
  --gpu 0
cd ../..

python scripts/visualize_fullpred.py comparable_methods/Pointnet_Pointnet2_pytorch/fullpred --out comparable_methods/Pointnet_Pointnet2_pytorch/fullpred/vis --thr 0.5 --show_bg