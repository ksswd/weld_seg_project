#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

# export PYTHONPATH=./
# python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
#   --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6 \
#   --dataset_root ./Stanford3dDataset_v1.2 \
#   --output_root ./data/s3dis

export PYTHONPATH=./
python tools/train.py \
  --config-file configs/s3dis/semseg-pt-v2m2-0-base.py \
  --num-gpus 1 \
  --options save_path=exp/s3dis/semseg-pt-v2m2-0-base_bs2 batch_size=2 num_worker=6

