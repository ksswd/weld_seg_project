#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

# cd data/new
# python data_aug.py --input_dir raw_no_side_labeled --output_dir augmentation_final --num_augment 10
# cd ../../
python main.py --mode preprocess --raw_data_dir data/new/aug_data --processed_data_dir data/new/processed_data