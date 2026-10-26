#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J weld-seg-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:0

set -euo pipefail

python tools/convert_weld_ply_to_pointcept.py \
  --in-root /lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset \
  --out-root data/weld
