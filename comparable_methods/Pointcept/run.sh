#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J weld-seg-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1


set -e
## 数据格式处理
# python tools/convert_weld_csv_to_pointcept.py
## 预训练
SAVE_ROOT=weights bash scripts/train.sh -g 1 -d weld -c pretrain-weld-ptv3 -n weld_pretrain

## 微调（加载预训练权重）
SAVE_ROOT=weights bash scripts/train.sh -g 1 -d weld -c semseg-pt-v3m1-0-base -n weld_finetune -w weights/weld/weld_pretrain/model/model_best.pth

## 推理
SAVE_ROOT=weights bash scripts/test.sh -g 1 -d weld -c semseg-pt-v3m1-0-base -n weld_finetune -w model_best

# set -exo pipefail

# PROJECT_DIR=/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/comparable_methods/Pointcept
# POINTOPS_DIR=$PROJECT_DIR/libs/pointops

# cd "$PROJECT_DIR"
# mkdir -p logs

# source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
# conda activate pointcept

# echo "===== Python / Torch / CUDA ====="
# python -V
# python -c "import torch; print('torch=', torch.__version__); print('torch cuda=', torch.version.cuda)"
# which nvcc || true
# nvcc --version || true

# echo "===== install basic deps ====="
# conda install -y conda-forge::libxcrypt

# echo "===== inspect problematic cccl ====="
# if [ -d "$CONDA_PREFIX/targets/x86_64-linux/include/cccl" ]; then
#     echo "Found incompatible cccl, renaming it away..."
#     mv "$CONDA_PREFIX/targets/x86_64-linux/include/cccl" \
#        "$CONDA_PREFIX/targets/x86_64-linux/include/cccl.bak.$(date +%s)"
# fi

# echo "===== inspect thrust / crypt ====="
# find "$CONDA_PREFIX" -path "*thrust/complex.h" 2>/dev/null || true
# find "$CONDA_PREFIX/include" -name crypt.h 2>/dev/null || true

# echo "===== prepare include path ====="
# export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
# export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH:-}"
# export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"

# # 避免继续吃到 targets 里的坏头文件
# export CFLAGS="-I$CONDA_PREFIX/include ${CFLAGS:-}"
# export CXXFLAGS="-I$CONDA_PREFIX/include ${CXXFLAGS:-}"
# export CPPFLAGS="-I$CONDA_PREFIX/include ${CPPFLAGS:-}"

# # 指定编译器
# export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
# export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# # nvcc host compiler
# export CUDAHOSTCXX="$CXX"

# # CUDA 路径
# export CUDA_HOME="$CONDA_PREFIX"
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# echo "===== clean build ====="
# cd "$POINTOPS_DIR"
# rm -rf build dist *.egg-info
# find . -name "*.so" -delete

# echo "===== build pointops ====="
# MAX_JOBS=4 python setup.py install

# echo "===== verify ====="
# python - << 'PY'
# import pointops
# print("pointops import success")
# PY

# bash scripts/train.sh \
# -g 1 \
# -d weld \
# -c semseg-pt-v3m1-0-base \
# -n weld_100epoches

# run Pointcept test (PTv3)
# cd ../Pointcept
# bash scripts/test.sh \
#   -g 1 \
#   -d weld \
#   -c semseg-pt-v3m1-0-base \
#   -n weld_100epoches \
#   -w model_best

