#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan   # 作业提交的指定分区队列为titan
#SBATCH --qos=titan           # 指定作业的QOS
#SBATCH -J weld-seg-job       # 作业在调度系统中的作业名为weld-seg-job
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

# 打印当前主机名
echo "Running on host: $(hostname)"

conda --version

# 创建并激活虚拟环境
ENV_NAME="Weld-Seg"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

if conda env list | grep -q $ENV_NAME; then
    echo "Environment $ENV_NAME exists, activating..."
    conda activate $ENV_NAME
else
    echo "Creating new environment $ENV_NAME..."
    conda create -y -n $ENV_NAME python=3.8
    conda activate $ENV_NAME
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo "System Information:"
lscpu
echo "CUDA devices:"
nvidia-smi

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

echo "Starting execution at $(date)"

# python main.py --mode preprocess
python main.py --mode train
# python main.py --mode finetune
# python visualize.py
# python scripts/plot_loss_from_log.py job.41541.out --out loss_plot.png --csv losses.csv

echo "Finished execution at $(date)"