#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan   # 作业提交的指定分区队列为titan
#SBATCH --qos=titan           # 指定作业的QOS
#SBATCH -J weld-seg-job       # 作业在调度系统中的作业名为weld-seg-job
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 


# python main.py --mode preprocess
python main.py --mode train
# python main.py --mode finetune
# python visualize.py
# python scripts/plot_loss_from_log.py job.41541.out --out loss_plot.png --csv losses.csv

python main.py --mode preprocess > preprocess_log.txt 2>&1
python main.py --mode train > train_log.txt 2>&1
python main.py --mode test > test_log.txt 2>&1

python main.py --mode pretrain > pretrain_log.txt 2>&1
python main.py --mode finetune > finetune_log.txt 2>&1