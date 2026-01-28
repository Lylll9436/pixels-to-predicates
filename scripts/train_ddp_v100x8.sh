#!/bin/bash
#SBATCH --job-name=dual_layer_ddp
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -o logs/ddp_%j.out
#SBATCH -e logs/ddp_%j.err

# ============================================================
# DDP Training Script for 8x V100 GPUs
# SEU Big Data Computing Center (https://sc.seu.edu.cn)
# Project: Dual-Layer Urban Perception Framework
# ============================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

# Clear modules and set up conda environment
module purge
export PATH=/seu_share2/home/zhangyu/220230195/.conda/envs/ml/bin:$PATH
export LD_LIBRARY_PATH=/seu_share2/home/zhangyu/220230195/.conda/envs/ml/lib:$LD_LIBRARY_PATH

# NCCL settings for multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# DDP settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Working directory
cd /seu_nvme/home/zhangyu/220230195/MyDatasets/server_training_package.tar_20260128144611/output

# Create logs directory if not exists
mkdir -p logs
mkdir -p result/dual_layer_4cities_ddp

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Launch DDP training with torchrun
# - nproc_per_node=8: Use all 8 V100 GPUs
# - Base LR 1.02e-05 will be scaled by world_size (8) in the script
torchrun --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/12_dual_layer_trainer_ddp.py \
    --repr-file packed_all_cities/graph_representations.pt \
    --ids-file packed_all_cities/graph_ids.json \
    --graphs-dir stage_03_scene_graphs_all \
    --macro-graph macro_graph_4cities.pt \
    --macro-mapping macro_graph_4cities.mapping.pkl \
    --comparisons-file llm_pairs/llm_pairs_4cities.csv \
    --learning-rate 1.02e-05 \
    --micro-dim 128 \
    --hidden-dim 256 \
    --num-gnn-layers 2 \
    --num-heads 4 \
    --dropout 0.13 \
    --fusion-mode gated \
    --batch-size 32 \
    --weight-decay 1.05e-05 \
    --epochs 100 \
    --patience 15 \
    --result-dir result/dual_layer_4cities_ddp

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
