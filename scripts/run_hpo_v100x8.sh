#!/bin/bash
#SBATCH --job-name=dual_layer_hpo
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o logs/hpo_%j.out
#SBATCH -e logs/hpo_%j.err

# ============================================================
# Bayesian HPO Script for Dual-Layer Model
# SEU Big Data Computing Center
# Project: Dual-Layer Urban Perception Framework
# ============================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=============================================="

# 环境设置
module purge
export PATH=/seu_share2/home/zhangyu/220230195/.conda/envs/ml/bin:$PATH
export LD_LIBRARY_PATH=/seu_share2/home/zhangyu/220230195/.conda/envs/ml/lib:$LD_LIBRARY_PATH

# 工作目录
cd /seu_nvme/home/zhangyu/220230195/MyDatasets/server_training_package.tar_20260128144611/output

# 创建目录
mkdir -p logs
mkdir -p result/hpo

# 打印环境信息
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Optuna version: $(python -c 'import optuna; print(optuna.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 运行 HPO（单 GPU，多 trials）
# 注意：HPO 使用单 GPU 串行运行更稳定，避免多进程冲突
python src/12_dual_layer_trainer_ddp.py \
    --hpo \
    --n-trials 200 \
    --study-name dual_layer_hpo_v1 \
    --storage sqlite:///result/hpo/optuna_study.db \
    --hpo-epochs 20 \
    --repr-file packed_all_cities/graph_representations.pt \
    --ids-file packed_all_cities/graph_ids.json \
    --graphs-dir stage_03_scene_graphs_all \
    --macro-graph macro_graph_4cities.pt \
    --macro-mapping macro_graph_4cities.mapping.pkl \
    --comparisons-file llm_pairs/llm_pairs_4cities.csv \
    --fusion-mode gated \
    --result-dir result/hpo \
    --log-dir logs \
    --seed 42

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
