#!/bin/bash
#SBATCH --job-name=dual_layer_hpo_a100
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o logs/hpo_a100_%j.out
#SBATCH -e logs/hpo_a100_%j.err

# ============================================================
# Parallel Bayesian HPO Script for Dual-Layer Model (4x A100)
# SEU Big Data Computing Center
# Project: Dual-Layer Urban Perception Framework
# 
# Strategy: Launch 4 parallel Optuna workers, each on one GPU
#           All workers share the same SQLite database
#           Optuna handles trial coordination automatically
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

# HPO 配置
STUDY_NAME="dual_layer_hpo_a100_v1"
STORAGE="sqlite:///result/hpo/optuna_study_a100.db"
TRIALS_PER_WORKER=50  # 4 workers × 50 = 200 total trials
HPO_EPOCHS=20

# 公共参数
COMMON_ARGS="--hpo \
    --study-name ${STUDY_NAME} \
    --storage ${STORAGE} \
    --hpo-epochs ${HPO_EPOCHS} \
    --repr-file packed_all_cities/graph_representations.pt \
    --ids-file packed_all_cities/graph_ids.json \
    --graphs-dir stage_03_scene_graphs_all \
    --macro-graph macro_graph_4cities.pt \
    --macro-mapping macro_graph_4cities.mapping.pkl \
    --comparisons-file llm_pairs/llm_pairs_4cities.csv \
    --fusion-mode gated \
    --result-dir result/hpo \
    --log-dir logs \
    --seed 42"

# 启动 4 个并行 HPO worker
echo ""
echo "=============================================="
echo "Launching 4 parallel HPO workers..."
echo "Study: ${STUDY_NAME}"
echo "Storage: ${STORAGE}"
echo "Trials per worker: ${TRIALS_PER_WORKER}"
echo "Total trials: $((4 * TRIALS_PER_WORKER))"
echo "=============================================="
echo ""

declare -a WORKER_PIDS

for GPU_ID in 0 1 2 3; do
    echo "[$(date)] Starting worker ${GPU_ID} on GPU ${GPU_ID}..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/12_dual_layer_trainer_ddp.py \
        ${COMMON_ARGS} \
        --n-trials ${TRIALS_PER_WORKER} \
        > logs/hpo_worker_${SLURM_JOB_ID}_gpu${GPU_ID}.log 2>&1 &
    
    # 记录 worker PID
    WORKER_PIDS[$GPU_ID]=$!
    echo "[$(date)] Worker ${GPU_ID} started with PID ${WORKER_PIDS[$GPU_ID]}"
done

echo "=============================================="
echo "All workers launched. Waiting for completion..."
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo "=============================================="

# 等待所有 worker 完成
wait

echo ""
echo "=============================================="
echo "HPO Results Summary"
echo "=============================================="
python -c "
import optuna

study = optuna.load_study(
    study_name='${STUDY_NAME}',
    storage='${STORAGE}'
)

print(f'Total trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial.number}')
print(f'Best value (validation AUC): {study.best_trial.value:.4f}')
print()
print('Best hyperparameters:')
for key, value in study.best_trial.params.items():
    print(f'  {key}: {value}')
"

echo "=============================================="
echo "Done! Check logs/hpo_worker_${SLURM_JOB_ID}_gpu*.log for individual worker logs"
echo "End time: $(date)"
echo "=============================================="
