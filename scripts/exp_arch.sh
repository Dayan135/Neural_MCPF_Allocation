#!/bin/bash
# Experiment 1: Architecture comparison at N=2, 5×5 grid
# 3 model types × 3 seeds = 9 tasks
# Submit: sbatch scripts/exp_arch.sh
#
# Pre-requisites (run once before submitting):
#   mkdir -p logs
#   cd dataset_generation
#   python build_dataset.py --split train --num_samples 5000 --num_agents 2 --num_workers 4
#   python build_dataset.py --split val   --num_samples 1000 --num_agents 2 --num_workers 4
#   python build_dataset.py --split test  --num_samples 1000 --num_agents 2 --num_workers 4

#SBATCH --job-name=mcpf_arch
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_arch_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_arch_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

MODEL_TYPES=(mlp deepsets transformer)
SEEDS=(0 1 2)

MODEL=${MODEL_TYPES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_NAME="arch_${MODEL}_s${SEED}"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL seed=$SEED"

cd "$PROJECT/training"
python train.py \
    --model_type "$MODEL" \
    --N 2 --M 2 \
    --hidden 64 --num_heads 4 --num_layers 3 \
    --epochs 100 --batch_size 256 --lr 1e-3 --lam 0.1 \
    --seed "$SEED" \
    --run_name "$RUN_NAME"

cd "$PROJECT/evaluation"
python evaluate.py \
    --checkpoint "$PROJECT/checkpoints/${RUN_NAME}/best.pt" \
    --data_dir   "$PROJECT/data" \
    --split test
