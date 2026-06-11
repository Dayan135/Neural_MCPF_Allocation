#!/bin/bash
#SBATCH --job-name=mcpf_poc
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/poc_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/poc_%j.err

# Submit from project root:
#   mkdir -p logs
#   sbatch scripts/poc_2agent_2goal.sh

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

# Build LKH for Linux if not already built
if [ ! -f "$PROJECT/RobustMCPF/LKH-3.0.11/LKH" ]; then
    bash "$PROJECT/scripts/setup_robustmcpf.sh"
fi

# Generate dataset: 500 train / 100 val / 100 test  (N=2, M=2, 5×5 grid)
cd "$PROJECT/dataset_generation"
python build_dataset.py --split train --num_samples 500 --num_agents 2
python build_dataset.py --split val   --num_samples 100 --num_agents 2
python build_dataset.py --split test  --num_samples 100 --num_agents 2

# Train
cd "$PROJECT/training"
python train.py --N 2 --M 2 --epochs 50 --batch_size 64

# Evaluate
cd "$PROJECT/evaluation"
python evaluate.py \
    --checkpoint "$PROJECT/checkpoints/best.pt" \
    --data_dir   "$PROJECT/data" \
    --split test
