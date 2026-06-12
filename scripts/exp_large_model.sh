#!/bin/bash
# Exp 11: h128/L6 universal transformer on the large_diverse dataset.
# 3 seeds as a Slurm array.  Requires a GPU node.
#
# Submit: sbatch scripts/exp_large_model.sh
# (run after gen_large_data.sh completes and all 28 configs have train/val/test)

#SBATCH --job-name=mcpf_large_model
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/large_model_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/large_model_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

SEED=$SLURM_ARRAY_TASK_ID

# All 28 (N,M) configs: N∈{2,3,4,5} × M∈{2..8}
DATA_BASE="$PROJECT/data/large_diverse"
DATA_DIRS=""
for N in 2 3 4 5; do
    for M in 2 3 4 5 6 7 8; do
        DIR="${DATA_BASE}/n${N}m${M}"
        DATA_DIRS="${DATA_DIRS:+${DATA_DIRS},}${DIR}"
    done
done

cd "$PROJECT/training"
python train.py \
    --mixed \
    --data_dirs "$DATA_DIRS" \
    --model_type transformer \
    --hidden 128 --num_layers 6 --num_heads 4 \
    --use_goal_dists \
    --lr 5e-4 --grad_clip 1.0 \
    --epochs 150 --batch_size 256 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "large_s${SEED}"

echo "Training complete: large_s${SEED}"
