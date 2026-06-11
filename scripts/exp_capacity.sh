#!/bin/bash
# Experiment 4: Capacity scale-up — Transformer, N=3, 8×8, 20k train
# hidden ∈ {64, 128, 256} × layers ∈ {3, 6} minus the existing baseline (h64/L3)
# = 5 configs × 3 seeds = 15 tasks
#
# Pre-requisite: data/n3_8x8_pool exists (gen_scale_data.sh already ran).
# Baseline:      datascale_t20000_s{0,1,2} (hidden=64, L=3) — already in checkpoints.
# Compare:       python compare.py --pattern "cap_*" --data_dir ../data/n3_8x8_pool
#                (add datascale_t20000_* to the pattern to include baseline in one table)

#SBATCH --job-name=mcpf_capacity
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_cap_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_cap_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
DATA_DIR="$PROJECT/data/n3_8x8_pool"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

# 5 configs (skip h64/L3 — already have datascale_t20000 baseline):
#   0: h64  L6
#   1: h128 L3
#   2: h128 L6
#   3: h256 L3
#   4: h256 L6
HIDDENS=(64  128 128 256 256)
LAYERS=(6    3   6   3   6)
SEEDS=(0 1 2)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
HIDDEN=${HIDDENS[$CONFIG_IDX]}
NUM_LAYERS=${LAYERS[$CONFIG_IDX]}
RUN_NAME="cap_h${HIDDEN}_l${NUM_LAYERS}_s${SEED}"

echo "Task $SLURM_ARRAY_TASK_ID: hidden=$HIDDEN layers=$NUM_LAYERS seed=$SEED"

cd "$PROJECT/training"
python train.py \
    --model_type transformer \
    --N 3 --M 3 \
    --hidden "$HIDDEN" --num_heads 4 --num_layers "$NUM_LAYERS" \
    --epochs 150 --batch_size 256 --lr 1e-3 --lam 0.1 \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --run_name "$RUN_NAME"

cd "$PROJECT/evaluation"
python evaluate.py \
    --checkpoint "$PROJECT/checkpoints/${RUN_NAME}/best.pt" \
    --data_dir   "$DATA_DIR" \
    --split test
