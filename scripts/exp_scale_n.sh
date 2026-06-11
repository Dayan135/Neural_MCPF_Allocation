#!/bin/bash
# Experiment 2: Scale N — MLP vs Transformer across N∈{2,3,4,5} on 8×8 grid
# 4 N values × 2 models × 3 seeds = 24 tasks
# Submit: sbatch scripts/exp_scale_n.sh
#
# Pre-requisite: scripts/gen_scale_data.sh has produced data/n{2,3,4,5}_8x8.
# Chain directly: sbatch --dependency=afterok:<gendata_job_id> scripts/exp_scale_n.sh

#SBATCH --job-name=mcpf_scaleN
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-23
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_scaleN_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_scaleN_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

N_VALUES=(2 3 4 5)
MODEL_TYPES=(mlp transformer)
SEEDS=(0 1 2)

# task_id = n_idx*6 + model_idx*3 + seed_idx
N=${N_VALUES[$((SLURM_ARRAY_TASK_ID / 6))]}
MODEL=${MODEL_TYPES[$(((SLURM_ARRAY_TASK_ID % 6) / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_NAME="scaleN_n${N}_${MODEL}_s${SEED}"
DATA_DIR="$PROJECT/data/n${N}_8x8"

echo "Task $SLURM_ARRAY_TASK_ID: N=$N model=$MODEL seed=$SEED"

cd "$PROJECT/training"
python train.py \
    --model_type "$MODEL" \
    --N "$N" --M "$N" \
    --hidden 64 --num_heads 4 --num_layers 3 \
    --epochs 100 --batch_size 256 --lr 1e-3 --lam 0.1 \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --run_name "$RUN_NAME"

cd "$PROJECT/evaluation"
python evaluate.py \
    --checkpoint "$PROJECT/checkpoints/${RUN_NAME}/best.pt" \
    --data_dir   "$DATA_DIR" \
    --split test
