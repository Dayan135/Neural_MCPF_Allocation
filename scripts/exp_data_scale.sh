#!/bin/bash
# Experiment 3: Data scale — Transformer, N=3, 8×8 grid
# Train sizes {1k, 5k, 10k, 20k} × 3 seeds = 12 tasks
# Submit: sbatch scripts/exp_data_scale.sh
#
# Pre-requisites (generate a large pool once, reuse subsets via --num_samples):
#   cd dataset_generation
#   python build_dataset.py --split train --num_samples 20000 --num_agents 3 \
#       --grid_w 8 --grid_h 8 --out_dir ../data/n3_8x8 --num_workers 4
#   python build_dataset.py --split val   --num_samples 2000  --num_agents 3 \
#       --grid_w 8 --grid_h 8 --out_dir ../data/n3_8x8 --num_workers 4
#   python build_dataset.py --split test  --num_samples 2000  --num_agents 3 \
#       --grid_w 8 --grid_h 8 --out_dir ../data/n3_8x8 --num_workers 4

#SBATCH --job-name=mcpf_datascale
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_datascale_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/exp_datascale_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
DATA_DIR="$PROJECT/data/n3_8x8"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

TRAIN_SIZES=(1000 5000 10000 20000)
SEEDS=(0 1 2)

TRAIN_SIZE=${TRAIN_SIZES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_NAME="datascale_t${TRAIN_SIZE}_s${SEED}"

echo "Task $SLURM_ARRAY_TASK_ID: train_size=$TRAIN_SIZE seed=$SEED"

# Write a temporary data dir with only TRAIN_SIZE train samples
TMP_DATA="$PROJECT/data/tmp_${RUN_NAME}"
mkdir -p "$TMP_DATA/train" "$TMP_DATA/val" "$TMP_DATA/test"

# Slice the pre-generated train data
python -c "
import numpy as np, sys
n = int(sys.argv[1])
src = sys.argv[2]
dst = sys.argv[3]
D = np.load(f'{src}/train/D_matrices.npy')[:n]
Y = np.load(f'{src}/train/Y_matrices.npy')[:n]
np.save(f'{dst}/train/D_matrices.npy', D)
np.save(f'{dst}/train/Y_matrices.npy', Y)
import shutil
shutil.copy(f'{src}/val/D_matrices.npy',  f'{dst}/val/D_matrices.npy')
shutil.copy(f'{src}/val/Y_matrices.npy',  f'{dst}/val/Y_matrices.npy')
shutil.copy(f'{src}/test/D_matrices.npy', f'{dst}/test/D_matrices.npy')
shutil.copy(f'{src}/test/Y_matrices.npy', f'{dst}/test/Y_matrices.npy')
" "$TRAIN_SIZE" "$DATA_DIR" "$TMP_DATA"

cd "$PROJECT/training"
python train.py \
    --model_type transformer \
    --N 3 --M 3 \
    --hidden 64 --num_heads 4 --num_layers 3 \
    --epochs 100 --batch_size 256 --lr 1e-3 --lam 0.1 \
    --seed "$SEED" \
    --data_dir "$TMP_DATA" \
    --run_name "$RUN_NAME"

cd "$PROJECT/evaluation"
python evaluate.py \
    --checkpoint "$PROJECT/checkpoints/${RUN_NAME}/best.pt" \
    --data_dir   "$TMP_DATA" \
    --split test

rm -rf "$TMP_DATA"
