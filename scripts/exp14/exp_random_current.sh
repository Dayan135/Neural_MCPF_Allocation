#!/bin/bash
# Model C1 (Exp 14): h128/L6 universal transformer trained on the random-diverse dataset
# (9 (N,M) configs, random 32x32 grids, walls 0-0.5). Copies exp_paper_current.sh
# exactly; only --data_dirs differs (random_diverse instead of paper_maps, no map axis).
# Submit after gen_random_diverse_data.sh:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp14/exp_random_current.sh

#SBATCH --job-name=mcpf_rand_current
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_current_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_current_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

SEED=$SLURM_ARRAY_TASK_ID

# 9 (N,M) configs of the random-diverse dataset (no map axis).
DATA_BASE="$PROJECT/data/random_diverse"
DATA_DIRS=""
for N in 5 10 15; do
    for M in 10 20 30; do
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
    --run_name "random_current_s${SEED}"

echo "Training complete: random_current_s${SEED}"
