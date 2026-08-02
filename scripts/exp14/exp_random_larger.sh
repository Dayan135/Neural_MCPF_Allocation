#!/bin/bash
# Model C2 (Exp 14): h256/L8 universal transformer trained on the random-diverse dataset.
# Copies exp_paper_larger.sh exactly; only --data_dirs differs (random_diverse, no map axis).
# Run in parallel with exp_random_current.sh:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp14/exp_random_larger.sh

#SBATCH --job-name=mcpf_rand_larger
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU
# h256/L8 needs a 24GB card (OOMs on the 10.5GB rtx_2080); reduce fragmentation too.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=$SLURM_ARRAY_TASK_ID

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
    --hidden 256 --num_layers 8 --num_heads 4 \
    --use_goal_dists \
    --lr 3e-4 --grad_clip 1.0 \
    --epochs 150 --batch_size 256 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "random_larger_s${SEED}"

echo "Training complete: random_larger_s${SEED}"
