#!/bin/bash
# Tier A closing experiment: h128/L6 universal transformer trained on
# the empty-32-32 map only,
# N/M grid: N∈{60,120,180} × M∈{100,225,350} (see scripts/gen_paper_maps_tierA_data.sh).
# 3 seeds as a Slurm array. Requires a GPU node.
#
# Submit after gen_paper_maps_tierA_data.sh completes:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp_tierA_empty.sh

#SBATCH --job-name=mcpf_tierA_empty
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierA_empty_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierA_empty_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

SEED=$SLURM_ARRAY_TASK_ID

# This map's (N,M) configs only, map-major to match gen_paper_maps_tierA_data.sh.
DATA_BASE="$PROJECT/data/paper_maps_tierA"
DATA_DIRS=""
for MAP in empty-32-32; do
    for N in 60 120 180; do
        for M in 100 225 350; do
            DIR="${DATA_BASE}/${MAP}/n${N}m${M}"
            DATA_DIRS="${DATA_DIRS:+${DATA_DIRS},}${DIR}"
        done
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
    --run_name "tierA_empty_s${SEED}"

echo "Training complete: tierA_empty_s${SEED}"
