#!/bin/bash
# Tier B closing experiment: h128/L6 universal transformer trained on
# the room-32-32 map only,
# N/M grid: N∈{30,55,80} × M∈{50,100,150} (see scripts/gen_paper_maps_tierB_data.sh).
# 3 seeds as a Slurm array. Requires a GPU node.
#
# Submit after gen_paper_maps_tierB_data.sh completes:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp_tierB_room.sh

#SBATCH --job-name=mcpf_tierB_room
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierB_room_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierB_room_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=$SLURM_ARRAY_TASK_ID

# This map's (N,M) configs only, map-major to match gen_paper_maps_tierB_data.sh.
DATA_BASE="$PROJECT/data/paper_maps_tierB"
DATA_DIRS=""
for MAP in room-32-32-4; do
    for N in 30 55 80; do
        for M in 50 100 150; do
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
    --epochs 50 --batch_size 32 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "tierB_room_s${SEED}"

echo "Training complete: tierB_room_s${SEED}"
