#!/bin/bash
# Tier B closing experiment: h128/L6 universal transformer trained on
# all 4 real benchmark maps jointly,
# N/M grid: N∈{30,55,80} × M∈{50,100,150} (see scripts/gen_paper_maps_tierB_data.sh).
# 3 seeds as a Slurm array. Requires a GPU node.
#
# --sample_fraction 0.25: pooling all 36 (map,N,M) configs at their full
# 20k/2k/2k each blew up total data to 864k samples/epoch and OOM'd after
# ~14min (36-config CPU-side dataset + big epochs). 0.25 caps the joint
# run to the SAME total sample budget as one per-map run (9 configs x full
# data), round-robin fair - every config keeps the same fraction, randomly
# subsampled, reproducible via --seed.
#
# Submit after gen_paper_maps_tierB_data.sh completes:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp_tierB_joint.sh

#SBATCH --job-name=mcpf_tierB_joint
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierB_joint_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/tierB_joint_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=$SLURM_ARRAY_TASK_ID

# All (map,N,M) configs, map-major to match gen_paper_maps_tierB_data.sh.
DATA_BASE="$PROJECT/data/paper_maps_tierB"
DATA_DIRS=""
for MAP in empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4; do
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
    --sample_fraction 0.25 \
    --model_type transformer \
    --hidden 128 --num_layers 6 --num_heads 4 \
    --use_goal_dists \
    --lr 5e-4 --grad_clip 1.0 \
    --epochs 50 --batch_size 32 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "tierB_joint_s${SEED}"

echo "Training complete: tierB_joint_s${SEED}"
