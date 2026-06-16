#!/bin/bash
# Paper-maps experiment, LARGER architecture: h256/L8 universal transformer
# (~4-6x the Exp 11 h128/L6 model's 1.2M params), trained jointly on all 36
# paper_maps configs (4 maps × N∈{5,10,15} × M∈{10,20,30}).
# 3 seeds as a Slurm array. Requires a GPU node.
#
# Same data and protocol as exp_paper_current.sh, only the model size differs
# (width 128->256, depth 6->8; lr lowered 5e-4->3e-4 for stability at this size).
# Run in parallel with exp_paper_current.sh:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp_paper_larger.sh

#SBATCH --job-name=mcpf_paper_larger
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/paper_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/paper_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

SEED=$SLURM_ARRAY_TASK_ID

# All 36 (map,N,M) configs, map-major to match gen_paper_maps_data.sh.
DATA_BASE="$PROJECT/data/paper_maps"
DATA_DIRS=""
for MAP in empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4; do
    for N in 5 10 15; do
        for M in 10 20 30; do
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
    --hidden 256 --num_layers 8 --num_heads 4 \
    --use_goal_dists \
    --lr 3e-4 --grad_clip 1.0 \
    --epochs 150 --batch_size 256 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "paper_larger_s${SEED}"

echo "Training complete: paper_larger_s${SEED}"
