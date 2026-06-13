#!/bin/bash
# Offline metrics for the h128/L6 large model on the diverse test set.
# Per-config (28 configs) × 3 seeds: per-goal acc, full-assignment acc, cost ratio.
# No solver — pure inference, so CPU is fine and fast.
#
# Submit: sbatch scripts/eval_large_offline.sh

#SBATCH --job-name=mcpf_eval_large
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_large_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_large_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
DATA="$PROJECT/data/large_diverse"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

cd "$PROJECT/evaluation"

for SEED in 0 1 2; do
    CKPT="$PROJECT/checkpoints/large_s${SEED}/best.pt"
    for N in 2 3 4 5; do
        for M in 2 3 4 5 6 7 8; do
            echo "##### seed=$SEED N=$N M=$M #####"
            python evaluate.py \
                --checkpoint "$CKPT" \
                --data_dir   "$DATA/n${N}m${M}" \
                --split test
        done
    done
done

echo "Offline eval complete."
