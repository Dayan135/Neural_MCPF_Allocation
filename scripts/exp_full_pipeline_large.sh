#!/bin/bash
# Full-pipeline execution-cost sweep for the h128/L6 large model (large_s0),
# at the SAME conditions as the Exp 10 sweep (8x8, obstacle_prob=0.1) so the
# 28-config table is directly comparable to the h64/L3 universal_s0 numbers.
#
# N∈{2,3,4,5} × M∈{2..8} = 28 configs, 200 instances each.
# Submit: sbatch scripts/exp_full_pipeline_large.sh

#SBATCH --job-name=mcpf_fullpipe_large
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_large_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_large_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
CKPT="$PROJECT/checkpoints/large_s0/best.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

export MKL_THREADING_LAYER=GNU

cd "$PROJECT/evaluation"

for N in 2 3 4 5; do
    for M in 2 3 4 5 6 7 8; do
        echo "##### N=$N M=$M #####"
        python full_pipeline_eval.py \
            --checkpoint "$CKPT" \
            --grid_w 8 --grid_h 8 \
            --num_agents "$N" --num_goals "$M" \
            --n_instances 200 \
            --csv "$PROJECT/results/fullpipe_large/n${N}m${M}.csv"
    done
done

echo "Large-model full pipeline sweep complete."
