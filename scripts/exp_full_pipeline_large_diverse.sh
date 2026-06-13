#!/bin/bash
# Full-pipeline execution-cost sweep for large_s0 ON ITS TRAINING DISTRIBUTION:
# per-instance random grids (8-12 x 8-12) and wall density (0.1-0.5).
# This is the on-distribution counterpart to exp_full_pipeline_large.sh (which
# fixes 8x8/p=0.1 for direct comparison to the Exp 10 baseline).
#
# N∈{2,3,4,5} × M∈{2..8} = 28 configs, 200 instances each.
# Submit: sbatch scripts/exp_full_pipeline_large_diverse.sh

#SBATCH --job-name=mcpf_fp_large_div
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fp_large_div_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fp_large_div_%j.err

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
            --grid_w 8 --grid_w_max 12 \
            --grid_h 8 --grid_h_max 12 \
            --obstacle_prob 0.1 --obstacle_prob_max 0.5 \
            --num_agents "$N" --num_goals "$M" \
            --n_instances 200 \
            --csv "$PROJECT/results/fullpipe_large_diverse/n${N}m${M}.csv"
    done
done

echo "Large-model diverse full pipeline sweep complete."
