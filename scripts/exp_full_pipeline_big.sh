#!/bin/bash
# Big full-pipeline sweep: N∈{2,3,4,5} × M∈{2..8} = 28 configs, 200 instances each.
# universal_s0 was trained on N∈{2,3,4} × M∈{2..6} (8×8 only) — so the N=5 rows
# and M∈{7,8} columns are ZERO-SHOT extrapolation beyond the training range.
#
# Submit: sbatch scripts/exp_full_pipeline_big.sh

#SBATCH --job-name=mcpf_fullpipe_big
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_big_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_big_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
CKPT="$PROJECT/checkpoints/universal_s0/best.pt"

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
            --csv "$PROJECT/results/fullpipe_big/n${N}m${M}.csv"
    done
done

echo "Big full pipeline sweep complete."
