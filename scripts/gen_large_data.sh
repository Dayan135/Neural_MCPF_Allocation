#!/bin/bash
# Diverse dataset for Exp 11: N∈{2,3,4,5} × M∈{2..8} = 28 configs.
# Per instance: random W,H ∈ [8,12], obstacle_prob ∈ [0.1, 0.5].
# Generates train(30k), val(3k), test(3k) per config — 840k training samples total.
#
# Each array task handles one (N,M) pair → no temp-file collision between tasks.
# Submit: sbatch scripts/gen_large_data.sh

#SBATCH --job-name=mcpf_gen_large
#SBATCH --array=0-27
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_large_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_large_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 28 (N,M) pairs, row-major: N∈{2,3,4,5} × M∈{2..8}
CONFIGS=(
    "2 2" "2 3" "2 4" "2 5" "2 6" "2 7" "2 8"
    "3 2" "3 3" "3 4" "3 5" "3 6" "3 7" "3 8"
    "4 2" "4 3" "4 4" "4 5" "4 6" "4 7" "4 8"
    "5 2" "5 3" "5 4" "5 5" "5 6" "5 7" "5 8"
)
IFS=' ' read -r N M <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: N=$N M=$M"

OUTDIR="$PROJECT/data/large_diverse/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 30000 || echo 3000 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --num_agents "$N" --num_goals "$M" \
        --grid_w 8 --grid_w_max 12 \
        --grid_h 8 --grid_h_max 12 \
        --obstacle_prob 0.1 --obstacle_prob_max 0.5 \
        --out_dir "$OUTDIR" \
        --num_workers 8 --seed 1337
done

echo "Done: N=$N M=$M"
