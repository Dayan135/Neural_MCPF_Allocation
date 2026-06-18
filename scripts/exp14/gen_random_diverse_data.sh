#!/bin/bash
# Random-diverse dataset for the random-vs-fixed-map comparison (model C, Exp 14).
# Random 32x32 grids, per-instance wall density uniform in [0.0, 0.5], placement varies.
# N∈{5,10,15} × M∈{10,20,30} = 9 (N,M) configs (NO map axis — random grids).
# Per config: train(80k), val(2k), test(2k). 80k matches model B's per-(N,M)-shape
# training volume (B saw 4 maps × 20k = 80k per shape; 720k total across 9 shapes).
# Submit: sbatch scripts/exp14/gen_random_diverse_data.sh

#SBATCH --job-name=mcpf_gen_randdiv
#SBATCH --array=0-8
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_randdiv_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_randdiv_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 9 (N, M) pairs. No map axis: grids are random per instance.
NS=(5 10 15)
MS=(10 20 30)

CONFIGS=()
for N in "${NS[@]}"; do
    for M in "${MS[@]}"; do
        CONFIGS+=("$N $M")
    done
done

IFS=' ' read -r N M <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: N=$N M=$M"

OUTDIR="$PROJECT/data/random_diverse/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 80000 || echo 2000 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --grid_w 32 --grid_h 32 \
        --obstacle_prob 0.0 --obstacle_prob_max 0.5 \
        --num_agents "$N" --num_goals "$M" \
        --out_dir "$OUTDIR" \
        --num_workers 16 --seed 1337
done

echo "Done: N=$N M=$M"
