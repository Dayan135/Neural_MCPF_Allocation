#!/bin/bash
# Stage 1 (dataset-selection) crossover eval: all 5 Tier A (dataset-B) checkpoints
# (joint + 4 per-map) evaluated on Tier B's (dataset-A) own N,M grid: N∈{30,55,80} x
# M∈{50,100,150}, all 4 maps. Completes the common-grid comparison requested in
# docs/exp16_data_request.md §4 — Tier B's checkpoints are already evaluated on this
# same grid (indist/mapextrap) and on Tier A's own grid (extrap = this grid's
# counterpart at the large end), so this is the missing quadrant: the large-data-regime
# (dataset-B/Tier A) models evaluated at the small/regular scale.
# 180-task array (5 roles x 4 maps x 9 configs).
# Writes per-instance CSVs to results/fullpipe_tierA_on_tierB_grid/<role>/<map>_n{N}m{M}.csv.
#
#   sbatch scripts/eval_tierA_on_tierB_grid.sh

#SBATCH --job-name=mcpf_etierA_onB
#SBATCH --array=0-179
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_onB_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_onB_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

ROLES=(joint empty random maze room)
MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
NS=(30 55 80)
MS=(50 100 150)

CONFIGS=()
for ROLE in "${ROLES[@]}"; do
    for MAP in "${MAPS[@]}"; do
        for N in "${NS[@]}"; do
            for M in "${MS[@]}"; do
                CONFIGS+=("$ROLE $MAP $N $M")
            done
        done
    done
done

IFS=' ' read -r ROLE MAP N M <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: role=$ROLE map=$MAP N=$N M=$M"

mkdir -p "$PROJECT/results/fullpipe_tierA_on_tierB_grid/$ROLE"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierA_${ROLE}_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 100 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 82800 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierA_on_tierB_grid/$ROLE/${MAP}_n${N}m${M}.csv"

echo "Done: role=$ROLE map=$MAP N=$N M=$M"
