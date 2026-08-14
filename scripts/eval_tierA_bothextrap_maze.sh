#!/bin/bash
# TIERA both extrapolation eval: tierA_maze_s0 (trained on maze only) on ALL 4 maps
# (beyond its own training N/M grid): N∈{190,205,220} × M∈{370,400,430}.
# 36-task array (4 maps x 9 configs) — same per-map checkpoint as eval_tierA_indist_maze.sh
# / eval_tierA_extrap_maze.sh, now swept across all 4 maps instead of just maze to
# measure generalization to unseen maps (per the professor's per-map-vs-joint comparison).
# Writes per-instance CSVs to results/fullpipe_tierA_bothextrap/maze/<map>_n{N}m{M}.csv.
#
# Submit after tierA_maze_s0 has finished training:
#   sbatch scripts/eval_tierA_bothextrap_maze.sh

#SBATCH --job-name=mcpf_etierA_bo_maze
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_bothextrap_maze_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_bothextrap_maze_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
NS=(190 205 220)
MS=(370 400 430)

CONFIGS=()
for MAP in "${MAPS[@]}"; do
    for N in "${NS[@]}"; do
        for M in "${MS[@]}"; do
            CONFIGS+=("$MAP $N $M")
        done
    done
done

IFS=' ' read -r MAP N M <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: map=$MAP N=$N M=$M"

mkdir -p "$PROJECT/results/fullpipe_tierA_bothextrap/maze"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierA_maze_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 50 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 82800 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierA_bothextrap/maze/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
