#!/bin/bash
# TIERB in-distribution eval: tierB_maze_s0 on the maze-32-32 map only
# (its own training grid): N∈{30,55,80} × M∈{50,100,150}.
# 9-task array (this map's 9 configs only).
# Writes per-instance CSVs to results/fullpipe_tierB_indist/maze/<map>_n{N}m{M}.csv.
#
# Submit after tierB_maze_s0 has finished training:
#   sbatch scripts/eval_tierB_indist_maze.sh

#SBATCH --job-name=mcpf_etierB_in_maze
#SBATCH --array=0-8
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_indist_maze_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_indist_maze_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(maze-32-32-2)
NS=(30 55 80)
MS=(50 100 150)

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

mkdir -p "$PROJECT/results/fullpipe_tierB_indist/maze"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierB_maze_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 200 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 27000 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierB_indist/maze/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
