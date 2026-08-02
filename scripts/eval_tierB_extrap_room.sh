#!/bin/bash
# TIERB extrapolation eval: tierB_room_s0 on the room-32-32 map only
# (beyond its own training grid): N∈{60,120,180} × M∈{100,225,350}.
# 9-task array; target = Tier A's own training grid.
# Writes per-instance CSVs to results/fullpipe_tierB_extrap/room/<map>_n{N}m{M}.csv.
#
# Submit after tierB_room_s0 has finished training:
#   sbatch scripts/eval_tierB_extrap_room.sh

#SBATCH --job-name=mcpf_etierB_ex_room
#SBATCH --array=0-8
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_extrap_room_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_extrap_room_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(room-32-32-4)
NS=(60 120 180)
MS=(100 225 350)

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

mkdir -p "$PROJECT/results/fullpipe_tierB_extrap/room"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierB_room_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 100 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 82800 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierB_extrap/room/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
