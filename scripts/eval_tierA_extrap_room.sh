#!/bin/bash
# TIERA extrapolation eval: tierA_room_s0 on the room-32-32 map only
# (beyond its own training grid): N∈{190,205,220} × M∈{370,400,430}.
# 9-task array; N/M chosen to stay under the free-cell ceiling.
# Writes per-instance CSVs to results/fullpipe_tierA_extrap/room/<map>_n{N}m{M}.csv.
#
# Submit after tierA_room_s0 has finished training:
#   sbatch scripts/eval_tierA_extrap_room.sh

#SBATCH --job-name=mcpf_etierA_ex_room
#SBATCH --array=0-8
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_extrap_room_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_extrap_room_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(room-32-32-4)
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

mkdir -p "$PROJECT/results/fullpipe_tierA_extrap/room"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierA_room_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 50 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 82800 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierA_extrap/room/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
