#!/bin/bash
# TIERA map extrapolation eval: tierA_random_s0 (trained on random only) on ALL 4 maps
# (its own training N/M grid): N∈{60,120,180} × M∈{100,225,350}.
# 36-task array (4 maps x 9 configs) — same per-map checkpoint as eval_tierA_indist_random.sh
# / eval_tierA_extrap_random.sh, now swept across all 4 maps instead of just random to
# measure generalization to unseen maps (per the professor's per-map-vs-joint comparison).
# Writes per-instance CSVs to results/fullpipe_tierA_mapextrap/random/<map>_n{N}m{M}.csv.
#
# Submit after tierA_random_s0 has finished training:
#   sbatch scripts/eval_tierA_mapextrap_random.sh

#SBATCH --job-name=mcpf_etierA_mp_rand
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_mapextrap_random_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierA_mapextrap_random_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
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

mkdir -p "$PROJECT/results/fullpipe_tierA_mapextrap/random"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierA_random_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 100 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 82800 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierA_mapextrap/random/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
