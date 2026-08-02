#!/bin/bash
# TIERB in-distribution eval: tierB_joint_s0 on all 4 maps
# (its own training grid): N∈{30,55,80} × M∈{50,100,150}.
# 36-task array (4 maps x 9 configs).
# Writes per-instance CSVs to results/fullpipe_tierB_indist/joint/<map>_n{N}m{M}.csv.
#
# Submit after tierB_joint_s0 has finished training:
#   sbatch scripts/eval_tierB_indist_joint.sh

#SBATCH --job-name=mcpf_etierB_in_join
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_indist_joint_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/eval_tierB_indist_joint_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
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

mkdir -p "$PROJECT/results/fullpipe_tierB_indist/joint"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/tierB_joint_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 200 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 27000 \
    --instance_timeout 600 \
    --csv "$PROJECT/results/fullpipe_tierB_indist/joint/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
