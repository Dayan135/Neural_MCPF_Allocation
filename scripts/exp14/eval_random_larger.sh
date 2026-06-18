#!/bin/bash
# Full-pipeline eval (Exp 14): model C2 (random_larger_s0, h256/L8) on all 36 real-map
# configs. Per-instance CSVs to results/fullpipe_random/larger/. Same seed/instances as
# the Exp 12 B-model eval → C2-vs-B2 on identical problems.
# Submit: sbatch scripts/exp14/eval_random_larger.sh

#SBATCH --job-name=mcpf_evalrand_larger
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
NS=(5 10 15)
MS=(10 20 30)

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

mkdir -p "$PROJECT/results/fullpipe_random/larger"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/random_larger_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 200 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 27000 \
    --csv "$PROJECT/results/fullpipe_random/larger/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
