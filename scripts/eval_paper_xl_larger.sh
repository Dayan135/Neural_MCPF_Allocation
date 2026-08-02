#!/bin/bash
# Zero-shot extrapolation eval: paper_larger_s0 (h256/L8) at LARGER N/M than trained.
# Training range N{5,10,15} x M{10,20,30}; here N{20,35,50} x M{50,75,100}. 36-task array.
# Counterpart to eval_paper_xl_current.sh — tests whether the bigger model's capacity
# pays off when extrapolating far, even though it lost to h128/L6 in-range.
#
# Submit:
#   sbatch scripts/eval_paper_xl_current.sh
#   sbatch scripts/eval_paper_xl_larger.sh

#SBATCH --job-name=mcpf_xl_larger
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/xl_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/xl_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4)
NS=(20 35 50)
MS=(50 75 100)

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

mkdir -p "$PROJECT/results/fullpipe_paper_xl/larger"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/paper_larger_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 100 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 41400 \
    --csv "$PROJECT/results/fullpipe_paper_xl/larger/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
