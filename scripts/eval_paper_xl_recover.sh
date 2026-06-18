#!/bin/bash
# Recover the 2 XL configs that timed out in eval_paper_xl_{current,larger}.sh:
# room-32-32-4 N20M75 and N35M100, for both models = 4 tasks. Those died because a
# single pathological CBS/LKH solve outran the 12h wall before the between-instance
# --max_seconds check could fire. This run adds --instance_timeout (hard per-instance
# SIGALRM cap) so one bad solve is skipped instead of blowing the wall; --max_seconds
# still writes a partial CSV if the whole job runs long.
#
# Submit: sbatch scripts/eval_paper_xl_recover.sh

#SBATCH --job-name=mcpf_xl_recover
#SBATCH --array=0-3
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/xl_recover_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/xl_recover_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

MAP=room-32-32-4
# 4 tasks: {current,larger} x {n20m75, n35m100}
MODELS=(current current larger larger)
NS=(20 35 20 35)
MS=(75 100 75 100)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
N=${NS[$SLURM_ARRAY_TASK_ID]}
M=${MS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL map=$MAP N=$N M=$M"

mkdir -p "$PROJECT/results/fullpipe_paper_xl/${MODEL}"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/paper_${MODEL}_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 100 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --instance_timeout 600 \
    --max_seconds 39600 \
    --csv "$PROJECT/results/fullpipe_paper_xl/${MODEL}/${MAP}_n${N}m${M}.csv"

echo "Done: model=$MODEL map=$MAP N=$N M=$M"
