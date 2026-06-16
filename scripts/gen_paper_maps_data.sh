#!/bin/bash
# Paper-realistic dataset: the 4 small RobustMCPF benchmark maps × 9 (N,M) pairs.
# Maps: empty-32-32, random-32-32-20, maze-32-32-2, room-32-32-4 (all 32x32).
# N∈{5,10,15} × M∈{10,20,30} — a moderate step up from Exp 11 (N≤5, M≤8) toward the
# paper's N≤50/M≤100 range, kept bounded so CBS/LKH solve time per instance stays sane.
# Per config: train(20k), val(2k), test(2k).
#
# Each array task handles one (map,N,M) triple → no LKH temp-file collision between tasks.
# Grid is fixed per task (--map_file); only agent/goal placement varies per instance.
# Submit: sbatch scripts/gen_paper_maps_data.sh

#SBATCH --job-name=mcpf_gen_papermaps
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_papermaps_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_papermaps_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 36 (map, N, M) triples: 4 maps × {5,10,15} × {10,20,30}, map-major.
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

OUTDIR="$PROJECT/data/paper_maps/${MAP}/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 20000 || echo 2000 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --map_file "${MAP}.map" \
        --num_agents "$N" --num_goals "$M" \
        --out_dir "$OUTDIR" \
        --num_workers 16 --seed 1337
done

echo "Done: map=$MAP N=$N M=$M"
