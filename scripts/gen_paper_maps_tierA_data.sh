#!/bin/bash
# Tier A ("large/sparse") dataset for the per-map-vs-joint closing experiments:
# the 4 real RobustMCPF benchmark maps x 9 (N,M) pairs, N up to 180 / M up to 350 —
# calibrated (scripts/calibrate_solve_time.py) so the hardest cell's median solve
# time is ~30-40s ("up to a minute" per the professor's sizing criterion), with a
# long tail of pathological instances well past a minute.
# Maps: empty-32-32, random-32-32-20, maze-32-32-2, room-32-32-4 (all 32x32).
# N∈{60,120,180} × M∈{100,225,350}. Per config: train(3k), val(500), test(500) —
# far below Tier B's 20k/2k/2k: at ~10-40s/solve even with 16 parallel workers the
# hardest cells would blow the wall budget at Tier B's sample count.
#
# Each array task handles one (map,N,M) triple -> no LKH temp-file collision between tasks.
# Grid is fixed per task (--map_file); only agent/goal placement varies per instance.
# --instance_timeout 600 is a wall-clock safety net (rare pathological CBS solves can
# run long regardless of N/M); it rejects and resamples the offending instance instead
# of hanging the job.
#
# NOTE: real per-sample cost on this cluster ran far slower than local (Mac)
# calibration predicted (likely NFS latency on LKH's small temp files under 16-way
# concurrency) — most M=350 cells didn't finish within the original 24h budget.
# build_dataset.py now checkpoints per chunk and resumes on re-submission (loads
# existing partial D/G/Y.npy, generates only the remainder), so re-running this
# script is safe/cheap even for already-completed cells.
# Submit: sbatch scripts/gen_paper_maps_tierA_data.sh

#SBATCH --job-name=mcpf_gen_tierA
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_tierA_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_tierA_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 36 (map, N, M) triples: 4 maps × {60,120,180} × {100,225,350}, map-major.
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

OUTDIR="$PROJECT/data/paper_maps_tierA/${MAP}/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 3000 || echo 500 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --map_file "${MAP}.map" \
        --num_agents "$N" --num_goals "$M" \
        --out_dir "$OUTDIR" \
        --num_workers 16 --seed 1337 \
        --instance_timeout 600
done

echo "Done: map=$MAP N=$N M=$M"
