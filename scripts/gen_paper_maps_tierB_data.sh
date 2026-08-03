#!/bin/bash
# Tier B ("moderate/dense") dataset for the per-map-vs-joint closing experiments:
# the 4 real RobustMCPF benchmark maps x 9 (N,M) pairs, N up to 80 / M up to 150 —
# calibrated (scripts/calibrate_solve_time.py) so a single solve is ~5-10s on the
# hardest cell, cheap enough to afford a large sample count.
# Maps: empty-32-32, random-32-32-20, maze-32-32-2, room-32-32-4 (all 32x32).
# N∈{30,55,80} × M∈{50,100,150}. Per config: train(20k), val(2k), test(2k) — same
# volume as Exp 12's paper_maps grid.
#
# Each array task handles one (map,N,M) triple -> no LKH temp-file collision between tasks.
# Grid is fixed per task (--map_file); only agent/goal placement varies per instance.
# --instance_timeout 600 is a wall-clock safety net (rare pathological CBS solves can
# run long regardless of N/M — seen even at N=105/M=210 during calibration); it rejects
# and resamples the offending instance instead of hanging the job.
#
# NOTE: real per-sample cost on this cluster ran ~10-20x slower than local (Mac)
# calibration predicted (likely NFS latency on LKH's small temp files under 16-way
# concurrency) — the original 12h budget only let the M=50 cells finish; all M>=100
# cells timed out. build_dataset.py now checkpoints per chunk and resumes on
# re-submission (loads existing partial D/G/Y.npy, generates only the remainder),
# so re-running this script is safe/cheap even for already-completed cells.
# Submit: sbatch scripts/gen_paper_maps_tierB_data.sh

#SBATCH --job-name=mcpf_gen_tierB
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_tierB_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_tierB_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 36 (map, N, M) triples: 4 maps × {30,55,80} × {50,100,150}, map-major.
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

OUTDIR="$PROJECT/data/paper_maps_tierB/${MAP}/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 20000 || echo 2000 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --map_file "${MAP}.map" \
        --num_agents "$N" --num_goals "$M" \
        --out_dir "$OUTDIR" \
        --num_workers 16 --seed 1337 \
        --instance_timeout 600
done

echo "Done: map=$MAP N=$N M=$M"
