#!/bin/bash
# Generate the universal mixed-size dataset.
#
# N∈{2,3,4} × M∈{2,3,4,5,6} — 15 configurations.
# 15k train / 3k val / 3k test per config.
# Output: data/universal_8x8/n{N}m{M}/
#
# Sequential (single job): concurrent LKH calls with the same split name
# collide on temp files, so configs must run one at a time.
# Estimated wall time: ~24h on 8-core cpu node (~1.5h per config × 15 + overhead).
#
# Submit: sbatch scripts/gen_universal_data.sh
# Chain:  sbatch --dependency=afterok:<this_job_id> scripts/exp_universal.sh

#SBATCH --job-name=mcpf_gen_universal
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=28:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_universal_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_universal_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

cd "$PROJECT/dataset_generation"

for N in 2 3 4; do
    for M in 2 3 4 5 6; do
        TAG="n${N}m${M}"
        OUT="$PROJECT/data/universal_8x8/${TAG}"
        echo "=== N=$N M=$M -> $OUT ==="
        python build_dataset.py --split train --num_samples 15000 \
            --num_agents "$N" --num_goals "$M" \
            --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
        python build_dataset.py --split val   --num_samples 3000  \
            --num_agents "$N" --num_goals "$M" \
            --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
        python build_dataset.py --split test  --num_samples 3000  \
            --num_agents "$N" --num_goals "$M" \
            --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    done
done

echo "All 15 universal datasets generated."
