#!/bin/bash
# Generate datasets for N≠M experiments (more goals than agents).
#
# Produces:
#   data/n2m4_8x8_G   — 2 agents, 4 goals
#   data/n3m6_8x8_G   — 3 agents, 6 goals
#   data/n2m6_8x8_G   — 2 agents, 6 goals (harder: ~3 goals per agent)
#
# Each dataset: 10k train / 2k val / 2k test, 8×8 grid, includes G_matrices.npy.
# Sequential: concurrent LKH calls sharing a split name would collide on temp files.
#
# Submit:  sbatch scripts/gen_nm_data.sh
# Chain:   sbatch --dependency=afterok:<this_job_id> scripts/exp_nm.sh

#SBATCH --job-name=mcpf_gen_nm
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_nm_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_nm_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

cd "$PROJECT/dataset_generation"

for CONFIG in "2 4 n2m4" "3 6 n3m6" "2 6 n2m6"; do
    N=$(echo "$CONFIG" | awk '{print $1}')
    M=$(echo "$CONFIG" | awk '{print $2}')
    TAG=$(echo "$CONFIG" | awk '{print $3}')
    OUT="$PROJECT/data/${TAG}_8x8_G"
    echo "=== N=$N M=$M -> $OUT ==="
    python build_dataset.py --split train --num_samples 10000 \
        --num_agents "$N" --num_goals "$M" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split val   --num_samples 2000  \
        --num_agents "$N" --num_goals "$M" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split test  --num_samples 2000  \
        --num_agents "$N" --num_goals "$M" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
done

echo "All N≠M datasets generated."
