#!/bin/bash
# Generate datasets for Exp 5 (goal-goal distance ablation).
# Produces data/n{2,3,4,5}_8x8_G — identical layout to the regular datasets
# but also containing G_matrices.npy (goal-to-goal BFS distances).
# Sequential: concurrent LKH calls sharing a split name would collide on temp files.
#
# Submit:  sbatch scripts/gen_scale_data_g.sh
# Chain:   sbatch --dependency=afterok:<this_job_id> scripts/exp_goal_dists.sh

#SBATCH --job-name=mcpf_gendata_g
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gendata_g_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gendata_g_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

cd "$PROJECT/dataset_generation"

for N in 2 3 4 5; do
    OUT="$PROJECT/data/n${N}_8x8_G"
    echo "=== N=$N -> $OUT ==="
    python build_dataset.py --split train --num_samples 10000 --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split val   --num_samples 2000  --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split test  --num_samples 2000  --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
done

echo "All G-datasets generated."
