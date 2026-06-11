#!/bin/bash
# Generate all datasets for Exp 2 (scale N) and Exp 3 (data scale).
#   - data/n{2,3,4,5}_8x8: 10k train / 2k val / 2k test per N
#   - data/n3_8x8_pool:    20k train pool (Exp 3 slices subsets); val/test copied from n3_8x8
# Sequential on purpose: concurrent runs with the same split name would collide
# on LKH temp files (configStr is f"{split}_{attempt}").
# Submit: sbatch scripts/gen_scale_data.sh
# Chain experiments: sbatch --dependency=afterok:<this_job_id> scripts/exp_scale_n.sh

#SBATCH --job-name=mcpf_gendata
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gendata_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gendata_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

cd "$PROJECT/dataset_generation"

for N in 2 3 4 5; do
    OUT="$PROJECT/data/n${N}_8x8"
    echo "=== N=$N -> $OUT ==="
    python build_dataset.py --split train --num_samples 10000 --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split val   --num_samples 2000  --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
    python build_dataset.py --split test  --num_samples 2000  --num_agents "$N" \
        --grid_w 8 --grid_h 8 --out_dir "$OUT" --num_workers 8
done

# Exp 3 pool: 20k train with a different base seed so it doesn't replicate
# the n3_8x8 train set; val/test reused from n3_8x8 for comparability.
POOL="$PROJECT/data/n3_8x8_pool"
echo "=== Exp 3 pool -> $POOL ==="
python build_dataset.py --split train --num_samples 20000 --num_agents 3 \
    --grid_w 8 --grid_h 8 --out_dir "$POOL" --num_workers 8 --seed 1042
mkdir -p "$POOL/val" "$POOL/test"
cp "$PROJECT/data/n3_8x8/val/D_matrices.npy"  "$POOL/val/"
cp "$PROJECT/data/n3_8x8/val/Y_matrices.npy"  "$POOL/val/"
cp "$PROJECT/data/n3_8x8/test/D_matrices.npy" "$POOL/test/"
cp "$PROJECT/data/n3_8x8/test/Y_matrices.npy" "$POOL/test/"

echo "All datasets generated."
