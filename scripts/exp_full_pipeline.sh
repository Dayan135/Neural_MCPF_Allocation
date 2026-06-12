#!/bin/bash
# Full MAPF execution-cost comparison: NN allocation vs combinatorial solver.
# Both pipelines end in CBS collision-free path planning; costs are true
# execution costs (Solution[5]), not BFS approximations.
#
# Uses checkpoints/universal_s0/best.pt over representative (N, M) configs.
# CPU partition: both pipelines are CPU-bound (LKH + CBS; NN forward is ~ms on CPU).
#
# Submit: sbatch scripts/exp_full_pipeline.sh

#SBATCH --job-name=mcpf_fullpipe
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_%j.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/fullpipe_%j.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
CKPT="$PROJECT/checkpoints/universal_s0/best.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env

# Some cpu nodes (e.g. ise-cpu-intl-03) hit the missing-iomp5 MKL symbol,
# same as the login node — force the GNU threading layer.
export MKL_THREADING_LAYER=GNU

cd "$PROJECT/evaluation"

for CONFIG in "2 2" "3 3" "4 4" "2 4" "4 6" "2 6"; do
    N=$(echo "$CONFIG" | awk '{print $1}')
    M=$(echo "$CONFIG" | awk '{print $2}')
    echo "##### N=$N M=$M #####"
    python full_pipeline_eval.py \
        --checkpoint "$CKPT" \
        --grid_w 8 --grid_h 8 \
        --num_agents "$N" --num_goals "$M" \
        --n_instances 200
done

echo "Full pipeline evaluation complete."
