# Random-vs-Fixed-Map Training Implementation Plan

> **Source of truth:** the committed `.sh` files in this folder, not the snippets embedded below.
> Since this plan was written they received operational tweaks (commit `7048803`): gen `--mem`
> 16G→32G (dense-grid OOM) and eval `--instance_timeout 600` (cap a pathological CBS solve). The
> embedded snippets below are not updated for these.

**Goal:** Train two models (C1 h128/L6, C2 h256/L8) on random 32×32 grids with 0–50% wall density at N∈{5,10,15}×M∈{10,20,30}, evaluate them on the 4 real benchmark maps, and compare against the fixed-map models (B1/B2) to decide whether random-diverse or fixed-map training transfers better. Separately, evaluate the old model (A) at large N/M on the real maps locally. Conclude with a zero-shot XL extrapolation on whichever family wins.

## Experiment numbering (canonical)

This plan produces / re-labels the following experiments (continues `RESULTS.md`):

| Exp | Title | Models | Status | Plan tasks |
|-----|-------|--------|--------|-----------|
| 11 | h128/L6 large model on diverse 8–12 dataset | (prior) | done | — (unchanged) |
| **12** | Paper maps, h128 vs h256 (in-distribution) | B1 vs B2 | done — coworker's Exp 12, **number unchanged** | — |
| **13.a** | Old model on real maps, **small** N/M (N≤5×M≤8) | A | done | — (`report/real_maps/REPORT.md`) |
| **13.b** | Old model on real maps, **large** N/M | A vs B1 | planned (local) | Tasks 6–7 |
| **14** | Random-diverse vs fixed-map training | C1,C2 vs B1,B2 | planned (cluster) | Tasks 1–5, 8 |
| **15** | Zero-shot XL extrapolation (N20–50×M50–100) | winner of Exp 14 | planned (conditional) | Task 9 |

Ordering rationale (Option B): B1/B2 already exist (coworker's Exp 12), so introducing them first
(Exp 12) means the A-vs-B1 comparison in Exp 13.b references an already-established model. Model A's
full story — small N/M (13.a), large N/M, and the A-vs-B1 comparison (13.b) — stays in one place.

Renumbering note: only the coworker's current RESULTS.md **Exp 13 → Exp 15** (his XL run becomes the
"B wins" branch of Exp 15); his **Exp 12 keeps its number**. That single RESULTS.md header edit is a
separate, confirm-first step (it touches his committed section) — not part of the script tasks below.

**Architecture:** Five new Slurm scripts (copies of the coworker's `gen_/exp_/eval_paper_maps_*` scripts with data sources repointed) for the cluster C1/C2 experiment, plus one aggregation script and a local model-A evaluation procedure. No changes to the model, loss, solver, or `build_dataset.py`/`full_pipeline_eval.py` (those already support every flag needed). Cluster scripts target the coworker's environment; the local task runs in the repo `.venv`.

**Tech Stack:** Python 3, PyTorch (cluster: GPU; local: CPU), RobustMCPF/LKH, SLURM, conda `mcpf_env` (cluster) / `.venv` (local).

## Global Constraints

- **Cluster scripts** target `PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation`, `conda activate mcpf_env`, `--account=erant --qos=normal`, `export MKL_THREADING_LAYER=GNU`. They are authored here and committed; the coworker submits them with `sbatch`. They are NOT runnable locally beyond syntax/enumeration checks.
- **Local task** runs from repo root `/home/yabofek/git-repos-wsl/school/Neural_MCPF_Allocation` using `.venv/bin/python`.
- **Controlled comparison — C copies B exactly except data.** C1 = h128/L6 (lr 5e-4); C2 = h256/L8 (lr 3e-4). Both: `transformer`, `--num_heads 4`, `--use_goal_dists`, `--mixed`, `--grad_clip 1.0`, `--epochs 150`, `--batch_size 256`, `--lam 0.1`, 3 seeds.
- **Random training data:** fixed `--grid_w 32 --grid_h 32`, per-instance walls `--obstacle_prob 0.0 --obstacle_prob_max 0.5`, **9 (N,M) configs**, **80k train / 2k val / 2k test per config** (matches B's 80k-per-shape / 720k-total training volume), `--num_workers 16 --seed 1337`.
- **Evaluation:** `full_pipeline_eval.py --map_file` (true CBS execution cost), `--n_instances 200 --seed 987654321 --max_seconds 27000` so C/B/A all share instances; per-instance CSVs aggregated post-hoc with `agg_paper_maps.py`.
- **Generated artifacts** (`data/`, `checkpoints/`, `results/`, `logs/`) are git-ignored. Only scripts + reports are committed. Commit scope: `env` for data/gen, `agent` for model/eval, `report` for write-ups.
- Branch prerequisite (already done this session): `origin/feat/paper-maps-scale` is merged, so scalable `order_goals`, `--max_seconds`, `--instance_timeout`, and `agg_paper_maps.py` are present.

## File structure

| File | Responsibility |
|------|----------------|
| `scripts/exp14/gen_random_diverse_data.sh` | Cluster: generate C training data (9-config array, random 32×32 walls 0–0.5). |
| `scripts/exp14/exp_random_current.sh` | Cluster: train C1 (h128/L6), 3-seed array, on random_diverse data. |
| `scripts/exp14/exp_random_larger.sh` | Cluster: train C2 (h256/L8), 3-seed array, on random_diverse data. |
| `scripts/exp14/eval_random_current.sh` | Cluster: full-pipeline eval C1 on the 4 maps (36-config array). |
| `scripts/exp14/eval_random_larger.sh` | Cluster: full-pipeline eval C2 on the 4 maps (36-config array). |
| `evaluation/agg_compare.py` | Post-task: read B (`fullpipe_paper`) + C (`fullpipe_random`) CSVs → random-vs-fixed comparison tables. |

Local model-A task (Tasks 6–7) creates no committed code beyond a results report; it reuses existing `build_dataset.py`, `evaluate.py`, `full_pipeline_eval.py`, `agg_paper_maps.py`.

---

## Task 1: Cluster data-generation script (C training data)

**Files:**
- Create: `scripts/exp14/gen_random_diverse_data.sh`

**Interfaces:**
- Produces (on cluster, when run): `data/random_diverse/n{N}m{M}/{train,val,test}/{D,G,Y}_matrices.npy` for the 9 (N,M) configs.

- [ ] **Step 1: Create the script**

```bash
#!/bin/bash
# Random-diverse dataset for the random-vs-fixed-map comparison (model C).
# Random 32x32 grids, per-instance wall density uniform in [0.0, 0.5], placement varies.
# N∈{5,10,15} × M∈{10,20,30} = 9 (N,M) configs (NO map axis — random grids).
# Per config: train(80k), val(2k), test(2k). 80k matches model B's per-(N,M)-shape
# training volume (B saw 4 maps × 20k = 80k per shape; 720k total across 9 shapes).
# Submit: sbatch scripts/exp14/gen_random_diverse_data.sh

#SBATCH --job-name=mcpf_gen_randdiv
#SBATCH --array=0-8
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_randdiv_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/gen_randdiv_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

# 9 (N, M) pairs. No map axis: grids are random per instance.
NS=(5 10 15)
MS=(10 20 30)

CONFIGS=()
for N in "${NS[@]}"; do
    for M in "${MS[@]}"; do
        CONFIGS+=("$N $M")
    done
done

IFS=' ' read -r N M <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: N=$N M=$M"

OUTDIR="$PROJECT/data/random_diverse/n${N}m${M}"
cd "$PROJECT/dataset_generation"

for SPLIT in train val test; do
    NSAMP=$( [[ "$SPLIT" == "train" ]] && echo 80000 || echo 2000 )
    python build_dataset.py \
        --split "$SPLIT" --num_samples "$NSAMP" \
        --grid_w 32 --grid_h 32 \
        --obstacle_prob 0.0 --obstacle_prob_max 0.5 \
        --num_agents "$N" --num_goals "$M" \
        --out_dir "$OUTDIR" \
        --num_workers 16 --seed 1337
done

echo "Done: N=$N M=$M"
```

- [ ] **Step 2: Syntax check**

Run: `bash -n scripts/exp14/gen_random_diverse_data.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Verify the config array enumerates 9 (N,M) pairs**

Run:
```bash
bash -c 'NS=(5 10 15); MS=(10 20 30); CONFIGS=(); for N in "${NS[@]}"; do for M in "${MS[@]}"; do CONFIGS+=("$N $M"); done; done; echo "count=${#CONFIGS[@]}"; printf "%s\n" "${CONFIGS[@]}"'
```
Expected: `count=9` then the 9 pairs `5 10` … `15 30`. Confirms `--array=0-8` covers exactly the configs.

- [ ] **Step 4: Local smoke test of the inner generator command (tiny N)**

Confirms `build_dataset.py` accepts the random-grid + wall-range flags (uses local `.venv`, 20 samples):
```bash
cd dataset_generation
../.venv/bin/python build_dataset.py --split test --num_samples 20 \
  --grid_w 32 --grid_h 32 --obstacle_prob 0.0 --obstacle_prob_max 0.5 \
  --num_agents 5 --num_goals 10 --out_dir /tmp/randdiv_smoke --num_workers 4 --seed 1337
cd ..
.venv/bin/python -c "import numpy as np; D=np.load('/tmp/randdiv_smoke/test/D_matrices.npy'); print('shape', D.shape)"
rm -rf /tmp/randdiv_smoke
```
Expected: generation completes, prints `shape (20, 5, 10)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp14/gen_random_diverse_data.sh
git commit -m "env: add random-diverse data-gen Slurm job for model C (random vs fixed maps)"
```

---

## Task 2: Cluster training scripts (C1 + C2)

**Files:**
- Create: `scripts/exp14/exp_random_current.sh` (C1, h128/L6)
- Create: `scripts/exp14/exp_random_larger.sh` (C2, h256/L8)

**Interfaces:**
- Consumes: `data/random_diverse/n{N}m{M}/` from Task 1.
- Produces (on cluster): `checkpoints/random_current_s{0,1,2}/best.pt`, `checkpoints/random_larger_s{0,1,2}/best.pt`.

- [ ] **Step 1: Create `scripts/exp14/exp_random_current.sh` (C1)**

```bash
#!/bin/bash
# Model C1: h128/L6 universal transformer trained on the random-diverse dataset
# (9 (N,M) configs, random 32x32 grids, walls 0-0.5). Copies exp_paper_current.sh
# exactly; only --data_dirs differs (random_diverse instead of paper_maps, no map axis).
# Submit after gen_random_diverse_data.sh:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp14/exp_random_current.sh

#SBATCH --job-name=mcpf_rand_current
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_current_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_current_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

SEED=$SLURM_ARRAY_TASK_ID

# 9 (N,M) configs of the random-diverse dataset (no map axis).
DATA_BASE="$PROJECT/data/random_diverse"
DATA_DIRS=""
for N in 5 10 15; do
    for M in 10 20 30; do
        DIR="${DATA_BASE}/n${N}m${M}"
        DATA_DIRS="${DATA_DIRS:+${DATA_DIRS},}${DIR}"
    done
done

cd "$PROJECT/training"
python train.py \
    --mixed \
    --data_dirs "$DATA_DIRS" \
    --model_type transformer \
    --hidden 128 --num_layers 6 --num_heads 4 \
    --use_goal_dists \
    --lr 5e-4 --grad_clip 1.0 \
    --epochs 150 --batch_size 256 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "random_current_s${SEED}"

echo "Training complete: random_current_s${SEED}"
```

- [ ] **Step 2: Create `scripts/exp14/exp_random_larger.sh` (C2)**

Identical to Step 1 except the SBATCH names, the GPU request, the CUDA alloc env, the architecture flags (h256/L8, lr 3e-4), and the run name:

```bash
#!/bin/bash
# Model C2: h256/L8 universal transformer trained on the random-diverse dataset.
# Copies exp_paper_larger.sh exactly; only --data_dirs differs (random_diverse, no map axis).
# Run in parallel with exp_random_current.sh:
#   sbatch --dependency=afterok:<gen_job_id> scripts/exp14/exp_random_larger.sh

#SBATCH --job-name=mcpf_rand_larger
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/rand_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU
# h256/L8 needs a 24GB card (OOMs on the 10.5GB rtx_2080); reduce fragmentation too.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=$SLURM_ARRAY_TASK_ID

DATA_BASE="$PROJECT/data/random_diverse"
DATA_DIRS=""
for N in 5 10 15; do
    for M in 10 20 30; do
        DIR="${DATA_BASE}/n${N}m${M}"
        DATA_DIRS="${DATA_DIRS:+${DATA_DIRS},}${DIR}"
    done
done

cd "$PROJECT/training"
python train.py \
    --mixed \
    --data_dirs "$DATA_DIRS" \
    --model_type transformer \
    --hidden 256 --num_layers 8 --num_heads 4 \
    --use_goal_dists \
    --lr 3e-4 --grad_clip 1.0 \
    --epochs 150 --batch_size 256 \
    --lam 0.1 \
    --seed "$SEED" \
    --run_name "random_larger_s${SEED}"

echo "Training complete: random_larger_s${SEED}"
```

- [ ] **Step 3: Syntax check both**

Run: `bash -n scripts/exp14/exp_random_current.sh && bash -n scripts/exp14/exp_random_larger.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Verify DATA_DIRS builds 9 comma-separated dirs**

Run:
```bash
bash -c 'DATA_BASE=/x; DATA_DIRS=""; for N in 5 10 15; do for M in 10 20 30; do DIR="${DATA_BASE}/n${N}m${M}"; DATA_DIRS="${DATA_DIRS:+${DATA_DIRS},}${DIR}"; done; done; echo "$DATA_DIRS" | tr "," "\n" | wc -l'
```
Expected: `9`

- [ ] **Step 5: Diff against the templates to confirm only intended fields changed**

Run: `diff <(sed -E 's/random_current/paper_current/g; s/randdiv|rand_current//g' scripts/exp14/exp_random_current.sh) scripts/exp_paper_current.sh || true`
Expected: differences limited to job-name, log paths, `--data_dirs` base (`random_diverse` vs `paper_maps` with map axis), and `--run_name`. Architecture/lr/epochs lines identical. (Visual check — no strict assert.)

- [ ] **Step 6: Commit**

```bash
git add scripts/exp14/exp_random_current.sh scripts/exp14/exp_random_larger.sh
git commit -m "agent: add C1/C2 training Slurm jobs (h128/h256 on random-diverse data)"
```

---

## Task 3: Cluster evaluation scripts (C1 + C2 on the 4 maps)

**Files:**
- Create: `scripts/exp14/eval_random_current.sh`
- Create: `scripts/exp14/eval_random_larger.sh`

**Interfaces:**
- Consumes: `checkpoints/random_{current,larger}_s0/best.pt` from Task 2.
- Produces (on cluster): `results/fullpipe_random/{current,larger}/{map}_n{N}m{M}.csv` (36 each).

- [ ] **Step 1: Create `scripts/exp14/eval_random_current.sh`**

Copy of `eval_paper_maps_current.sh` with the checkpoint and output dir repointed to the random-trained C1 model. The 36-config (map×N×M) grid is unchanged — we evaluate the random-trained model *on the real maps*.

```bash
#!/bin/bash
# Full-pipeline eval: model C1 (random_current_s0, h128/L6, trained on random-diverse
# data) on all 36 real-map configs. Per-instance CSVs to results/fullpipe_random/current/.
# Same seed/instances as the Exp 12 B-model eval → C1-vs-B1 on identical problems.
# Submit: sbatch scripts/exp14/eval_random_current.sh

#SBATCH --job-name=mcpf_evalrand_current
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_current_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_current_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

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

mkdir -p "$PROJECT/results/fullpipe_random/current"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/random_current_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 200 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 27000 \
    --csv "$PROJECT/results/fullpipe_random/current/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
```

- [ ] **Step 2: Create `scripts/exp14/eval_random_larger.sh`**

Identical to Step 1 except every `current`→`larger` (job-name, checkpoint `random_larger_s0`, output dir `results/fullpipe_random/larger`, log paths):

```bash
#!/bin/bash
# Full-pipeline eval: model C2 (random_larger_s0, h256/L8) on all 36 real-map configs.
# Per-instance CSVs to results/fullpipe_random/larger/. Submit: sbatch scripts/exp14/eval_random_larger.sh

#SBATCH --job-name=mcpf_evalrand_larger
#SBATCH --array=0-35
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_larger_%A_%a.out
#SBATCH --error=/home/dayanb/course_multiagent/Neural_MCPF_Allocation/logs/evalrand_larger_%A_%a.err

set -euo pipefail

PROJECT=/home/dayanb/course_multiagent/Neural_MCPF_Allocation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpf_env
export MKL_THREADING_LAYER=GNU

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

mkdir -p "$PROJECT/results/fullpipe_random/larger"

cd "$PROJECT"
python evaluation/full_pipeline_eval.py \
    --checkpoint "$PROJECT/checkpoints/random_larger_s0/best.pt" \
    --map_file "${MAP}.map" \
    --num_agents "$N" --num_goals "$M" \
    --n_instances 200 --seed 987654321 \
    --tag "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --max_seconds 27000 \
    --csv "$PROJECT/results/fullpipe_random/larger/${MAP}_n${N}m${M}.csv"

echo "Done: map=$MAP N=$N M=$M"
```

- [ ] **Step 3: Syntax check both**

Run: `bash -n scripts/exp14/eval_random_current.sh && bash -n scripts/exp14/eval_random_larger.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Verify 36-config enumeration**

Run:
```bash
bash -c 'MAPS=(empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4); NS=(5 10 15); MS=(10 20 30); C=(); for MAP in "${MAPS[@]}"; do for N in "${NS[@]}"; do for M in "${MS[@]}"; do C+=("$MAP $N $M"); done; done; done; echo "count=${#C[@]}"'
```
Expected: `count=36` (matches `--array=0-35`).

- [ ] **Step 5: Commit**

```bash
git add scripts/exp14/eval_random_current.sh scripts/exp14/eval_random_larger.sh
git commit -m "agent: add C1/C2 full-pipeline eval Slurm jobs (random-trained models on real maps)"
```

---

## Task 4: Comparison aggregation script (C vs B)

**Files:**
- Create: `evaluation/agg_compare.py`

**Interfaces:**
- Consumes: per-instance CSVs in `results/fullpipe_random/{current,larger}/` (C, from Task 3) and `results/fullpipe_paper/{current,larger}/` (B, existing Exp 12 output). Reuses `agg_paper_maps.load_group` for parsing.
- Produces: stdout comparison tables (per model size: C vs B cost-ratio / exact-match / speedup, per config + aggregate).

- [ ] **Step 1: Write the script**

```python
"""
Compare random-trained (C) vs fixed-map-trained (B) models on the 4 real maps.

Reads the per-instance full-pipeline CSVs both model families wrote:
  B (fixed maps):    results/fullpipe_paper/{current,larger}/<map>_n{N}m{M}.csv   (Exp 12)
  C (random-diverse): results/fullpipe_random/{current,larger}/<map>_n{N}m{M}.csv (this work)

Prints, per model size (current=h128, larger=h256): per-config B-vs-C cost ratio /
exact-match / speedup, and the mean over all 36 configs. Reuses agg_paper_maps.load_group.

Usage:
    python evaluation/agg_compare.py
    python evaluation/agg_compare.py --fixed_base results/fullpipe_paper \\
                                     --random_base results/fullpipe_random
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_paper_maps import load_group  # noqa: E402


def compare_group(fixed_base, random_base, group):
    b = load_group(fixed_base, group)
    c = load_group(random_base, group)
    common = sorted(set(b) & set(c))
    print(f"\n===== {group} (h128 if current, h256 if larger) — {len(common)} shared configs =====")
    print(f"{'config':<26}{'B ratio':>9}{'C ratio':>9}{'Δratio':>8}"
          f"{'B exact':>9}{'C exact':>9}{'B spd':>8}{'C spd':>8}")
    db, dc = [], []
    for name in common:
        rb, rc = b[name], c[name]
        dratio = rc["ratio"] - rb["ratio"]
        db.append(rb); dc.append(rc)
        print(f"{name:<26}{rb['ratio']:>9.3f}{rc['ratio']:>9.3f}{dratio:>+8.3f}"
              f"{rb['exact']:>9.1%}{rc['exact']:>9.1%}{rb['speedup']:>7.1f}x{rc['speedup']:>7.1f}x")
    mb_ratio = np.mean([x["ratio"] for x in db]); mc_ratio = np.mean([x["ratio"] for x in dc])
    mb_exact = np.mean([x["exact"] for x in db]); mc_exact = np.mean([x["exact"] for x in dc])
    print(f"{'MEAN':<26}{mb_ratio:>9.3f}{mc_ratio:>9.3f}{mc_ratio - mb_ratio:>+8.3f}"
          f"{mb_exact:>9.1%}{mc_exact:>9.1%}")
    verdict = "C (random) better" if mc_ratio < mb_ratio else "B (fixed) better"
    print(f"  verdict: {verdict} on mean cost ratio "
          f"(lower is better; B={mb_ratio:.4f}, C={mc_ratio:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_base", default="results/fullpipe_paper")
    ap.add_argument("--random_base", default="results/fullpipe_random")
    args = ap.parse_args()
    for group in ("current", "larger"):
        compare_group(args.fixed_base, args.random_base, group)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and runs against synthetic CSVs**

Build two tiny fake CSV trees matching the real schema and confirm the script parses them, prints a verdict, and exits 0:
```bash
.venv/bin/python - <<'EOF'
import os, csv, subprocess, tempfile, random
root = tempfile.mkdtemp()
def write(base, group, cfg, ratio_center):
    d = os.path.join(base, group); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{cfg}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["inst_seed","cost_nn","cost_solver","nn_k","solver_k",
                    "conflicts_nn","conflicts_solver","alloc_ms","nn_plan_ms","solver_ms"])
        for i in range(20):
            solver = 100.0
            nn = solver * ratio_center
            w.writerow([i, nn, solver, 1, 1, 0, 0, 1.0, 5.0, 12.0])
fixed = os.path.join(root, "fullpipe_paper"); rand = os.path.join(root, "fullpipe_random")
write(fixed, "current", "empty-32-32_n5m10", 1.05)
write(rand,  "current", "empty-32-32_n5m10", 1.02)
r = subprocess.run([".venv/bin/python","evaluation/agg_compare.py",
                    "--fixed_base",fixed,"--random_base",rand], capture_output=True, text=True)
print(r.stdout); print("STDERR:", r.stderr); print("rc:", r.returncode)
assert r.returncode == 0 and "verdict" in r.stdout
print("OK")
EOF
```
Expected: prints a `current` table with B ratio 1.050 / C ratio 1.020 / Δratio −0.030, a verdict line "C (random) better", and `OK`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/agg_compare.py
git commit -m "agent: add C-vs-B random/fixed comparison aggregation script"
```

---

## Task 5: Cluster run instructions (handoff doc for the coworker)

**Files:**
- Create: `scripts/exp14/README_random_vs_fixed.md`

**Interfaces:**
- Consumes: Tasks 1–4 scripts.
- Produces: an ordered submission recipe so the coworker can run the cluster half unattended.

- [ ] **Step 1: Write the handoff doc**

```markdown
# Random-vs-fixed-map experiment — cluster submission order

All from the project root on slurm.bgu.ac.il (`/home/dayanb/course_multiagent/Neural_MCPF_Allocation`),
after `mkdir -p logs`. Trains models C1 (h128) and C2 (h256) on random-diverse data, then evaluates
them on the 4 real maps. Compare to the existing Exp 12 B-model results.

1. Generate data (9 configs, ~720k samples total):
       GEN=$(sbatch --parsable scripts/exp14/gen_random_diverse_data.sh)

2. Train C1 + C2 after generation finishes (run both in parallel):
       sbatch --dependency=afterok:$GEN scripts/exp14/exp_random_current.sh
       sbatch --dependency=afterok:$GEN scripts/exp14/exp_random_larger.sh

3. After training, evaluate on the 4 real maps (run both in parallel):
       sbatch scripts/exp14/eval_random_current.sh
       sbatch scripts/exp14/eval_random_larger.sh

4. Aggregate (login node, no GPU). Pulls the C verdict and compares to B:
       MKL_THREADING_LAYER=GNU python evaluation/agg_paper_maps.py --base results/fullpipe_random
       MKL_THREADING_LAYER=GNU python evaluation/agg_compare.py

Per-instance CSVs land in results/fullpipe_random/{current,larger}/. results/ is git-ignored;
copy the CSVs back (or run agg_compare.py on the cluster) to build the final report.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/exp14/README_random_vs_fixed.md
git commit -m "env: add cluster submission recipe for random-vs-fixed experiment"
```

---

## Task 6: Local — generate model-A test data on the real maps

**Files:**
- Produces (local, git-ignored): `data/paper_maps_local/{map}/n{N}m{M}/test/{D,G,Y}_matrices.npy` (36 configs).

**Interfaces:**
- Consumes: vendored maps in `RobustMCPF/Maps/`, built LKH binary.
- Produces: 200-instance test sets used by Task 7's offline eval.

- [ ] **Step 1: Generate 200 test instances per config across all 36 configs**

```bash
cd /home/yabofek/git-repos-wsl/school/Neural_MCPF_Allocation/dataset_generation
for MAP in empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4; do
  for CFG in "5 10" "5 20" "5 30" "10 10" "10 20" "10 30" "15 10" "15 20" "15 30"; do
    N=$(echo $CFG | cut -d' ' -f1); M=$(echo $CFG | cut -d' ' -f2)
    echo "=== GEN $MAP n${N}m${M} ==="
    ../.venv/bin/python build_dataset.py \
      --split test --num_samples 200 \
      --map_file ${MAP}.map \
      --num_agents $N --num_goals $M \
      --out_dir ../data/paper_maps_local/${MAP}/n${N}m${M} \
      --num_workers 4 --seed 987654321
  done
done
cd ..
```
Expected: 36 "Saved 200 samples" lines. High-M maze/room cells are the slow ones (more solver work + rejections).

- [ ] **Step 2: Verify shapes and column-sum invariant**

```bash
.venv/bin/python - <<'EOF'
import numpy as np, glob
ok = 0
for p in sorted(glob.glob("data/paper_maps_local/*/*/test/D_matrices.npy")):
    D = np.load(p); Y = np.load(p.replace("D_","Y_"))
    assert np.allclose(Y.sum(1), 1), p
    ok += 1
print(f"{ok} configs, all column sums == 1")
EOF
```
Expected: `36 configs, all column sums == 1`.

- [ ] **Step 3: (no commit — generated data is git-ignored)**

`git status --porcelain data/` should print nothing (confirms `data/` is ignored). Nothing to commit.

---

## Task 7: Local — evaluate model A and write the report

**Files:**
- Produces (local, git-ignored): `results/oldmodel_real_maps_offline.txt`, `results/fullpipe_oldmodel/{map}_n{N}m{M}.csv` (36).
- Create (committed): `report/oldmodel_large_nm/REPORT.md`.

**Interfaces:**
- Consumes: model A checkpoint `checkpoints/large_s0/best.pt`; Task 6 data; existing `evaluate.py`, `full_pipeline_eval.py`, `agg_paper_maps.py`.

- [ ] **Step 1: Offline eval (accuracy) over the 36 configs**

```bash
cd /home/yabofek/git-repos-wsl/school/Neural_MCPF_Allocation
mkdir -p results
: > results/oldmodel_real_maps_offline.txt
for MAP in empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4; do
  for CFG in n5m10 n5m20 n5m30 n10m10 n10m20 n10m30 n15m10 n15m20 n15m30; do
    echo "=== ${MAP} ${CFG} ===" >> results/oldmodel_real_maps_offline.txt
    .venv/bin/python evaluation/evaluate.py \
      --checkpoint checkpoints/large_s0/best.pt \
      --data_dir data/paper_maps_local/${MAP}/${CFG} --split test 2>&1 \
      | grep -E "per_goal|full_assign|mean_cost" >> results/oldmodel_real_maps_offline.txt
  done
done
cat results/oldmodel_real_maps_offline.txt
```
Expected: 36 blocks of metrics. Per-goal accuracy will likely fall off sharply versus the small-N/M
range in the committed report (this is pure scale extrapolation for model A).

- [ ] **Step 2: Full-pipeline eval (true execution cost) over the 36 configs**

Uses the same seed/instances as the Exp 12 B-model eval, so model A is directly comparable to B1.
The `--max_seconds` / `--instance_timeout` guards (merged from the coworker's branch) prevent hangs at M=30.
```bash
mkdir -p results/fullpipe_oldmodel
for MAP in empty-32-32 random-32-32-20 maze-32-32-2 room-32-32-4; do
  for CFG in "5 10" "5 20" "5 30" "10 10" "10 20" "10 30" "15 10" "15 20" "15 30"; do
    N=$(echo $CFG | cut -d' ' -f1); M=$(echo $CFG | cut -d' ' -f2)
    echo "=== FULLPIPE $MAP n${N}m${M} ==="
    .venv/bin/python evaluation/full_pipeline_eval.py \
      --checkpoint checkpoints/large_s0/best.pt \
      --map_file ${MAP}.map \
      --num_agents $N --num_goals $M \
      --n_instances 200 --seed 987654321 \
      --max_seconds 3600 --instance_timeout 600 \
      --csv results/fullpipe_oldmodel/${MAP}_n${N}m${M}.csv
  done
done
```
Expected: 36 CSVs in `results/fullpipe_oldmodel/`. Watch for `n<200` (budget/timeout skips) on the
hardest maze/room high-M cells — recorded, not fatal.

- [ ] **Step 3: Aggregate model A's full-pipeline results**

`agg_paper_maps.py` expects a base dir with `current`/`larger` subdirs; model A is a single model, so
point it at a base where `current` is a symlink/copy of the oldmodel dir, or aggregate directly:
```bash
.venv/bin/python - <<'EOF'
import sys, os
sys.path.insert(0, "evaluation")
from agg_paper_maps import load_group
import numpy as np
# Treat results/fullpipe_oldmodel as a single group at base=results, group=fullpipe_oldmodel.
g = load_group("results", "fullpipe_oldmodel")
print(f"{'config':<26}{'ratio':>8}{'exact':>8}{'spd':>7}")
rs = []
for name in sorted(g):
    v = g[name]; rs.append(v["ratio"])
    print(f"{name:<26}{v['ratio']:>8.3f}{v['exact']:>8.1%}{v['speedup']:>6.1f}x")
print(f"\nMEAN cost ratio over {len(g)} configs: {np.mean(rs):.4f}")
EOF
```
Expected: a 36-row table + mean cost ratio.

- [ ] **Step 4: Write the report**

Create `report/oldmodel_large_nm/REPORT.md` with: setup (model A at large N/M on the 4 maps,
200 inst/config), the offline accuracy table (from Step 1), the full-pipeline execution-cost table
(from Step 3), and an **A-vs-B1 comparison** pulling B1's Exp 12 per-config numbers from `RESULTS.md`
(model A and B1 share h128/L6 and the same 200 instances/seed, so the only difference is training
data: random-small-N/M vs fixed-maps-large-N/M). State the verdict on whether model A extrapolates
upward in N/M on real maps. Match the figure style of the committed `evaluation/plot_real_maps.py`
if figures are wanted (optional).

- [ ] **Step 5: Commit the report**

```bash
git add report/oldmodel_large_nm/REPORT.md
git commit -m "report: model A (small-N/M) extrapolated to large N/M on real maps; A-vs-B1"
```

---

## Task 8: Exp 14 report — random-vs-fixed verdict + conclusions

**Files:**
- Create (committed): `report/random_vs_fixed/REPORT.md`

**Interfaces:**
- Consumes: `evaluation/agg_compare.py` output (Task 4) over the cluster CSVs (`results/fullpipe_random` C, `results/fullpipe_paper` B). Runs after the cluster eval (Task 3) completes and CSVs are available locally or on the login node.

This is the deliverable of **Exp 14**: not just the comparison tables, but the written conclusion
on whether random-diverse training beats fixed-map training on the real maps.

- [ ] **Step 1: Produce the comparison tables**

```bash
cd /home/yabofek/git-repos-wsl/school/Neural_MCPF_Allocation
.venv/bin/python evaluation/agg_compare.py \
  --fixed_base results/fullpipe_paper --random_base results/fullpipe_random \
  | tee results/random_vs_fixed_compare.txt
```
Expected: per-config + mean B-vs-C tables for `current` (h128) and `larger` (h256), each ending in a
`verdict:` line (lower mean cost ratio wins).

- [ ] **Step 2: Write the report**

Create `report/random_vs_fixed/REPORT.md` covering: setup (C1/C2 = B1/B2 recipe, only training data
swapped: random 32×32 walls 0–50% vs the 4 fixed maps; 80k/config matching B's per-shape volume;
eval on the 4 real maps, 200 inst/config, same seed → identical instances); the **C-vs-B comparison
tables** (from Step 1) at both sizes; per-map and per-(N,M) breakdown; and **conclusions** — does
random-diverse training transfer to real maps better, worse, or on par with fixed-map training, and
at which model size / map / M does the gap appear. Relate back to Exp 13.a/13.b (the old model
already generalized to real maps) and state which family (B or C) wins, since that selects the model
for Exp 15. Match the figure style of `evaluation/plot_real_maps.py` if figures are wanted.

- [ ] **Step 3: Commit**

```bash
git add report/random_vs_fixed/REPORT.md
git commit -m "report: Exp 14 — random-diverse vs fixed-map training on real maps (C vs B)"
```

---

## Task 9: Exp 15 — zero-shot XL extrapolation on the Exp-14 winner

**Files:**
- Create (only if C wins Exp 14): `scripts/exp14/eval_random_xl_current.sh`, `scripts/exp14/eval_random_xl_larger.sh`
- Create (committed): `report/xl_extrapolation/REPORT.md`

**Interfaces:**
- Consumes: the Exp 14 verdict (Task 8). The XL grid is **N∈{20,35,50} × M∈{50,75,100}** on the 4
  maps, 100 instances/config — identical to the coworker's existing XL run (his current RESULTS.md
  Exp 13, which is renumbered to Exp 15).

**Branch on the Exp 14 verdict:**

- [ ] **Step 1 (if B1/B2 wins Exp 14): reuse the existing XL run — no new compute.**

The coworker's XL extrapolation already exists (`eval_paper_xl_{current,larger}.sh`, results in
`results/fullpipe_paper_xl/…`, written up as his current RESULTS.md Exp 13). Exp 15 = that run,
renumbered. Action: in `RESULTS.md` relabel his XL section header to **Exp 15** (part of the
confirm-first RESULTS.md renumber). No scripts, no cluster jobs. Skip Steps 2–4.

- [ ] **Step 2 (if C1/C2 wins Exp 14): create the C XL eval scripts.**

Copy `scripts/eval_paper_xl_current.sh` → `scripts/exp14/eval_random_xl_current.sh`, changing only the
checkpoint to `checkpoints/random_current_s0/best.pt` and the output dir to
`results/fullpipe_random_xl/current/`. Keep the XL config grid (N∈{20,35,50}×M∈{50,75,100}, 4 maps),
`--n_instances 100`, `--max_seconds`, `--instance_timeout 600`, and the recover-script pattern
exactly as his. Likewise `eval_paper_xl_larger.sh` → `eval_random_xl_larger.sh` with
`checkpoints/random_larger_s0/best.pt` and `results/fullpipe_random_xl/larger/`.

- [ ] **Step 3 (if C wins): syntax-check + commit the scripts**

```bash
bash -n scripts/exp14/eval_random_xl_current.sh && bash -n scripts/exp14/eval_random_xl_larger.sh && echo OK
git add scripts/exp14/eval_random_xl_current.sh scripts/exp14/eval_random_xl_larger.sh
git commit -m "agent: add C1/C2 zero-shot XL extrapolation eval jobs (Exp 15)"
```
Expected: `OK`. Coworker submits these on the cluster; aggregate with `agg_paper_maps.py
--base results/fullpipe_random_xl`.

- [ ] **Step 4: Write the Exp 15 report**

Create `report/xl_extrapolation/REPORT.md`: the XL cost-ratio / diff / speedup tables for the winning
family, and whether the win from Exp 14 holds, widens, or reverses at paper-scale N/M (toward the
paper's N≤50/M≤100). If B won (Step 1), this report cites the existing Exp-13/now-15 numbers and
adds the framing that B was the chosen family. Commit:
```bash
git add report/xl_extrapolation/REPORT.md
git commit -m "report: Exp 15 — zero-shot XL extrapolation on the Exp 14 winner"
```

---

## Self-Review

**Spec coverage (by experiment):**
- **Exp 13.b** — Model A at large N/M on real maps, offline + full-pipeline, A-vs-B1 → Tasks 6–7. ✓
- **Exp 14** — C1/C2 training data (random 32×32, walls 0–0.5, 80k/config) → Task 1; C1/C2 training (his recipe) → Task 2; C1/C2 eval on 4 maps (his method) → Task 3; C-vs-B comparison → Task 4 (`agg_compare.py`); cluster handoff → Task 5; report + verdict + conclusions → Task 8. ✓
- **Exp 15** — zero-shot XL extrapolation on the Exp-14 winner (B → reuse existing run; C → new XL scripts) + report → Task 9. ✓
- Exp 12 (B1 vs B2, coworker's, number unchanged) and Exp 13.a (A small N/M, done) are pre-existing; only the RESULTS.md renumber (his 13→15) remains — flagged as confirm-first, outside the script tasks. ✓
- Branch-merge prerequisite → done this session (noted in Global Constraints). ✓

**Placeholder scan:** No TBD/TODO. Every script is full content; every verify step has an exact command + expected output. The report-authoring steps (Tasks 7, 8, 9 Step 4) describe required sections with their data sources rather than pre-writing prose — acceptable since the numbers don't exist until the runs complete. Task 9 is intentionally conditional (branches on the Exp 14 verdict); both branches are fully specified.

**Type/name consistency:** `random_diverse` data dir, `random_{current,larger}_s{seed}` checkpoints, `results/fullpipe_random{,_xl}/{current,larger}/` CSVs, and `agg_compare.py` group names (`current`/`larger`) are consistent across Tasks 1→2→3→4→8→9. `load_group(base, group)` reused from `agg_paper_maps.py` with its real signature. "Exp 12" refers to the B-models (fixed maps) throughout the body.

**Scope:** One cluster pipeline (Exp 14, with conditional Exp 15) + one local eval (Exp 13.b). Coherent; no decomposition needed.
