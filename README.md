# Neural MCPF Allocation

A supervised neural network for **goal allocation** in basic Multi-Agent Path Finding (MAPF).
Given N agents and M goals on a grid, the model predicts which agent should visit each goal
(general multi-agent TSP — an agent may take 0, 1, or several goals), minimizing total path cost.
Ground-truth labels are produced by the [RobustMCPF](#credits--rights--robustmcpf-third-party)
exact solver in its basic (no-orientation, no-delay) mode.

See [CLAUDE.md](CLAUDE.md) for the architecture and design decisions.

## Setup

```bash
conda activate mcpf_env          # torch 2.12, numpy, scipy
pip install -r requirements.txt  # tqdm, matplotlib, pytest
```

### RobustMCPF solver (one-time)

The solver is **not vendored** in this repo. Provision it with:

```bash
./scripts/setup_robustmcpf.sh
```

This clones [RobustMCPF](#credits--rights--robustmcpf-third-party) at a pinned commit,
compiles the LKH-3 TSP solver for your platform, and applies the one-line BasicMAPF patch
(`scripts/basic_mapf.patch`). Only LKH is built — BasicMAPF does not use GLKH.

## Data generation

Each sample is built by: random grid → place agents/goals → BFS distance matrix `D` →
run the RobustMCPF solver for the optimal allocation `Y`. Samples with unreachable goals or
solver failures are skipped. Output is saved as `data/{split}/D_matrices.npy` and
`Y_matrices.npy` (shapes `(S, N, M)`).

```bash
cd dataset_generation
python build_dataset.py --split train --num_samples 10000
python build_dataset.py --split val   --num_samples 1000
python build_dataset.py --split test  --num_samples 1000
```

Key flags (all optional): `--grid_w 5 --grid_h 5 --num_agents 2 --obstacle_prob 0.1 --seed 42`.
Scale up by raising `--num_agents` and the grid dimensions.

## Training

```bash
cd training
python train.py --N 2 --epochs 100 --batch_size 256 --lr 1e-3 --lam 0.1
```

Trains `GoalAllocMLP` with the hybrid loss `L_CE + λ·L_MinSum`, Adam + ReduceLROnPlateau, and
saves the best-val-loss checkpoint to `checkpoints/best.pt`. Each epoch logs train/val loss
(CE and MinSum terms) and per-goal accuracy. Set `--M` if the number of goals differs from agents;
sweep `--lam` (e.g. 0, 0.01, 0.1, 1.0) for the loss-weight ablation.

## Evaluation

```bash
cd evaluation
python evaluate.py --checkpoint ../checkpoints/best.pt --split test
```

Reports per-goal accuracy, full-assignment accuracy, and cost ratio (NN cost / solver-optimal cost).

## Tests

```bash
pytest -m "not slow"   # fast suite: distance, grids, loss, network (no solver, ~1.5s)
pytest                 # also runs the RobustMCPF/LKH integration tests (marked `slow`)
```

---

## Credits & Rights — `RobustMCPF/` (third-party)

RobustMCPF is an **external repository** and is **not** original work of this project. It is the
exact combinatorial solver used to generate ground-truth goal allocations. It is **not** committed
here (it is large, ships solver binaries, and is third-party) — `scripts/setup_robustmcpf.sh`
clones it on demand and `RobustMCPF/` is git-ignored.

- **Source:** https://github.com/yehonatan280198/RobustMCPF
- **Paper:** *Robust Multiagent Combinatorial Path Finding* — accepted at **AAAI-2026**.
- **Authors:** Yehonatan Kidushim, Avraham Natan, Roni Stern, Meir Kalech.

**Rights / licensing:** the upstream repository ships **no LICENSE file**. All rights to the
RobustMCPF code, data, and the accompanying technical appendix remain with the original authors.
It is used here only for research/coursework reproduction. Before any redistribution or use beyond
this course, confirm the intended terms with the original authors. Do not relicense it. Our only
modification is the one-line BasicMAPF patch in `scripts/basic_mapf.patch` (applied by the setup
script), which enables the basic-MAPF mode the course requires.

All other files in this repository (the dataset pipeline, model, training, and evaluation code) are
the work of this project.
