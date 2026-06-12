# Neural MCPF Allocation

Part of the `course_multiagent/` project (Multi-Agent Systems course). A supervised neural network
that learns **goal allocation** for basic MAPF: given N agents and M goals on a grid, decide which
agent visits which goal(s), minimizing the total path cost (Min-Sum). Ground-truth labels come from
the RobustMCPF exact solver; the NN is trained to imitate it and (eventually) replace it for speed.

## Problem type — general mTSP (not 1-to-1)

The solver implements **multi-agent TSP**: an agent can visit 0, 1, or several goals, whichever
minimizes total tour cost. This is *not* a balanced 1-to-1 assignment — e.g. with two agents close
together and two distant goals, the solver gives **both goals to one agent** (cost 7) rather than
splitting them (cost 14). Every goal is visited by exactly one agent, but agents are unconstrained.

Consequence for the model:
- Ground truth `Y[i,j] = 1` iff agent `i` visits goal `j`. **Columns sum to 1; rows are free.**
- Output is a **column-wise softmax** (for each goal, a distribution over agents) — **not** a
  Sinkhorn/doubly-stochastic permutation. An earlier 1-to-1 + Sinkhorn design was discarded once we
  confirmed the solver does mTSP. Do not reintroduce Sinkhorn unless the problem is reframed as
  balanced assignment.

## Phase-1 scope

Basic MAPF: **no orientation/rotation, no delays, deterministic.** Small scale: 5×5 grids, N=2 to
start; `--num_agents` / `--grid_w` / `--grid_h` make this configurable for scale-up experiments.

## Data representation

- **Input `D`** `(B, N, M)`: BFS shortest-path distance, agent `i` → goal `j`, walls respected.
  Normalized to [0,1] by `(W-1)+(H-1)` (true max BFS distance on an open grid).
- **Output `P`** `(B, N, M)`: per-goal assignment probabilities, `P[:, :, j].sum(dim=1) == 1`.
- **Ground truth `Y`** `(B, N, M)`: binary, column sums = 1, from the solver.

## Loss

`Total = L_CE + λ · L_MinSum` (`model/losses.py:mTSP_loss`, default λ=0.1)
- `L_CE = -(1/M) Σ_j Σ_i Y[i,j]·log P[i,j]` — per-goal cross-entropy over agents (mimic solver).
- `L_MinSum = (1/M) Σ P[i,j]·D[i,j]` — expected assignment cost (penalize distant allocations).

## Layout

| Path | Role |
|------|------|
| `solver_wrapper.py` | `run_basic_mapf(map_dims, agents, goals)` → `{allocation, paths, cost}`. The one bridge to RobustMCPF. |
| `dataset_generation/grid_gen.py` | Random grids + agent/goal placement (placement logic inlined from GenerateInstances — that module runs map-loading at import time and can't be imported). |
| `dataset_generation/distance.py` | BFS distance matrix + normalization. |
| `dataset_generation/oracle.py` | Runs the solver, builds `Y`, validates column sums. |
| `dataset_generation/build_dataset.py` | Orchestrates generation; argparse; `--num_workers` parallel solver calls; per-split rng (`_split_rng`); writes `data/{split}/{D,Y}_matrices.npy`. |
| `model/network.py` | Four architectures + `build_model` factory: `GoalAllocMLP` (flatten → 2 hidden layers), `GoalAllocDeepSets` (shared per-goal MLP), `GoalAllocTransformer` (row-column attention on D, per-size positional embeddings), `GoalAllocTransformerUniversal` (no positional embeddings, size-agnostic G injection — one model for any N, M; `universal=True`). All end in column softmax. |
| `model/losses.py` | `mTSP_loss(P, Y, D, lam)`. |
| `training/train.py` | Dataset, Adam + ReduceLROnPlateau, `--model_type {mlp,deepsets,transformer}`, `--run_name` checkpoint subdir. |
| `evaluation/evaluate.py` | Offline metrics: per-goal acc, full-assignment acc, cost ratio. Rebuilds the model from checkpoint args. |
| `evaluation/compare.py` | Aggregates runs across checkpoint dirs: mean±std per run-name group (seed suffix stripped). |
| `evaluation/plot_results.py` | Report figures from hard-coded results: scaling curves (fig1) + N=3 ablation bars (fig2) → `evaluation/figures/` (git-ignored). |
| `tests/` | pytest suite (`conftest.py` does sys.path setup + grid fixtures). |
| `pytest.ini` | testpaths, `--strict-markers`, registers the `slow` marker. |
| `scripts/setup_robustmcpf.sh` | Builds the LKH binary for the local platform (`mkdir SRC/OBJ && make`). |
| `scripts/basic_mapf.patch` | Our one-line BasicMAPF patch — kept for reference; already applied in the vendored tree. |
| `scripts/poc_2agent_2goal.sh` | Slurm job: build LKH → generate 500/100/100 samples → train 50 epochs → evaluate. |
| `scripts/gen_scale_data.sh` | Slurm job (cpu partition, qos=normal): generates all Exp-2/3 datasets sequentially (parallel jobs sharing a split name would collide on LKH temp files). |
| `scripts/exp_arch.sh` | Slurm array (9): {mlp,deepsets,transformer} × 3 seeds at N=2, 5×5. |
| `scripts/exp_scale_n.sh` | Slurm array (24): N∈{2..5} × {mlp,transformer} × 3 seeds, 8×8, 10k train. |
| `scripts/exp_data_scale.sh` | Slurm array (12): transformer N=3 8×8, train size {1k,5k,10k,20k} × 3 seeds, slices `data/n3_8x8_pool`. |
| `scripts/gen_nm_data.sh` | Slurm job: generates N≠M datasets (n2m4, n3m6, n2m6) with G. |
| `scripts/exp_nm.sh` | Slurm array (9): transformer+G on the three N≠M configs × 3 seeds. |
| `scripts/gen_universal_data.sh` | Slurm job (~24h cpu): 15 configs N∈{2,3,4}×M∈{2..6}, 15k/3k/3k each → `data/universal_8x8/n{N}m{M}`. |
| `scripts/exp_universal.sh` | Slurm array (3): universal model on all 15 configs jointly (`--mixed`), then per-config eval. |
| `evaluation/pipeline_eval.py` | NN-vs-solver pipeline eval: tour cost from D+G (optimal per-agent goal order), inference/solver timing, `--n_timing`. |
| `RobustMCPF/` | Third-party solver — **now vendored** (committed); owner granted permission. Binaries and build artifacts are git-ignored via `RobustMCPF/.gitignore`. |

## RobustMCPF integration — must-knows

- **BasicMAPF mode:** pass `algorithm="BasicMAPF"`. This name routes through the `else` branch in
  `Run_Robust_Cbss_Framework` (→ `kBestSequencing`/LKH-TSP), `LowLevelPlan`, `Verify`, and
  `FindConflict`. One source patch was required at `Run_Robust_Cbss_Framework.py:89`
  (`if self.algorithm not in ["IDP", "BasicMAPF"]:`, to skip positive-constraint nodes). The patch
  is already applied in the vendored tree; `scripts/basic_mapf.patch` is kept for reference only.
- **CWD matters:** `kBestSequencing` builds ATSP files and invokes the LKH binary via `os.getcwd()`.
  `solver_wrapper.run_basic_mapf` `chdir`s into `RobustMCPF/` for the call and restores CWD after.
  Each call uses a unique `configStr` to avoid temp-file collisions.
- **Formats:** agents are `[(flat_idx, direction), ...]` (direction ignored in basic mode); goals are
  `[flat_idx, ...]`; `flat_idx = row*Cols + col`. Allocation comes from
  `solver.K_optimal_sequences[1]["Allocations"]` as `{agent: [start_loc, goal_loc, ...]}`.
- **LKH binary** is not committed (platform-specific). Build it once with `scripts/setup_robustmcpf.sh`.

## Running

See [README.md](README.md) for full data-generation and training commands. Quick reference:
```bash
cd dataset_generation && python build_dataset.py --split train --num_samples 10000
cd ../training       && python train.py --N 2 --epochs 100
cd ../evaluation     && python evaluate.py --checkpoint ../checkpoints/best.pt --split test
```

Environment: conda env `mcpf_env` (numpy, scipy, tqdm, matplotlib, pytest via conda; torch installed
via pip due to MKL conflict with conda-channel torch on the cluster):
```bash
conda install -n mcpf_env numpy scipy tqdm matplotlib pytest -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
After creating the env, build LKH: `bash scripts/setup_robustmcpf.sh`

### Cluster (slurm.bgu.ac.il)

Remote path: `/home/dayanb/course_multiagent/Neural_MCPF_Allocation`
Submit PoC job: `sbatch scripts/poc_2agent_2goal.sh` (from project root, after `mkdir -p logs`)

## Results

**Seed-leakage caveat:** before 2026-06-11, `build_dataset.py` seeded every split with the same
base seed, so val/test instances replayed the train set's seed stream. The original PoC numbers
(per-goal 0.855, full 0.720) and the first Exp-1 run were measured on contaminated test sets and
are void. Fixed by `_split_rng` (rng seeded with `[base_seed, split_offset]`); regression test in
`tests/test_build_dataset.py`.

### Exp 1 — architecture comparison (clean data, 2026-06-11)

N=2, M=2, 5×5, 5k train / 1k val / 1k test, 100 epochs, batch=256, hidden=64, λ=0.1, 3 seeds.

| Model | Per-goal acc | Full-assignment acc | Mean cost ratio |
|-------|-------------|---------------------|-----------------|
| DeepSets | 0.858±0.000 | 0.727±0.000 | 0.9689±0.0000 |
| MLP | 0.875±0.004 | 0.756±0.005 | 0.9760±0.0013 |
| Transformer | 0.887±0.001 | 0.782±0.002 | 0.9905±0.0023 |

Transformer wins on both accuracies. Caveat on cost ratio: it uses Σ Y_pred·D (assignment cost on
the BFS matrix), *not* the true mTSP tour cost — below-1 values reflect that metric mismatch, not
beating the solver; closer to 1.0 means closer agreement. Aggregate runs with
`evaluation/compare.py --pattern "arch_*"` (on the login node, prefix with
`MKL_THREADING_LAYER=GNU` — numpy's MKL hits a missing-iomp5 symbol there).

### Exp 2 — scale N (8×8, 10k train / 2k val / 2k test, 3 seeds, 2026-06-11)

Full-assignment accuracy (per-goal in parens):

| N | MLP | Transformer |
|---|-----|-------------|
| 2 | 0.772±0.003 (0.882) | 0.789±0.006 (0.890) |
| 3 | 0.549±0.005 (0.812) | 0.590±0.004 (0.830) |
| 4 | 0.382±0.002 (0.776) | 0.443±0.008 (0.799) |
| 5 | 0.270±0.002 (0.758) | 0.337±0.004 (0.785) |

Transformer beats MLP at every N and the gap widens with N (+1.7pt at N=2 → +6.7pt at N=5).
Full-assignment accuracy decays roughly as (per-goal)^M for both, as expected.

### Exp 3 — data scale (transformer, N=3, 8×8, 3 seeds, 2026-06-11)

| Train size | Per-goal acc | Full-assignment acc |
|-----------|-------------|---------------------|
| 1k | 0.814±0.002 | 0.553±0.003 |
| 5k | 0.824±0.001 | 0.578±0.001 |
| 10k | 0.831±0.003 | 0.599±0.005 |
| 20k | 0.835±0.002 | 0.603±0.003 |

Gains are flattening by 20k (+0.4pt full from 10k→20k) — at this model size, more data alone
won't push much further; capacity (hidden/num_layers) is the next lever.
**Gotcha:** Exp-3 checkpoints store a deleted temp `data_dir`, so `compare.py` must be given
`--data_dir ../data/n3_8x8_pool` explicitly; it now skips runs whose data shape mismatches the
model's (N, M) instead of silently evaluating wrong-size data (the size-agnostic transformer
would happily run on it).

### Exp 4 — capacity (transformer, N=3, 8×8, 20k train, 3 seeds, 2026-06-11)

Baseline h64/L3 = the `datascale_t20000` runs (0.603 full). Full-assignment accuracy:

| hidden | L=3 | L=6 |
|--------|-----|-----|
| 64 | 0.603±0.003 | 0.615±0.005 |
| 128 | 0.605±0.003 | 0.617±0.001 |
| 256 | 0.605±0.007 | 0.242±0.242 (diverged) |

**Capacity is not the bottleneck**: +1.4pt at best (depth helps slightly, width does nothing).
h256/L6 diverges on some seeds at lr=1e-3 — lower the lr if ever revisiting that size.

### Exp 5 — goal-goal distance ablation (transformer h64/L3, 8×8, 10k train, 3 seeds, 2026-06-11)

Input ablation: with vs without G ∈ R^(M×M) (goal-to-goal BFS distances, injected via a
Linear(M→d) goal-context projection — `use_goal_dists`). Full-assignment accuracy:

| N | without G | with G | Δ |
|---|-----------|--------|---|
| 2 | 0.790±0.002 | 0.932±0.005 | +14.2pt |
| 3 | 0.595±0.010 | 0.775±0.008 | +18.0pt |
| 4 | 0.455±0.006 | 0.632±0.007 | +17.7pt |
| 5 | 0.337±0.004 | 0.478±0.011 | +14.1pt (Exp 6, `gn5_*` runs on `data/n5_8x8_G`) |

**The missing tour-structure signal was the real bottleneck** — D alone can't express "these
goals are adjacent, bundle them"; G provides it and dominates any capacity gain by >10×.
The noG rows replicate Exp 2's transformer numbers on freshly generated data (0.789/0.590/0.443),
a good consistency check. At N=2 with G, mean cost ratio is 1.0000 — near-perfect solver agreement.
Train with `--use_goal_dists` on `data/n{2,3,4}_8x8_G` (datasets that include `G_matrices.npy`).

### Exp 7 — universal mixed-size model (2026-06-12)

One `GoalAllocTransformerUniversal` (h64/L3, 151k params — no positional embeddings, G via
scalar-embed + sum-pool) trained jointly on all 15 configs N∈{2,3,4} × M∈{2,3,4,5,6},
15k train / 3k val / 3k test per config (`data/universal_8x8/n{N}m{M}`), 3 seeds.
Train with `--mixed --data_dirs <comma-separated>` (`MixedSizeBatchSampler` keeps each batch
shape-homogeneous; no padding). Full-assignment accuracy vs the per-size Exp-5 specialists:

| Config | Per-size specialist | Universal | Δ |
|--------|--------------------|-----------| --|
| N=M=2 | 0.932±0.005 | 0.913±0.002 | −1.9pt |
| N=M=3 | 0.775±0.008 | 0.810±0.002 | +3.5pt |
| N=M=4 | 0.632±0.007 | 0.656±0.007 | +2.4pt |
| N=M=5 **zero-shot** | 0.478±0.011 | 0.513±0.003 | +3.5pt |

**Zero-shot N=5 beats the specialist trained on N=5** — the model never saw N=5 (trained on
N≤4) yet transfers upward, evidence it learned the allocation principle rather than size-specific
patterns. Only regression is the easiest config (N=2). Mean cost ratio ~0.99 everywhere;
inference ~0.06 ms/instance. Caveat: not data-matched vs baselines (15k/config vs 10k, plus
225k total cross-config transfer). Training is fast: ~25 min/seed on rtx3090 for 100 epochs.

Full 15-config matrix (full-assignment accuracy, 3 seeds, std ≤ 0.010 everywhere):

| N \ M | 2 | 3 | 4 | 5 | 6 |
|-------|---|---|---|---|---|
| 2 | 0.913 | 0.874 | 0.755 | 0.632 | 0.558 |
| 3 | 0.873 | 0.810 | 0.708 | 0.585 | 0.483 |
| 4 | 0.832 | 0.775 | 0.656 | 0.553 | 0.435 |

Accuracy decays with M (more goals = more per-goal decisions that must all be right) and is
nearly symmetric in N. M>N (multi-goal tours) is harder than M<N (idle agents): n2m6 0.558 vs
n4m2 0.832 (per-goal: M dominates difficulty). Offline cost ratios (Σ Y·D metric) stay 0.97–1.00.

### Exp 8 — full-pipeline NN-vs-solver (universal_s0, 2026-06-12)

`evaluation/pipeline_eval.py`: NN argmax allocation → tour cost (D first hop + G goal-goal hops,
optimal per-agent goal order; same metric applied to the solver's allocation for an
apples-to-apples ratio). Solver wall-time measured on 50 reconstructed instances per config.

| Config | Full acc | Tour-cost ratio | NN ms | Solver ms | Speedup |
|--------|---------|-----------------|-------|-----------|---------|
| n3m3 | 0.809 | 1.010 | 0.052 | 64.5 | ~1250× |
| n2m4 | 0.754 | 1.026 | 0.058 | 43.9 | ~760× |
| n4m6 | 0.442 | 1.056 | 0.233 | 59.7 | ~260× |

**The case for the NN**: even where exact-match accuracy drops to 0.44 (n4m6), the tour cost is
only 5.6% above optimal — most "wrong" assignments are near-ties. Three orders of magnitude
faster than the solver at 1–6% cost suboptimality. NN time is batched amortized (batch=512);
single-instance latency would be higher but still ≫100× faster.

## Tests

```bash
pytest -m "not slow"   # fast suite — pure functions, no solver/LKH (~1.5s)
pytest                 # everything, incl. solver-integration tests
```

Solver-free tests cover the silent-corruption risks (BFS distances + normalization, grid/placement
invariants, loss correctness + clamp guard, the column-softmax `sum==1` invariant). Tests that
invoke RobustMCPF/LKH are marked `@pytest.mark.slow` (`tests/test_oracle.py`) — they guard the
integration boundary that broke before (allocation format, every-goal-assigned-once, CWD restore).
Deliberately **not** tested: full `build_dataset`/`train.py` runs and RobustMCPF internals.

## Conventions

Commit scopes (from `course_multiagent/CLAUDE.md`): `agent`, `env`, `tests`, `report`, `viz`.
Use `env` for data/grid pipeline, `agent` for model/training. Do **not** commit generated
`data/*.npy`, `checkpoints/`, or `logs/`. To change solver behavior, edit `scripts/basic_mapf.patch`
and re-apply it to `RobustMCPF/` manually — do not edit `RobustMCPF/` files directly without
updating the patch.
