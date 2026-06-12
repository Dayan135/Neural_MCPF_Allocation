# Experimental Results

All experiments: 100 epochs, batch=256, Adam lr=1e-3, λ=0.1, 3 seeds unless noted.

**Seed-leakage caveat:** before 2026-06-11, `build_dataset.py` seeded every split with the same
base seed, so val/test instances replayed the train set's seed stream. The original PoC numbers
(per-goal 0.855, full 0.720) and the first Exp-1 run were measured on contaminated test sets and
are void. Fixed by `_split_rng` (rng seeded with `[base_seed, split_offset]`); regression test in
`tests/test_build_dataset.py`.

## Exp 1 — architecture comparison (clean data, 2026-06-11)

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

## Exp 2 — scale N (8×8, 10k train / 2k val / 2k test, 3 seeds, 2026-06-11)

Full-assignment accuracy (per-goal in parens):

| N | MLP | Transformer |
|---|-----|-------------|
| 2 | 0.772±0.003 (0.882) | 0.789±0.006 (0.890) |
| 3 | 0.549±0.005 (0.812) | 0.590±0.004 (0.830) |
| 4 | 0.382±0.002 (0.776) | 0.443±0.008 (0.799) |
| 5 | 0.270±0.002 (0.758) | 0.337±0.004 (0.785) |

Transformer beats MLP at every N and the gap widens with N (+1.7pt at N=2 → +6.7pt at N=5).
Full-assignment accuracy decays roughly as (per-goal)^M for both, as expected.

## Exp 3 — data scale (transformer, N=3, 8×8, 3 seeds, 2026-06-11)

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

## Exp 4 — capacity (transformer, N=3, 8×8, 20k train, 3 seeds, 2026-06-11)

Baseline h64/L3 = the `datascale_t20000` runs (0.603 full). Full-assignment accuracy:

| hidden | L=3 | L=6 |
|--------|-----|-----|
| 64 | 0.603±0.003 | 0.615±0.005 |
| 128 | 0.605±0.003 | 0.617±0.001 |
| 256 | 0.605±0.007 | 0.242±0.242 (diverged) |

**Capacity is not the bottleneck**: +1.4pt at best (depth helps slightly, width does nothing).
h256/L6 diverges on some seeds at lr=1e-3 — lower the lr if ever revisiting that size.

## Exp 5 — goal-goal distance ablation (transformer h64/L3, 8×8, 10k train, 3 seeds, 2026-06-11)

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

## Exp 7 — universal mixed-size model (2026-06-12)

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

## Exp 8 — full-pipeline NN-vs-solver (universal_s0, 2026-06-12)

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
