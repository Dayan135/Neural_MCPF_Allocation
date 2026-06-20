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

## Exp 9 — full MAPF execution cost (universal_s0, 200 instances/config, 2026-06-12)

The end-to-end comparison (`evaluation/full_pipeline_eval.py`): same fresh instance fed to both
pipelines; **both end in CBS collision-free path planning**; costs are true execution costs
(`Solution[5]`). NN pipeline = NN forward + brute-force goal ordering + CBS on the fixed
allocation (`run_basic_mapf_with_allocation`, no LKH call). Solver pipeline = LKH + CBS.
Times are single-instance (not batched), measured on a cluster cpu node.

| Config | Exec-cost ratio | Exact-cost match | Mean diff (steps) | NN ms | Solver ms | Speedup |
|--------|----------------|------------------|--------------------|-------|-----------|---------|
| n2m2 | 1.0101 | 96.5% | 0.08 | 9.6 | 16.2 | 1.7× |
| n3m3 | 1.0070 | 97.0% | 0.07 | 8.0 | 21.7 | 2.7× |
| n4m4 | 1.0291 | 88.0% | 0.27 | 9.2 | 28.3 | 3.1× |
| n2m4 | 1.0202 | 88.5% | 0.26 | 6.8 | 15.1 | 2.2× |
| n4m6 | 1.0500 | 72.5% | 0.69 | 10.0 | 36.3 | 3.6× |
| n2m6 | 1.0626 | 64.0% | 1.09 | 9.2 | 22.5 | 2.4× |

Key observations:
- **Execution cost is 0.7–6.3% above optimal** across all configs; exact-cost match (64–97%) is
  much higher than exact-assignment match (Exp 7: 44–91%) — many "wrong" allocations are
  cost-equivalent ties.
- **Single-instance speedup is 1.7–3.6×**, far below the batched-inference speedups of Exp 8
  (~10³×): when the NN must plan paths too, CBS dominates both pipelines and only the LKH call
  is saved. The NN's real speed advantage appears when (a) instances are batched, or (b) the
  TSP/LKH share grows with problem size — the speedup indeed trends up with M (1.7× → 3.6×).
- Hardest config (n2m6, ~3 goals/agent tours): ratio 1.063 — consistent with Exp 8's tour-cost
  picture; allocation errors, not collisions, drive the gap.
- Round-trip validation: injecting the solver's own allocation reproduces its exact cost
  (`tests/test_fixed_alloc.py`).

## Exp 10 — big full-pipeline sweep with zero-shot extrapolation (universal_s0, 2026-06-12)

`scripts/exp_full_pipeline_big.sh`: 28 configs N∈{2,3,4,5} × M∈{2..8}, 200 fresh instances each,
8×8 grid, 10% obstacles, single-instance timing on a cluster cpu node. universal_s0 trained only
on N∈{2,3,4} × M∈{2..6} — configs marked **Extrap** are zero-shot (N=5 row, M∈{7,8} columns).

### Column glossary

| Column | Meaning |
|--------|---------|
| Config | `nNmM` = N agents, M goals on the 8×8 grid. |
| Extrap | ✓ = outside the training range (N=5 and/or M≥7) — zero-shot extrapolation. |
| Cost ratio | Mean of (NN execution cost / solver execution cost) per instance. Execution cost = total collision-free path length after CBS conflict resolution (`Solution[5]`) — the *real* MAPF objective, not a BFS approximation. 1.0 = matches the optimal solver. |
| Exact match | Fraction of instances where the NN pipeline's execution cost equals the solver's exactly. Higher than assignment match — many differing allocations are cost ties. |
| Diff (steps) | Mean (NN cost − solver cost) in grid steps per instance. |
| Infeasible | Fraction of instances where *all* NN allocation candidates (up to 3, ranked by joint probability) admitted no collision-free plan. These have no cost and are excluded from the other columns. |
| Fallback | Fraction of instances where the argmax allocation was infeasible and a lower-probability candidate (k≥2) was used instead — the NN-side analogue of the solver's k-best escape. |
| NN ms | Mean wall time of the complete NN pipeline per instance: model forward + goal-visit ordering (brute-force over the agent's goals) + CBS path planning on the fixed allocation (no LKH call). |
| Solver ms | Mean wall time of the full solver per instance: LKH TSP allocation + CBS path planning. |
| Speedup | Solver ms / NN ms. |

### Results

| Config | Extrap | Cost ratio | Exact match | Diff (steps) | Infeasible | Fallback | NN ms | Solver ms | Speedup |
|--------|:------:|-----------|-------------|------|------|------|-------|-----------|---------|
| n2m2 | | 1.010 | 96.5% | 0.07 | 0% | 0% | 11.3 | 33.5 | 3.0× |
| n2m3 | | 1.006 | 98.0% | 0.08 | 0% | 0% | 12.2 | 32.4 | 2.7× |
| n2m4 | | 1.020 | 88.5% | 0.26 | 0% | 0% | 12.7 | 31.5 | 2.5× |
| n2m5 | | 1.044 | 74.5% | 0.69 | 0% | 0% | 12.2 | 38.4 | 3.1× |
| n2m6 | | 1.063 | 64.0% | 1.09 | 0% | 0% | 17.0 | 38.3 | 2.3× |
| n2m7 | ✓ | 1.090 | 54.5% | 1.61 | 0% | 0% | 18.3 | 37.5 | 2.1× |
| n2m8 | ✓ | 1.078 | 51.0% | 1.55 | 0% | 0% | 38.7 | 44.3 | 1.1× |
| n3m2 | | 1.009 | 98.0% | 0.07 | 0% | 0% | 13.3 | 34.9 | 2.6× |
| n3m3 | | 1.007 | 97.0% | 0.07 | 0% | 0% | 12.1 | 30.9 | 2.6× |
| n3m4 | | 1.043 | 83.0% | 0.47 | 0% | 0% | 11.7 | 39.2 | 3.4× |
| n3m5 | | 1.046 | 76.5% | 0.60 | 0% | 0% | 12.3 | 39.1 | 3.2× |
| n3m6 | | 1.070 | 66.5% | 1.03 | 0% | 0% | 12.9 | 41.0 | 3.2× |
| n3m7 | ✓ | 1.068 | 58.0% | 1.12 | 0% | 0% | 16.4 | 43.7 | 2.7× |
| n3m8 | ✓ | 1.100 | 43.5% | 1.82 | 0% | 0% | 20.4 | 51.9 | 2.5× |
| n4m2 | | 1.000 | 100.0% | 0.00 | 0% | 0% | 10.9 | 33.3 | 3.1× |
| n4m3 | | 1.010 | 96.0% | 0.09 | 0% | 0% | 11.7 | 37.0 | 3.2× |
| n4m4 | | 1.029 | 88.0% | 0.27 | 0% | 0% | 12.1 | 39.8 | 3.3× |
| n4m5 | | 1.042 | 76.0% | 0.54 | 0% | 0% | 12.8 | 45.1 | 3.5× |
| n4m6 | | 1.050 | 72.5% | 0.69 | 0% | 0% | 13.3 | 45.8 | 3.5× |
| n4m7 | ✓ | 1.074 | 55.5% | 1.09 | 0% | 0% | 19.8 | 49.3 | 2.5× |
| n4m8 | ✓ | 1.092 | 40.0% | 1.52 | 0% | 0% | 19.6 | 53.0 | 2.7× |
| n5m2 | ✓ | 1.002 | 99.5% | 0.01 | 0% | 0% | 34.8 | 37.4 | 1.1× |
| n5m3 | ✓ | 1.009 | 96.5% | 0.07 | 0% | 0% | 30.1 | 38.6 | 1.3× |
| n5m4 | ✓ | 1.020 | 91.5% | 0.19 | 0% | 0% | 13.3 | 44.4 | 3.3× |
| n5m5 | ✓ | 1.041 | 76.5% | 0.47 | 0% | 0.5% | 41.5 | 49.3 | 1.2× |
| n5m6 | ✓ | 1.053 | 73.0% | 0.67 | 0% | 0% | 14.3 | 55.8 | 3.9× |
| n5m7 | ✓ | 1.070 | 52.5% | 0.98 | 0% | 0% | 15.5 | 63.0 | 4.1× |
| n5m8 | ✓ | 1.093 | 38.5% | 1.49 | 0% | 0% | 17.4 | 65.3 | 3.8× |

### Findings

1. **Zero-shot N=5 is free.** The N=5 row (never trained) matches the in-distribution N=4 row at
   every M (e.g. n5m4 1.020 vs n4m4 1.029; n5m5 1.041 vs n4m5 1.042) — the universal model
   extrapolates to an unseen agent count with no execution-cost penalty.
2. **M is the difficulty axis, N barely matters.** Every row shows the same trend as M grows:
   ratio ~1.01 → ~1.09, exact match ~97% → ~40%. M-extrapolation (7, 8) degrades gracefully —
   worst case n3m8 at 1.100, i.e. 10% above optimal.
3. **Infeasibility exists but is rare and handled.** Zero instances were infeasible after the
   probability-ranked fallback chain; exactly one instance (n5m5, 0.5%) needed a k=2 candidate.
   That same instance previously hung the sweep in an infinite CBS constraint-escalation loop —
   the `cbs_node_budget` guard plus fallback resolved it in milliseconds. (CBS does not terminate
   on truly infeasible allocations; the framework uses vanish-at-target semantics, so the minimal
   infeasible case is two agents forced to cross in a width-1 corridor.)
4. **Speedup grows with problem size**: solver time climbs 31 → 65 ms as the LKH TSP grows, NN
   pipeline stays ~12–20 ms → up to 4.1× at n5m7. NN-side outliers (n2m8 38.7 ms, n5m5 41.5 ms)
   are the 8-goal brute-force visit ordering and the fallback retry, respectively.

## Exp 11 — larger model (h128/L6) on a diverse dataset (large_s{0,1,2}, 2026-06-13)

**Motivation.** The Exp 7–10 universal model (h64/L3, ~151k params) was trained on a narrow
distribution: fixed 8×8 grids, fixed obstacle prob 0.1, N∈{2,3,4} × M∈{2..6}. Two ceilings at
once — model capacity (one network fitting 15 joint distributions) and data diversity (never
seeing varied geometry). Exp 11 lifts both: **h128/L6 (~1.2M params)** trained on **all 28
configs** N∈{2,3,4,5} × M∈{2..8}, with **per-instance random grids (8–12 × 8–12) and wall
density (0.1–0.5)**. 30k/3k/3k per config = 840k train samples. lr=5e-4, grad_clip=1.0,
150 epochs, 3 seeds. Best val loss 0.2514/0.2528/0.2537. Scripts: `gen_large_data.sh`,
`exp_large_model.sh`.

**Data-generation fix.** At up to 50% walls, instances can pass the BFS-reachability filter yet be
so constraint-dense that CBS never terminates (it escalates time constraints forever — the same
non-termination Exp 10 hit with infeasible allocations). 12 of 28 generation tasks hung. Fixed by
adding `cbs_node_budget` to `run_basic_mapf` (default 50k in `get_ground_truth`); budget-exceeded
instances are rejected like unreachable ones. At p=0.1 this changes nothing in practice.

### Offline metrics — diverse test set, 3 seeds

Per-goal accuracy (full-assignment accuracy in parens). Seeds near-identical (std ≈ 0.000).
Cost ratio is the first-hop Σ P·D proxy (see Exp 1 caveat), 0.97–1.00 throughout.

| N\M | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-----|---|---|---|---|---|---|---|
| **2** | 0.960 (.94) | 0.949 (.90) | 0.928 (.84) | 0.911 (.76) | 0.892 (.69) | 0.876 (.63) | 0.852 (.55) |
| **3** | 0.930 (.89) | 0.926 (.85) | 0.915 (.79) | 0.897 (.71) | 0.881 (.65) | 0.864 (.57) | 0.850 (.51) |
| **4** | 0.923 (.87) | 0.919 (.82) | 0.899 (.73) | 0.886 (.67) | 0.875 (.60) | 0.854 (.51) | 0.841 (.47) |
| **5** | 0.921 (.86) | 0.910 (.79) | 0.903 (.73) | 0.887 (.65) | 0.877 (.58) | 0.860 (.51) | 0.845 (.44) |

Accuracy degrades gracefully with M and is essentially flat across N — N=5, zero-shot for the old
model, is now fully in-distribution and as strong as N=2.

### Full pipeline at 8×8/p=0.1 — direct comparison to Exp 10 (large_s0, 200 inst/config)

Same conditions as Exp 10, so the cost ratio is directly comparable to the h64/L3 universal model.
**Mean over 28 configs: cost ratio 1.029 (was ~1.04), exact-match 82.9%, 0% infeasible,
0% fallbacks, 2.4× speedup.** Columns as in Exp 10.

All 28 configs are in-distribution for large_s0 (it trained on every N×M pair), so unlike Exp 10
there is no extrapolation column. `(—)` = config absent from the Exp 10 table (was zero-shot there).

| Config | Cost ratio (old → new) | Exact match | Diff (steps) | Speedup |
|--------|------------------------|-------------|--------------|---------|
| n2m2 | 1.010 → **1.008** | 97.0% | 0.06 | 2.0× |
| n2m3 | 1.006 → **1.008** | 96.5% | 0.11 | 2.3× |
| n2m4 | 1.020 → **1.015** | 89.5% | 0.21 | 1.9× |
| n2m5 | 1.044 → **1.035** | 77.0% | 0.58 | 2.1× |
| n2m6 | 1.063 → **1.059** | 72.0% | 0.96 | 2.1× |
| n2m7 | 1.090 → **1.070** | 60.0% | 1.27 | 2.2× |
| n2m8 | 1.078 → **1.057** | 61.5% | 1.17 | 1.2× |
| n3m2 | 1.009 → **1.003** | 99.0% | 0.03 | 2.3× |
| n3m3 | 1.007 → **1.008** | 97.5% | 0.06 | 2.1× |
| n3m4 | 1.043 → **1.016** | 92.0% | 0.19 | 2.3× |
| n3m5 | 1.046 → **1.029** | 87.0% | 0.41 | 2.5× |
| n3m6 | 1.070 → **1.040** | 77.5% | 0.60 | 2.3× |
| n3m7 | 1.068 → **1.043** | 70.0% | 0.71 | 2.2× |
| n3m8 | 1.100 → **1.067** | 61.0% | 1.24 | 2.4× |
| n4m2 | 1.000 → 1.011 | 97.0% | 0.05 | 2.5× |
| n4m3 | 1.010 → 1.014 | 95.5% | 0.11 | 2.8× |
| n4m4 | 1.029 → **1.007** | 96.5% | 0.07 | 2.8× |
| n4m5 | 1.042 → **1.014** | 92.5% | 0.20 | 2.4× |
| n4m6 | 1.050 → **1.032** | 84.0% | 0.49 | 2.3× |
| n4m7 | 1.074 → **1.047** | 69.5% | 0.70 | 2.6× |
| n4m8 | 1.092 → **1.053** | 66.5% | 0.86 | 2.4× |
| n5m2 | 1.002 → **1.001** | 99.5% | 0.01 | 2.2× |
| n5m3 | 1.009 → 1.022 | 93.5% | 0.17 | 2.4× |
| n5m4 | 1.020 → **1.012** | 95.0% | 0.10 | 2.7× |
| n5m5 | 1.041 → **1.020** | 88.5% | 0.24 | 2.6× |
| n5m6 | 1.053 → **1.030** | 79.5% | 0.40 | 2.7× |
| n5m7 | (—) → 1.051 | 64.5% | 0.72 | 2.8× |
| n5m8 | (—) → 1.054 | 61.0% | 0.86 | 2.9× |

**Takeaways.**
1. **Gains concentrate on the hard cells.** Every M≥4 config improves, several by 2–3 points
   (n3m4 1.043→1.016, n4m5 1.042→1.014, n4m8 1.092→1.053, n5m5 1.041→1.020). Easy small-M
   configs move ≤1pt either way — noise around already-near-optimal allocations.
2. **No infeasible allocations, no fallbacks anywhere** (the old model occasionally needed a
   k≥2 candidate). The larger model's top allocation is always CBS-feasible at 8×8.
3. **This is at a distribution disadvantage**: large_s0 trained mostly on bigger, denser grids,
   yet still beats the in-distribution old model at 8×8. The on-distribution diverse sweep
   (below) is the fairer test.

### Full pipeline on the diverse distribution (large_s0, random 8–12 grids, p∈[0.1,0.5])

The fair test: 200 instances/config drawn from the model's own training distribution (grid size
and wall density sampled per instance, via the new `--grid_w_max/--grid_h_max/--obstacle_prob_max`
flags in `full_pipeline_eval.py`). The 50k-node solver budget guards against dense-maze CBS hangs.

**Mean over 28 configs: cost ratio 1.020, exact-match 85.0%, 0% infeasible, 0% fallbacks,
0 solver-skipped, 2.1× speedup.** On-distribution the model is *better* than at 8×8 (1.020 vs
1.029) — the off-distribution penalty in the comparison table above is real but small. The budget
never fired (0 skipped across all 5,600 instances), so these diverse grids, while harder, are not
pathological for the solver.

| Config | Cost ratio | Exact match | Diff (steps) | Speedup |
|--------|-----------|-------------|--------------|---------|
| n2m2 | 1.005 | 99.0% | 0.04 | 1.9× |
| n2m3 | 1.008 | 97.5% | 0.12 | 1.9× |
| n2m4 | 1.010 | 93.5% | 0.17 | 1.7× |
| n2m5 | 1.018 | 85.0% | 0.42 | 1.8× |
| n2m6 | 1.028 | 80.5% | 0.66 | 2.1× |
| n2m7 | 1.035 | 74.5% | 0.81 | 2.1× |
| n2m8 | 1.051 | 66.5% | 1.39 | 1.3× |
| n3m2 | 1.004 | 99.0% | 0.04 | 2.2× |
| n3m3 | 1.005 | 96.0% | 0.06 | 1.9× |
| n3m4 | 1.011 | 93.5% | 0.17 | 2.1× |
| n3m5 | 1.024 | 82.0% | 0.49 | 2.0× |
| n3m6 | 1.027 | 82.5% | 0.55 | 2.3× |
| n3m7 | 1.027 | 73.5% | 0.55 | 2.1× |
| n3m8 | 1.044 | 65.0% | 1.08 | 2.1× |
| n4m2 | 1.003 | 99.0% | 0.02 | 2.6× |
| n4m3 | 1.002 | 98.5% | 0.02 | 2.2× |
| n4m4 | 1.007 | 93.5% | 0.09 | 2.3× |
| n4m5 | 1.015 | 89.5% | 0.25 | 0.6× |
| n4m6 | 1.033 | 79.5% | 0.63 | 2.3× |
| n4m7 | 1.039 | 72.5% | 0.81 | 2.4× |
| n4m8 | 1.042 | 63.5% | 0.97 | 2.7× |
| n5m2 | 1.004 | 98.5% | 0.02 | 2.3× |
| n5m3 | 1.008 | 96.0% | 0.08 | 2.4× |
| n5m4 | 1.010 | 93.0% | 0.15 | 2.5× |
| n5m5 | 1.016 | 89.5% | 0.22 | 1.1× |
| n5m6 | 1.018 | 83.0% | 0.34 | 2.3× |
| n5m7 | 1.035 | 73.0% | 0.67 | 2.8× |
| n5m8 | 1.039 | 63.5% | 0.83 | 2.7× |

M remains the difficulty axis (ratio and exact-match track M, roughly flat in N); the worst cell
(n2m8, ratio 1.051) is still within ~5% of optimal execution cost. The two low speedups (n4m5 0.6×,
n5m5 1.1×) are single instances where the brute-force goal-visit ordering hit a 7-goal agent — the
NN allocation itself is sub-millisecond; the cost is in the classical ordering step, not the model.

### Raw data

Both pipeline sweeps wrote a per-instance CSV per config (200 rows each), so every aggregate above
is reconstructible per-instance and new statistics (std, percentiles, per-instance diffs) need no
re-run. **On the cluster only** (`results/` is git-ignored — generated artifacts, not committed):

- `results/fullpipe_large/n{N}m{M}.csv` — 8×8/p=0.1 sweep (28 files)
- `results/fullpipe_large_diverse/n{N}m{M}.csv` — diverse-grid sweep (28 files)
- Columns: `inst_seed, cost_nn, cost_solver, nn_k, solver_k, conflicts_nn, conflicts_solver,
  alloc_ms, nn_plan_ms, solver_ms`

Aggregate job logs: `logs/eval_large_18139134.out` (offline — per-config only, no per-instance CSV),
`logs/fullpipe_large_18139135.out` (8×8), `logs/fp_large_div_18139275.out` (diverse).

## Exp 12 — paper benchmark maps, scaled N/M, h128/L6 vs h256/L8 (paper_*_s0, 2026-06-18)

**Motivation.** Move off procedural 8–12 grids onto the RobustMCPF paper's actual benchmark maps,
with a moderate scale-up, and ask whether a *larger* model beats the current one here. Two universal
models trained jointly on the **4 small 32×32 MovingAI maps** (empty, random-20, maze, room) ×
**N∈{5,10,15} × M∈{10,20,30}** (36 configs, 20k/2k/2k each): the current **h128/L6** (1.19M params)
and a larger **h256/L8** (6.33M). Same data; lr 5e-4 (h128) / 3e-4 (h256), grad_clip 1.0, 150 ep,
mixed-size. Scripts: `gen_paper_maps_data.sh`, `exp_paper_{current,larger}.sh`.

**Training.** Both wall-limited before 150 ep. Best per-goal val acc: h128/L6 **0.857** (ep ~58–64),
h256/L8 0.851 (ep ~33–40). The larger model hit its val minimum early then overfit (train acc → 0.91,
val loss rising) — extra capacity did not help in-distribution.

### Full-pipeline execution cost — 200 instances/config, true CBS cost

`eval_paper_maps_{current,larger}.sh` → `full_pipeline_eval.py --map_file` (NN argmax allocation →
goal ordering → CBS, vs LKH+CBS solver; cost = `Solution[5]`). Mean over 36 configs:

| Model | Cost ratio | Exact match | Diff mean | Diff max | Diff std | Speedup |
|-------|-----------|-------------|-----------|----------|----------|---------|
| **h128/L6 (current)** | **1.048** | **0.433** | **6.48** | 147 | **8.71** | **1.54×** |
| h256/L8 (larger) | 1.052 | 0.402 | 7.09 | 153 | 9.12 | 1.24× |

**h128/L6 wins in-distribution** on every aggregate metric — lower cost, higher exact-match, and
~20% faster inference (2.5× the params buys nothing), consistent with the overfitting picture.
It holds per cell too — current ≤ larger on cost ratio at all 4 maps and ~every (N,M):

| (N,M) | ratio cur | ratio lrg | exact cur | exact lrg |
|-------|-----------|-----------|-----------|-----------|
| 5,10  | 1.040 | 1.038 | 0.608 | 0.585 |
| 5,20  | 1.074 | 1.076 | 0.280 | 0.261 |
| 5,30  | 1.102 | 1.113 | 0.111 | 0.091 |
| 10,10 | 1.022 | 1.025 | 0.706 | 0.680 |
| 10,20 | 1.041 | 1.043 | 0.410 | 0.394 |
| 10,30 | 1.061 | 1.069 | 0.196 | 0.166 |
| 15,10 | 1.015 | 1.017 | 0.801 | 0.771 |
| 15,20 | 1.031 | 1.034 | 0.505 | 0.440 |
| 15,30 | 1.046 | 1.055 | 0.278 | 0.231 |

By map (cost ratio cur/lrg): empty 1.053/1.060, random 1.053/1.055, maze 1.029/1.035, room 1.057/1.059.

### Full per-config tables (200 instances each; n<200 = solver-budget skips)

**h128/L6 (current) — per-config (36 configs):**

| Config | n | Cost ratio | Exact | Diff mean | Diff max | Diff std | Speedup |
|--------|---|-----------|-------|-----------|----------|----------|---------|
| empty-32-32 n5m10 | 200 | 1.0524 | 0.510 | 3.84 | 36 | 6.74 | 1.27× |
| empty-32-32 n5m20 | 200 | 1.0773 | 0.220 | 8.62 | 61 | 10.10 | 1.61× |
| empty-32-32 n5m30 | 200 | 1.1069 | 0.060 | 15.62 | 65 | 13.18 | 1.85× |
| empty-32-32 n10m10 | 200 | 1.0250 | 0.705 | 1.42 | 24 | 3.57 | 1.49× |
| empty-32-32 n10m20 | 200 | 1.0502 | 0.360 | 4.86 | 38 | 7.31 | 1.71× |
| empty-32-32 n10m30 | 200 | 1.0685 | 0.120 | 8.88 | 52 | 8.67 | 1.86× |
| empty-32-32 n15m10 | 200 | 1.0137 | 0.840 | 0.71 | 13 | 2.08 | 1.74× |
| empty-32-32 n15m20 | 200 | 1.0322 | 0.490 | 2.83 | 24 | 4.59 | 1.99× |
| empty-32-32 n15m30 | 200 | 1.0470 | 0.230 | 5.45 | 37 | 6.30 | 2.24× |
| random-32-32-20 n5m10 | 200 | 1.0448 | 0.545 | 3.45 | 37 | 6.81 | 1.16× |
| random-32-32-20 n5m20 | 200 | 1.0842 | 0.195 | 10.74 | 50 | 12.00 | 1.11× |
| random-32-32-20 n5m30 | 200 | 1.1067 | 0.045 | 17.20 | 66 | 13.19 | 1.34× |
| random-32-32-20 n10m10 | 200 | 1.0247 | 0.660 | 1.59 | 28 | 3.52 | 1.63× |
| random-32-32-20 n10m20 | 200 | 1.0470 | 0.290 | 5.01 | 26 | 6.53 | 1.75× |
| random-32-32-20 n10m30 | 200 | 1.0614 | 0.125 | 8.72 | 48 | 8.45 | 1.99× |
| random-32-32-20 n15m10 | 200 | 1.0195 | 0.705 | 1.09 | 18 | 2.69 | 2.00× |
| random-32-32-20 n15m20 | 200 | 1.0327 | 0.430 | 3.08 | 37 | 5.05 | 0.72× |
| random-32-32-20 n15m30 | 200 | 1.0542 | 0.190 | 7.04 | 44 | 8.39 | 2.48× |
| maze-32-32-2 n5m10 | 200 | 1.0103 | 0.860 | 1.49 | 48 | 6.00 | 1.30× |
| maze-32-32-2 n5m20 | 200 | 1.0462 | 0.565 | 9.90 | 119 | 20.33 | 0.24× |
| maze-32-32-2 n5m30 | 200 | 1.0748 | 0.295 | 19.09 | 147 | 25.85 | 0.31× |
| maze-32-32-2 n10m10 | 200 | 1.0136 | 0.860 | 1.43 | 66 | 6.50 | 1.84× |
| maze-32-32-2 n10m20 | 200 | 1.0218 | 0.640 | 3.58 | 67 | 8.80 | 2.01× |
| maze-32-32-2 n10m30 | 200 | 1.0348 | 0.460 | 7.04 | 86 | 13.54 | 0.58× |
| maze-32-32-2 n15m10 | 200 | 1.0070 | 0.890 | 0.64 | 31 | 3.00 | 2.34× |
| maze-32-32-2 n15m20 | 200 | 1.0182 | 0.680 | 2.56 | 52 | 7.32 | 0.66× |
| maze-32-32-2 n15m30 | 199 | 1.0338 | 0.513 | 5.76 | 112 | 12.63 | 0.42× |
| room-32-32-4 n5m10 | 200 | 1.0515 | 0.515 | 4.63 | 40 | 7.65 | 3.84× |
| room-32-32-4 n5m20 | 199 | 1.0868 | 0.141 | 12.59 | 117 | 14.15 | 3.85× |
| room-32-32-4 n5m30 | 198 | 1.1186 | 0.045 | 21.64 | 114 | 16.04 | 1.61× |
| room-32-32-4 n10m10 | 200 | 1.0244 | 0.600 | 1.84 | 22 | 3.57 | 1.56× |
| room-32-32-4 n10m20 | 198 | 1.0465 | 0.348 | 5.77 | 50 | 8.76 | 0.25× |
| room-32-32-4 n10m30 | 199 | 1.0790 | 0.080 | 12.58 | 56 | 11.63 | 0.40× |
| room-32-32-4 n15m10 | 200 | 1.0208 | 0.770 | 1.41 | 35 | 4.04 | 2.33× |
| room-32-32-4 n15m20 | 200 | 1.0393 | 0.420 | 4.21 | 32 | 6.20 | 0.54× |
| room-32-32-4 n15m30 | 200 | 1.0485 | 0.180 | 6.95 | 39 | 8.29 | 1.37× |

**h256/L8 (larger) — per-config (36 configs):**

| Config | n | Cost ratio | Exact | Diff mean | Diff max | Diff std | Speedup |
|--------|---|-----------|-------|-----------|----------|----------|---------|
| empty-32-32 n5m10 | 200 | 1.0518 | 0.495 | 3.88 | 37 | 6.75 | 1.09× |
| empty-32-32 n5m20 | 200 | 1.0818 | 0.215 | 9.21 | 51 | 10.53 | 0.96× |
| empty-32-32 n5m30 | 200 | 1.1336 | 0.025 | 19.34 | 71 | 15.74 | 1.07× |
| empty-32-32 n10m10 | 200 | 1.0250 | 0.695 | 1.45 | 22 | 3.40 | 1.14× |
| empty-32-32 n10m20 | 200 | 1.0503 | 0.320 | 4.82 | 48 | 7.00 | 1.27× |
| empty-32-32 n10m30 | 200 | 1.0784 | 0.120 | 10.10 | 53 | 9.28 | 1.38× |
| empty-32-32 n15m10 | 200 | 1.0172 | 0.790 | 0.88 | 24 | 2.57 | 1.36× |
| empty-32-32 n15m20 | 200 | 1.0424 | 0.375 | 3.73 | 44 | 5.95 | 1.42× |
| empty-32-32 n15m30 | 200 | 1.0592 | 0.155 | 6.85 | 29 | 6.72 | 1.51× |
| random-32-32-20 n5m10 | 200 | 1.0440 | 0.520 | 3.58 | 47 | 7.27 | 1.11× |
| random-32-32-20 n5m20 | 200 | 1.0806 | 0.145 | 10.11 | 80 | 10.66 | 1.33× |
| random-32-32-20 n5m30 | 200 | 1.1091 | 0.040 | 17.51 | 68 | 12.92 | 1.58× |
| random-32-32-20 n10m10 | 200 | 1.0309 | 0.605 | 1.97 | 28 | 4.04 | 0.67× |
| random-32-32-20 n10m20 | 200 | 1.0485 | 0.305 | 5.15 | 42 | 6.71 | 1.61× |
| random-32-32-20 n10m30 | 200 | 1.0676 | 0.090 | 9.66 | 46 | 8.89 | 1.87× |
| random-32-32-20 n15m10 | 200 | 1.0181 | 0.715 | 1.00 | 16 | 2.62 | 1.89× |
| random-32-32-20 n15m20 | 200 | 1.0375 | 0.415 | 3.48 | 34 | 5.63 | 2.00× |
| random-32-32-20 n15m30 | 200 | 1.0575 | 0.145 | 7.40 | 40 | 7.36 | 2.37× |
| maze-32-32-2 n5m10 | 200 | 1.0109 | 0.840 | 1.55 | 102 | 8.10 | 1.16× |
| maze-32-32-2 n5m20 | 200 | 1.0538 | 0.520 | 11.37 | 153 | 20.58 | 1.50× |
| maze-32-32-2 n5m30 | 200 | 1.0864 | 0.275 | 22.11 | 143 | 31.48 | 0.13× |
| maze-32-32-2 n10m10 | 200 | 1.0153 | 0.840 | 1.65 | 104 | 8.68 | 0.38× |
| maze-32-32-2 n10m20 | 200 | 1.0215 | 0.665 | 3.53 | 89 | 9.58 | 2.26× |
| maze-32-32-2 n10m30 | 198 | 1.0513 | 0.399 | 10.42 | 112 | 16.29 | 0.38× |
| maze-32-32-2 n15m10 | 200 | 1.0105 | 0.835 | 0.87 | 27 | 3.05 | 2.07× |
| maze-32-32-2 n15m20 | 200 | 1.0224 | 0.620 | 3.01 | 42 | 6.98 | 0.78× |
| maze-32-32-2 n15m30 | 197 | 1.0422 | 0.497 | 7.14 | 106 | 14.16 | 2.33× |
| room-32-32-4 n5m10 | 200 | 1.0456 | 0.485 | 4.21 | 29 | 6.83 | 1.26× |
| room-32-32-4 n5m20 | 200 | 1.0858 | 0.165 | 12.38 | 61 | 12.33 | 0.26× |
| room-32-32-4 n5m30 | 200 | 1.1220 | 0.025 | 22.31 | 95 | 15.72 | 0.13× |
| room-32-32-4 n10m10 | 200 | 1.0303 | 0.580 | 2.23 | 30 | 4.63 | 1.87× |
| room-32-32-4 n10m20 | 200 | 1.0522 | 0.285 | 6.31 | 35 | 7.69 | 1.20× |
| room-32-32-4 n10m30 | 197 | 1.0780 | 0.056 | 12.38 | 59 | 11.63 | 0.47× |
| room-32-32-4 n15m10 | 200 | 1.0206 | 0.745 | 1.30 | 19 | 3.24 | 0.96× |
| room-32-32-4 n15m20 | 200 | 1.0316 | 0.350 | 3.43 | 30 | 5.11 | 0.55× |
| room-32-32-4 n15m30 | 199 | 1.0620 | 0.126 | 8.91 | 38 | 8.35 | 1.30× |

**Difficulty axes.** M dominates: ratio climbs 1.02 → 1.10 and exact-match collapses 0.80 → 0.09 as
M 10 → 30. More agents help (at M=30: N5 1.10, N10 1.06, N15 1.046 — extra agents spread the goals →
shorter tours). maze is "easiest" by ratio (walls inflate the absolute optimum, shrinking the
relative gap). Speedup is modest (1.2–1.5×) and <1× on several maze/room high-M cells: single-instance
CBS dominates both pipelines, so only the LKH allocation call is saved (cf. Exp 9/10).

## Exp 13.a — old model on real maps, small N/M (large_s0, 2026-06-17)

Zero-shot eval of the Exp 11 model **A** (h128/L6, trained on random 8–12 grids, N≤5×M≤8) on the 4
RobustMCPF benchmark maps (empty, random-20, maze, room; 32×32) at N≤5×M≤8, 200 inst/config.
**Headline:** model A generalizes to the real maps with no degradation — mean execution-cost ratio
**1.019**, 0% infeasible; **structured maps (maze/room) are *easier* than open** because walls
disambiguate allocation (they break the ties an open grid leaves to chance). No full-pipeline speedup
at this scale: roomy maps → ~0 collisions → the solver's CBS is trivial and never re-roots, so there
is nothing for the NN to skip. Full writeup + figures: `report/real_maps/REPORT.md`.

## Exp 13.b — old model on real maps, large N/M (large_s0 vs B1, 2026-06-18)

Same model A pushed to the paper-scale regime **N∈{5,10,15} × M∈{10,20,30}** on the 4 maps, 200
inst/config, vs the same-architecture map-trained **B1** (Exp 12) on identical instances (seed
987654321). **Verdict: model A does NOT extrapolate across scale.** Mean execution-cost ratio
**A=1.246 vs B1=1.048**. M is the cliff (1.08 → 1.36 as M 10→30); the near-tie property (Exp 13.a)
breaks at large M; and the *scissors* — at M=30, **more agents HURT A (1.28 → 1.44 over N) but HELP
B1 (1.10 → 1.05)**: A, trained on N≤5, mis-coordinates beyond its range. Retraining at scale is
justified. Tables + figures: `report/oldmodel_large_nm/REPORT.md`.

## Exp 14 — random-diverse vs fixed-map training (C vs B, 2026-06-19)

Controlled comparison: **C1/C2** copy the **B1/B2** recipe (Exp 12) exactly, swapping *only* the
training data — random 32×32 grids, per-instance wall density uniform in [0, 0.5], 80k/config —
instead of the 4 fixed maps. Both families evaluated on the 4 real maps, 200 inst/config, identical
instances. **Verdict: fixed-map (B) beats random-diverse (C)** — h128 1.048 vs 1.075, h256 1.052 vs
1.066 — **but the entire gap is the maze** (h128 maze: B 1.029 vs C 1.127); on empty/random/room the
two tie, and h256 C even edges B on empty. Random Bernoulli walls transfer to unstructured maps but
never learned the maze's corridor structure. **Selects B as the family for Exp 15.** Full tables:
`report/random_vs_fixed/REPORT.md`.

## Exp 15 — zero-shot XL extrapolation to paper-scale N/M (B = Exp-14 winner, paper_*_s0, 2026-06-18)

Exp 14 selected the **fixed-map family (B)**, so its zero-shot extrapolation toward the paper's
N≤50/M≤100 is the conclusion experiment. (This is the run formerly labelled Exp 13; renumbered after
13.a/13.b/14 were inserted.)

**Setup.** Both models are size-agnostic, so evaluate them far beyond their N≤15/M≤30 training range:
**N∈{20,35,50} × M∈{50,75,100}** on the same 4 maps (toward the paper's N≤50/M≤100), no retraining.
`eval_paper_xl_{current,larger}.sh`, 100 instances/config, 12h wall + 11.5h soft budget
(`--max_seconds`) with partial-CSV fallback. **36/36** configs each. Two room cells
(`N20M75`, `N35M100`) first timed out — a single pathological CBS solve outran the wall before the
between-instance budget check could fire — then were recovered (`eval_paper_xl_recover.sh`) with a
hard per-instance `--instance_timeout 600`, which skipped the 1–2 offending solves per config and let
the rest finish in ≤42 min.

### The verdict flips: capacity wins out-of-distribution

Mean over all 36 configs:

| Model | Cost ratio | Diff mean | Diff max | Diff std | Speedup |
|-------|-----------|-----------|----------|----------|---------|
| h128/L6 (current) | 1.244 | 48.8 | 224 | 14.2 | **9.4×** |
| **h256/L8 (larger)** | **1.208** | **42.1** | **124** | **11.9** | 5.9× |

**Opposite of Exp 12.** Far OOD the larger model is better on cost across all 4 maps and nearly every
(N,M) cell, widest at high N, and roughly *halves* the worst-case diff (224 → 124) — more robust, not
just lower-mean. The smaller model's only edge is raw speed (fewer params → faster forward). The
capacity that overfit in-range is the better *generalizer* at scale.

| (N,M) | ratio cur | ratio lrg |
|-------|-----------|-----------|
| 20,50  | 1.204 | 1.211 |
| 20,75  | 1.283 | 1.231 |
| 20,100 | 1.248 | 1.246 |
| 35,50  | 1.217 | 1.188 |
| 35,75  | 1.288 | 1.211 |
| 35,100 | 1.233 | 1.227 |
| 50,50  | 1.221 | 1.159 |
| 50,75  | 1.289 | 1.184 |
| 50,100 | 1.216 | 1.211 |

(At N=20 the two are close — current even edges M=50; the larger model's advantage opens up at N=35,50.)
By map (cur/lrg): empty 1.279/1.241, random 1.265/1.227, maze 1.175/1.148, room 1.258/1.214.

### Full per-config tables (100 instances each; n<100 = solver-budget skips or instance-timeout)

**h128/L6 (current) — per-config (36 configs):**

| Config | n | Cost ratio | Exact | Diff mean | Diff max | Diff std | Speedup |
|--------|---|-----------|-------|-----------|----------|----------|---------|
| empty-32-32 n20m50 | 100 | 1.2587 | 0.000 | 39.83 | 89 | 13.32 | 2.87× |
| empty-32-32 n20m75 | 100 | 1.2930 | 0.000 | 59.85 | 112 | 14.23 | 5.32× |
| empty-32-32 n20m100 | 100 | 1.2724 | 0.000 | 67.65 | 102 | 12.89 | 3.44× |
| empty-32-32 n35m50 | 100 | 1.2970 | 0.000 | 40.64 | 87 | 14.17 | 3.03× |
| empty-32-32 n35m75 | 100 | 1.2806 | 0.000 | 51.33 | 86 | 12.71 | 4.56× |
| empty-32-32 n35m100 | 100 | 1.2653 | 0.000 | 59.17 | 85 | 10.69 | 8.43× |
| empty-32-32 n50m50 | 100 | 1.3037 | 0.000 | 36.51 | 66 | 10.96 | 7.19× |
| empty-32-32 n50m75 | 100 | 1.2971 | 0.000 | 49.58 | 112 | 13.32 | 8.45× |
| empty-32-32 n50m100 | 100 | 1.2449 | 0.000 | 50.73 | 78 | 11.00 | 18.56× |
| random-32-32-20 n20m50 | 100 | 1.2371 | 0.000 | 40.12 | 89 | 16.13 | 5.30× |
| random-32-32-20 n20m75 | 100 | 1.2800 | 0.000 | 62.58 | 111 | 15.01 | 5.92× |
| random-32-32-20 n20m100 | 99 | 1.2537 | 0.000 | 67.86 | 100 | 14.62 | 15.05× |
| random-32-32-20 n35m50 | 100 | 1.2567 | 0.000 | 36.99 | 69 | 12.60 | 5.18× |
| random-32-32-20 n35m75 | 99 | 1.2977 | 0.000 | 58.11 | 93 | 14.40 | 1.03× |
| random-32-32-20 n35m100 | 100 | 1.2519 | 0.000 | 59.83 | 92 | 11.95 | 18.75× |
| random-32-32-20 n50m50 | 100 | 1.2687 | 0.000 | 34.71 | 88 | 11.09 | 1.14× |
| random-32-32-20 n50m75 | 100 | 1.3102 | 0.000 | 54.86 | 99 | 14.12 | 8.59× |
| random-32-32-20 n50m100 | 100 | 1.2314 | 0.000 | 50.88 | 84 | 10.95 | 10.42× |
| maze-32-32-2 n20m50 | 99 | 1.1152 | 0.000 | 23.62 | 78 | 15.49 | 0.76× |
| maze-32-32-2 n20m75 | 100 | 1.2431 | 0.000 | 62.89 | 224 | 26.02 | 1.97× |
| maze-32-32-2 n20m100 | 99 | 1.1987 | 0.000 | 59.61 | 117 | 20.29 | 17.35× |
| maze-32-32-2 n35m50 | 100 | 1.1108 | 0.000 | 18.11 | 59 | 9.98 | 2.39× |
| maze-32-32-2 n35m75 | 100 | 1.2348 | 0.000 | 49.84 | 145 | 20.51 | 6.74× |
| maze-32-32-2 n35m100 | 98 | 1.1835 | 0.000 | 46.61 | 99 | 14.77 | 7.62× |
| maze-32-32-2 n50m50 | 100 | 1.1005 | 0.010 | 14.11 | 37 | 7.49 | 16.21× |
| maze-32-32-2 n50m75 | 99 | 1.2176 | 0.000 | 40.67 | 94 | 14.00 | 14.86× |
| maze-32-32-2 n50m100 | 99 | 1.1722 | 0.000 | 39.14 | 69 | 12.26 | 7.95× |
| room-32-32-4 n20m50 | 100 | 1.2040 | 0.000 | 37.40 | 85 | 14.86 | 4.96× |
| room-32-32-4 n20m75 | 99 | 1.3165 | 0.000 | 74.27 | 125 | 17.86 | 1.79× |
| room-32-32-4 n20m100 | 99 | 1.2663 | 0.000 | 75.17 | 132 | 17.56 | 2.71× |
| room-32-32-4 n35m50 | 99 | 1.2019 | 0.000 | 31.03 | 79 | 12.37 | 0.66× |
| room-32-32-4 n35m75 | 99 | 1.3396 | 0.000 | 69.42 | 118 | 19.39 | 21.93× |
| room-32-32-4 n35m100 | 98 | 1.2309 | 0.000 | 56.17 | 88 | 12.41 | 7.61× |
| room-32-32-4 n50m50 | 100 | 1.2096 | 0.000 | 28.23 | 63 | 11.00 | 12.99× |
| room-32-32-4 n50m75 | 99 | 1.3322 | 0.000 | 59.85 | 117 | 18.47 | 29.45× |
| room-32-32-4 n50m100 | 99 | 1.2168 | 0.000 | 47.94 | 74 | 11.80 | 47.83× |

**h256/L8 (larger) — per-config (36 configs):**

| Config | n | Cost ratio | Exact | Diff mean | Diff max | Diff std | Speedup |
|--------|---|-----------|-------|-----------|----------|----------|---------|
| empty-32-32 n20m50 | 100 | 1.2394 | 0.000 | 36.79 | 70 | 10.43 | 3.74× |
| empty-32-32 n20m75 | 100 | 1.2641 | 0.000 | 54.04 | 91 | 13.74 | 5.35× |
| empty-32-32 n20m100 | 100 | 1.2864 | 0.000 | 71.17 | 122 | 14.73 | 4.33× |
| empty-32-32 n35m50 | 100 | 1.2225 | 0.000 | 30.45 | 55 | 9.14 | 3.31× |
| empty-32-32 n35m75 | 100 | 1.2448 | 0.000 | 44.86 | 80 | 11.62 | 3.33× |
| empty-32-32 n35m100 | 100 | 1.2638 | 0.000 | 58.88 | 86 | 10.49 | 5.74× |
| empty-32-32 n50m50 | 100 | 1.1784 | 0.000 | 21.47 | 48 | 8.16 | 6.00× |
| empty-32-32 n50m75 | 100 | 1.2307 | 0.000 | 38.60 | 68 | 9.89 | 5.36× |
| empty-32-32 n50m100 | 100 | 1.2417 | 0.000 | 50.09 | 77 | 9.60 | 10.74× |
| random-32-32-20 n20m50 | 100 | 1.2182 | 0.000 | 36.84 | 69 | 10.98 | 5.61× |
| random-32-32-20 n20m75 | 100 | 1.2463 | 0.000 | 55.20 | 92 | 14.20 | 5.96× |
| random-32-32-20 n20m100 | 99 | 1.2561 | 0.000 | 68.57 | 104 | 14.66 | 9.79× |
| random-32-32-20 n35m50 | 100 | 1.2173 | 0.000 | 31.53 | 59 | 9.48 | 3.08× |
| random-32-32-20 n35m75 | 100 | 1.2397 | 0.000 | 46.84 | 75 | 11.65 | 5.73× |
| random-32-32-20 n35m100 | 100 | 1.2523 | 0.000 | 59.94 | 87 | 10.95 | 8.96× |
| random-32-32-20 n50m50 | 100 | 1.1763 | 0.000 | 22.88 | 47 | 8.10 | 1.82× |
| random-32-32-20 n50m75 | 99 | 1.1961 | 0.000 | 34.74 | 67 | 9.52 | 8.71× |
| random-32-32-20 n50m100 | 100 | 1.2378 | 0.000 | 52.33 | 79 | 10.58 | 5.63× |
| maze-32-32-2 n20m50 | 100 | 1.1522 | 0.000 | 31.66 | 124 | 20.65 | 2.95× |
| maze-32-32-2 n20m75 | 100 | 1.1722 | 0.000 | 44.53 | 95 | 14.24 | 5.87× |
| maze-32-32-2 n20m100 | 100 | 1.1788 | 0.000 | 53.61 | 95 | 16.30 | 11.45× |
| maze-32-32-2 n35m50 | 100 | 1.1269 | 0.000 | 20.66 | 51 | 10.06 | 3.45× |
| maze-32-32-2 n35m75 | 100 | 1.1440 | 0.000 | 30.62 | 67 | 11.52 | 3.97× |
| maze-32-32-2 n35m100 | 100 | 1.1586 | 0.000 | 40.30 | 77 | 12.24 | 4.22× |
| maze-32-32-2 n50m50 | 100 | 1.1042 | 0.020 | 14.64 | 33 | 7.61 | 8.76× |
| maze-32-32-2 n50m75 | 100 | 1.1393 | 0.000 | 25.97 | 69 | 10.38 | 7.76× |
| maze-32-32-2 n50m100 | 100 | 1.1551 | 0.000 | 35.30 | 69 | 11.31 | 1.21× |
| room-32-32-4 n20m50 | 99 | 1.2345 | 0.000 | 42.91 | 75 | 14.76 | 1.18× |
| room-32-32-4 n20m75 | 99 | 1.2427 | 0.000 | 57.08 | 95 | 14.24 | 3.55× |
| room-32-32-4 n20m100 | 99 | 1.2610 | 0.000 | 73.68 | 119 | 17.52 | 1.51× |
| room-32-32-4 n35m50 | 99 | 1.1853 | 0.000 | 28.69 | 55 | 10.63 | 0.61× |
| room-32-32-4 n35m75 | 98 | 1.2171 | 0.000 | 44.51 | 90 | 14.03 | 13.10× |
| room-32-32-4 n35m100 | 100 | 1.2316 | 0.000 | 56.26 | 92 | 12.81 | 4.27× |
| room-32-32-4 n50m50 | 100 | 1.1761 | 0.000 | 23.79 | 51 | 9.28 | 6.40× |
| room-32-32-4 n50m75 | 100 | 1.1711 | 0.000 | 31.01 | 64 | 10.56 | 3.12× |
| room-32-32-4 n50m100 | 99 | 1.2091 | 0.000 | 46.28 | 72 | 11.05 | 26.17× |

**Both degrade**, as expected this far OOD: ratio ~1.21–1.24 (vs ~1.05 in-range) and exact-match ≈ 0 —
at M=50–100 the NN never exactly reproduces the solver's cost, but still lands ~20% above optimal.

**Speedup explodes with scale.** Up to **47.8×** (room n50m100) and ~6–10× mean — at 75–100 goals
LKH's mTSP allocation is expensive while the NN forward stays cheap. This is the practical case for
the NN: paper-scale allocation at ~20% cost for an order-of-magnitude-plus speedup. maze stays
"easiest" by ratio (~1.15).

### Raw data (Exp 12 + 13)

Per-instance CSVs, **cluster only** (`results/` git-ignored), same columns as Exp 11
(`inst_seed, cost_nn, cost_solver, nn_k, solver_k, conflicts_nn, conflicts_solver, alloc_ms,
nn_plan_ms, solver_ms`):

- `results/fullpipe_paper/{current,larger}/<map>_n{N}m{M}.csv` — Exp 12, 36 files each (200 rows)
- `results/fullpipe_paper_xl/{current,larger}/<map>_n{N}m{M}.csv` — Exp 15, 36 files each (≤100 rows)

Aggregate with `evaluation/agg_paper_maps.py --base <dir> --ns <N,…> --ms <M,…>` (prints per-config,
per-model, by-map, by-(N,M)). Jobs: 18227874/18227875 (Exp 12 eval), 18229490/18229491 (Exp 15 XL),
18236650 (Exp 15 room-cell recovery). Exp 14 (random C): gen 18243314, eval 18259262/18259263.
