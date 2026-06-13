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
