# Exp 14 — Random-diverse vs fixed-map training, evaluated on the real benchmark maps

**Verdict: fixed-map training (B) beats random-diverse training (C) at both model sizes.**
The advantage is concentrated almost entirely on the *structured* `maze` map; on the
*unstructured* maps (`empty`, `random-20`) the two are statistically tied, and the larger
random-trained model even edges ahead on `empty`.

## Setup

A controlled comparison: models **C1/C2** copy the **B1/B2** recipe (Exp 12) *exactly* —
`GoalAllocTransformerUniversal`, mixed-size training, `--use_goal_dists`, `--grad_clip 1.0`,
150 epochs, batch 256, λ=0.1, 3 seeds; C1/B1 = h128/L6 (lr 5e-4), C2/B2 = h256/L8 (lr 3e-4).
**Only the training data differs:**

| Family | Training data |
|--------|---------------|
| **B** (fixed) | the 4 real 32×32 benchmark maps × N∈{5,10,15} × M∈{10,20,30}, 20k/config (Exp 12) |
| **C** (random) | random 32×32 grids, **per-instance wall density uniform in [0, 0.5]**, same N×M, **80k/config** |

C's 80k/config matches B's per-(N,M)-shape volume (B saw 4 maps × 20k = 80k per shape; 720k
total each). Both families are then evaluated with `full_pipeline_eval.py --map_file` on the
**4 real maps** (NN argmax allocation → goal ordering → CBS, vs LKH+CBS solver; cost = `Solution[5]`),
200 instances/config, `--seed 987654321` — so B and C see **identical instances** per config
(paired comparison). Aggregated with `evaluation/agg_compare.py`.

Scripts: `scripts/exp14/{gen_random_diverse_data,exp_random_current,exp_random_larger,eval_random_current,eval_random_larger}.sh`.
C checkpoints: `random_{current,larger}_s0` (training timed out at ~22h before epoch 150, but
`best.pt` was captured at the val-loss minimum — val had already plateaued/overfit, same as B).

## Headline — mean over 36 configs

| Size | B (fixed) ratio | C (random) ratio | Δ | B exact | C exact | Winner |
|------|----------------:|-----------------:|--:|--------:|--------:|:------:|
| **h128** | **1.048** | 1.075 | +0.027 | **43.3%** | 33.0% | **B** |
| **h256** | **1.052** | 1.066 | +0.014 | **40.2%** | 34.5% | **B** |

Lower cost ratio is better. B wins on both mean execution cost and exact-cost-match at both sizes.

## By map — the gap is `maze`

| Map | h128 B / C (Δ) | h256 B / C (Δ) |
|-----|---------------:|---------------:|
| empty-32-32 | 1.053 / 1.056 (+0.004) | 1.060 / 1.058 (**−0.002**) |
| random-32-32-20 | 1.053 / 1.056 (+0.003) | 1.055 / 1.058 (+0.004) |
| **maze-32-32-2** | **1.029 / 1.127 (+0.098)** | **1.035 / 1.083 (+0.048)** |
| room-32-32-4 | 1.057 / 1.060 (+0.002) | 1.059 / 1.064 (+0.005) |

On `empty` and `random`, B and C are within ±0.004 — a tie — and on `empty` the larger random
model (C2) is actually *better* than B2. The entire mean-level win for B comes from **`maze`**,
where C's cost ratio blows out to 1.127 (h128) / 1.083 (h256) and exact-match collapses
(h128 maze: B 64% vs C 28%; the worst cell, maze n5m30, is B 1.075 vs **C 1.362**).

## By M — the deficit grows with goal count

| M | h128 B / C (Δ) | h256 B / C (Δ) |
|---|---------------:|---------------:|
| 10 | 1.026 / 1.034 (+0.009) | 1.027 / 1.029 (+0.002) |
| 20 | 1.049 / 1.064 (+0.015) | 1.051 / 1.061 (+0.010) |
| 30 | 1.070 / 1.126 (+0.057) | 1.079 / 1.108 (+0.029) |

C's disadvantage is small at M=10 and widens sharply at M=30 — because high-M instances require
long multi-goal tours, and on `maze` those tours must thread the corridors C never learned.

## Interpretation

Random Bernoulli-wall grids resemble the `empty`/`random-20` maps (scattered or no obstacles),
so C transfers to them perfectly well. But they contain **no corridor/room structure**, so C
never learned to route through `maze`'s narrow passages and bottlenecks — exactly where allocation
quality matters most for tour cost. B saw the actual `maze` map in training and handles it. In
short: **training on the target maps wins specifically because of structured geometry; for
unstructured maps, cheap random-grid data is just as good.** This also explains why the gap is
larger for h128 (+0.027) than h256 (+0.014): the bigger model generalizes structure a bit better
from random data, narrowing — but not closing — the maze deficit.

## Consequence for Exp 15

This selects **B (fixed-map) as the family** to carry forward. Per the plan's "B wins" branch,
**Exp 15 (zero-shot XL extrapolation, N∈{20,35,50}×M∈{50,75,100}) reuses the existing B XL run**
(`results/fullpipe_paper_xl/…`, written up in `RESULTS.md`) — no new C XL scripts or cluster jobs
needed.

## Full per-config tables

Cost ratio = NN execution cost / solver execution cost (lower better, 1.0 = matches solver).
Exact = fraction of instances where NN and solver execution costs are equal. Δ = C ratio − B ratio
(positive = B better). 200 instances/config; speedup columns omitted here (single-instance CBS
variance — see the raw CSVs).

#### h128 (current)

| Config | B ratio | C ratio | Δ | B exact | C exact |
|--------|--------:|--------:|--:|--------:|--------:|
| empty-32-32 n5m10 | 1.052 | 1.053 | +0.001 | 51.0% | 49.5% |
| empty-32-32 n5m20 | 1.077 | 1.086 | +0.009 | 22.0% | 22.0% |
| empty-32-32 n5m30 | 1.107 | 1.114 | +0.007 | 6.0% | 3.5% |
| empty-32-32 n10m10 | 1.025 | 1.027 | +0.002 | 70.5% | 69.0% |
| empty-32-32 n10m20 | 1.050 | 1.049 | −0.001 | 36.0% | 34.5% |
| empty-32-32 n10m30 | 1.069 | 1.074 | +0.005 | 12.0% | 14.0% |
| empty-32-32 n15m10 | 1.014 | 1.019 | +0.006 | 84.0% | 79.0% |
| empty-32-32 n15m20 | 1.032 | 1.035 | +0.003 | 49.0% | 44.5% |
| empty-32-32 n15m30 | 1.047 | 1.050 | +0.003 | 23.0% | 21.0% |
| random-32-32-20 n5m10 | 1.045 | 1.046 | +0.001 | 54.5% | 54.5% |
| random-32-32-20 n5m20 | 1.084 | 1.087 | +0.002 | 19.5% | 19.5% |
| random-32-32-20 n5m30 | 1.107 | 1.111 | +0.004 | 4.5% | 4.0% |
| random-32-32-20 n10m10 | 1.025 | 1.027 | +0.003 | 66.0% | 65.5% |
| random-32-32-20 n10m20 | 1.047 | 1.053 | +0.006 | 29.0% | 28.0% |
| random-32-32-20 n10m30 | 1.061 | 1.075 | +0.014 | 12.5% | 10.5% |
| random-32-32-20 n15m10 | 1.020 | 1.022 | +0.002 | 70.5% | 70.0% |
| random-32-32-20 n15m20 | 1.033 | 1.032 | −0.001 | 43.0% | 47.0% |
| random-32-32-20 n15m30 | 1.054 | 1.050 | −0.004 | 19.0% | 17.1% |
| maze-32-32-2 n5m10 | 1.010 | 1.061 | +0.051 | 86.0% | 60.0% |
| maze-32-32-2 n5m20 | 1.046 | 1.128 | +0.082 | 56.5% | 18.0% |
| maze-32-32-2 n5m30 | 1.075 | 1.362 | +0.288 | 29.5% | 1.0% |
| maze-32-32-2 n10m10 | 1.014 | 1.033 | +0.020 | 86.0% | 59.5% |
| maze-32-32-2 n10m20 | 1.022 | 1.064 | +0.042 | 64.0% | 25.5% |
| maze-32-32-2 n10m30 | 1.035 | 1.235 | +0.201 | 46.0% | 1.5% |
| maze-32-32-2 n15m10 | 1.007 | 1.024 | +0.017 | 89.0% | 63.0% |
| maze-32-32-2 n15m20 | 1.018 | 1.062 | +0.044 | 68.0% | 24.0% |
| maze-32-32-2 n15m30 | 1.034 | 1.174 | +0.140 | 51.3% | 2.5% |
| room-32-32-4 n5m10 | 1.052 | 1.050 | −0.001 | 51.5% | 46.0% |
| room-32-32-4 n5m20 | 1.087 | 1.088 | +0.001 | 14.1% | 15.5% |
| room-32-32-4 n5m30 | 1.119 | 1.130 | +0.011 | 4.5% | 3.0% |
| room-32-32-4 n10m10 | 1.024 | 1.029 | +0.004 | 60.0% | 59.5% |
| room-32-32-4 n10m20 | 1.046 | 1.048 | +0.001 | 34.8% | 32.5% |
| room-32-32-4 n10m30 | 1.079 | 1.080 | +0.001 | 8.0% | 6.1% |
| room-32-32-4 n15m10 | 1.021 | 1.018 | −0.003 | 77.0% | 73.5% |
| room-32-32-4 n15m20 | 1.039 | 1.037 | −0.002 | 42.0% | 31.0% |
| room-32-32-4 n15m30 | 1.049 | 1.058 | +0.009 | 18.0% | 11.6% |

#### h256 (larger)

| Config | B ratio | C ratio | Δ | B exact | C exact |
|--------|--------:|--------:|--:|--------:|--------:|
| empty-32-32 n5m10 | 1.052 | 1.039 | −0.013 | 49.5% | 52.0% |
| empty-32-32 n5m20 | 1.082 | 1.088 | +0.006 | 21.5% | 17.5% |
| empty-32-32 n5m30 | 1.134 | 1.135 | +0.001 | 2.5% | 4.5% |
| empty-32-32 n10m10 | 1.025 | 1.029 | +0.004 | 69.5% | 70.5% |
| empty-32-32 n10m20 | 1.050 | 1.055 | +0.004 | 32.0% | 35.5% |
| empty-32-32 n10m30 | 1.078 | 1.082 | +0.004 | 12.0% | 14.0% |
| empty-32-32 n15m10 | 1.017 | 1.013 | −0.004 | 79.0% | 83.0% |
| empty-32-32 n15m20 | 1.042 | 1.033 | −0.009 | 37.5% | 48.5% |
| empty-32-32 n15m30 | 1.059 | 1.045 | −0.014 | 15.5% | 25.0% |
| random-32-32-20 n5m10 | 1.044 | 1.055 | +0.011 | 52.0% | 50.0% |
| random-32-32-20 n5m20 | 1.081 | 1.085 | +0.004 | 14.5% | 19.0% |
| random-32-32-20 n5m30 | 1.109 | 1.128 | +0.019 | 4.0% | 4.0% |
| random-32-32-20 n10m10 | 1.031 | 1.029 | −0.002 | 60.5% | 70.5% |
| random-32-32-20 n10m20 | 1.048 | 1.050 | +0.001 | 30.5% | 27.5% |
| random-32-32-20 n10m30 | 1.068 | 1.077 | +0.010 | 9.0% | 9.0% |
| random-32-32-20 n15m10 | 1.018 | 1.017 | −0.002 | 71.5% | 73.5% |
| random-32-32-20 n15m20 | 1.038 | 1.030 | −0.008 | 41.5% | 41.5% |
| random-32-32-20 n15m30 | 1.058 | 1.056 | −0.002 | 14.5% | 14.0% |
| maze-32-32-2 n5m10 | 1.011 | 1.034 | +0.024 | 84.0% | 64.5% |
| maze-32-32-2 n5m20 | 1.054 | 1.097 | +0.043 | 52.0% | 23.0% |
| maze-32-32-2 n5m30 | 1.086 | 1.225 | +0.138 | 27.5% | 2.5% |
| maze-32-32-2 n10m10 | 1.015 | 1.022 | +0.006 | 84.0% | 70.5% |
| maze-32-32-2 n10m20 | 1.021 | 1.060 | +0.038 | 66.5% | 31.7% |
| maze-32-32-2 n10m30 | 1.051 | 1.140 | +0.089 | 39.9% | 4.5% |
| maze-32-32-2 n15m10 | 1.011 | 1.012 | +0.001 | 83.5% | 73.5% |
| maze-32-32-2 n15m20 | 1.022 | 1.043 | +0.021 | 62.0% | 32.0% |
| maze-32-32-2 n15m30 | 1.042 | 1.117 | +0.075 | 49.7% | 8.5% |
| room-32-32-4 n5m10 | 1.046 | 1.046 | +0.001 | 48.5% | 46.5% |
| room-32-32-4 n5m20 | 1.086 | 1.096 | +0.010 | 16.5% | 13.5% |
| room-32-32-4 n5m30 | 1.122 | 1.139 | +0.017 | 2.5% | 2.0% |
| room-32-32-4 n10m10 | 1.030 | 1.034 | +0.004 | 58.0% | 58.0% |
| room-32-32-4 n10m20 | 1.052 | 1.052 | −0.000 | 28.5% | 29.1% |
| room-32-32-4 n10m30 | 1.078 | 1.085 | +0.007 | 5.6% | 5.0% |
| room-32-32-4 n15m10 | 1.021 | 1.016 | −0.005 | 74.5% | 76.0% |
| room-32-32-4 n15m20 | 1.032 | 1.043 | +0.011 | 35.0% | 31.5% |
| room-32-32-4 n15m30 | 1.062 | 1.066 | +0.004 | 12.6% | 10.5% |

## Raw data

Per-instance CSVs (cluster only, `results/` git-ignored), same schema as Exp 11/12:
- B: `results/fullpipe_paper/{current,larger}/<map>_n{N}m{M}.csv` (Exp 12)
- C: `results/fullpipe_random/{current,larger}/<map>_n{N}m{M}.csv` (this experiment)

Reproduce the tables: `python evaluation/agg_compare.py --fixed_base results/fullpipe_paper
--random_base results/fullpipe_random`. Jobs: gen 18243314, eval C 18259262 (h128) / 18259263 (h256).
