# Exp 16 — manifest (Tier A + Tier B, both legs)

Delivered against `docs/exp16_data_request.md`. **Both tiers are now included.** Tier B
(dataset A) finished retraining on 2026-08-15 after a stale-checkpoint issue was caught and
fixed (see "Tier B retrain note" below) and has been evaluated; Stage 1 (dataset-A-vs-B
head-to-head) is completed by an additional eval sweep, `eval_tierA_on_tierB_grid.sh`, run
specifically to fill the one missing quadrant (see below).

## Naming: Tier A = dataset B, Tier B = dataset A

The request doc calls the two training regimes "(A) regular: many instances at regular size" and
"(B) large: fewer instances at large size, scaled up to where the exact solver takes ~1 min."
This branch's experiments were run under the names "Tier A" (large/sparse) and "Tier B"
(moderate/dense), calibrated with `scripts/calibrate_solve_time.py`:

| This repo's name | Request-doc name | Own grid (N, M) | Samples/config | Solve time (calibration) |
|---|---|---|---|---|
| **Tier A** | **dataset B** (large) | (60,100) (120,225) (180,350) | train 3000 / val 500 / test 500 | ~30-40s median on the hardest cell |
| **Tier B** | **dataset A** (regular) | (30,50) (55,100) (80,150) | train 20000 / val 2000 / test 2000 | ~5-10s on the hardest cell |

`stage2_expert_vs_general/` holds both legs, `tierA_*` and `tierB_*` prefixed. Stage 1's crossover
data lives in `stage1_dataset_selection/` (see below).

## Tier B retrain note

The four Tier B per-map checkpoints were originally trained 2026-08-07, five days before a fix to
`model/network.py`'s goal-context pooling (`.sum(dim=2)` → `.mean(dim=2)`, landed 2026-08-11).
At Tier B's M≤150 the bug does not overflow to NaN, so the original training looked healthy
(smooth convergence, plausible validation accuracy) but the un-normalized sum injected an
M-scaled magnitude bias into every goal embedding, degrading allocation quality without crashing
training. Cross-checking identical held-out instances against the (already post-fix) Tier B
joint checkpoint showed the original per-map checkpoints producing dramatically worse allocations
(e.g. room n=80,m=150: solver cost 258, joint-model cost 266, stale-room-model cost 598). The
four per-map models were retrained from scratch with the unchanged, already-correct training
scripts (`scripts/exp_tierB_{empty,random,maze,room}.sh`); the joint model was unaffected (already
trained post-fix). All Tier B numbers in this delivery are from the retrained checkpoints.

## Model roles (10, all h128/L6 universal transformer)

`joint` (generalist, trained on all 4 maps) and one specialist per map: `empty`, `random`,
`maze`, `room` — × 2 tiers = 10 models. See `model_inventory.csv`. `total_samples` there is the
nominal data available per role; both tiers' joint runs additionally used
`--sample_fraction 0.25`, capping what they actually train on per epoch to roughly one
specialist's budget (round-robin across all 36 configs) so joint-vs-per-map isn't confounded by
joint simply seeing 4x more data.

## What's included

- `stage2_expert_vs_general/tier{A,B}_{joint,empty,random,maze,room}/{map}_n{N}m{M}.csv` —
  per-instance pipeline-eval CSVs, each tier's own-size grid, **all 4 maps per model**
  (specialist off-diagonal cells come from the `mapextrap` eval sweep; the diagonal/native-map
  cell is included there too and is redundant with, but consistent with, the `indist` sweep).
  Columns: `inst_seed, cost_nn, cost_solver, nn_k, solver_k, conflicts_nn, conflicts_solver,
  alloc_ms, nn_plan_ms, solver_ms` (written by `evaluation/full_pipeline_eval.py`), 100
  instances/cell, seed 987654321 (same seed for both tiers — paired comparison where grids
  overlap). Tier A coverage: 174/180. Tier B coverage: 180/180 (full).
- `stage1_dataset_selection/{role}_{A,B}/{map}_n{N}m{M}.csv` — the crossover comparison: each
  role's dataset-A (Tier B) and dataset-B (Tier A) checkpoint, evaluated on a **common grid
  spanning both regimes** (Tier B's own small grid + Tier A's own large grid), same seed
  (987654321), same protocol as Stage 2. Built from four pre-existing sweeps
  (`tierB_indist`/`tierB_mapextrap` = dataset-A model on the small grid; `tierB_extrap` =
  dataset-A model on the large grid, since Tier B's extrapolation target is defined as exactly
  Tier A's own grid; `tierA_indist`/`tierA_mapextrap` = dataset-B model on the large grid) plus
  one new sweep run specifically to complete this quadrant: `eval_tierA_on_tierB_grid.sh`
  (dataset-B/Tier A checkpoints evaluated on dataset-A/Tier B's own small grid — this comparison
  did not exist anywhere in the prior data and required a fresh 180-cell eval, since evaluating
  each model only on its own training range would not answer which dataset generalizes better).
- `stage2_agg/` and the new `stage1_agg/` — aggregate CSVs per (model, map, N, M): `cost_ratio,
  exact_match, diff_mean, diff_max, diff_std, alloc_ms, nn_plan_ms, solver_ms, n`. No dedicated
  `infeasible`/`fallback` columns — the per-instance CSV doesn't carry them directly; `nn_k`
  (candidates tried before a feasible plan was found) is the per-instance proxy, `nn_k > 1`
  meaning a fallback was needed. Aggregate infeasible/fallback rates are printed by
  `full_pipeline_eval.py` at run time but not persisted to file, so they aren't recoverable from
  the CSVs alone.
- `model_inventory.csv` — 10 rows. Tier A's `best_val_loss`/`best_epoch`/`train_wall_hours` are
  blank (training logs weren't pulled for those runs). Tier B's four per-map roles have
  `best_val_loss` and `train_wall_hours` from the actual retrain (`best_epoch` not captured);
  `tierB_joint`'s training history spans several resubmissions during earlier OOM debugging on
  this branch and isn't cleanly attributable to one run, so those fields are left blank rather
  than guessed — the checkpoint itself is validated (used successfully in every Tier B eval, and
  produces execution-cost numbers in the same sane range as the specialists, ~1.02-1.07). Only
  seed 0 was evaluated per role for both tiers; Tier A had 3 seeds trained (0-2), Tier B had 2
  (0-1, cut from a planned 3 partway through to let all four per-map configs run in parallel
  under the cluster's per-user GPU quota) — seeds beyond 0 were trained but not evaluated.
- `solver_scale_wall_raw.csv` (+ `solver_scale_wall.csv`) — fresh local calibration, covering
  Tier A's own grid plus its extrapolation grid (up to N=220, M=430), all 4 maps, 3
  instances/cell. Tier-independent.

## What's explicitly missing (not fabricated)

- **Offline accuracy** (`evaluation/evaluate.py` per model) — not run for any of the 10 checkpoints.
- **Training logs in the requested per-epoch CSV format** — the raw Slurm stdout logs exist on the
  cluster (contain `epoch, train_loss, val_loss, ...` per line) but haven't been parsed into
  `train_logs/{model_id}.txt` for all 10 models.
- **`bothextrap`** (far-size × other-map cells, both tiers) — eval scripts exist for Tier A
  (`eval_tierA_bothextrap_*.sh`) and were partially run for both tiers on the cluster, but not
  pulled into this delivery; not needed for RQ1-RQ3 as scoped.
- **Job IDs / node types / torch version** — training used the `gpu` partition (`rtx_6000`), eval
  used the `cpu` partition (per the Slurm headers in `scripts/exp_tier{A,B}_*.sh` /
  `scripts/eval_tier{A,B}_*.sh` / `scripts/eval_tierA_on_tierB_grid.sh`), but exact job IDs and
  torch version aren't recorded anywhere pulled to this machine.

## Continuity with Exp 12's B1

Exp 12's **B1** (`RESULTS.md`) is h128/L6, jointly trained on the 4 real maps at
N∈{5,10,15}×M∈{10,20,30}. Both `tierA_joint` and `tierB_joint` are **retrains, not reuses**: same
architecture, but materially larger N/M ranges (up to 180×350 and 80×150 respectively, vs B1's
15×30). None of the three are the same model.
