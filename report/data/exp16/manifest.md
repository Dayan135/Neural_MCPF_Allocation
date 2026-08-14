# Exp 16 — manifest (Tier A / dataset-B leg only)

Delivered against `docs/exp16_data_request.md`. **Only Tier A is included here.** Tier B is
still running on the cluster; this directory will be updated with Tier B's data (and Stage 1
becomes possible) once that finishes.

## Naming: Tier A = dataset B, Tier B = dataset A

The request doc calls the two training regimes "(A) regular: many instances at regular size" and
"(B) large: fewer instances at large size, scaled up to where the exact solver takes ~1 min."
This branch's experiments were run under the names "Tier A" (large/sparse) and "Tier B"
(moderate/dense), calibrated with `scripts/calibrate_solve_time.py`:

| This repo's name | Request-doc name | Own grid (N, M) | Samples/config | Solve time (calibration) |
|---|---|---|---|---|
| **Tier A** | **dataset B** (large) | (60,100) (120,225) (180,350) | train 3000 / val 500 / test 500 | ~30-40s median on the hardest cell |
| Tier B | dataset A (regular) | (30,50) (55,100) (80,150) | train 20000 / val 2000 / test 2000 | ~5-10s on the hardest cell |

Everything under `stage2_expert_vs_general/` in this directory is **Tier A only**, i.e. the
dataset-B leg of the 5×2 model matrix.

## Model roles (5, all h128/L6 universal transformer)

`joint` (generalist, trained on all 4 maps) and one specialist per map: `empty`, `random`,
`maze`, `room`. See `model_inventory.csv`. `total_samples` there is the nominal data available
per role (36 configs × 4000 for joint, 9 × 4000 for a specialist); the joint run additionally used
`--sample_fraction 0.25`, capping what it actually trains on per epoch to roughly one specialist's
budget (round-robin across its 36 configs) so joint-vs-per-map isn't confounded by joint simply
seeing 4x more data.

## What's included

- `stage2_expert_vs_general/tierA_{joint,empty,random,maze,room}/{map}_n{N}m{M}.csv` — per-instance
  pipeline-eval CSVs, own-size grid (N∈{60,120,180}×M∈{100,225,350}), **all 4 maps per model**
  (specialist off-diagonal cells come from `eval_tierA_mapextrap_*.sh`, which evaluates each
  specialist on all 4 maps, not just its own — the diagonal cell is included there too and is
  redundant with, but consistent with, `eval_tierA_indist_*.sh`). Columns: `inst_seed, cost_nn,
  cost_solver, nn_k, solver_k, conflicts_nn, conflicts_solver, alloc_ms, nn_plan_ms, solver_ms`
  (written by `evaluation/full_pipeline_eval.py`), 100 instances/cell, seed 987654321.
  Coverage: 174/180 of the full 5-role × 4-map × 9-config grid (34-36 per role) — the handful of
  missing cells are configs that hadn't finished on the cluster at pull time; tables report `n`
  per cell so this is visible rather than silently interpolated.
- `stage2_agg/` — aggregate CSVs per (model, map, N, M): `cost_ratio, exact_match, diff_mean,
  diff_max, diff_std, alloc_ms, nn_plan_ms, solver_ms, n` (built by `report/gen_exp16_tables.py`
  from the per-instance CSVs above). No dedicated `infeasible`/`fallback` columns — the
  per-instance CSV doesn't carry them directly; `nn_k` (candidates tried before a feasible plan
  was found) is the per-instance proxy, `nn_k > 1` meaning a fallback was needed. Aggregate
  infeasible/fallback rates are printed by `full_pipeline_eval.py` at run time but not persisted
  to file, so they aren't recoverable from the CSVs alone.
- `model_inventory.csv` — 5 rows (Tier A only). `best_val_loss`, `best_epoch`, `train_wall_hours`
  are blank: no training logs were pulled down for these runs, only the eval CSVs above. Only
  seed 0 was evaluated per role (checkpoints `tierA_{role}_s0/best.pt`); seeds 1-2 were trained
  (3-seed Slurm array) but not evaluated for this report.
- `solver_scale_wall_raw.csv` (+ `solver_scale_wall.csv` once post-processed) — fresh local
  calibration (this machine has LKH built + `mcpf_env`), covering Tier A's own grid plus its
  extrapolation grid (up to N=220, M=430), all 4 maps, 3 instances/cell. Tier-independent, so
  valid regardless of Tier B's status. Raw columns: `map, N, M, mean_s, median_s, p90_s, max_s,
  n_instances, n_fail, n_timeout`.

## What's explicitly missing (not fabricated)

- **Tier B / dataset A entirely** — still running on the cluster. Nothing here should be read as
  a dataset-A number.
- **Stage 1 (dataset-A-vs-B head-to-head)** — needs both tiers; not attempted.
- **Offline accuracy** (`evaluation/evaluate.py` per model) — not run for these checkpoints.
- **Training logs** (per-epoch loss/accuracy) — live on the cluster only, not pulled down.
- **`bothextrap`** (far-size × other-map cells) — eval scripts exist (`eval_tierA_bothextrap_*.sh`)
  but were never submitted; no data.
- **Job IDs / node types / torch version** — not recorded anywhere pulled to this machine;
  training used `gpu` partition (`rtx_6000`), eval used `cpu` partition (per the Slurm headers in
  `scripts/exp_tierA_*.sh` / `scripts/eval_tierA_*.sh`), but exact job IDs aren't available from
  this session.

## Continuity with Exp 12's B1

Exp 12's **B1** (`RESULTS.md`) is h128/L6, jointly trained on the 4 real maps at
N∈{5,10,15}×M∈{10,20,30}. The new `tierA_joint` model is a **retrain, not a reuse**: same
architecture, but a materially larger N/M range (up to 180×350 vs B1's 15×30). They are not the
same model.
