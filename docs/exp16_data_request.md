# Exp 16 — data request for the final report

**Working name:** Exp 16 (RESULTS.md currently ends at Exp 15).
**Purpose:** everything the final report needs back from the cluster once the 10 models are
trained. Every item below feeds a specific table or figure.

> **Important:** the report is written on a laptop with **no access to the cluster**. Anything
> not sent here cannot be recovered without a round trip. Please treat this as the complete
> extraction list, not a starting point.

---

## 1. What Exp 16 is

- **5 model roles** — 1 **generalist** trained on all 4 benchmark maps, plus 4 **experts**, each
  trained on one map only (`empty-32-32`, `random-32-32-20`, `maze-32-32-2`, `room-32-32-4`).
- **× 2 training datasets** — **(A) regular**: many instances at regular size; **(B) large**:
  fewer instances at large size, scaled up to where the exact solver takes ~1 min.
- **= 10 models**, all `h128/L6`.

Three stages in the report:

| Stage | Question | Output |
|---|---|---|
| **0** | How far does N, M scale before the exact solver becomes impractical (~60 s)? | evaluation envelope |
| **1** | Per role, which dataset is better — A or B? **5 head-to-head comparisons** | 5 winners |
| **2** | Do the 4 experts beat the generalist? Accuracy **and** timing | routing recommendation |

---

## 2. What to send — data, not graphs

**Send per-instance CSVs.** They are the primary artifact: every table and figure in the report
is computed from them locally. Aggregates cannot be un-aggregated — if only means are sent and
we later need error bars, percentiles, a paired test, or a scatter plot, the data has to be
regenerated on the cluster.

| Artifact | Needed? |
|---|---|
| **Per-instance CSVs** | **Yes — the primary deliverable** |
| Aggregate tables (per model/map/config) | Yes — as a cross-check against our own aggregation |
| Training logs (per epoch) | Yes — small text files, needed for a convergence figure |
| **Generated graphs** | **No** — optional throwaway PNGs only, as a sanity check |

Figures are produced locally so they match the report's visual language (navy `#1F497D` / accent
blue, LaTeX palette, the conventions in `evaluation/plot_*.py`) and can be iterated without a
round trip. matplotlib + numpy are available on the writing machine.

### Where to put the files

`results/` is **git-ignored** — files copied there will be silently untracked. Please deliver
into a tracked path:

```
report/data/exp16/
├── manifest.md                  # what was run, job IDs, cluster node types, dates
├── solver_scale_wall.csv        # stage 0
├── model_inventory.csv          # the 10 models
├── train_logs/{model_id}.txt    # per-epoch training logs, 10 files
├── offline/{model_id}.csv       # offline accuracy metrics
├── stage1_dataset_selection/{role}_{dataset}/{map}_n{N}m{M}.csv
└── stage2_expert_vs_general/{model_id}/{map}_n{N}m{M}.csv
```

Total volume is a few MB — trivially committable.

---

## 3. Stage 0 — the solver scale wall  *(Figure F7)*

Sweep N and M per map until mean solver wall-time reaches ~60 s. This defines the report's
evaluation envelope.

**Deliver:** `solver_scale_wall.csv`

| Column | Meaning |
|---|---|
| `map` | map name |
| `N`, `M` | agents, goals |
| `n` | instances attempted |
| `solver_ms_mean`, `solver_ms_median`, `solver_ms_p90` | LKH + CBS wall time |
| `timeout_frac` | fraction hitting the wall / node budget |
| `budget_skipped` | instances rejected by `cbs_node_budget` |

**Please also state explicitly** — the report has to define three separate ceilings:

1. **Labeling ceiling** — the largest (N, M) where the solver still returns ground-truth labels
   in acceptable time. This bounds what dataset B can contain.
2. **Evaluation ceiling** — the largest (N, M) where a solver baseline still exists, so a cost
   ratio can be computed at all.
3. **Beyond it** — what gets reported where the solver no longer returns. Without a denominator
   the cost ratio is undefined, so this needs a decision: NN feasibility + raw cost only, or
   comparison against a lower bound. **Please flag which you did.**

---

## 4. Stage 1 — dataset selection, 5 head-to-head comparisons  *(Table T3a)*

For each of the 5 roles, compare its **dataset-A model against its dataset-B model**, and pick
a winner. Selection is made on **validation**; Stage 2 then reports on **test**.

### Evaluation grid — please read

Do **not** evaluate each model only on its own training range — that guarantees each wins on its
own turf and tells us nothing. Evaluate **both models of a role on one common (N, M) grid that
spans both regimes**: the regular-size range *and* the large-size range, up to the Stage-0
evaluation ceiling.

This turns a single aggregate into a **crossover curve** — dataset A presumably wins at small
N/M, dataset B at large N/M, and the interesting number is where they cross. That is a far
better result than "B wins on average", and it is the same cost to produce.

**Scope:** each role is compared on **its own domain** — expert-maze A vs expert-maze B on
`maze`; the generalist A vs generalist B on all 4 maps.

**Deliver:** per-instance CSVs under `stage1_dataset_selection/{role}_{dataset}/`, schema in §6.
Plus an aggregate row per (role, dataset, map, N, M), and a one-line statement of **which
dataset won for each of the 5 roles, and on what basis**.

---

## 5. Stage 2 — experts vs generalist  *(Tables T3b, Figures F8 — highest priority)*

Take the 5 Stage-1 winners. **Evaluate all 5 on all 4 maps** — the full 5 × 4 matrix, not just
the diagonal.

- **Diagonal** (each expert on its own map, generalist on each map) answers *"are experts
  better?"*
- **Off-diagonal** (each expert on the other 3 maps) answers *"what does it cost when a router
  guesses wrong?"* — which is what turns the result into a deployment recommendation rather than
  an observation. These models already exist, so this is 4 evals instead of 1 per model.

**Deliver:** per-instance CSVs under `stage2_expert_vs_general/{model_id}/`, plus aggregates.

---

## 6. CSV schema and metrics

### 6.1 Per-instance CSV (both stages)

Same schema as Exp 11/15, so `evaluation/agg_paper_maps.py` still works:

```
{map}_n{N}m{M}.csv
columns: inst_seed, cost_nn, cost_solver, nn_k, solver_k,
         conflicts_nn, conflicts_solver, alloc_ms, nn_plan_ms, solver_ms
```

### 6.2 Aggregates per (model, map, N, M)

`cost_ratio`, `exact_match`, `diff_mean`, `diff_max`, `diff_std`, `infeasible`, `fallback`,
`alloc_ms`, `nn_plan_ms`, `solver_ms`, `n`.

### 6.3 Offline accuracy — please don't skip this

The report compares models on **accuracy as well as timing**, and execution cost alone does not
give the accuracy half. From `evaluation/evaluate.py`, per model per (map, N, M) on the test
split, into `offline/{model_id}.csv`:

`per_goal_acc`, `full_assignment_acc`, `cost_ratio_proxy`, `n`

### 6.4 Timing protocol — record this or the comparison is unusable

- **Same node** for the NN and solver runs within a cell — they are being compared directly.
- **Single-instance** timing (not batched), matching Exps 9–15 so the numbers stay comparable
  with earlier results.
- **Warm-up excluded** (discard the first inference).
- If batched allocation-only timing is also measured — worth having, it is the ~10³× figure —
  report it **separately** and label it clearly. Do not mix it into the pipeline timings.
- Record in `manifest.md`: node type, CPU model, whether the NN ran on GPU or CPU, and
  torch version.

### 6.5 Training log per model

Per-epoch: `epoch, train_loss, val_loss, train_ce, val_ce, train_minsum, val_minsum,
train_pergoal_acc, val_pergoal_acc`. Same format as the existing `report/train_logs/*.txt` is
fine. 10 files.

### 6.6 Model inventory — `model_inventory.csv`, one row per model

`model_id`, `role`, `train_maps`, `dataset`, `N_range`, `M_range`, `n_configs`,
`samples_per_config`, `total_samples`, `hidden`, `layers`, `params`, `epochs`, `lr`,
`batch_size`, `grad_clip`, `seeds`, `best_val_loss`, `best_val_pergoal_acc`, `best_epoch`,
`train_wall_hours`, `gpu`.

**Also flag any training anomalies** — models that hit the wall-clock limit before finishing
their epoch budget, diverged, or were restarted. This has happened before (Exp 12's models were
both wall-limited before 150 epochs; Exp 4's h256/L6 diverged on some seeds), and the report
should not present a wall-limited model as a converged one.

**One extra thing, please:** a sentence on **why only the smaller model** was carried forward.
The report reserves 1–2 sentences for this and we do not want to invent a rationale. The current
placeholder is derived from Exp 12 (h128 beat h256 in-distribution on every aggregate; 2.5× the
params bought nothing and cost ~20% inference speed) — if the actual reasoning differs, say so
and it gets replaced.

---

## 7. Protocol rules — cheap now, expensive to backfill

### 7.1 Score the generalist **per map**, never pooled
A single pooled number cannot distinguish "uniformly mediocre on all four maps" from "great on
three, bad on the maze". That distinction *is* the research question. Pooling makes the
expert-vs-generalist comparison unfalsifiable.

### 7.2 Evaluate every expert on **all four** maps
See §5. The off-diagonal is what yields the cost of misrouting.

### 7.3 Identical instance seeds across every model
All models must see the **same instances** for a given (map, N, M) cell. Paired comparisons on
identical instances are far stronger statistically than comparing independent means, and cost
nothing to arrange — `full_pipeline_eval.py` already supports fixing the instance seed.

### 7.4 Select on validation, report on test
Stage 1 picks the dataset; Stage 2 compares the winners. If both use the same test set the
comparison is biased. One line in the report pre-empts the criticism — but only if it is true.

---

## 8. Priority order

If time or compute runs short:

1. **Stage 2 diagonal** — 5 winners on their own domains, per-instance CSVs (blocks T3b, F8, §6.3)
2. **Stage 2 off-diagonal** — the other 3 maps per expert (upgrades the conclusion to routing)
3. **Stage 1** — the 5 dataset head-to-heads (blocks T3a)
4. **Stage 0** — solver scale wall (blocks F7 and the evaluation envelope)
5. **Offline accuracy** + **model inventory** + **training logs** (blocks T2, the accuracy half,
   and reproducibility)

Per-instance CSVs are required at **every** level — they are not a separate lower-priority item.

---

## 9. One continuity question

Exp 12's **B1** is already `h128/L6` trained on all 4 maps at N∈{5,10,15} × M∈{10,20,30} — that
is exactly the **generalist × dataset-A** cell of the new 5 × 2 matrix.

Is the new generalist a re-run of B1, a reuse of it, or a retrain with different parameters? The
report can use this as a continuity anchor between the earlier experiments and the final study,
but only if we can state which it is.

---

## 10. Checklist

- [ ] Files delivered into `report/data/exp16/` (**not** `results/` — git-ignored)
- [ ] `manifest.md` — what ran, job IDs, node types, dates
- [ ] **Stage 0** — `solver_scale_wall.csv` + the three ceilings stated
- [ ] **Stage 1** — 5 head-to-heads on a common grid spanning both regimes; winners named
- [ ] **Stage 2** — full 5 × 4 matrix, per-instance CSVs + aggregates
- [ ] **Offline accuracy** — `offline/{model_id}.csv` for all evaluated models
- [ ] **Timing** — single-instance, same node, warm-up excluded; hardware recorded
- [ ] `model_inventory.csv` (10 rows) + the small-model rationale sentence
- [ ] `train_logs/` — 10 per-epoch logs
- [ ] Protocol: generalist per-map ✓ · experts on all 4 maps ✓ · identical seeds ✓ ·
      select-on-val ✓
- [ ] Continuity: relationship between the new generalist and Exp 12's B1 stated
