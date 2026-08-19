# Exp 16 — review notes and open decisions

Written 2026-08-19, after reviewing the Section 6 fill-in and the delivered data. Two parts: what
we changed in the report, and what still needs your input before submission.

The canonical file is `report/final_report_overleaf.tex` (the Overleaf working copy, now tracked).
`report/final_report.tex` is the older repo copy and has diverged — please don't edit that one.

---

## Part 1 — what we changed

| Commit | Change |
|---|---|
| `8b5112d` | Tracked the Overleaf copy in git. It was untracked, so the canonical draft existed only outside version control |
| `9846a99` | Renamed the two training datasets **large** / **regular** everywhere, replacing Dataset A/B and Tier A/B |
| `d2739ad` | Addressed the five highlighted review notes (contributions, matrix shapes, a worked numeric example, the loss rationale, the D3 title) |
| `db39bea` | Cleared the last TODO box, the deferred-figure comment and a stale header note |

**On the rename.** The letters were inverted between the report and the raw files: the report had
Dataset A = large, while the manifest and the data request had Tier A = dataset B = large. The
appendix carried four lines telling readers to swap the letter when cross-referencing. Descriptive
names remove the problem at source. This also fixed a live contradiction — `fig_tierB_convergence.png`
was titled "Tier B (dataset A)" while its caption read "Tier B (Dataset B)".

The generator scripts were relabeled and their figures and tables regenerated from the committed
data. **No numbers changed**: `report/data/` is untouched, and the regenerated tables differ only
in caption text and in the winner column reading `regular` instead of `B`. The single-letter keys
survive inside `gen_exp16_stage1.py` because they address the on-disk directory layout and the
column names in `stage1_winners.csv`; renaming those would have rewritten committed data for
nothing.

**Verification.** Table 3 was re-aggregated independently from the raw per-instance CSVs and
reproduces your published values to within 0.001.

---

## Part 2 — decisions we need from you

### 1. The 220x430 claim in the abstract has no data behind it — highest priority

The abstract states:

> "Pushed further, to 220x430 on the four real benchmark maps, the same generalist stays within
> 9.6% of optimal at 6.6-16.1x speedup."

None of that is supported by anything under `report/data/exp16/`:

| Claim | What the delivered data shows |
|---|---|
| evaluated at 220x430 | Largest evaluation anywhere is **180x350**. 220x430 appears only in `solver_scale_wall.csv`, which is solver timing, 3 instances per cell, with no network run |
| within 9.6% of optimal | Generalist by-map range is 1.057-1.074, i.e. within **7.4%**. Worst single cell is 1.178 |
| 6.6-16.1x speedup | Generalist is **8.8-12.1x**. Across all 20 model-map rows it is 2.3-17.1x |

The manifest notes that `eval_tierA_extrap_*.sh` and `bothextrap` were "partially run for both tiers
on the cluster, but not pulled into this delivery", so you may well have these results already.

**Can you pull those CSVs?** If yes, they also need a table or figure in Section 6.3 — an abstract
claim with no support in the body will not survive a careful reader. If they can't be recovered,
the sentence has to be replaced with sourced numbers.

### 2. RQ3 is answered on one leg only, and it's the losing leg

RQ2 concludes the regular dataset is the better training distribution. RQ3 is then answered using
**large-dataset models only** (Table 3, Figure 8). The regular-dataset models have complete 5x4
coverage on disk, and they tell a different story:

| Map | large: joint / specialist | regular: joint / specialist |
|---|---|---|
| empty | 1.071 / 1.072 | 1.053 / **1.045** |
| random | 1.074 / 1.066 | 1.054 / **1.047** |
| maze | 1.057 / 1.057 | 1.036 / **1.035** |
| room | 1.068 / 1.066 | 1.052 / **1.047** |

At the large dataset the two tie. At the regular dataset the specialists win on all four maps,
consistently, with roughly twice the sample count (1800 vs 800-900 per cell).

So the headline "specialists do not beat the generalist" currently rests on the weaker,
lower-powered leg that RQ2 just declared the loser. The maze-misrouting finding survives both legs,
so that part is solid. **Should the regular leg be added?** `gen_exp16_tables.py` is already
structured to do it, and its header now notes the coverage exists.

### 3. Offline accuracy was never run

Not run for any of the 10 checkpoints, per the manifest. The professor asked for accuracy **and**
timing; Section 6 reports execution cost, exact match and speedup only. Defensible given finding D4
(execution cost is the better measure), but it's a deviation from the ask. Worth running
`evaluation/evaluate.py` if it's cheap.

### 4. "Dataset choice for RQ2 is made on validation data"

Section 6.2 says this. Stage 1 appears to use `full_pipeline_eval` on freshly seeded evaluation
grids rather than the validation split. Is the sentence true as written? If not it's a one-line
correction, but it matters — that sentence is what protects the two-stage selection from a
selection-bias objection.

### 5. How often does the goal-ordering fallback fire?

Relevant to item 6 below. Instrumenting `order_goals` to count fallbacks would settle it.

---

## Part 3 — three passages: your call

These were in the report, removed during our manual Overleaf pass, and we started restoring them
before deciding they should be your call instead. For each: what was there, why its absence is a
problem, and what we suggest.

### 6. The eight-goal ordering caveat (Section 3.1)

**What was removed.** A sentence saying the two pipelines' orderings agree while an agent holds at
most eight goals, because our enumeration is exhaustive and therefore optimal for a given
allocation, so cost differences isolate the allocator.

**Why its absence is a problem.** Section 3.1 now says only that the two pipelines "differ in how
the visit order is obtained", and stops. A reader is left unsure whether the comparison is
contaminated. It also matters more at your scales than it did before: at `n60m350` the mean is 5.8
goals per agent, so the busiest agent in an instance will hold roughly 10-12 — past the eight-goal
cutoff, where `order_goals` drops to nearest-neighbor plus 2-opt. In those cells part of the
measured gap comes from our ordering heuristic rather than from the allocator, which weakens the
claim that only the allocation box changes.

**Suggested fix.** Restore the caveat, updated for the new scale, stating plainly that attribution
is clean up to eight goals per agent and weakens above it. Best paired with item 5: if the fallback
fires rarely, say so and the caveat is cheap; if it fires often in the high-M cells, it belongs in
the limitations too.

### 7. The solver-ceiling sentence (Section 3.3)

**What was removed.** A short passage saying the architecture places no upper bound on N or M, and
that what bounds the experiments is the exact solver — every result is a ratio against its output,
so the baseline has to finish for a comparison to exist at all.

**Why its absence is a problem.** Section 3.3 now asserts "the architecture therefore places no
upper bound on N or M" and stops there, which invites the obvious question of why the experiments
stop where they do. The answer is a genuine strength of the work — the limit is the baseline's, not
the model's — and leaving it out makes the evaluation range look like a capability ceiling.

**Suggested fix.** Restore one or two sentences pointing at Section 6, phrased without quoting a
size so it doesn't read as a fixed limit. This dovetails with item 1: if the 220x430 data is
recovered, that becomes the natural number to cite.

### 8. The Discussion still quotes superseded numbers

**What's wrong.** Nothing was removed here — the Discussion was simply never updated. It says:

> "which yields 5.9-9.4x end-to-end speedup at roughly 20% above optimal cost"

Those are Exp 15's numbers. Section 6 now reports the generalist at **8.8-12.1x speedup and
5.7-7.4% above optimal**, with an individual specialist reaching 17.1x on its own map.

**Why it's a problem.** The Discussion is where a reader looks for the practical verdict, and it
currently understates the result while contradicting Section 6 two pages earlier. Of the three
items here this is the only one we'd call a defect rather than an improvement.

**Suggested fix.** Replace the sentence with the Section 6 figures. If item 1 resolves, the extreme
end of the range can be cited too. We drafted this and reverted it pending your review, so the
wording is ready if you want it.

---

## Summary of what to send back

- [ ] The 220x430 extrapolation CSVs, or agreement to restate the abstract (item 1)
- [ ] Decision on adding the regular-dataset leg to RQ3 (item 2)
- [ ] Offline accuracy, if cheap to run (item 3)
- [ ] Confirm or correct the validation-selection sentence (item 4)
- [ ] Ordering-fallback frequency (item 5)
- [ ] Decisions on the three passages (items 6, 7, 8)
