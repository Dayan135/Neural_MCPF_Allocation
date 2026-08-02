# Presentation Plan & Private Speaker Notes

**Talk:** Neural MCPF Allocation — learning to replace the combinatorial allocator in multi-agent path finding.
**Audience:** course teacher + classmates (Multi-Agent Systems). **Duration:** ~20 min + Q&A.
**Format:** Marp (Markdown) slides, English. **Style:** clean academic (white bg, navy accent), minimal on-slide text, visual-driven.

This file is the **private readme** (not shown to the audience): per slide it gives the on-slide content, what to *say*, the relevant code files, and the exact numbers. The audience deck is generated separately from `GENERATION_PROMPT.md`.

## The 4 core messages (everything serves these)
1. **Problem & bottleneck** — multi-agent goal allocation (mTSP inside MAPF) is solved optimally by RobustMCPF (LKH-TSP + CBS), but the allocation step is the expensive, badly-scaling part.
2. **The idea** — a small, size-agnostic transformer imitates the solver's allocation from distance inputs (D, G) → near-optimal (~1–6% over optimal in range).
3. **It generalizes** — one model for any N, M; transfers across problem scale and (with the right data) across map geometry.
4. **Payoff: speed at scale** — replacing allocation gives up to **5.9–9.4×** at paper scale within ~20% of optimal. Use it where speed/throughput/scale matter, not for tiny exact-critical instances.

## Code map (for cross-referencing during prep / Q&A)
| Topic | Files |
|---|---|
| Solver bridge (ground truth) | `solver_wrapper.py`, `RobustMCPF/` (LKH-TSP + CBS) |
| Data generation | `dataset_generation/build_dataset.py`, `grid_gen.py` (maps/placement), `oracle.py` (`get_ground_truth`) |
| Input transform D, G | `dataset_generation/distance.py` (`compute_distance_matrix`, `compute_goal_distance_matrix`, `normalize_D`) |
| Network | `model/network.py` (`GoalAllocTransformerUniversal`, `_RowColBlock`, `build_model`) |
| Loss | `model/losses.py` (`mTSP_loss` = CE + λ·MinSum) |
| Training | `training/train.py` (`--mixed` MixedSizeBatchSampler) |
| Offline eval | `evaluation/evaluate.py` (accuracy, Σ P·D proxy) |
| Execution-cost eval | `evaluation/full_pipeline_eval.py` (NN→ordering→CBS vs LKH+CBS) |
| Aggregation | `evaluation/agg_paper_maps.py`, `agg_compare.py` |
| Tests | `tests/test_{distance,grid_gen,losses,network,oracle,build_dataset,fixed_alloc}.py` |
| Maps | `RobustMCPF/Maps/{empty,random-32-32-20,maze-32-32-2,room-32-32-4}.map` |
| Cluster jobs | `scripts/exp14/`, `scripts/exp_paper_*.sh`, `scripts/eval_paper_*.sh` |
| Figures | `evaluation/plot_maps.py`, `plot_real_maps.py`, `plot_oldmodel_large_nm.py`, `plot_random_vs_fixed.py` |

---

# Per-slide spec

Legend — **ON SLIDE**: minimal audience-facing content. **VISUAL**: asset in `assets/` or `[CREATE]` diagram. **NOTES**: what to say (~speaking time). **CODE**: files to mention/know.

### Slide 1 — Title  [Core]
- **ON SLIDE:** "Neural MCPF Allocation — Learning the Allocation Step of Multi-Agent Path Finding". Authors, course, date. One-line tagline: *"A transformer that replaces a combinatorial solver's allocation — near-optimal, any size, much faster at scale."*
- **VISUAL:** subtle — small maze/grid motif or the maps panel faded.
- **NOTES (~20s):** One sentence framing: we taught a neural net to do the expensive decision a classic MAPF solver makes — who visits which goal — and studied when that's a good trade.

### Slide 2 — Motivation  [Core]
- **ON SLIDE:** "Who goes where?" + 3 icons: warehouse robots, delivery fleet, search-and-rescue. Phrase: *many agents, many goals, minimize total work.*
- **VISUAL:** [CREATE or stock] warehouse/robot-fleet image.
- **NOTES (~60s):** Fleets of robots/agents must split up a set of targets. The core decision is the *allocation*: which agent takes which goals. Bad allocation = wasted travel and collisions. This decision is combinatorial and is the heart of multi-agent coordination. Real systems (Amazon warehouses, drone delivery) solve versions of this constantly and need it fast.

### Slide 3 — The problem (formal)  [Core]
- **ON SLIDE:** "Multi-agent goal allocation = mTSP inside MAPF." Bullets (short): each goal visited once; agents take 0/1/many goals; minimize **total tour cost**; then plan collision-free paths.
- **VISUAL:** [CREATE] tiny 5×5 grid, 2 agents + 2 goals, two options: *split* (cost 14) vs *bundle to one agent* (cost 7). Highlight that bundling wins.
- **NOTES (~75s):** Formally it's a multi-agent TSP (mTSP), not a 1-to-1 assignment: an agent may take several goals if that's cheaper. Classic example — two agents near each other, two far goals: giving both goals to one agent (cost 7) beats splitting (cost 14). Every goal visited exactly once; agents unconstrained. After allocation, paths must be collision-free (the MAPF part). **CODE:** the mTSP regime is why our output is a per-goal distribution (columns sum to 1), not a permutation — see `model/network.py` docstring & `CLAUDE.md`.

### Slide 4 — The exact algorithm (baseline)  [Core]
- **ON SLIDE:** "RobustMCPF — the exact solver." Pipeline chips: *LKH-TSP allocation (k-best) → CBS collision-free planning → optimal paths.*
- **VISUAL:** [CREATE] simple 2-box pipeline: [LKH allocation] → [CBS planning] → paths.
- **NOTES (~60s):** Our ground truth is RobustMCPF in BasicMAPF mode: it uses the LKH TSP solver to find the optimal allocation/tours, then CBS (Conflict-Based Search) to turn that into collision-free paths. It's exact/near-optimal but the TSP allocation is combinatorial. **CODE:** `solver_wrapper.run_basic_mapf` bridges to `RobustMCPF/`; allocation from `K_optimal_sequences`, paths + cost from CBS (`Solution[5]`).

### Slide 5 — Bottleneck + our idea  [Core]
- **ON SLIDE:** "The allocation step is the bottleneck." → "Replace it with a neural net." + *expected:* imitate the solver, gain speed (esp. at scale).
- **VISUAL:** arrow swapping the LKH box for an NN box.
- **NOTES (~60s):** LKH allocation cost grows with problem size; CBS is needed by both approaches. So we replace *only* the allocation with a learned model that's near-instant after training. Hypothesis going in: the NN imitates the solver's allocations closely and gives a large speedup, especially as N, M grow. (Spoiler we'll revisit: the speed win is real but concentrated at scale.)

### Slide 6 — System overview  [Core]
- **ON SLIDE:** side-by-side: **Solver:** instance → *LKH alloc* → CBS → paths. **Ours:** instance → *NN alloc* → goal ordering → CBS → paths. Highlight only the alloc box differs.
- **VISUAL:** [CREATE] two parallel pipelines, NN box highlighted.
- **NOTES (~45s):** Apples-to-apples: both pipelines share goal-ordering + CBS; only the allocator differs, so any cost/speed difference is attributable to the allocation. **CODE:** `evaluation/full_pipeline_eval.py` runs both on the same instances/seed.

### Slide 7 — Inputs: how we represent the problem  [Core]
- **ON SLIDE:** "Two distance matrices." **D** (N×M): agent→goal. **G** (M×M): goal→goal. Normalized to [0,1]. Phrase: *distances, not coordinates.*
- **VISUAL:** assets `[CREATE] D_G_matrices.png` (small heatmaps of an example D and G). (offer to generate)
- **NOTES (~75s):** We feed BFS shortest-path distances (walls respected), not raw coordinates — distances are what determine tour cost and they transfer across maps. D tells each agent how far each goal is; **G is the key feature** — it encodes which goals are near each other, i.e. which to bundle. In an ablation, adding G lifted accuracy by **+14–18 points**, more than any architecture/size/data change. **CODE:** `dataset_generation/distance.py` (`compute_distance_matrix`, `compute_goal_distance_matrix`, `normalize_D` divides by (W−1)+(H−1)).

### Slide 8 — The network  [Core]
- **ON SLIDE:** "A universal transformer." chips: *row attention (agent↔goals) · column attention (goal↔agents) · column softmax · loss = CE + λ·MinSum.*
- **VISUAL:** [CREATE] attention schematic on the N×M grid (row + column arrows).
- **NOTES (~75s):** Each agent-goal pair is embedded; stacked row/column attention blocks let agents compare their goals and goals compare their agents — capturing the combinatorial structure. Output is a **column-wise softmax**: for each goal, a probability distribution over agents (columns sum to 1). Trained with cross-entropy to imitate the solver, plus a small MinSum term (λ=0.1) penalizing distant assignments. **CODE:** `model/network.py` `_RowColBlock`; `model/losses.py` `mTSP_loss`.

### Slide 9 — How one model handles ANY N, M  [Core — key technical]
- **ON SLIDE:** the transform chain: **D, G → per-pair tokens (N×M grid) → attention → N×M logits → column softmax → allocation matrix.** Phrase: *no positional embeddings → any size.*
- **VISUAL:** [CREATE] `anyNM_pipeline.png` — D(N×M) + G(M×M) → grid of d-dim tokens → blocks → logits(N×M) → argmax-per-column → binary allocation Y(N×M). (offer to generate)
- **NOTES (~90s, the technical centerpiece):**
  - *Input → tokens:* every scalar D[i,j] is projected to a d-dim vector (Linear 1→d), giving an N×M grid of tokens. G is injected per goal by embedding each G[j,k] scalar and **summing over k** (sum-pool) → a goal-context vector added to that goal's column. Sum-pool means M can be any size.
  - *Why any N,M:* there are **no agent/goal positional embeddings** — agents and goals are unordered in mTSP, and attention works on variable-length sequences, so the same weights process a 2×2 or a 50×100 grid.
  - *Output → allocation:* a Linear(d→1) gives one logit per cell → N×M logits; **column softmax** (over agents) → per-goal distribution; **argmax per goal** → the binary allocation matrix Y (each goal's agent). 
  - **CODE:** `model/network.py` `GoalAllocTransformerUniversal.forward` — `input_proj`, `g_scalar_proj(...).sum(dim=2)`, `out_proj`, `softmax(dim=1)`; decode via argmax in `evaluation/evaluate.py` (`decode_assignment`).

### Slide 10 — Small vs big model  [Core]
- **ON SLIDE:** "Two sizes, same recipe." h128/L6 = **1.2M params**; h256/L8 = **6.3M**. Foreshadow: *small wins in-range, big wins far out-of-range.*
- **VISUAL:** [CREATE] params bar (1.2M vs 6.3M).
- **NOTES (~45s):** We study a small and a large universal model, identical except width/depth. Preview: in-distribution the small one is better and faster (the big one overfits); at extreme extrapolation the big one generalizes better. We'll see both.

### Slide 11 — Method: data, training, evaluation  [Core]
- **ON SLIDE:** 3 chips: **Data** (solver labels D,G,Y) · **Train** (imitate, mixed-size) · **Eval** (offline accuracy + true CBS execution cost).
- **VISUAL:** [CREATE] data→train→eval flow.
- **NOTES (~75s):** Data: place agents/goals on grids, run the solver, store D, G and the optimal allocation Y; unreachable/over-budget instances rejected. Train one universal model on many (N,M) shapes (a sampler keeps batches shape-homogeneous). Evaluate two ways: **offline** (does the NN pick the solver's agent? per-goal/full-assignment accuracy) and the one that matters — **execution cost**: run NN→ordering→CBS and compare the true collision-free path length to the solver's. Tested with a pytest suite (distances, placement, loss, column-sum invariant, solver integration). **CODE:** `build_dataset.py`, `oracle.py`, `train.py`, `evaluate.py`, `full_pipeline_eval.py`, `tests/`.

### Slide 12 — Results: in-distribution  [Core]
- **ON SLIDE:** "Near the solver, much cheaper to run." Headline numbers: cost ratio **~1.02–1.05**, exact-cost match high; speedup up to a few × single-instance, ~10³× batched (allocation-only).
- **VISUAL:** `assets/indist_costratio.png` (small-vs-large cost ratio) or `assets/heatmap_cost.png`.
- **NOTES (~75s):** Across 28 small configs the NN's execution cost is ~1–6% above optimal with 0% infeasible; the large diverse model hits mean 1.020. Most disagreements with the solver are cost-equivalent ties, so exact-cost match is much higher than exact-allocation match. Speed: in the full pipeline CBS dominates so single-instance speedup is modest, but allocation-only/batched it's ~1000×. **CODE:** Exp 9–11 via `full_pipeline_eval.py`; numbers in `RESULTS.md`.

### Slide 13 — The four real maps  [Core]
- **ON SLIDE:** "From random grids to real benchmark maps." names + the panel.
- **VISUAL:** `assets/maps_panel.png`.
- **NOTES (~45s):** We then moved onto the 4 standard RobustMCPF benchmark maps — empty (open), random-20 (scattered), maze (corridors), room (rooms+doors) — spanning unstructured to highly structured. Notice only maze/room have real *structure*; that distinction drives the next results.

### Slide 14 — Results: generalization studies  [Core]
- **ON SLIDE:** three mini-findings: **(a)** old model transfers across geometry, not scale; **(b)** random vs fixed-map training ties — except the maze; **(c)** at XL scale the big model wins, **5.9–9.4×** faster.
- **VISUAL:** `assets/scissors_13b.png` (the scissors) + `assets/random_vs_fixed_bymap.png` (maze gap). (use one per click / or side by side)
- **NOTES (~110s, the story climax):**
  - (a) *Geometry vs scale:* the small model trained only on tiny random grids generalizes zero-shot to real maps at small N/M (cost ratio 1.019) — but pushed to large N/M it fails (1.246 vs the map-trained model's 1.048). The "scissors": at M=30, **more agents hurt the old model (1.28→1.44) but help the in-range one (1.10→1.05)**. Lesson: it generalizes across map *shape*, not problem *scale*.
  - (b) *What training data?* We trained a model on random grids with varied walls (0–50%) vs on the 4 real maps, same recipe. They **tie everywhere except the maze** (fixed-map 1.029 vs random 1.127) — random scattered walls never reproduce corridor structure. Lesson: target-map training only matters for *structured* geometry.
  - (c) *Extreme scale (N≤50, M≤100):* the bigger model extrapolates best (1.208 vs 1.244) and the speedup climbs to **5.9–9.4×** — the regime where replacing the solver pays off most.
  - **CODE:** Exps 13.b/14/15 in `RESULTS.md`; figures `plot_oldmodel_large_nm.py`, `plot_random_vs_fixed.py`.

### Slide 15 — When to use the NN vs the exact solver  [Core]
- **ON SLIDE:** 2-column decision table. **Use NN:** large N/M · batched/high-throughput · unstructured maps · speed > exactness. **Use solver:** tiny instances · exactness-critical · unseen structured maps · one-off.
- **VISUAL:** [CREATE] simple 2-column table (green/grey).
- **NOTES (~60s):** The NN replaces only allocation, so it wins where allocation/TSP cost dominates and many instances are solved: big problems (5.9–9.4× at ~20% overhead), batched inference (orders of magnitude), and unstructured maps where cheap random training transfers. The exact solver still wins on small instances (already millisecond-fast), exactness-critical settings, and structured maps the NN hasn't trained on.

### Slide 16 — Conclusion  [Core]
- **ON SLIDE:** recap the 4 core messages as 4 one-liners + 1 future-work line.
- **VISUAL:** maybe the maps panel or a small summary graphic.
- **NOTES (~45s):** A small universal transformer learns the solver's allocation from distances, handles any N,M, and is near-optimal; it generalizes across scale and (with the right data) geometry; and it's a real speedup where it counts — at scale and in batch. Next: a tour-aware loss and a learned goal-ordering head to push quality and remove the last classical bottleneck.

### Slide 17 — Q&A / Thanks  [Core]
- **ON SLIDE:** "Thank you — questions?" + repo / contact.
- **NOTES:** Keep backup slides (O1–O7) ready for likely questions.

---

# Optional / backup slides

### O1 — Loss in detail  [OPT]
- **ON SLIDE:** L = L_CE + λ·L_MinSum (λ=0.1); CE = per-goal cross-entropy vs solver; MinSum = Σ P·D.
- **NOTES:** CE drives imitation; MinSum is a mild geometric regularizer (prevents uniform early). **CODE:** `model/losses.py`.

### O2 — Full results (heatmaps)  [OPT]
- **VISUAL:** `assets/heatmap_cost.png`, `assets/heatmap_match.png`.
- **NOTES:** Difficulty gradient is horizontal — driven by M (goals), nearly flat in N. Worst cell still within ~5%.

### O3 — Overfitting / small vs big  [OPT]
- **VISUAL:** `assets/convergence.png`.
- **NOTES:** Val loss bottoms ~epoch 22 then rises; best-val-loss checkpointing = effective early stopping. Explains why the big model overfits in-distribution but generalizes better far-OOD.

### O4 — Maze deep-dive (why C fails)  [OPT]
- **VISUAL:** `assets/maps_panel.png` (point at maze) + `assets/random_vs_fixed_bymap.png`.
- **NOTES:** Random Bernoulli walls ≈ empty/random/room statistically, but never produce long corridors → C never learned to thread them; the entire B-vs-C gap is the maze (worst cell maze n5m30: 1.075 vs 1.362).

### O5 — Metrics defined  [OPT]
- **NOTES:** Offline = allocation agreement (per-goal / full-assignment). Execution cost = true CBS path length ratio (NN/solver); exact match = identical integer cost. **CODE:** `evaluate.py`, `full_pipeline_eval.py`.

### O6 — Limitations + next steps  [OPT]
- **NOTES:** Brute-force goal ordering is O(k!) (mitigated by NN+2-opt fallback); CBS not differentiable; tour-aware loss + learned ordering are the next levers. **CODE:** `full_pipeline_eval.order_goals`.

### O7 — Demo  [OPT — built]
- **Built** via `presentation/make_demo.py` (model A on room-32-32-4, N=4 M=6, mode `tie_diff`):
  - `assets/demo_run_compare.gif` — **synchronized side-by-side** (solver | NN) running in lockstep;
    best for the slide. Both cost 62; NN sends the two right goals to a different agent (green vs red).
  - `assets/demo_nn_vs_solver.png` — static side-by-side (same story). Colors = allocation.
  - `assets/demo_run.gif`, `assets/demo_run_solver.gif` — standalone NN / solver animations.
  - `assets/demo_final_frame.png` — last frame (PDF fallback; GIFs are static in PDF).
- **Say:** "Same instance, real structured map. The NN makes a *different* allocation than the exact
  solver, yet CBS runs it collision-free at the *identical* cost (62) — a concrete example of the
  cost-equivalent ties that make execution-cost match so high." Other flavors:
  `make_demo.py --mode exact` (identical to solver) or `--mode gap` (NN slightly above optimal); other
  maps/sizes e.g. `--map maze-32-32-2 --n 5 --m 6`.

---

# Styling guide (for the generation prompt)
- **Theme:** clean academic. White background; primary navy **#1F497D**, accent blue **#0070C0**; success green **#198754** and warn amber for callouts (matches the LaTeX report palette).
- **Type:** sans-serif (e.g., Inter/Helvetica). Big slide titles, sparse body. One idea per slide.
- **Density:** minimal — title + ≤4 short phrases + one visual. Detail lives in speaker notes.
- **Figures:** large, centered, with a one-line caption; keep the navy/red color language consistent (navy = solver/fixed/B, red = NN/random/C where applicable).
- **Footer:** small slide number + short talk title. **Aspect:** 16:9.
- **Speaker notes:** include the NOTES text as presenter notes (Marp HTML comments) so they don't show on slides.

# Diagrams / artifacts
**Created** (in `assets/`, via `presentation/make_diagrams.py` — regenerable):
1. `bundle_vs_split.png` — slide 3 ✅
2. `pipeline_solver_vs_nn.png` — slides 4/6 ✅
3. `D_G_matrices.png` — slide 7 ✅
4. `anyNM_pipeline.png` — slide 9 ✅
5. `params_bar.png` — slide 10 ✅
6. `decision_table.png` — slide 15 ✅

**Optional diagrams — now also created** (in `assets/`):
7. `motivation.png` — slide 2 (agents/goals "who goes where", warehouse-style) ✅
8. `attention_sketch.png` — slide 8 (row + column attention on the agent×goal grid) ✅
9. `method_flow.png` — slide 11 (data → train → eval) ✅

**Still placeholders (truly optional):**
- slide 5: an arrow swapping the LKH box for the NN box (or reuse `pipeline_solver_vs_nn.png`).
- O7: the interactive demo (stretch).

Result figures already created earlier also live in `assets/`: `maps_panel`, `scissors_13b`,
`random_vs_fixed_bymap`, `indist_costratio`, `heatmap_cost`, `heatmap_match`, `convergence`.
