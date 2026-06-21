# Generation Prompt — Neural MCPF Allocation talk (paste into any LM)

You are an expert technical-presentation designer. Generate a **Marp** presentation in **Markdown**
(a single `slides.md` file) for a ~20-minute academic talk to a Multi-Agent Systems course (teacher +
classmates). Output **only** the Marp Markdown.

## Hard requirements
- **Marp format.** Begin with front-matter:
  ```
  ---
  marp: true
  theme: default
  paginate: true
  size: 16:9
  style: |
    section { font-family: Inter, Helvetica, Arial, sans-serif; }
    h1, h2 { color: #1F497D; }
    strong { color: #0070C0; }
    section::after { color: #888; font-size: 0.6em; }
  ---
  ```
- **Language:** English. **Theme:** clean academic — white background, navy `#1F497D` titles, accent
  blue `#0070C0`, success green `#198754` for "use NN" / positive, grey for "use solver".
- **Minimal on-slide text:** a title + at most ~4 short phrases/bullets + one visual per slide. The
  detail goes in **presenter notes**, written as HTML comments `<!-- ... -->` after each slide (Marp
  treats these as speaker notes; they must NOT render on the slide).
- **Images:** reference files in `assets/` exactly as named below (e.g. `![w:900](assets/maps_panel.png)`).
  Where a slide says `[DIAGRAM: ...]`, insert a clear placeholder (a styled box or a `TODO` note with
  the described content) — do not invent a real image path.
- **Color language in figures/wording:** navy = solver / fixed-map / model "B"; red = NN / random / model "C".
- Keep one idea per slide. Use `---` between slides. 17 core slides, then a divider, then 7 optional/backup slides.

## Core narrative (the 4 messages to land)
1) Optimal multi-agent goal allocation (mTSP in MAPF) is solved by RobustMCPF (LKH-TSP + CBS), but the
   allocation step is the costly, badly-scaling part. 2) A small size-agnostic transformer imitates that
   allocation from distance inputs (D, G), near-optimal (~1–6% over optimal in range). 3) One model for
   any N, M; it generalizes across problem scale and (with the right data) map geometry. 4) Payoff is
   speed at scale: up to 5.9–9.4× at paper scale within ~20% of optimal — use it where speed/scale matter.

## Slides (title · on-slide content · visual · presenter notes)

**1. Title** — "Neural MCPF Allocation — Learning the Allocation Step of Multi-Agent Path Finding";
authors/course/date; tagline "A transformer that replaces a combinatorial solver's allocation —
near-optimal, any size, much faster at scale." Visual: faded `assets/maps_panel.png`.
Notes: one-sentence framing of the project.

**2. Motivation** — "Who goes where?"; phrase "many agents, many goals, minimize total work"; 3 domains
(warehouse robots, delivery fleets, search & rescue). Visual: `assets/motivation.png`.
Notes: the core decision is allocation — which agent takes which goals; bad allocation wastes travel and
causes collisions; real systems need it fast.

**3. The problem (formal)** — "Multi-agent goal allocation = mTSP inside MAPF"; bullets: each goal visited
once · agents take 0/1/many goals · minimize total tour cost · then plan collision-free paths.
Visual: `assets/bundle_vs_split.png`.
Notes: it's mTSP not 1-to-1 — an agent may take several goals if cheaper (bundle cost 7 beats split 14);
this is why the model outputs a per-goal distribution over agents (columns sum to 1), not a permutation.

**4. Exact algorithm (baseline)** — "RobustMCPF — the exact solver"; chips: LKH-TSP allocation (k-best) →
CBS collision-free planning → optimal paths. Visual: `[DIAGRAM: [LKH allocation]→[CBS]→paths]`.
Notes: ground truth = RobustMCPF (BasicMAPF); LKH finds optimal allocation/tours, CBS makes paths
collision-free; exact but the TSP allocation is combinatorial. (Code: solver_wrapper.py, RobustMCPF/.)

**5. Bottleneck + idea** — "The allocation step is the bottleneck" → "Replace it with a neural net";
note: expected = imitate solver + gain speed at scale. Visual: arrow swapping LKH box → NN box.
Notes: LKH cost grows with size; CBS is shared by both; replace only allocation with a near-instant
learned model. Hypothesis: close to solver quality + big speedup, especially as N, M grow.

**6. System overview** — side-by-side: Solver: instance→LKH alloc→CBS→paths; Ours:
instance→NN alloc→goal ordering→CBS→paths; highlight only the alloc box differs.
Visual: `assets/pipeline_solver_vs_nn.png`.
Notes: apples-to-apples — both share ordering+CBS, only the allocator differs (full_pipeline_eval.py).

**7. Inputs** — "Two distance matrices": **D** (N×M) agent→goal, **G** (M×M) goal→goal; normalized [0,1];
phrase "distances, not coordinates". Visual: `assets/D_G_matrices.png`.
Notes: BFS shortest-path distances respect walls and transfer across maps; **G is the key feature** (which
goals are near each other → bundle); adding G lifted accuracy +14–18 points, more than any other change.
(Code: dataset_generation/distance.py.)

**8. The network** — "A universal transformer"; chips: row attention (agent↔goals) · column attention
(goal↔agents) · column softmax · loss = CE + λ·MinSum. Visual: `assets/attention_sketch.png`.
Notes: embed each agent-goal pair; stacked row/column attention captures the combinatorial structure;
output is column-wise softmax (per goal, a distribution over agents); trained with cross-entropy to imitate
the solver + small MinSum term (λ=0.1). (Code: model/network.py, model/losses.py.)

**9. How one model handles ANY N, M** *(key technical slide)* — chain:
**D, G → per-pair tokens (N×M grid) → attention → N×M logits → column softmax → allocation matrix**;
phrase "no positional embeddings → any size". Visual: `assets/anyNM_pipeline.png`.
Notes (centerpiece): every scalar D[i,j] → d-dim token (N×M grid); G injected per goal by embedding each
G[j,k] and **summing over k** (sum-pool → any M); **no agent/goal positional embeddings** (agents/goals
unordered; attention handles variable length) so the same weights run a 2×2 or a 50×100; output Linear(d→1)
→ N×M logits → column softmax → **argmax per goal** = the binary allocation matrix.
(Code: model/network.py GoalAllocTransformerUniversal.forward; decode in evaluation/evaluate.py.)

**10. Small vs big** — "Two sizes, same recipe": h128/L6 = 1.2M params, h256/L8 = 6.3M; foreshadow:
small wins in-range, big wins far out-of-range. Visual: `assets/params_bar.png`.
Notes: identical except width/depth; in-distribution the small model is better+faster (big overfits); at
extreme extrapolation the big one generalizes better.

**11. Method** — chips: Data (solver labels D,G,Y) · Train (imitate, mixed-size) · Eval (offline accuracy +
true CBS execution cost). Visual: `assets/method_flow.png`.
Notes: generate instances, run solver, store D,G,Y (reject unreachable/over-budget); train one universal
model on many (N,M) shapes; evaluate offline (does NN pick solver's agent?) and — what matters — execution
cost (NN→ordering→CBS vs solver, true path length); pytest suite guards it.
(Code: build_dataset.py, oracle.py, train.py, evaluate.py, full_pipeline_eval.py, tests/.)

**12. Results: in-distribution** — "Near the solver, much cheaper to run"; numbers: cost ratio ~1.02–1.05,
high exact-cost match, 0% infeasible; speedup up to a few× single-instance, ~10³× batched (allocation-only).
Visual: `assets/indist_costratio.png` (or `assets/heatmap_cost.png`).
Notes: ~1–6% over optimal across 28 configs; large diverse model mean 1.020; most disagreements are
cost-equivalent ties (exact-cost match ≫ exact-allocation match); CBS dominates full-pipeline time so
single-instance speedup is modest, batched/allocation-only ~1000×.

**13. The four real maps** — "From random grids to real benchmark maps"; names. Visual: `assets/maps_panel.png`.
Notes: empty (open), random-20 (scattered), maze (corridors), room (rooms+doors); only maze/room are truly
*structured* — that drives the next results.

**14. Results: generalization** *(climax)* — three findings: (a) transfers across geometry, not scale;
(b) random vs fixed training ties — except the maze; (c) at XL scale the big model wins, 5.9–9.4× faster.
Visual: `assets/scissors_13b.png` and `assets/random_vs_fixed_bymap.png` (two figures).
Notes: (a) small model zero-shot on real maps at small N/M = 1.019, but at large N/M fails (1.246 vs
map-trained 1.048); scissors at M=30 — more agents hurt old model (1.28→1.44), help in-range one (1.10→1.05).
(b) random-grid training ties fixed-map everywhere except maze (1.029 vs 1.127): random walls never make
corridors. (c) at N≤50/M≤100 the big model extrapolates best (1.208 vs 1.244), speedup 5.9–9.4×.

**15. When to use NN vs solver** — 2-column table. **Use NN (green):** large N/M · batched/high-throughput ·
unstructured maps · speed>exactness. **Use solver (grey):** tiny instances · exactness-critical · unseen
structured maps · one-off. Visual: `assets/decision_table.png` (or render the table on-slide).
Notes: NN replaces only allocation, so it wins where allocation/TSP dominates and many instances are solved;
solver still wins small/exact/unseen-structured.

**16. Conclusion** — 4 one-line takeaways (the 4 messages) + 1 future-work line (tour-aware loss + learned
goal-ordering). Visual: small summary graphic or faded maps panel.

**17. Q&A / Thanks** — "Thank you — questions?" + repo link.

---
### Optional / backup (after a divider; show only if time / for Q&A)
- **O1 Loss detail:** L = L_CE + λ·L_MinSum (λ=0.1); CE imitates solver, MinSum = Σ P·D regularizer.
- **O2 Full results:** `assets/heatmap_cost.png`, `assets/heatmap_match.png`; difficulty gradient is in M, flat in N.
- **O3 Overfitting:** `assets/convergence.png`; val loss bottoms ~epoch 22; best-val checkpoint = early stop; explains small-vs-big.
- **O4 Maze deep-dive:** `assets/maps_panel.png` + `assets/random_vs_fixed_bymap.png`; random walls ≠ corridors; entire B-vs-C gap is maze.
- **O5 Metrics defined:** offline = allocation agreement; execution cost = true CBS path-length ratio; exact match = identical integer cost.
- **O6 Limitations + next steps:** O(k!) brute-force ordering (mitigated by NN+2-opt); CBS non-differentiable; tour-aware loss + learned ordering next.
- **O7 Demo:** real run of model A on the room map. **Best for the slide:** `assets/demo_run_compare.gif`
  — synchronized side-by-side, exact solver (left) vs NN (right), both cost 62, animating in lockstep
  (the NN sends the two right goals to a different agent — green vs the solver's red). Standalone GIFs if
  you prefer your own layout: `assets/demo_run_solver.gif`, `assets/demo_run.gif`. Static proof:
  `assets/demo_nn_vs_solver.png`. PDF fallback (GIFs are static in PDF): `assets/demo_final_frame.png`.
  Notes: same instance, different allocation, identical execution cost (62) — a vivid example of the
  cost-equivalent ties behind the high exact-match rates.

## Final instructions to the generating model
- Produce the complete `slides.md` now, with presenter notes as `<!-- -->` comments on every slide.
- Honor the minimal-text rule: move any long sentence into the notes.
- For each `[DIAGRAM: ...]`, insert a clearly-labeled placeholder describing the figure to add later.
- Do not fabricate numbers; use only those given above.
