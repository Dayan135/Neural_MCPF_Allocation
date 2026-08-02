# Random-vs-fixed-map experiment (Exp 14) — cluster submission order

All commands from the project root on slurm.bgu.ac.il
(`/home/dayanb/course_multiagent/Neural_MCPF_Allocation`), after `mkdir -p logs`.

Trains models **C1 (h128/L6)** and **C2 (h256/L8)** on random-diverse data (random 32×32 grids,
per-instance wall density 0–50%, N∈{5,10,15}×M∈{10,20,30}), then evaluates them on the 4 real
benchmark maps. The C results are compared to the existing fixed-map models B1/B2 (Exp 12) to
decide whether random-diverse or fixed-map training transfers better.

The C recipe is identical to B's (`exp_paper_*`) except the training data: 9 (N,M) configs of
random grids instead of 36 (map×N,M) fixed-map configs. Per config 80k train / 2k val / 2k test —
80k matches B's per-(N,M)-shape volume (B saw 4 maps × 20k = 80k per shape; 720k total).

## Submission order

```bash
# 1. Generate the random-diverse training data (9 configs, ~720k samples total).
GEN=$(sbatch --parsable scripts/exp14/gen_random_diverse_data.sh)

# 2. Train C1 + C2 after generation finishes (run both in parallel).
sbatch --dependency=afterok:$GEN scripts/exp14/exp_random_current.sh
sbatch --dependency=afterok:$GEN scripts/exp14/exp_random_larger.sh

# 3. After training, evaluate both on the 4 real maps (run both in parallel).
sbatch scripts/exp14/eval_random_current.sh
sbatch scripts/exp14/eval_random_larger.sh

# 4. Aggregate (login node, no GPU). C-only tables, then the C-vs-B verdict.
MKL_THREADING_LAYER=GNU python evaluation/agg_paper_maps.py --base results/fullpipe_random
MKL_THREADING_LAYER=GNU python evaluation/agg_compare.py
```

## Outputs

- Data: `data/random_diverse/n{N}m{M}/{train,val,test}/`
- Checkpoints: `checkpoints/random_{current,larger}_s{0,1,2}/best.pt`
- Eval CSVs: `results/fullpipe_random/{current,larger}/{map}_n{N}m{M}.csv`

`data/`, `checkpoints/`, `results/`, `logs/` are git-ignored. To build the Exp 14 report locally,
copy the `results/fullpipe_random/` CSVs back (the existing `results/fullpipe_paper/` B CSVs are
needed too for `agg_compare.py`), or run `agg_compare.py` on the cluster and copy its text output.

## Next (Exp 15)

If C wins Exp 14, add `scripts/eval_random_xl_{current,larger}.sh` (copies of
`eval_paper_xl_{current,larger}.sh` repointed to the C checkpoints) for the zero-shot XL sweep
(N∈{20,35,50}×M∈{50,75,100}). If B wins, Exp 15 is the existing `eval_paper_xl_*` run.
