"""
Shared aggregation helpers for the Tier A/B (Exp 16-17) per-map-vs-joint closeout
figures. Reads the raw per-instance CSVs pulled from the cluster into
results/fullpipe_tier{A,B}_{indist,extrap,mapextrap}/<checkpoint>/<map>_n{N}m{M}.csv
(columns written by evaluation/full_pipeline_eval.py: inst_seed, cost_nn,
cost_solver, nn_k, solver_k, conflicts_nn, conflicts_solver, alloc_ms,
nn_plan_ms, solver_ms).

Not committed to results/ (git-ignored) — run locally after rsyncing the raw
CSVs down from the cluster.
"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
MAP_LABEL = {"empty-32-32": "empty", "random-32-32-20": "random",
             "maze-32-32-2": "maze", "room-32-32-4": "room"}
CHECKPOINTS = ["joint", "empty", "random", "maze", "room"]

GRIDS = {
    "A": {"own": {"N": [60, 120, 180], "M": [100, 225, 350]},
          "extrap": {"N": [190, 205, 220], "M": [370, 400, 430]}},
    "B": {"own": {"N": [30, 55, 80], "M": [50, 100, 150]},
          "extrap": {"N": [60, 120, 180], "M": [100, 225, 350]}},  # = Tier A's own grid
}


def load_csv(base, checkpoint, map_name, n, m):
    """Return per-instance (cost_nn, cost_solver, alloc_ms+nn_plan_ms, solver_ms) arrays, or None."""
    path = os.path.join(RESULTS, base, checkpoint, f"{map_name}_n{n}m{m}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cn = np.array([float(r["cost_nn"]) for r in rows])
    cs = np.array([float(r["cost_solver"]) for r in rows])
    nn_ms = np.array([float(r["alloc_ms"]) + float(r["nn_plan_ms"]) for r in rows])
    sv_ms = np.array([float(r["solver_ms"]) for r in rows])
    return cn, cs, nn_ms, sv_ms


def cell_stats(base, checkpoint, map_name, n, m):
    """Mean cost ratio + speedup for one (checkpoint, map, N, M) cell, or None if missing."""
    d = load_csv(base, checkpoint, map_name, n, m)
    if d is None:
        return None
    cn, cs, nn_ms, sv_ms = d
    ratio = np.where(cs > 0, cn / cs, 1.0)
    return dict(n_inst=len(cn), ratio=ratio.mean(), exact=(cn == cs).mean(),
                speedup=sv_ms.mean() / nn_ms.mean())


def grid_avg(base, checkpoint, map_name, ns, ms):
    """Average cost ratio (and n instances) over an N x M grid of cells for one (checkpoint, map)."""
    ratios, exacts, speedups, ninst = [], [], [], 0
    for n in ns:
        for m in ms:
            s = cell_stats(base, checkpoint, map_name, n, m)
            if s is None:
                continue
            ratios.append(s["ratio"])
            exacts.append(s["exact"])
            speedups.append(s["speedup"])
            ninst += s["n_inst"]
    if not ratios:
        return None
    return dict(n_inst=ninst, ratio=np.mean(ratios), exact=np.mean(exacts),
                speedup=np.mean(speedups), n_cells=len(ratios))


def by_m(base, checkpoint, map_name, ns, ms):
    """Mean cost ratio per M value, averaged over N (for trend lines)."""
    out = {}
    for m in ms:
        ratios = []
        for n in ns:
            s = cell_stats(base, checkpoint, map_name, n, m)
            if s is not None:
                ratios.append(s["ratio"])
        if ratios:
            out[m] = np.mean(ratios)
    return out
