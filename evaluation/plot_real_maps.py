"""
Generate report figures for the real-maps evaluation of the old (Exp 11) model.

Reads:
  - Phase 6 per-instance CSVs in results/fullpipe_real_maps/<map>_<cfg>.csv
  - Phase 5 offline full-assignment accuracy (hard-coded below from
    results/offline_real_maps.txt)

Writes PNGs to report/real_maps/.
Run: .venv/bin/python evaluation/plot_real_maps.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(ROOT, "results", "fullpipe_real_maps")
OUT_DIR = os.path.join(ROOT, "report", "real_maps")
os.makedirs(OUT_DIR, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
CFGS = ["n2m2", "n3m3", "n4m4", "n5m5", "n3m6", "n5m8"]  # difficulty-ascending

COLORS = {
    "empty-32-32":     "#5e81ac",
    "random-32-32-20": "#a3be8c",
    "maze-32-32-2":    "#bf616a",
    "room-32-32-4":    "#d08770",
}
MARKERS = {
    "empty-32-32": "o", "random-32-32-20": "s",
    "maze-32-32-2": "^", "room-32-32-4": "D",
}

# Phase 5 offline full-assignment accuracy (from results/offline_real_maps.txt)
OFFLINE_FULL = {
    "empty-32-32":     [0.960, 0.885, 0.795, 0.705, 0.670, 0.350],
    "random-32-32-20": [0.960, 0.935, 0.820, 0.755, 0.715, 0.575],
    "maze-32-32-2":    [0.970, 0.980, 0.915, 0.815, 0.785, 0.565],
    "room-32-32-4":    [0.965, 0.945, 0.790, 0.750, 0.700, 0.540],
}
# Exp 11 random-grid baseline (full-assignment) for the same configs
BASELINE_FULL = [0.94, 0.85, 0.73, 0.65, 0.65, 0.44]


def load_csv(mp, cf):
    path = os.path.join(CSV_DIR, f"{mp}_{cf}.csv")
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return f("cost_nn"), f("cost_solver"), f("alloc_ms"), f("nn_plan_ms"), f("solver_ms")


def aggregate():
    """Return dict[map] -> dict of per-config metric lists."""
    out = {}
    for mp in MAPS:
        ratio, exact, nn_ms, sv_ms, speed = [], [], [], [], []
        for cf in CFGS:
            cn, cs, al, pl, sv = load_csv(mp, cf)
            ratio.append(np.where(cs > 0, cn / cs, 1.0).mean())
            exact.append((cn == cs).mean())
            nn_ms.append((al + pl).mean())
            sv_ms.append(sv.mean())
            speed.append(sv.mean() / (al + pl).mean())
        out[mp] = dict(ratio=ratio, exact=exact, nn_ms=nn_ms, sv_ms=sv_ms, speed=speed)
    return out


def fig_offline_acc():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CFGS))
    for mp in MAPS:
        ax.plot(x, OFFLINE_FULL[mp], marker=MARKERS[mp], color=COLORS[mp],
                label=mp, linewidth=2, markersize=7)
    ax.plot(x, BASELINE_FULL, "k--", linewidth=1.5, alpha=0.6,
            label="random-grid baseline (Exp 11)")
    ax.set_xticks(x); ax.set_xticklabels(CFGS)
    ax.set_xlabel("config (N, M)"); ax.set_ylabel("full-assignment accuracy")
    ax.set_title("Phase 5: offline allocation accuracy on real maps")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "rm_fig1_offline_acc.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_cost_ratio(agg):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CFGS))
    for mp in MAPS:
        ax.plot(x, agg[mp]["ratio"], marker=MARKERS[mp], color=COLORS[mp],
                label=mp, linewidth=2, markersize=7)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.6, label="optimal (solver)")
    ax.set_xticks(x); ax.set_xticklabels(CFGS)
    ax.set_xlabel("config (N, M)"); ax.set_ylabel("execution-cost ratio (NN / solver)")
    ax.set_title("Phase 6: true execution-cost ratio on real maps")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "rm_fig2_cost_ratio.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_exact_match(agg):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CFGS)); w = 0.2
    for i, mp in enumerate(MAPS):
        ax.bar(x + (i - 1.5) * w, agg[mp]["exact"], w, color=COLORS[mp], label=mp)
    ax.set_xticks(x); ax.set_xticklabels(CFGS)
    ax.set_xlabel("config (N, M)"); ax.set_ylabel("exact execution-cost match rate")
    ax.set_title("Phase 6: fraction of instances matching solver cost exactly")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "rm_fig3_exact_match.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_speedup(agg):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CFGS)); w = 0.2
    for i, mp in enumerate(MAPS):
        ax.bar(x + (i - 1.5) * w, agg[mp]["speed"], w, color=COLORS[mp], label=mp)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.7, label="parity (NN = solver)")
    ax.set_xticks(x); ax.set_xticklabels(CFGS)
    ax.set_xlabel("config (N, M)"); ax.set_ylabel("speedup (solver ms / NN ms)")
    ax.set_title("Phase 6: single-instance speedup (CPU). <1 = NN slower")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "rm_fig4_speedup.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_timing(agg):
    """NN vs solver wall time per config, averaged over maps (excl. maze outliers note)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CFGS))
    nn = np.mean([agg[mp]["nn_ms"] for mp in MAPS], axis=0)
    sv = np.mean([agg[mp]["sv_ms"] for mp in MAPS], axis=0)
    ax.plot(x, nn, "o-", color="#bf616a", linewidth=2, label="NN pipeline (forward + order + CBS)")
    ax.plot(x, sv, "s-", color="#5e81ac", linewidth=2, label="solver (LKH + CBS)")
    ax.set_xticks(x); ax.set_xticklabels(CFGS)
    ax.set_xlabel("config (N, M)"); ax.set_ylabel("mean wall time (ms), avg over 4 maps")
    ax.set_title("Phase 6: pipeline wall time (single-instance, CPU)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "rm_fig5_timing.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    agg = aggregate()
    fig_offline_acc()
    fig_cost_ratio(agg)
    fig_exact_match(agg)
    fig_speedup(agg)
    fig_timing(agg)
    print("\nAll figures written to", OUT_DIR)
