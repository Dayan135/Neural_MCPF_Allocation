"""
Figures for Exp 14 — random-diverse (C) vs fixed-map (B) training, on the real maps.

Aggregates are sourced from report/random_vs_fixed/REPORT.md (raw CSVs are cluster-only).
Writes PNGs to report/random_vs_fixed/.

Run: .venv/bin/python evaluation/plot_random_vs_fixed.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(ROOT, "report", "random_vs_fixed")
os.makedirs(OUT_DIR, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]

# Mean cost ratio by map (B fixed / C random) — REPORT.md "By map" table.
BY_MAP = {
    "h128": {"empty-32-32": (1.053, 1.056), "random-32-32-20": (1.053, 1.056),
             "maze-32-32-2": (1.029, 1.127), "room-32-32-4": (1.057, 1.060)},
    "h256": {"empty-32-32": (1.060, 1.058), "random-32-32-20": (1.055, 1.058),
             "maze-32-32-2": (1.035, 1.083), "room-32-32-4": (1.059, 1.064)},
}
# Mean cost ratio by M (B / C) — REPORT.md "By M" table.
BY_M = {
    "h128": {10: (1.026, 1.034), 20: (1.049, 1.064), 30: (1.070, 1.126)},
    "h256": {10: (1.027, 1.029), 20: (1.051, 1.061), 30: (1.079, 1.108)},
}
# Maze-only mean cost ratio by M (B / C) — computed from REPORT.md maze per-config rows.
MAZE_BY_M = {
    "h128": {10: (1.010, 1.039), 20: (1.029, 1.085), 30: (1.048, 1.257)},
    "h256": {10: (1.012, 1.023), 20: (1.032, 1.067), 30: (1.060, 1.161)},
}
MS = [10, 20, 30]
B_COLOR, C_COLOR = "#5e81ac", "#bf616a"


def fig_by_map(size="h128"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MAPS)); w = 0.38
    b = [BY_MAP[size][m][0] for m in MAPS]
    c = [BY_MAP[size][m][1] for m in MAPS]
    ax.bar(x - w/2, b, w, color=B_COLOR, label="B (fixed-map)")
    ax.bar(x + w/2, c, w, color=C_COLOR, label="C (random-diverse)")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    for xi, (bv, cv) in enumerate(zip(b, c)):
        ax.annotate(f"{bv:.3f}", (xi - w/2, bv), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{cv:.3f}", (xi + w/2, cv), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("-32-32", "").replace("-32-32-20", "-20") for m in MAPS])
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_ylim(1.0, max(c) * 1.04)
    ax.set_title(f"Exp 14 ({size}): the gap is ALL maze — B vs C by map")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig1_by_map_{size}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_maze_by_M():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for size, ls in (("h128", "-"), ("h256", "--")):
        b = [MAZE_BY_M[size][m][0] for m in MS]
        c = [MAZE_BY_M[size][m][1] for m in MS]
        ax.plot(MS, b, ls, color=B_COLOR, marker="s", linewidth=2, label=f"B (fixed) {size}")
        ax.plot(MS, c, ls, color=C_COLOR, marker="o", linewidth=2, label=f"C (random) {size}")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xticks(MS); ax.set_xlabel("M (goals)  —  maze-32-32-2 only")
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_title("Exp 14: on the maze, C blows up with M; B stays flat")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_maze_by_M.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def fig_by_M(size="h128"):
    fig, ax = plt.subplots(figsize=(7, 5))
    b = [BY_M[size][m][0] for m in MS]
    c = [BY_M[size][m][1] for m in MS]
    ax.plot(MS, b, "s-", color=B_COLOR, linewidth=2, markersize=8, label="B (fixed-map)")
    ax.plot(MS, c, "o-", color=C_COLOR, linewidth=2, markersize=8, label="C (random-diverse)")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xticks(MS); ax.set_xlabel("M (goals)  —  mean over 4 maps")
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_title(f"Exp 14 ({size}): C's deficit grows with M (driven by maze)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"fig3_by_M_{size}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    fig_by_map("h128")
    fig_maze_by_M()
    fig_by_M("h128")
    print("\nAll figures written to", OUT_DIR)
