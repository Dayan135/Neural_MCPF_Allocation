"""
Figures for Exp 13.b — old model A vs map-trained B1 at large N/M on real maps.

Reads model A full-pipeline CSVs (results/fullpipe_oldmodel/<map>_n{N}m{M}.csv) and
compares to B1 (h128/L6, fixed-map) per-config cost ratios from RESULTS.md Exp 12
(hard-coded below). Writes PNGs to report/oldmodel_large_nm/.

Run: .venv/bin/python evaluation/plot_oldmodel_large_nm.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(ROOT, "results", "fullpipe_oldmodel")
OUT_DIR = os.path.join(ROOT, "report", "oldmodel_large_nm")
os.makedirs(OUT_DIR, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
NS = [5, 10, 15]
MS = [10, 20, 30]
COLORS = {"empty-32-32": "#5e81ac", "random-32-32-20": "#a3be8c",
          "maze-32-32-2": "#bf616a", "room-32-32-4": "#d08770"}
MARKERS = {"empty-32-32": "o", "random-32-32-20": "s",
           "maze-32-32-2": "^", "room-32-32-4": "D"}

# B1 (h128/L6 fixed-map) per-config cost ratios — RESULTS.md Exp 12.
B1 = {
    "empty-32-32":    {"n5m10": 1.0524, "n5m20": 1.0773, "n5m30": 1.1069, "n10m10": 1.0250, "n10m20": 1.0502, "n10m30": 1.0685, "n15m10": 1.0137, "n15m20": 1.0322, "n15m30": 1.0470},
    "random-32-32-20": {"n5m10": 1.0448, "n5m20": 1.0842, "n5m30": 1.1067, "n10m10": 1.0247, "n10m20": 1.0470, "n10m30": 1.0614, "n15m10": 1.0195, "n15m20": 1.0327, "n15m30": 1.0542},
    "maze-32-32-2":   {"n5m10": 1.0103, "n5m20": 1.0462, "n5m30": 1.0748, "n10m10": 1.0136, "n10m20": 1.0218, "n10m30": 1.0348, "n15m10": 1.0070, "n15m20": 1.0182, "n15m30": 1.0338},
    "room-32-32-4":   {"n5m10": 1.0515, "n5m20": 1.0868, "n5m30": 1.1186, "n10m10": 1.0244, "n10m20": 1.0465, "n10m30": 1.0790, "n15m10": 1.0208, "n15m20": 1.0393, "n15m30": 1.0485},
}


def a_ratio(mp, cf):
    rows = list(csv.DictReader(open(os.path.join(CSV_DIR, f"{mp}_{cf}.csv"))))
    cn = np.array([float(r["cost_nn"]) for r in rows])
    cs = np.array([float(r["cost_solver"]) for r in rows])
    return np.where(cs > 0, cn / cs, 1.0).mean()


def fig_ratio_by_M():
    """Mean cost ratio vs M (over 4 maps × 3 N): A vs B1 — the M cliff."""
    a = [np.mean([a_ratio(mp, f"n{N}m{M}") for mp in MAPS for N in NS]) for M in MS]
    b = [np.mean([B1[mp][f"n{N}m{M}"] for mp in MAPS for N in NS]) for M in MS]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(MS, a, "o-", color="#bf616a", linewidth=2.5, markersize=9, label="A (old, trained N≤5 M≤8)")
    ax.plot(MS, b, "s-", color="#5e81ac", linewidth=2.5, markersize=9, label="B1 (map-trained, this scale)")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, label="optimal")
    for x, y in zip(MS, a):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    for x, y in zip(MS, b):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
    ax.set_xticks(MS)
    ax.set_xlabel("M (goals)")
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_title("Exp 13.b: M is the cliff — old model A vs map-trained B1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_ratio_by_M.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def fig_N_divergence():
    """Cost ratio vs N at M=30 (over 4 maps): A worsens, B1 improves."""
    a = [np.mean([a_ratio(mp, f"n{N}m30") for mp in MAPS]) for N in NS]
    b = [np.mean([B1[mp][f"n{N}m30"] for mp in MAPS]) for N in NS]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(NS, a, "o-", color="#bf616a", linewidth=2.5, markersize=9, label="A (old, trained N≤5)")
    ax.plot(NS, b, "s-", color="#5e81ac", linewidth=2.5, markersize=9, label="B1 (trained N≤15)")
    for x, y in zip(NS, a):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    for x, y in zip(NS, b):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
    ax.set_xticks(NS)
    ax.set_xlabel("N (agents)  —  fixed M=30")
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_title("Exp 13.b: more agents HURT A but HELP B1 (M=30)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_N_divergence_m30.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def fig_A_by_map():
    """Model A cost ratio vs M, one line per map (mean over N) — maze easiest."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for mp in MAPS:
        y = [np.mean([a_ratio(mp, f"n{N}m{M}") for N in NS]) for M in MS]
        ax.plot(MS, y, marker=MARKERS[mp], color=COLORS[mp], linewidth=2, markersize=8, label=mp)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xticks(MS)
    ax.set_xlabel("M (goals)")
    ax.set_ylabel("model A execution-cost ratio (mean over N)")
    ax.set_title("Exp 13.b: model A by map — maze relatively easiest")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_A_by_map.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_ratio_by_M()
    fig_N_divergence()
    fig_A_by_map()
    print("\nAll figures written to", OUT_DIR)
