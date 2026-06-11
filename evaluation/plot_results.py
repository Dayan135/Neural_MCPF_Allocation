"""
Generate report figures from hard-coded experiment results.

Figures saved to evaluation/figures/.
Run: python evaluation/plot_results.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment data (mean, std) — sourced from CLAUDE.md
# ---------------------------------------------------------------------------

N_VALUES = [2, 3, 4, 5]

# Full-assignment accuracy: (mean, std)
RESULTS = {
    "MLP (no G)": {
        # Exp 2 — 8×8, 10k train
        2: (0.772, 0.003),
        3: (0.549, 0.005),
        4: (0.382, 0.002),
        5: (0.270, 0.002),
    },
    "Transformer (no G)": {
        # Exp 5 noG rows (freshly generated, consistent with Exp 2)
        2: (0.790, 0.002),
        3: (0.595, 0.010),
        4: (0.455, 0.006),
        5: (0.337, 0.004),  # Exp 2 transformer (no G data for N=5 yet)
    },
    "Transformer + G": {
        # Exp 5 G rows (N=2,3,4) + Exp 6 G row (N=5, filled in when ready)
        2: (0.932, 0.005),
        3: (0.775, 0.008),
        4: (0.632, 0.007),
        5: None,            # TODO: fill in after Exp 6 completes
    },
}

COLORS = {
    "MLP (no G)":          "#5e81ac",
    "Transformer (no G)":  "#88c0d0",
    "Transformer + G":     "#bf616a",
}

MARKERS = {
    "MLP (no G)":          "s",
    "Transformer (no G)":  "^",
    "Transformer + G":     "o",
}


# ---------------------------------------------------------------------------
# Figure 1 — Full-assignment accuracy vs N
# ---------------------------------------------------------------------------

def fig1_scale_n():
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, data in RESULTS.items():
        xs, ys, errs = [], [], []
        for n in N_VALUES:
            val = data.get(n)
            if val is not None:
                xs.append(n)
                ys.append(val[0])
                errs.append(val[1])
        ax.errorbar(
            xs, ys, yerr=errs,
            label=label,
            color=COLORS[label],
            marker=MARKERS[label],
            markersize=7,
            linewidth=1.8,
            capsize=4,
        )

    ax.set_xlabel("Number of agents N  (= number of goals M)", fontsize=11)
    ax.set_ylabel("Full-assignment accuracy", fontsize=11)
    ax.set_title("Scaling behaviour across N (8×8 grid, 10k train samples)", fontsize=11)
    ax.set_xticks(N_VALUES)
    ax.set_ylim(0.1, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "fig1_scale_n.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Ablation bar chart at N=3
# ---------------------------------------------------------------------------

def fig2_ablation_n3():
    # Successive additions to the baseline; show full-assignment acc at N=3
    labels = [
        "MLP baseline",
        "+ Transformer\narchitecture",
        "+ Capacity\n(h128/L6)",
        "+ Goal-goal\nfeature G",
    ]
    means  = [0.549, 0.590, 0.617, 0.775]
    stds   = [0.005, 0.004, 0.001, 0.008]
    colors = ["#5e81ac", "#81a1c1", "#88c0d0", "#bf616a"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55, capsize=4,
                  error_kw={"elinewidth": 1.5})

    # Annotate delta from previous bar
    for i in range(1, len(means)):
        delta = means[i] - means[i - 1]
        ax.text(x[i], means[i] + stds[i] + 0.015,
                f"+{delta:.3f}",
                ha="center", va="bottom", fontsize=9,
                color="#2e3440" if delta < 0.05 else "#bf616a",
                fontweight="bold" if delta > 0.05 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Full-assignment accuracy  (N=3)", fontsize=11)
    ax.set_title("What actually moves the needle at N=3", fontsize=11)
    ax.set_ylim(0.4, 0.92)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "fig2_ablation.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_scale_n()
    fig2_ablation_n3()
