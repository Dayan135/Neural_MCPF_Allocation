"""Generate report figures as PNG files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(__file__)

PRIMARY   = "#1F497D"
ACCENT    = "#0070C0"
SUCCESS   = "#198754"
GRAY      = "#888888"

def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

# ── Figure 1: Architecture comparison (Exp 1) ──────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.8))
models = ["DeepSets", "MLP", "Transformer"]
per_goal  = [0.858, 0.875, 0.887]
full_asgn = [0.727, 0.756, 0.782]
x = np.arange(len(models))
w = 0.32
ax.bar(x - w/2, per_goal,  w, color=ACCENT,   alpha=0.85, label="Per-goal acc")
ax.bar(x + w/2, full_asgn, w, color=PRIMARY,  alpha=0.85, label="Full-assignment acc")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.65, 0.93)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Figure 1 — Architecture Comparison (Exp 1, N=2, M=2)", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
# annotate values on top
for xi, v in zip(x - w/2, per_goal):
    ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w/2, full_asgn):
    ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
savefig("fig1_arch.png")

# ── Figure 2: Goal-goal distance ablation (Exp 5) ──────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ns = [2, 3, 4, 5]
without_g = [0.790, 0.595, 0.455, 0.337]
with_g    = [0.932, 0.775, 0.632, 0.478]
x = np.arange(len(ns))
w = 0.35
ax.bar(x - w/2, without_g, w, color=GRAY,    alpha=0.75, label="Without G")
ax.bar(x + w/2, with_g,    w, color=SUCCESS,  alpha=0.85, label="With G")
ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in ns], fontsize=11)
ax.set_ylim(0.25, 1.00)
ax.set_ylabel("Full-assignment Accuracy", fontsize=11)
ax.set_title("Figure 2 — Impact of Goal-Goal Distance Feature G (Exp 5)", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
# annotate deltas
for xi, wg, g in zip(x, without_g, with_g):
    delta = g - wg
    ax.annotate(f"+{delta*100:.1f}pt",
                xy=(xi + w/2, g), xytext=(xi + w/2, g + 0.04),
                ha="center", fontsize=8, color=SUCCESS,
                arrowprops=dict(arrowstyle="-", color=SUCCESS, lw=0.7))
savefig("fig2_ablation.png")

# ── Figure 3: Exp 11 small vs large model cost ratio (selected configs) ─────
fig, ax = plt.subplots(figsize=(7.0, 4.2))
configs  = ["n2m2", "n3m4", "n4m5", "n4m8", "n3m8", "n5m8"]
old_vals = [1.010, 1.043, 1.042, 1.092, 1.100, 1.093]
new_vals = [1.008, 1.016, 1.014, 1.053, 1.067, 1.054]
x = np.arange(len(configs))
w = 0.35
ax.barh(x + w/2, old_vals, w, color=GRAY,   alpha=0.75, label="Small model (Exp 10, h64/L3)")
ax.barh(x - w/2, new_vals, w, color=ACCENT, alpha=0.85, label="Large model (Exp 11, h128/L6)")
ax.set_yticks(x)
ax.set_yticklabels(configs, fontsize=10)
ax.set_xlim(0.99, 1.12)
ax.set_xlabel("Execution-cost ratio (lower is better)", fontsize=11)
ax.set_title("Figure 3 — Cost Ratio: Small vs. Large Model at 8×8 (Exp 11)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
ax.axvline(1.0, color="black", lw=0.8, linestyle=":")
savefig("fig3_exp11.png")

# ── Figures 4 & 5: N×M heatmaps (Exp 11 diverse distribution) ───────────────
Ms = [2, 3, 4, 5, 6, 7, 8]
Ns = [2, 3, 4, 5]

# cost ratio (lower = better)
cost_ratio = np.array([
    [1.005, 1.008, 1.010, 1.018, 1.028, 1.035, 1.051],  # N=2
    [1.004, 1.005, 1.011, 1.024, 1.027, 1.027, 1.044],  # N=3
    [1.003, 1.002, 1.007, 1.015, 1.033, 1.039, 1.042],  # N=4
    [1.004, 1.008, 1.010, 1.016, 1.018, 1.035, 1.039],  # N=5
])

# exact-cost match % (higher = better)
exact_match = np.array([
    [99.0, 97.5, 93.5, 85.0, 80.5, 74.5, 66.5],  # N=2
    [99.0, 96.0, 93.5, 82.0, 82.5, 73.5, 65.0],  # N=3
    [99.0, 98.5, 93.5, 89.5, 79.5, 72.5, 63.5],  # N=4
    [98.5, 96.0, 93.0, 89.5, 83.0, 73.0, 63.5],  # N=5
])

def heatmap(data, cmap, title, fname, fmt, cbar_label, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(Ms)))
    ax.set_xticklabels([f"M={m}" for m in Ms], fontsize=10)
    ax.set_yticks(np.arange(len(Ns)))
    ax.set_yticklabels([f"N={n}" for n in Ns], fontsize=10)
    ax.set_title(title, fontsize=11)
    # annotate cells; pick text colour for contrast
    norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    for i in range(len(Ns)):
        for j in range(len(Ms)):
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                    fontsize=8.5,
                    color="white" if norm[i, j] > 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_xlabel("Goals (M)", fontsize=10)
    ax.set_ylabel("Agents (N)", fontsize=10)
    savefig(fname)

# cost ratio: reversed cmap so low (good) is green, high (bad) is red
heatmap(cost_ratio, "RdYlGn_r",
        "Figure 4 — Execution-Cost Ratio across N×M (Exp 11, diverse)",
        "fig4_heatmap_cost.png", "{:.3f}",
        "Cost ratio (lower better)", vmin=1.00, vmax=1.06)

# exact match: green = high (good)
heatmap(exact_match, "RdYlGn",
        "Figure 5 — Exact-Cost Match across N×M (Exp 11, diverse)",
        "fig5_heatmap_match.png", "{:.0f}%",
        "Exact match % (higher better)", vmin=60, vmax=100)

print("All figures generated.")
