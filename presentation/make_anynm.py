"""
Slide-9 visualization: how one universal net maps ANY (N, M) in and out.

Animated step-by-step GIF that reveals the pipeline stage by stage, with TWO instances
(small 2×2 and larger 4×6) flowing through the SAME stages — shape labels show how the tensors
differ in size while the weights stay shared.

Outputs (presentation/assets/):
  anynm_steps.gif         — animated walk-through (for the live talk / HTML)
  anynm_steps_final.png   — last frame (PDF fallback)

Run: .venv/bin/python presentation/make_anynm.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
NAVY, BLUE, RED, GREEN, GREY = "#1F497D", "#0070C0", "#C0392B", "#198754", "#6c757d"
LIGHT, LIGHTRED, LIGHTGREEN = "#EBF1FB", "#F8E3DF", "#DEEEE4"

# (box label, shape template) — shape filled per row with n, m
STAGES = [
    ("instance",            "{n} agents · {m} goals"),
    ("D , G\n(distances)",  "D {n}×{m} · G {m}×{m}"),
    ("embed\n(scalar → ℝ^d)", "{n}×{m} tokens"),
    ("row + column\nattention ×L", "{n}×{m}×d"),
    ("per-token logit\n(ℝ^d → 1)", "{n}×{m}"),
    ("column\nsoftmax",     "{n}×{m}"),
    ("argmax →\nallocation", "{n}×{m}"),
]
# zones: (first stage, last stage, fill, label color, label)
BANDS = [
    (0, 1, "#dfe3e8", GREY, "data prep (pre)"),
    (2, 5, "#dbe7f6", NAVY, "the network — shared weights  ·  N×M in → N×M out"),
    (6, 6, "#dcefe3", GREEN, "decode (post)"),
]
FILL = [LIGHTRED, LIGHTRED, LIGHT, LIGHT, LIGHT, LIGHT, LIGHTGREEN]
EDGE = [RED, RED, NAVY, NAVY, NAVY, NAVY, GREEN]
TITLES = [
    "Each instance has N agents and M goals",
    "Distances D (N×M) and G (M×M) — any size",
    "Embed each agent–goal pair → an N×M grid of tokens  (no positional embeddings)",
    "Row + column attention mixes them — works on any sequence length",
    "One shared linear head → an N×M logit grid",
    "Column softmax: for each goal, a distribution over agents",
    "Argmax per goal → the N×M allocation matrix",
]
ROWS = [("small instance", 2, 2, 3.2), ("larger instance", 4, 6, 1.1)]

NS = len(STAGES)
X0, BW, GAP, BH = 2.3, 1.95, 0.45, 1.25
XS = [X0 + i * (BW + GAP) for i in range(NS)]
XMAX = XS[-1] + BW + 0.3


def draw(ax, upto):
    ax.clear()
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 6.0); ax.axis("off")
    # 3 zones (always shown): data prep (pre) · the network · decode (post)
    for a, b, fc, col, lab in BANDS:
        x0 = XS[a] - 0.25; x1 = XS[b] + BW + 0.25
        ax.add_patch(Rectangle((x0, 0.7), x1 - x0, 3.9, facecolor=fc, edgecolor="none",
                               alpha=0.55, zorder=0))
        ax.text((x0 + x1) / 2, 4.78, lab, ha="center", va="center", fontsize=10.5,
                color=col, fontweight="bold")
    ax.text(XMAX / 2, 5.68, "One network, any N × M", ha="center", fontsize=15,
            color=NAVY, fontweight="bold")
    ax.text(XMAX / 2, 5.28, TITLES[min(upto, NS - 1)], ha="center", fontsize=11, color="#333")
    for label, n, m, y in ROWS:
        ax.text(0.12, y + BH / 2, label, ha="left", va="center", fontsize=9.5, color=GREY)
    for i in range(min(upto + 1, NS)):
        cur = (i == upto)
        for label, n, m, y in ROWS:
            lw = 2.6 if cur else 1.5
            ax.add_patch(FancyBboxPatch((XS[i], y), BW, BH,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=FILL[i], edgecolor=EDGE[i], linewidth=lw, zorder=2))
            ax.text(XS[i] + BW / 2, y + BH / 2 + 0.12, STAGES[i][0], ha="center", va="center",
                    fontsize=8.8, color="black", zorder=3)
            ax.text(XS[i] + BW / 2, y - 0.16, STAGES[i][1].format(n=n, m=m), ha="center",
                    va="center", fontsize=8.4, color=EDGE[i], fontweight="bold", zorder=3)
            if i > 0:
                ax.add_patch(FancyArrowPatch((XS[i - 1] + BW, y + BH / 2), (XS[i], y + BH / 2),
                             arrowstyle="-|>", mutation_scale=12, color=GREY, linewidth=1.4, zorder=2))
    ax.text(XMAX / 2, 0.45,
            "d = token width (128 / 256)      ·      L = number of attention blocks (6 / 8)",
            ha="center", fontsize=10, color="#333")
    ax.text(XMAX / 2, 0.13,
            "Same weights process both sizes — the token grid just grows with N×M; there is no fixed-size vector.",
            ha="center", fontsize=9.5, color=GREY, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    frames = NS + 4  # hold a few frames at the end

    def update(f):
        draw(ax, min(f, NS - 1))
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=850, blit=False)
    gif = os.path.join(OUT, "anynm_steps.gif")
    anim.save(gif, writer=PillowWriter(fps=1.3)); print("wrote", gif)
    # Per-step build frames (FIXED size — no bbox tight — so boxes don't shift between steps).
    # Used as consecutive slides advanced with the arrow keys.
    for k in range(NS):
        draw(ax, k)
        p = os.path.join(OUT, f"anynm_step{k + 1}.png")
        fig.savefig(p, dpi=150); print("wrote", p)
    draw(ax, NS - 1); fig.savefig(os.path.join(OUT, "anynm_steps_final.png"), dpi=150,
                                  bbox_inches="tight")
    plt.close(fig); print("wrote", os.path.join(OUT, "anynm_steps_final.png"))


if __name__ == "__main__":
    main()
