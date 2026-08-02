"""
Generate the schematic diagrams for the presentation (clean academic palette).
Writes PNGs to presentation/assets/.

Run: .venv/bin/python presentation/make_diagrams.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, RegularPolygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

NAVY = "#1F497D"
BLUE = "#0070C0"
RED = "#C0392B"
GREEN = "#198754"
GREY = "#6c757d"
LIGHT = "#EBF1FB"
LIGHTRED = "#F8E3DF"
LIGHTGREEN = "#DEEEE4"


def _box(ax, x, y, w, h, text, fc, ec, fs=11, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="black"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.8, color=color))


# ── 1. bundle vs split ───────────────────────────────────────────────────────
def bundle_vs_split():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, title, cost, mode in [(axes[0], "Split: one goal each", 14, "split"),
                                  (axes[1], "Bundle: one agent takes both", 7, "bundle")]:
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for gx in range(5):
            for gy in range(5):
                ax.add_patch(Rectangle((gx - 0.5, gy - 0.5), 1, 1, fill=False,
                                       edgecolor="#dddddd", linewidth=0.8))
        a1, a2 = (0, 0), (1, 0)
        g1, g2 = (3, 4), (4, 4)
        ax.plot(*a1, "s", color=NAVY, ms=16); ax.text(a1[0], a1[1]-0.0, "A1", color="white",
                 ha="center", va="center", fontsize=8, fontweight="bold")
        ax.plot(*a2, "s", color=NAVY, ms=16); ax.text(a2[0], a2[1], "A2", color="white",
                 ha="center", va="center", fontsize=8, fontweight="bold")
        ax.plot(*g1, "*", color=RED, ms=20); ax.plot(*g2, "*", color=RED, ms=20)
        ax.text(g1[0], g1[1]+0.55, "G1", color=RED, ha="center", fontsize=9)
        ax.text(g2[0], g2[1]+0.55, "G2", color=RED, ha="center", fontsize=9)
        if mode == "split":
            _arrow(ax, a1[0], a1[1], g1[0], g1[1], NAVY)
            _arrow(ax, a2[0], a2[1], g2[0], g2[1], BLUE)
        else:
            _arrow(ax, a1[0], a1[1], g1[0], g1[1], NAVY)
            _arrow(ax, g1[0], g1[1], g2[0], g2[1], NAVY)
        ax.set_title(f"{title}\ntotal cost = {cost}", fontsize=11,
                     color=(RED if cost == 14 else GREEN), fontweight="bold")
    fig.suptitle("mTSP, not 1-to-1 assignment: bundling nearby goals can be cheaper",
                 fontsize=12, color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "bundle_vs_split.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 2. solver vs NN pipeline ─────────────────────────────────────────────────
def pipeline_solver_vs_nn():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 22); ax.set_ylim(0, 7); ax.axis("off")
    # Solver row (top)
    ax.text(0.2, 6.2, "Exact solver", fontsize=12, color=NAVY, fontweight="bold")
    _box(ax, 0.5, 4.6, 3.0, 1.2, "Instance\n(agents, goals, map)", LIGHT, NAVY, 9)
    _box(ax, 4.6, 4.6, 3.4, 1.2, "LKH-TSP\nallocation", LIGHT, NAVY, 10, NAVY, True)
    _box(ax, 9.1, 4.6, 3.2, 1.2, "CBS\ncollision planning", LIGHT, NAVY, 10)
    _box(ax, 13.4, 4.6, 2.6, 1.2, "Paths\n(optimal)", LIGHT, NAVY, 9)
    for x1, x2 in [(3.5, 4.6), (8.0, 9.1), (12.3, 13.4)]:
        _arrow(ax, x1, 5.2, x2, 5.2, NAVY)
    # Ours row (bottom)
    ax.text(0.2, 2.6, "Ours (NN allocator)", fontsize=12, color=RED, fontweight="bold")
    _box(ax, 0.5, 1.0, 3.0, 1.2, "Instance\n(agents, goals, map)", LIGHT, NAVY, 9)
    _box(ax, 4.6, 1.0, 3.4, 1.2, "NN\nallocation", LIGHTRED, RED, 10, RED, True)
    _box(ax, 9.1, 1.0, 3.2, 1.2, "CBS\ncollision planning", LIGHT, NAVY, 10)
    _box(ax, 13.4, 1.0, 2.6, 1.2, "Paths\n(suboptimal)", LIGHT, NAVY, 9)
    for x1, x2 in [(3.5, 4.6), (8.0, 9.1), (12.3, 13.4)]:
        _arrow(ax, x1, 1.6, x2, 1.6, RED if x1 == 3.5 else NAVY)
    ax.text(16.4, 3.4, "Only the allocation\nbox changes;\nCBS is shared.", fontsize=10,
            color=GREY, ha="left", va="center", style="italic")
    fig.tight_layout()
    p = os.path.join(OUT, "pipeline_solver_vs_nn.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 3. D and G matrices ──────────────────────────────────────────────────────
def D_G_matrices():
    rng = np.random.default_rng(3)
    N, M = 3, 4
    D = rng.uniform(0.1, 0.9, (N, M))
    G = rng.uniform(0.1, 0.9, (M, M)); G = (G + G.T) / 2; np.fill_diagonal(G, 0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(D, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("D  (N×M): agent → goal", color=NAVY, fontsize=12)
    axes[0].set_xlabel("goals"); axes[0].set_ylabel("agents")
    axes[0].set_xticks(range(M)); axes[0].set_yticks(range(N))
    for i in range(N):
        for j in range(M):
            axes[0].text(j, i, f"{D[i,j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if D[i, j] > 0.5 else "black")
    im1 = axes[1].imshow(G, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("G  (M×M): goal → goal", color=RED, fontsize=12)
    axes[1].set_xlabel("goals"); axes[1].set_ylabel("goals")
    axes[1].set_xticks(range(M)); axes[1].set_yticks(range(M))
    for i in range(M):
        for j in range(M):
            axes[1].text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if G[i, j] > 0.5 else "black")
    fig.suptitle("Inputs are normalized BFS distances (walls respected)", color=NAVY, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "D_G_matrices.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 4. any-N,M transform pipeline (the key diagram) ──────────────────────────
def anyNM_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.4)); ax.set_xlim(0, 24); ax.set_ylim(0, 7); ax.axis("off")
    y = 3.0; h = 1.8
    _box(ax, 0.4, y, 3.2, h, "D (N×M)\nG (M×M)\ndistances", LIGHT, NAVY, 10)
    _box(ax, 4.4, y, 4.0, h, "per-pair tokens\nN×M grid of d-dim\n(no positional emb.)", LIGHTRED, RED, 9.5, RED, True)
    _box(ax, 9.2, y, 3.6, h, "L × [row attn\n+ col attn + FFN]", LIGHT, NAVY, 10)
    _box(ax, 13.6, y, 2.8, h, "logits\n(N×M)", LIGHT, NAVY, 10)
    _box(ax, 17.2, y, 3.0, h, "column softmax\n(per goal)", LIGHT, BLUE, 9.5)
    _box(ax, 21.0, y, 2.7, h, "argmax →\nallocation Y\n(N×M)", LIGHTGREEN, GREEN, 9.5, GREEN, True)
    for x1, x2 in [(3.6, 4.4), (8.4, 9.2), (12.8, 13.6), (16.4, 17.2), (20.2, 21.0)]:
        _arrow(ax, x1, y + h / 2, x2, y + h / 2, NAVY)
    ax.text(12, 1.7, "Same weights for any N, M — attention handles variable size; G summed over goals (sum-pool)",
            fontsize=10.5, color=GREY, ha="center", style="italic")
    ax.text(12, 6.2, "How one model maps any-size problem → any-size allocation",
            fontsize=12.5, color=NAVY, ha="center", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "anyNM_pipeline.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 5. params bar ────────────────────────────────────────────────────────────
def params_bar():
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["h128 / L6\n(small)", "h256 / L8\n(large)"], [1.2, 6.3],
                  color=[NAVY, RED], width=0.55)
    for b, v in zip(bars, [1.2, 6.3]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v}M", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("parameters (millions)"); ax.set_ylim(0, 7.2)
    ax.set_title("Two universal models, same recipe", color=NAVY, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.5, 0.05, "small wins in-distribution  ·  large wins far out-of-distribution",
             ha="center", fontsize=9.5, color=GREY, style="italic")
    p = os.path.join(OUT, "params_bar.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 6. decision table ────────────────────────────────────────────────────────
def decision_table():
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.3, 0.3, 4.6, 5.2, "", LIGHTGREEN, GREEN)
    _box(ax, 5.1, 0.3, 4.6, 5.2, "", "#eeeeee", GREY)
    ax.text(2.6, 5.0, "Use the NN", ha="center", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(7.4, 5.0, "Use the exact solver", ha="center", fontsize=14, color=GREY, fontweight="bold")
    nn = ["• large N, M (5.9–9.4× faster)", "• batched / high-throughput", "• unstructured maps",
          "• speed > exactness"]
    sv = ["• tiny instances (already fast)", "• exactness-critical", "• unseen structured maps",
          "• one-off / offline"]
    for i, t in enumerate(nn):
        ax.text(0.6, 4.2 - i * 0.9, t, fontsize=11, color="black", ha="left")
    for i, t in enumerate(sv):
        ax.text(5.4, 4.2 - i * 0.9, t, fontsize=11, color="black", ha="left")
    fig.suptitle("When to replace the solver's allocator", color=NAVY, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "decision_table.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 7. motivation: many agents, many goals (warehouse-style) ─────────────────
def motivation():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 8.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for gx in range(10):
        for gy in range(9):
            ax.add_patch(Rectangle((gx - 0.5, gy - 0.5), 1, 1, fill=False,
                                   edgecolor="#eeeeee", linewidth=0.6))
    # a few "shelves" (warehouse walls)
    for (sx, sy, sw, sh) in [(3, 2, 1, 4), (6, 3, 1, 4)]:
        ax.add_patch(Rectangle((sx - 0.5, sy - 0.5), sw, sh, facecolor="#cfd6df",
                               edgecolor=GREY, linewidth=1.0))
    agents = {"A1": (0, 0), "A2": (0, 8), "A3": (9, 0)}
    # goals assigned to each agent (color-coded)
    assign = {
        "A1": [(2, 1), (1, 3)],
        "A2": [(2, 7), (4, 8)],
        "A3": [(8, 3), (8, 6)],
    }
    colors = {"A1": NAVY, "A2": BLUE, "A3": RED}
    for a, (ax_, ay_) in agents.items():
        c = colors[a]
        ax.plot(ax_, ay_, "s", color=c, ms=18)
        ax.text(ax_, ay_, a, color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        for (gx, gy) in assign[a]:
            ax.plot(gx, gy, "*", color=c, ms=20)
            _arrow(ax, ax_, ay_, gx, gy, c)
    ax.set_title("Many agents, many goals — who goes where?\n(allocate goals to agents, minimize total travel)",
                 fontsize=12, color=NAVY)
    fig.tight_layout()
    p = os.path.join(OUT, "motivation.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 8. attention sketch (row + column on the agent×goal grid) ────────────────
def attention_sketch():
    N, M = 3, 4
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.set_xlim(-1.4, M + 0.4); ax.set_ylim(-1.4, N + 0.4)
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(N):
        for j in range(M):
            hi_row = (i == 1)
            hi_col = (j == 2)
            fc = LIGHT
            if hi_row and hi_col:
                fc = "#cfe0f3"
            elif hi_row:
                fc = LIGHTRED
            elif hi_col:
                fc = LIGHTGREEN
            ax.add_patch(FancyBboxPatch((j, N - 1 - i, ), 0.86, 0.86,
                         boxstyle="round,pad=0.01,rounding_size=0.06",
                         facecolor=fc, edgecolor=GREY, linewidth=1.0))
    ax.text(M / 2 - 0.07, N + 0.0, "goals  →", ha="center", fontsize=10, color=GREY)
    ax.text(-1.1, N / 2 - 0.5, "agents ↓", rotation=90, va="center", fontsize=10, color=GREY)
    # row attention arrow (agent 2 over its goals)
    ax.add_patch(FancyArrowPatch((0.1, N - 1 - 1 + 0.43), (M - 0.1, N - 1 - 1 + 0.43),
                 arrowstyle="<|-|>", mutation_scale=14, color=RED, linewidth=2))
    ax.text(M / 2, N - 1 - 1 - 0.35, "row attention:\nagent compares its goals", ha="center",
            fontsize=9.5, color=RED)
    # column attention arrow (goal 3 over agents)
    ax.add_patch(FancyArrowPatch((2 + 0.43, 0.1), (2 + 0.43, N - 0.1),
                 arrowstyle="<|-|>", mutation_scale=14, color=GREEN, linewidth=2))
    ax.text(M + 0.05, N / 2 - 0.5, "column attention:\ngoal compares\nits agents", ha="left",
            fontsize=9.5, color=GREEN)
    ax.set_title("Row + column attention on the agent×goal grid", fontsize=12, color=NAVY)
    fig.tight_layout()
    p = os.path.join(OUT, "attention_sketch.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 9. data → train → eval flow ──────────────────────────────────────────────
def method_flow():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 21); ax.set_ylim(0, 5); ax.axis("off")
    y, h = 1.4, 2.2
    _box(ax, 0.5, y, 6.0, h, "DATA\nplace agents/goals → run solver\n→ store D, G, Y (reject bad)",
         LIGHT, NAVY, 10, NAVY, True)
    _box(ax, 7.5, y, 5.5, h, "TRAIN\nimitate solver (CE + MinSum)\nmixed-size: one model, any N,M",
         LIGHTRED, RED, 10, RED, True)
    _box(ax, 14.0, y, 6.3, h, "EVAL\noffline accuracy +\ntrue CBS execution cost",
         LIGHTGREEN, GREEN, 10, GREEN, True)
    _arrow(ax, 6.5, y + h / 2, 7.5, y + h / 2, NAVY)
    _arrow(ax, 13.0, y + h / 2, 14.0, y + h / 2, NAVY)
    ax.set_title("From solver labels to a trained, evaluated allocator", fontsize=12, color=NAVY)
    fig.tight_layout()
    p = os.path.join(OUT, "method_flow.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 10. Exp 15 — zero-shot XL extrapolation (h128 vs h256 by N) ──────────────
def xl_extrapolation():
    Ns = [20, 35, 50]
    h128 = [1.245, 1.246, 1.242]   # mean cost ratio by N (RESULTS.md Exp 15)
    h256 = [1.229, 1.209, 1.185]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(Ns)); w = 0.36
    ax.bar(x - w/2, [v - 1.0 for v in h128], w, bottom=1.0, color=NAVY, label="h128 (small)")
    ax.bar(x + w/2, [v - 1.0 for v in h256], w, bottom=1.0, color=RED, label="h256 (large)")
    for xi, v in zip(x - w/2, h128):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    for xi, v in zip(x + w/2, h256):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.6, label="optimal")
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("mean execution-cost ratio (NN / solver)")
    ax.set_ylim(1.0, 1.36)
    ax.set_title("Exp 15: zero-shot XL (M≤100) — the larger model wins at scale", color=NAVY, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="upper right", framealpha=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.04, "zero-shot, no retraining  ·  inference 5.9–9.4× faster than the solver",
             ha="center", fontsize=9.5, color=GREY, style="italic")
    p = os.path.join(OUT, "xl_extrapolation.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 11. Exp 15 detail — per-(N,M) cost-ratio heatmaps (h128, h256) + speedup ──
def xl_detail():
    Ms = [50, 75, 100]; Ns = [20, 35, 50]
    h128 = np.array([[1.204, 1.283, 1.248], [1.217, 1.288, 1.233], [1.221, 1.289, 1.216]])
    h256 = np.array([[1.211, 1.231, 1.246], [1.188, 1.211, 1.227], [1.159, 1.184, 1.211]])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    vmin, vmax = 1.15, 1.29
    im = None
    for ax, data, name in [(axes[0], h128, "h128 (small)"), (axes[1], h256, "h256 (large)")]:
        im = ax.imshow(data, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"M={m}" for m in Ms])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"N={n}" for n in Ns])
        ax.set_title(f"cost ratio — {name}", fontsize=11, color=NAVY)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=[axes[0], axes[1]], fraction=0.045, pad=0.02, label="NN / solver")
    sp = axes[2]
    sp.bar(["h128", "h256"], [9.4, 5.9], color=[NAVY, RED], width=0.55)
    for i, v in enumerate([9.4, 5.9]):
        sp.text(i, v + 0.15, f"{v}×", ha="center", fontsize=12, fontweight="bold")
    sp.set_ylim(0, 11); sp.set_ylabel("mean speedup (× vs solver)")
    sp.set_title("speed", fontsize=11, color=NAVY)
    sp.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exp 15 detail: cost ratio across N×M (green = closer to optimal) + speedup",
                 fontsize=12.5, color=NAVY)
    p = os.path.join(OUT, "xl_heatmaps_speedup.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


# ── 12. Model inventory table (A / B1 / B2 / C1 / C2) ────────────────────────
def model_inventory():
    cols = ["Model", "Net (params)", "Training data", "Scale", "Train size"]
    rows = [
        ["A (old)", "h128/L6 — 1.2M", "random grids 8–12, walls 0.1–0.5", "N ≤ 5 · M ≤ 8", "≈840k"],
        ["B1", "h128/L6 — 1.2M", "the 4 real benchmark maps (32×32)", "N 5–15 · M 10–30", "≈720k"],
        ["B2", "h256/L8 — 6.3M", "the 4 real benchmark maps (32×32)", "N 5–15 · M 10–30", "≈720k"],
        ["C1", "h128/L6 — 1.2M", "random 32×32 grids, walls 0–50%", "N 5–15 · M 10–30", "≈720k"],
        ["C2", "h256/L8 — 6.3M", "random 32×32 grids, walls 0–50%", "N 5–15 · M 10–30", "≈720k"],
    ]
    rowbg = ["#eeeeee", LIGHT, LIGHT, LIGHTRED, LIGHTRED]
    modelcol = [GREY, NAVY, NAVY, RED, RED]
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left",
                   colWidths=[0.10, 0.18, 0.34, 0.22, 0.12])
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.0)
    for j in range(len(cols)):
        c = tbl[0, j]; c.set_facecolor(NAVY)
        c.set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(rowbg[i])
            cell.set_edgecolor("white")
            if j == 0:
                cell.set_text_props(color=modelcol[i], fontweight="bold")
    ax.set_title("Our models — architecture × training data", color=NAVY, fontsize=13, pad=12)
    fig.tight_layout()
    p = os.path.join(OUT, "model_inventory.png"); fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    model_inventory()
    xl_detail()
    bundle_vs_split()
    pipeline_solver_vs_nn()
    D_G_matrices()
    anyNM_pipeline()
    params_bar()
    decision_table()
    motivation()
    attention_sketch()
    method_flow()
    xl_extrapolation()
    print("\nAll diagrams written to", OUT)
