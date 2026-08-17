"""
Exp 16 Stage 1 — dataset-A-vs-B head-to-head (RQ2), completing the TODO at
final_report.tex's R2 block. Reads report/data/exp16/stage1_dataset_selection/{role}_{A,B}/,
which holds each role's dataset-A (Tier B) and dataset-B (Tier A) checkpoint evaluated on a
common grid spanning both regimes (Tier B's own small grid + Tier A's own large grid — see
report/data/exp16/manifest.md for exactly which sweeps feed each quadrant).

Per docs/exp16_data_request.md §4: each role is compared on its own domain (a specialist on its
own map; the generalist on all 4 maps), selection nominally on validation but only test-time
full-pipeline CSVs exist here (same limitation as the rest of Exp 16 to date), so this is reported
as-is with that caveat.

Run from report/ with mcpf_env active: `python gen_exp16_stage1.py`
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP16 = os.path.join(ROOT, "report", "data", "exp16")
STAGE1 = os.path.join(EXP16, "stage1_dataset_selection")
AGG_OUT = os.path.join(EXP16, "stage1_agg")
FIG_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_closeout")
os.makedirs(AGG_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
MAP_LABEL = {"empty-32-32": "empty", "random-32-32-20": "random",
             "maze-32-32-2": "maze", "room-32-32-4": "room"}
ROLES = ["joint", "empty", "random", "maze", "room"]
OWN_DOMAIN = {"joint": MAPS, "empty": ["empty-32-32"], "random": ["random-32-32-20"],
              "maze": ["maze-32-32-2"], "room": ["room-32-32-4"]}

SMALL_N, SMALL_M = [30, 55, 80], [50, 100, 150]
LARGE_N, LARGE_M = [60, 120, 180], [100, 225, 350]


def load_cell(role, dataset, map_name, n, m):
    path = os.path.join(STAGE1, f"{role}_{dataset}", f"{map_name}_n{n}m{m}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cn = np.array([float(r["cost_nn"]) for r in rows])
    cs = np.array([float(r["cost_solver"]) for r in rows])
    ratio = np.where(cs > 0, cn / cs, 1.0)
    return dict(n=len(rows), cost_ratio=ratio.mean(), exact_match=(cn == cs).mean())


# ── Per-role winner: mean cost ratio, dataset A vs B, over the role's own domain,
#    unioned across BOTH grids (the actual common-grid comparison) — each dataset
#    is evaluated on BOTH the small grid (its native regime for A) and the large
#    grid (its native regime for B); this is the whole point of a crossover: both
#    checkpoints have to be scored on both regimes, not just their own turf. ────
winner_rows = []
for role in ROLES:
    doms = OWN_DOMAIN[role]
    ratios = {"A": [], "B": []}
    for ds in ["A", "B"]:
        for ns, ms in [(SMALL_N, SMALL_M), (LARGE_N, LARGE_M)]:
            for mp in doms:
                for n in ns:
                    for m in ms:
                        c = load_cell(role, ds, mp, n, m)
                        if c:
                            ratios[ds].append(c["cost_ratio"])
    if not ratios["A"] or not ratios["B"]:
        print(f"[skip] {role}: A={len(ratios['A'])} cells, B={len(ratios['B'])} cells")
        continue
    ma, mb = np.mean(ratios["A"]), np.mean(ratios["B"])
    winner = "A" if ma < mb else "B"
    winner_rows.append(dict(role=role, ratio_A=ma, ratio_B=mb, margin=abs(ma - mb),
                             winner=winner, n_A=len(ratios["A"]), n_B=len(ratios["B"])))
    print(f"{role:8s} A={ma:.4f} (n={len(ratios['A'])})  B={mb:.4f} (n={len(ratios['B'])})  "
          f"winner={winner}  margin={abs(ma-mb):.4f}")

with open(os.path.join(AGG_OUT, "stage1_winners.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["role", "ratio_A", "ratio_B", "margin", "winner", "n_A", "n_B"])
    w.writeheader()
    for r in winner_rows:
        w.writerow(r)

# ── Crossover figure: cost ratio vs M, two panels (small regime | large regime),
#    two lines per panel (dataset A, dataset B), averaged over each role's own
#    domain then over roles — mirrors fig8's own/extrapolated split so the two
#    grids' overlapping M-ranges don't produce a misleading merged x-axis. ─────
def by_m_all_roles(dataset, ns, ms):
    out = {}
    for m in ms:
        vals = []
        for role in ROLES:
            for mp in OWN_DOMAIN[role]:
                for n in ns:
                    c = load_cell(role, dataset, mp, n, m)
                    if c:
                        vals.append(c["cost_ratio"])
        if vals:
            out[m] = np.mean(vals)
    return out


COLOR_A, COLOR_B = "#1BAF7A", "#1F497D"
fig, (ax_small, ax_large) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, (ns, ms), title in [(ax_small, (SMALL_N, SMALL_M), "Small/regular regime (dataset A's own grid)"),
                             (ax_large, (LARGE_N, LARGE_M), "Large regime (dataset B's own grid)")]:
    a = by_m_all_roles("A", ns, ms)
    b = by_m_all_roles("B", ns, ms)
    ax.plot(sorted(a), [a[m] for m in sorted(a)], "-o", color=COLOR_A, lw=2, ms=6,
            label="Dataset A (Tier B, more data)")
    ax.plot(sorted(b), [b[m] for m in sorted(b)], "-o", color=COLOR_B, lw=2, ms=6,
            label="Dataset B (Tier A, less data)")
    ax.set_xlabel("Goals M", fontsize=10.5)
    ax.axhline(1.0, color="black", lw=0.8, linestyle=":")
    ax.set_title(title, fontsize=10.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
ax_small.set_ylabel("Execution-cost ratio (lower better)", fontsize=10.5)
ax_small.legend(fontsize=8.5, loc="upper left")
fig.suptitle("Stage 1 — Dataset A vs. B, mean over each role's own domain", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out_fig = os.path.join(FIG_OUT, "fig_stage1_crossover.png")
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote {out_fig}")

# ── LaTeX table snippet ──────────────────────────────────────────────────────
tex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_winner_table.tex")
with open(tex_path, "w") as f:
    f.write("% Auto-generated by gen_exp16_stage1.py. Do not edit by hand.\n")
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Stage 1: dataset A vs.\\ B, mean cost ratio over each role's own domain "
            "(own map for a specialist, all 4 maps for the generalist), unioned across both "
            "grids. Winner is the lower (better) mean.}\n")
    f.write("\\small\n\\begin{tabularx}{\\textwidth}{lXXXl}\n\\toprule\n")
    f.write("Role & Dataset A ratio & Dataset B ratio & Margin & Winner \\\\\n\\midrule\n")
    for r in winner_rows:
        role_disp = "generalist" if r["role"] == "joint" else r["role"]
        f.write(f"{role_disp} & {r['ratio_A']:.3f} & {r['ratio_B']:.3f} & "
                f"{r['margin']:.3f} & {'A' if r['winner']=='A' else 'B'} \\\\\n")
    f.write("\\bottomrule\n\\end{tabularx}\n\\label{tab:stage1winners}\n\\end{table}\n")
print(f"Wrote {tex_path}")
