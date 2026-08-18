"""
Exp 16 report tables/figures — Tier A (Dataset A leg) only.

Report convention (2026-08-18 rename): Dataset A = Tier A (large/sparse), Dataset B = Tier B
(regular/dense) — see gen_exp16_stage1.py's docstring for the full translation note against the
old on-disk convention. This script only builds the Stage-2 (specialist-vs-generalist, R3/R4) and
scale-extrapolation pieces, which are Tier-A-only by construction (RQ3 is reported for one leg;
see final_report.tex Section 6). Stage 1 (Dataset-A-vs-B, R2) needs both tiers and is built by
gen_exp16_stage1.py instead.

Reads the per-instance CSVs already reorganized into report/data/exp16/stage2_expert_vs_general/
(the tracked deliverable — not results/, which is git-ignored and cluster-pull-only).

Run from report/ with mcpf_env active: `python gen_exp16_tables.py`
"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP16 = os.path.join(ROOT, "report", "data", "exp16")
STAGE2 = os.path.join(EXP16, "stage2_expert_vs_general")
AGG_OUT = os.path.join(EXP16, "stage2_agg")
FIG_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_closeout")
os.makedirs(AGG_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
MAP_LABEL = {"empty-32-32": "empty", "random-32-32-20": "random",
             "maze-32-32-2": "maze", "room-32-32-4": "room"}
ROLES = ["joint", "empty", "random", "maze", "room"]
OWN_N, OWN_M = [60, 120, 180], [100, 225, 350]


def load_cell(role, map_name, n, m):
    path = os.path.join(STAGE2, f"tierA_{role}", f"{map_name}_n{n}m{m}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cn = np.array([float(r["cost_nn"]) for r in rows])
    cs = np.array([float(r["cost_solver"]) for r in rows])
    diff = cn - cs
    ratio = np.where(cs > 0, cn / cs, 1.0)
    alloc_ms = np.array([float(r["alloc_ms"]) for r in rows])
    nn_plan_ms = np.array([float(r["nn_plan_ms"]) for r in rows])
    solver_ms = np.array([float(r["solver_ms"]) for r in rows])
    return dict(n=len(rows), cost_ratio=ratio.mean(), exact_match=(cn == cs).mean(),
                diff_mean=diff.mean(), diff_max=diff.max(), diff_std=diff.std(),
                alloc_ms=alloc_ms.mean(), nn_plan_ms=nn_plan_ms.mean(),
                solver_ms=solver_ms.mean())


# ── Per-cell aggregates, written per role ───────────────────────────────────
FIELDS = ["map", "N", "M", "n", "cost_ratio", "exact_match", "diff_mean", "diff_max",
          "diff_std", "alloc_ms", "nn_plan_ms", "solver_ms"]
all_cells = {}  # (role, map) -> {(n,m): stats}
for role in ROLES:
    out_path = os.path.join(AGG_OUT, f"tierA_{role}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for mp in MAPS:
            for n in OWN_N:
                for m in OWN_M:
                    s = load_cell(role, mp, n, m)
                    if s is None:
                        continue
                    all_cells[(role, mp, n, m)] = s
                    writer.writerow({"map": mp, "N": n, "M": m, **s})
    print(f"Wrote {out_path}")


def role_map_avg(role, map_name):
    """Mean cost ratio / exact match / diff / timing / n across the own (N,M) grid."""
    ratios, exacts, diffs, allocs, nnplans, solvers, ninst = [], [], [], [], [], [], 0
    for n in OWN_N:
        for m in OWN_M:
            s = all_cells.get((role, map_name, n, m))
            if s is None:
                continue
            ratios.append(s["cost_ratio"])
            exacts.append(s["exact_match"])
            diffs.append(s["diff_mean"])
            allocs.append(s["alloc_ms"])
            nnplans.append(s["nn_plan_ms"])
            solvers.append(s["solver_ms"])
            ninst += s["n"]
    if not ratios:
        return None
    nn_ms = np.mean(allocs) + np.mean(nnplans)
    return dict(cost_ratio=np.mean(ratios), exact_match=np.mean(exacts), diff_mean=np.mean(diffs),
                nn_ms=nn_ms, solver_ms=np.mean(solvers), speedup=np.mean(solvers) / nn_ms,
                n=ninst, n_cells=len(ratios))


# ── 5x4 summary table (T3b/R3-R4 source) ────────────────────────────────────
summary_path = os.path.join(EXP16, "stage2_5x4_summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["role", "map", "cost_ratio", "exact_match", "diff_mean",
                                            "nn_ms", "solver_ms", "speedup", "n", "n_cells", "native"])
    writer.writeheader()
    for role in ROLES:
        for mp in MAPS:
            r = role_map_avg(role, mp)
            if r is None:
                continue
            native = (role != "joint" and MAP_LABEL[mp] == role)
            writer.writerow({"role": role, "map": MAP_LABEL[mp], "native": native, **r})
print(f"Wrote {summary_path}")

# ── Figure: 5x4 heatmap, same convention as fig9 (RdYlGn_r, capped at 1.5) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VMAX = 1.5
data = np.full((len(ROLES), len(MAPS)), np.nan)
ncell = np.full((len(ROLES), len(MAPS)), 0)
for i, role in enumerate(ROLES):
    for j, mp in enumerate(MAPS):
        r = role_map_avg(role, mp)
        if r:
            data[i, j] = r["cost_ratio"]
            ncell[i, j] = r["n_cells"]

fig, ax = plt.subplots(figsize=(7.2, 5.2))
im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=1.0, vmax=VMAX)
ax.set_xticks(np.arange(len(MAPS)))
ax.set_xticklabels([MAP_LABEL[m] for m in MAPS], fontsize=11)
ax.set_yticks(np.arange(len(ROLES)))
ax.set_yticklabels([r if r != "joint" else "joint (generalist)" for r in ROLES], fontsize=11)
ax.set_xlabel("Evaluated on map", fontsize=10.5)
ax.set_ylabel("Trained on", fontsize=10.5)
ax.set_title("Tier A (Dataset A) — specialist vs. generalist,\nexecution-cost ratio, own N,M grid",
             fontsize=11.5)
for i in range(len(ROLES)):
    for j in range(len(MAPS)):
        v = data[i, j]
        if np.isnan(v):
            continue
        is_native = (ROLES[i] != "joint" and MAP_LABEL[MAPS[j]] == ROLES[i])
        capped = min(v, VMAX)
        norm = (capped - 1.0) / (VMAX - 1.0)
        txt = f"{v:.3f}" + (" *" if is_native else "")
        if ncell[i, j] < 9:
            txt += f"\n(n_cells={ncell[i,j]})"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                color="white" if norm > 0.6 else "black",
                fontweight="bold" if is_native else "normal")
cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend="max")
cbar.set_label("Cost ratio (capped at 1.5)", fontsize=9.5)
fig.tight_layout()
out_fig = os.path.join(FIG_OUT, "fig_tierA_5x4_heatmap.png")
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote {out_fig}")

# ── Stage 0: solver scale wall (R1) ─────────────────────────────────────────
CALIB_RAW = os.path.join(EXP16, "solver_scale_wall_raw.csv")
if os.path.exists(CALIB_RAW):
    calib_rows = list(csv.DictReader(open(CALIB_RAW)))
    wall_path = os.path.join(EXP16, "solver_scale_wall.csv")
    with open(wall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["map", "N", "M", "n", "solver_ms_mean",
                                                "solver_ms_median", "solver_ms_p90", "timeout_frac"])
        writer.writeheader()
        for r in calib_rows:
            writer.writerow({
                "map": r["map"], "N": r["N"], "M": r["M"], "n": r["n_instances"],
                "solver_ms_mean": round(float(r["mean_s"]) * 1000, 1),
                "solver_ms_median": round(float(r["median_s"]) * 1000, 1),
                "solver_ms_p90": round(float(r["p90_s"]) * 1000, 1),
                "timeout_frac": round(int(r["n_timeout"]) / int(r["n_instances"]), 3),
            })
    print(f"Wrote {wall_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    CALIB_COLOR = {"empty-32-32": "#EB6834", "random-32-32-20": "#1BAF7A",
                   "maze-32-32-2": "#EDA100", "room-32-32-4": "#E34948"}
    for mp in MAPS:
        rows_m = [r for r in calib_rows if r["map"] == mp]
        rows_m.sort(key=lambda r: (int(r["N"]), int(r["M"])))
        xs = list(range(len(rows_m)))
        med = [float(r["median_s"]) for r in rows_m]
        p90 = [float(r["p90_s"]) for r in rows_m]
        ax.plot(xs, med, "-o", color=CALIB_COLOR[mp], lw=1.8, ms=4, label=MAP_LABEL[mp])
        ax.plot(xs, p90, "--", color=CALIB_COLOR[mp], lw=1, alpha=0.5)
    ax.axhline(60, color="black", lw=0.8, linestyle=":")
    ax.text(0, 62, "60s", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows_m)))
    ax.set_xticklabels([f"{r['N']},{r['M']}" for r in rows_m], rotation=45, fontsize=8, ha="right")
    ax.set_xlabel("(N, M)", fontsize=10)
    ax.set_ylabel("Solver wall-time, s (solid=median, dashed=p90)", fontsize=9.5)
    ax.set_title("Solver wall-time vs. problem size (local calibration, 3 instances/cell)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    calib_fig = os.path.join(FIG_OUT, "fig_solver_scale_wall.png")
    plt.savefig(calib_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {calib_fig}")

# ── Headline numbers for the report text (printed, not auto-inserted) ──────
diag = [role_map_avg(r, mp) for r in ROLES for mp in MAPS
        if (r == "joint") or (MAP_LABEL[mp] == r)]
diag = [d for d in diag if d]
offdiag = [role_map_avg(r, mp) for r in ROLES for mp in MAPS
           if r != "joint" and MAP_LABEL[mp] != r]
offdiag = [d for d in offdiag if d]
joint_only = [role_map_avg("joint", mp) for mp in MAPS]
joint_only = [d for d in joint_only if d]

# ── Appendix: full per-configuration LaTeX longtable ────────────────────────
tex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tierA_fullconfig_table.tex")
with open(tex_path, "w") as f:
    f.write("% Auto-generated by gen_exp16_tables.py. Do not edit by hand.\n")
    f.write("{\\small\n\\begin{longtable}{llccccccc}\n\\toprule\n")
    f.write("Model & Map & $N$ & $M$ & Ratio & Exact & Diff mean & Speedup & $n$ \\\\\n")
    f.write("\\midrule\n\\endhead\n")
    for role in ROLES:
        for mp in MAPS:
            for n in OWN_N:
                for m in OWN_M:
                    s = all_cells.get((role, mp, n, m))
                    if s is None:
                        continue
                    nn_ms = s["alloc_ms"] + s["nn_plan_ms"]
                    speedup = s["solver_ms"] / nn_ms if nn_ms > 0 else float("nan")
                    f.write(f"\\texttt{{{role}}} & {MAP_LABEL[mp]} & {n} & {m} & "
                            f"{s['cost_ratio']:.3f} & {s['exact_match']*100:.1f}\\% & "
                            f"{s['diff_mean']:.1f} & {speedup:.1f}$\\times$ & {s['n']} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\caption{Full per-configuration results, Dataset A leg (Tier A models, own $N,M$ "
            "grid). 174 of 180 possible (model, map, $N$, $M$) cells; missing cells were not yet "
            "finished on the cluster at pull time. $n$: instances evaluated in that one cell "
            "(out of a nominal 100).}\n")
    f.write("\\label{tab:fullconfig}\n\\end{longtable}\n}\n")
print(f"Wrote {tex_path}")

print("\n--- headline numbers (own N,M grid; Tier A / Dataset A only) ---")
print(f"Diagonal (specialist-on-own-map + joint-on-each-map) mean cost ratio: "
      f"{np.mean([d['cost_ratio'] for d in diag]):.3f}")
print(f"Off-diagonal (specialist-on-other-map) mean cost ratio: "
      f"{np.mean([d['cost_ratio'] for d in offdiag]):.3f}")
print(f"Joint (generalist) mean cost ratio, averaged per map then over maps: "
      f"{np.mean([d['cost_ratio'] for d in joint_only]):.3f}")
for mp in MAPS:
    j = role_map_avg("joint", mp)
    s = role_map_avg(MAP_LABEL[mp], mp)
    if j and s:
        print(f"  {MAP_LABEL[mp]:8s}: joint={j['cost_ratio']:.3f}  specialist={s['cost_ratio']:.3f}")
