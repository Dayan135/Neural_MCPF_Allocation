"""
Exp 16 training-convergence figure (final_report.tex appendix TODO, optional).
Only 4 of the 10 models have per-epoch logs delivered (report/data/exp16/train_logs/) —
Tier B's four specialists, pulled from the retrain this session watched directly. Tier A's
logs and Tier B's joint weren't pulled (see manifest.md). Run from report/ with mcpf_env active.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "report", "data", "exp16", "train_logs")
FIG_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_closeout")

ROLES = ["empty", "random", "maze", "room"]
COLOR = {"empty": "#EB6834", "random": "#1BAF7A", "maze": "#EDA100", "room": "#E34948"}

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.3))
for role in ROLES:
    path = os.path.join(LOGS, f"tierB_{role}_s0.txt")
    rows = list(csv.DictReader(open(path)))
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    ta = [float(r["train_pergoal_acc"]) for r in rows]
    va = [float(r["val_pergoal_acc"]) for r in rows]
    best_ep = vl.index(min(vl))
    c = COLOR[role]
    ax_loss.plot(ep, tl, "-", color=c, lw=1.4, alpha=0.55)
    ax_loss.plot(ep, vl, "--", color=c, lw=1.8, label=role)
    ax_loss.plot(ep[best_ep], vl[best_ep], "o", color=c, ms=7, mec="black", mew=0.8)
    ax_acc.plot(ep, ta, "-", color=c, lw=1.4, alpha=0.55)
    ax_acc.plot(ep, va, "--", color=c, lw=1.8, label=role)
    ax_acc.plot(ep[best_ep], va[best_ep], "o", color=c, ms=7, mec="black", mew=0.8)

for ax, ylabel, title in [(ax_loss, "Loss", "(a) Total loss"),
                           (ax_acc, "Per-goal accuracy", "(b) Per-goal accuracy")]:
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
ax_loss.legend(fontsize=8.5, loc="upper right", title="train (solid) / val (dashed)", title_fontsize=7.5)
fig.suptitle("Tier B (dataset A) specialists — training convergence, marked at the "
             "best-validation-loss checkpoint", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG_OUT, "fig_tierB_convergence.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Wrote {out}")

for role in ROLES:
    path = os.path.join(LOGS, f"tierB_{role}_s0.txt")
    rows = list(csv.DictReader(open(path)))
    vl = [float(r["val_loss"]) for r in rows]
    best = vl.index(min(vl)) + 1
    print(f"{role:8s} best val loss {min(vl):.4f} at epoch {best}/50 "
          f"(final epoch: {vl[-1]:.4f})")
