"""Parse train.py .out logs and plot loss/accuracy convergence (Exp 11, large model)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re, os, glob

OUT = os.path.dirname(__file__)
PRIMARY, ACCENT, SUCCESS, GRAY = "#1F497D", "#0070C0", "#198754", "#888888"

# Epoch    1/150 | train loss 0.3850 (ce 0.3592 ms 0.2581) acc 0.851 | val loss 0.3532 (ce 0.3276 ms 0.2557) acc 0.884
PAT = re.compile(
    r"Epoch\s+(\d+)/\d+ \| train loss ([\d.]+) \(ce ([\d.]+) ms ([\d.]+)\) acc ([\d.]+)"
    r" \| val loss ([\d.]+) \(ce ([\d.]+) ms ([\d.]+)\) acc ([\d.]+)"
)

def parse(path):
    ep, trl, trc, trm, tra, vl, vc, vm, va = ([] for _ in range(9))
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            g = m.groups()
            ep.append(int(g[0]))
            trl.append(float(g[1])); trc.append(float(g[2])); trm.append(float(g[3])); tra.append(float(g[4]))
            vl.append(float(g[5]));  vc.append(float(g[6]));  vm.append(float(g[7]));  va.append(float(g[8]))
    return {k: np.array(v) for k, v in
            zip(("ep","trl","trc","trm","tra","vl","vc","vm","va"),
                (ep,trl,trc,trm,tra,vl,vc,vm,va))}

runs = [parse(p) for p in sorted(glob.glob(os.path.join(OUT, "train_logs", "large_s*.txt")))]
assert runs, "no train_logs/large_s*.txt found"
ep = runs[0]["ep"]

def stack(key):
    a = np.vstack([r[key] for r in runs])
    return a.mean(axis=0), a.std(axis=0)

trl_m, trl_s = stack("trl"); vl_m, vl_s = stack("vl")
tra_m, tra_s = stack("tra"); va_m, va_s = stack("va")

# best val-loss epoch (mean curve) — where the checkpoint is taken
best_ep = int(ep[np.argmin(vl_m)])
best_vl = float(vl_m.min())

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- Loss panel ----
axL.plot(ep, trl_m, color=PRIMARY, lw=1.8, label="Train loss")
axL.fill_between(ep, trl_m-trl_s, trl_m+trl_s, color=PRIMARY, alpha=0.15)
axL.plot(ep, vl_m, color=ACCENT, lw=1.8, ls="--", label="Val loss")
axL.fill_between(ep, vl_m-vl_s, vl_m+vl_s, color=ACCENT, alpha=0.15)
axL.axvline(best_ep, color=GRAY, lw=1.0, ls=":")
axL.annotate(f"best val {best_vl:.3f}\n@ epoch {best_ep}",
             xy=(best_ep, best_vl), xytext=(best_ep+12, best_vl+0.18),
             fontsize=8.5, color="black",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
axL.set_xlabel("Epoch"); axL.set_ylabel("Total loss (CE + λ·MinSum)")
axL.set_title("(a) Loss convergence")
axL.grid(True, ls="--", alpha=0.4); axL.set_axisbelow(True)
axL.spines[["top","right"]].set_visible(False)
axL.legend(fontsize=9, loc="upper right")

# ---- Accuracy panel ----
axR.plot(ep, tra_m, color=PRIMARY, lw=1.8, label="Train acc")
axR.fill_between(ep, tra_m-tra_s, tra_m+tra_s, color=PRIMARY, alpha=0.15)
axR.plot(ep, va_m, color=SUCCESS, lw=1.8, ls="--", label="Val acc")
axR.fill_between(ep, va_m-va_s, va_m+va_s, color=SUCCESS, alpha=0.15)
axR.axvline(best_ep, color=GRAY, lw=1.0, ls=":")
axR.set_xlabel("Epoch"); axR.set_ylabel("Per-goal accuracy")
axR.set_title("(b) Per-goal accuracy")
axR.grid(True, ls="--", alpha=0.4); axR.set_axisbelow(True)
axR.spines[["top","right"]].set_visible(False)
axR.legend(fontsize=9, loc="lower right")

fig.suptitle("Figure 6 — Training Convergence, large model h128/L6 (mean of 3 seeds, ±1 std)",
             fontsize=11, y=1.00)
fig.tight_layout()
path = os.path.join(OUT, "fig6_convergence.png")
fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved {path}")
print(f"  best val loss {best_vl:.4f} @ epoch {best_ep}")
print(f"  final train loss {trl_m[-1]:.4f}  val loss {vl_m[-1]:.4f}  "
      f"(overfit gap {vl_m[-1]-trl_m[-1]:.3f})")
print(f"  final train acc {tra_m[-1]:.3f}  val acc {va_m[-1]:.3f}")
