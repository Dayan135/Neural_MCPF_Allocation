"""
Presentation demo: NN (model A) vs exact solver on a real benchmark map.

Produces three artifacts in presentation/assets/:
  demo_nn_vs_solver.png   — static side-by-side: each pipeline's allocation + CBS paths + cost
  demo_run.gif            — "watch the robots run": agents moving along the NN's collision-free paths
  demo_final_frame.png    — last frame of the GIF (PDF fallback)

Runs entirely locally (model A = checkpoints/large_s0/best.pt, LKH built). Picks a legible,
feasible instance (small N,M) where the NN is near-optimal, so the demo tells the success story.

Run: .venv/bin/python presentation/make_demo.py [--map room-32-32-4 --n 4 --m 6]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for p in (ROOT, os.path.join(ROOT, "dataset_generation"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from grid_gen import load_map_file, sample_agents_goals
from distance import compute_distance_matrix, compute_goal_distance_matrix, normalize_D
from solver_wrapper import run_basic_mapf, run_basic_mapf_with_allocation
from evaluate import load_model
from full_pipeline_eval import order_goals

OUT = os.path.join(ROOT, "presentation", "assets")
CKPT = os.path.join(ROOT, "checkpoints", "large_s0", "best.pt")
PALETTE = ["#1F497D", "#C0392B", "#198754", "#D08770", "#6f42c1", "#0070C0", "#b58900"]


def grid_array(md):
    return np.array(md["Map"], dtype=int).reshape(md["Rows"], md["Cols"])


def xy(loc, cols):
    return loc % cols, loc // cols  # (col=x, row=y)


def nn_pipeline(model, md, agents, goals, D_raw, G_raw, use_g, tag):
    cols = md["Cols"]; rows = md["Rows"]
    D = normalize_D(D_raw, cols, rows)
    Dt = torch.from_numpy(D).float()[None]
    Gt = torch.from_numpy(normalize_D(G_raw, cols, rows)).float()[None] if use_g else None
    with torch.no_grad():
        P = (model(Dt, G=Gt) if Gt is not None else model(Dt))[0].cpu().numpy()
    Y = np.zeros_like(P); Y[P.argmax(0), np.arange(P.shape[1])] = 1.0  # argmax per goal
    ordered = {}
    for a in range(len(agents)):
        assigned = list(np.where(Y[a] > 0.5)[0])
        ordered[a] = [goals[g] for g in order_goals(a, assigned, D_raw, G_raw)]
    res = run_basic_mapf_with_allocation(md, agents, goals, ordered, config_str=f"demo_nn_{tag}")
    return res


def alloc_owner(alloc, m):
    """goal_index -> agent_id tuple (so two allocations can be compared)."""
    owner = [-1] * m
    for a, gis in alloc.items():
        for gi in gis:
            owner[gi] = a
    return tuple(owner)


def find_instance(model, md, use_g, n, m, mode="tie_diff", max_path=70, ratio_cap=1.06, tries=160):
    """Search seeds for a legible, feasible instance matching the requested story.

    mode "tie_diff": NN cost == solver cost but the goal→agent allocation DIFFERS
                     (the cost-equivalent-tie story — visibly different panels).
    mode "exact"   : NN allocation identical to solver (both same paths/cost).
    mode "gap"     : NN slightly above optimal (1.0 < ratio <= ratio_cap).
    Falls back to the closest candidate if no exact match is found.
    """
    best = None
    for s in range(tries):
        try:
            agents, goals = sample_agents_goals(md, n, M=m, seed=1000 + s)
        except ValueError:
            continue
        D = compute_distance_matrix(md, agents, goals)
        if np.any(D == float("inf")):
            continue
        G = compute_goal_distance_matrix(md, goals)
        if np.any(G == float("inf")):
            continue
        sv = run_basic_mapf(md, agents, goals, config_str=f"demo_sv_{s}", cbs_node_budget=50000)
        if sv is None:
            continue
        nn = nn_pipeline(model, md, agents, goals, D, G, use_g, tag=str(s))
        if nn is None:
            continue
        ratio = nn["cost"] / sv["cost"] if sv["cost"] > 0 else 1.0
        maxlen = max(len(p) for p in nn["paths"].values())
        same_cost = (nn["cost"] == sv["cost"])
        diff_alloc = alloc_owner(nn["allocation"], m) != alloc_owner(sv["allocation"], m)
        cand = dict(seed=s, agents=agents, goals=goals, nn=nn, sv=sv, ratio=ratio,
                    maxlen=maxlen, same_cost=same_cost, diff_alloc=diff_alloc)
        if maxlen > max_path:
            pass
        elif mode == "tie_diff" and same_cost and diff_alloc:
            print(f"  chosen seed {s}: DIFFERENT allocation, SAME cost "
                  f"({nn['cost']}={sv['cost']}), max path {maxlen}")
            return cand
        elif mode == "exact" and same_cost and not diff_alloc:
            print(f"  chosen seed {s}: identical allocation, cost {nn['cost']}, max path {maxlen}")
            return cand
        elif mode == "gap" and 1.0 < ratio <= ratio_cap:
            print(f"  chosen seed {s}: NN {nn['cost']} vs solver {sv['cost']} "
                  f"(ratio {ratio:.3f}), max path {maxlen}")
            return cand
        # fallback scoring: prefer the requested story, then legibility
        if mode == "tie_diff":
            key = (0 if (same_cost and diff_alloc) else 1, 0 if same_cost else 1, ratio, maxlen)
        elif mode == "gap":
            key = (abs(ratio - 1.03), maxlen)
        else:
            key = (ratio, maxlen)
        if best is None or key < best["_key"]:
            cand["_key"] = key; best = cand
    print(f"  fallback seed {best['seed']}: cost {best['nn']['cost']} vs {best['sv']['cost']} "
          f"(ratio {best['ratio']:.3f}, diff_alloc={best['diff_alloc']}, max path {best['maxlen']})")
    return best


def draw_panel(ax, grid, agents, goals, result, title):
    cols = grid.shape[1]
    ax.imshow(grid, cmap="binary", origin="upper", interpolation="nearest", alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12)
    alloc = result["allocation"]            # {agent: [goal_idx,...]}
    goal_color = {}
    for a, gidxs in alloc.items():
        for gi in gidxs:
            goal_color[gi] = PALETTE[a % len(PALETTE)]
    # paths
    for a, path in result["paths"].items():
        c = PALETTE[a % len(PALETTE)]
        xs = [xy(loc, cols)[0] for loc, _ in path]
        ys = [xy(loc, cols)[1] for loc, _ in path]
        ax.plot(xs, ys, "-", color=c, linewidth=2.0, alpha=0.8)
    # goals
    for gi, g in enumerate(goals):
        gx, gy = xy(g, cols)
        ax.plot(gx, gy, "*", color=goal_color.get(gi, "#888888"), ms=15,
                markeredgecolor="white", markeredgewidth=0.6)
    # agent starts
    for a, (loc, _) in enumerate(agents):
        ax_, ay_ = xy(loc, cols)
        ax.plot(ax_, ay_, "s", color=PALETTE[a % len(PALETTE)], ms=12,
                markeredgecolor="white", markeredgewidth=0.8)


def side_by_side(grid, agents, goals, nn, sv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    draw_panel(axes[0], grid, agents, goals, sv,
               f"Exact solver\nexecution cost = {sv['cost']}")
    draw_panel(axes[1], grid, agents, goals, nn,
               f"Neural allocation (ours)\nexecution cost = {nn['cost']}")
    fig.suptitle("Same instance, real map — NN allocation vs exact solver\n"
                 "squares = agents · stars = goals (colored by assigned agent) · lines = collision-free paths",
                 fontsize=10.5, color="#1F497D")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, "demo_nn_vs_solver.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def _panel_artists(ax, grid, goals, result, label):
    """Draw static map+goals on ax and return animatable artists for one pipeline."""
    cols = grid.shape[1]
    ax.imshow(grid, cmap="binary", origin="upper", interpolation="nearest", alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontsize=11)
    goal_color = {gi: PALETTE[a % len(PALETTE)]
                  for a, gis in result["allocation"].items() for gi in gis}
    for gi, g in enumerate(goals):
        gx, gy = xy(g, cols)
        ax.plot(gx, gy, "*", color=goal_color.get(gi, "#888888"), ms=14,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    paths = result["paths"]
    pos = {a: [xy(loc, cols) for loc, _ in p] for a, p in paths.items()}
    trails = {a: ax.plot([], [], "-", color=PALETTE[a % len(PALETTE)], lw=2, alpha=0.7, zorder=2)[0]
              for a in paths}
    dots = {a: ax.plot([], [], "o", color=PALETTE[a % len(PALETTE)], ms=11,
                       markeredgecolor="white", markeredgewidth=0.8, zorder=4)[0]
            for a in paths}
    return dict(pos=pos, trails=trails, dots=dots)


def animate_panels(grid, goals, panels, out_name, suptitle):
    """Animate 1+ pipelines in lockstep. panels = [dict(result=, label=), ...]."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 6.4), squeeze=False)
    axes = axes[0]
    arts, T = [], 1
    for ax, panel in zip(axes, panels):
        a = _panel_artists(ax, grid, goals, panel["result"], panel["label"])
        arts.append(a)
        T = max(T, max(len(p) for p in a["pos"].values()))
    st = fig.suptitle("", fontsize=12, color="#1F497D")

    def update(t):
        for a in arts:
            for ag, p in a["pos"].items():
                pts = p[:t + 1]
                a["trails"][ag].set_data([q[0] for q in pts], [q[1] for q in pts])
                cx, cy = p[min(t, len(p) - 1)]
                a["dots"][ag].set_data([cx], [cy])
        st.set_text(f"{suptitle}   (step {min(t, T - 1) + 1}/{T})")
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, update, frames=T + 6, interval=180, blit=False)
    p = os.path.join(OUT, out_name)
    anim.save(p, writer=PillowWriter(fps=6)); plt.close(fig); print("wrote", p)


def final_frame(grid, agents, goals, nn, sv):
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    draw_panel(ax, grid, agents, goals, nn,
               f"NN allocation — collision-free execution\ncost {nn['cost']} (solver {sv['cost']})")
    fig.tight_layout()
    p = os.path.join(OUT, "demo_final_frame.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="room-32-32-4")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--mode", default="tie_diff", choices=["tie_diff", "exact", "gap"],
                    help="tie_diff: different allocation, same cost; exact: identical; gap: NN slightly above optimal")
    args = ap.parse_args()

    model = load_model(CKPT, torch.device("cpu"))
    use_g = torch.load(CKPT, map_location="cpu", weights_only=False)["args"].get("use_goal_dists", False)
    md = load_map_file(f"{args.map}.map")
    grid = grid_array(md)
    print(f"map {args.map} ({md['Rows']}x{md['Cols']}), N={args.n} M={args.m}, mode={args.mode}; searching...")
    inst = find_instance(model, md, use_g, args.n, args.m, mode=args.mode)
    agents, goals, nn, sv = inst["agents"], inst["goals"], inst["nn"], inst["sv"]
    side_by_side(grid, agents, goals, nn, sv)
    sv_label = f"Exact solver  (cost {sv['cost']})"
    nn_label = f"Neural allocation  (cost {nn['cost']})"
    # combined side-by-side (synchronized) — best for the slide
    animate_panels(grid, goals, [dict(result=sv, label=sv_label), dict(result=nn, label=nn_label)],
                   "demo_run_compare.gif", "Same instance: exact solver vs NN, running collision-free")
    # standalone panels (if you prefer to place two GIFs yourself)
    animate_panels(grid, goals, [dict(result=nn, label=nn_label)],
                   "demo_run.gif", "NN agents running collision-free paths")
    animate_panels(grid, goals, [dict(result=sv, label=sv_label)],
                   "demo_run_solver.gif", "Exact-solver agents running collision-free paths")
    final_frame(grid, agents, goals, nn, sv)
    print("\nDemo artifacts written to", OUT)


if __name__ == "__main__":
    main()
