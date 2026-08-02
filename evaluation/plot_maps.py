"""
Render the 4 RobustMCPF benchmark maps as images (for reports / presentation).

Reads RobustMCPF/Maps/<name>.map (MovingAI octile: '.'=free, '@'/'T'=wall) and writes
one PNG per map plus a combined 2x2 panel to report/maps/.

Run: .venv/bin/python evaluation/plot_maps.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAPS_DIR = os.path.join(ROOT, "RobustMCPF", "Maps")
OUT_DIR = os.path.join(ROOT, "report", "maps")
os.makedirs(OUT_DIR, exist_ok=True)

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
TITLES = {
    "empty-32-32": "empty-32-32 (open)",
    "random-32-32-20": "random-32-32-20 (scattered)",
    "maze-32-32-2": "maze-32-32-2 (corridors)",
    "room-32-32-4": "room-32-32-4 (rooms + doors)",
}


def load_grid(name):
    """Return HxW array: 0=free, 1=wall."""
    path = os.path.join(MAPS_DIR, f"{name}.map")
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("map") + 1
    body = [ln for ln in lines[start:] if ln.strip() != ""]
    grid = np.array([[0 if ch == "." else 1 for ch in row] for row in body], dtype=int)
    return grid


def wall_pct(g):
    return 100.0 * g.sum() / g.size


def save_one(name):
    g = load_grid(name)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(g, cmap="binary", interpolation="nearest")  # 0=white free, 1=black wall
    ax.set_title(f"{TITLES[name]}\n{g.shape[0]}x{g.shape[1]}, {wall_pct(g):.0f}% walls", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"map_{name}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}  ({g.shape[0]}x{g.shape[1]}, {wall_pct(g):.1f}% walls)")


def save_panel():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.5))
    for ax, name in zip(axes.ravel(), MAPS):
        g = load_grid(name)
        ax.imshow(g, cmap="binary", interpolation="nearest")
        ax.set_title(f"{TITLES[name]}  ({wall_pct(g):.0f}% walls)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The 4 RobustMCPF benchmark maps (32x32)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "maps_panel.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    for m in MAPS:
        save_one(m)
    save_panel()
    print("\nAll map images written to", OUT_DIR)
