"""
Aggregate results across multiple experiment runs and print a summary table.

Usage:
    python compare.py --checkpoint_dir ../checkpoints --pattern "arch_*"
                      [--data_dir ../data] [--split test]

Reads every best.pt under matching subdirectories, evaluates on the given
split, and prints mean ± std grouped by model_type (and optionally N).
"""

import argparse
import fnmatch
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluate import load_model, offline_metrics


def find_checkpoints(checkpoint_dir: str, pattern: str) -> list[str]:
    matches = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if fnmatch.fnmatch(name, pattern):
            path = os.path.join(checkpoint_dir, name, "best.pt")
            if os.path.isfile(path):
                matches.append(path)
    return matches


def group_key(run_name: str) -> str:
    """Group runs by name with the seed suffix stripped, so e.g.
    datascale_t1000_s0/s1/s2 aggregate together but t1000 vs t5000 stay apart."""
    return re.sub(r"_s\d+$", "", run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=os.path.join("..", "checkpoints"))
    parser.add_argument("--pattern", default="*", help="glob pattern for run subdirs")
    parser.add_argument("--data_dir", default=os.path.join("..", "data"),
                        help="root data dir; overridden per-run if checkpoint stores a data_dir")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = find_checkpoints(args.checkpoint_dir, args.pattern)

    if not checkpoints:
        print(f"No checkpoints found matching '{args.pattern}' in {args.checkpoint_dir}")
        return

    groups: dict[str, list[dict]] = defaultdict(list)

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run_args = ckpt["args"]

        # Data dir: use per-run data_dir if it exists, else fall back to CLI arg
        data_dir = run_args.get("data_dir", args.data_dir)
        split_dir = os.path.join(data_dir, args.split)
        if not os.path.isdir(split_dir):
            data_dir = args.data_dir
            split_dir = os.path.join(data_dir, args.split)

        try:
            D_all = np.load(os.path.join(split_dir, "D_matrices.npy"))
            Y_all = np.load(os.path.join(split_dir, "Y_matrices.npy"))
        except FileNotFoundError:
            print(f"  [skip] {ckpt_path}: data not found at {split_dir}")
            continue

        # Size-agnostic models (transformer) would silently run on data of the
        # wrong problem size — reject any N/M mismatch instead of reporting it.
        N, M = run_args["N"], run_args.get("M", run_args["N"])
        if D_all.shape[1:] != (N, M):
            print(f"  [skip] {ckpt_path}: data at {split_dir} has shape "
                  f"{D_all.shape[1:]}, model expects ({N}, {M})")
            continue

        model = load_model(ckpt_path, device)
        metrics = offline_metrics(model, D_all, Y_all, device)
        metrics["run"] = os.path.basename(os.path.dirname(ckpt_path))
        groups[group_key(metrics["run"])].append(metrics)
        print(f"  {metrics['run']}: per_goal={metrics['per_goal_accuracy']:.3f}  "
              f"full={metrics['full_assignment_accuracy']:.3f}  "
              f"cost_ratio={metrics['mean_cost_ratio']:.4f}")

    print(f"\n{'Group':<30} {'N':>3}  {'per_goal':>10}  {'full_assign':>12}  {'cost_ratio':>11}")
    print("-" * 72)
    for key in sorted(groups):
        runs = groups[key]
        pg = np.array([r["per_goal_accuracy"] for r in runs])
        fa = np.array([r["full_assignment_accuracy"] for r in runs])
        cr = np.array([r["mean_cost_ratio"] for r in runs])
        n_runs = len(runs)
        print(
            f"{key:<30} {n_runs:>3}  "
            f"{pg.mean():.3f}±{pg.std():.3f}  "
            f"{fa.mean():.3f}±{fa.std():.3f}  "
            f"{cr.mean():.4f}±{cr.std():.4f}"
        )


if __name__ == "__main__":
    main()
