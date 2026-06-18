"""
Compare random-trained (C) vs fixed-map-trained (B) models on the 4 real maps (Exp 14).

Reads the per-instance full-pipeline CSVs both model families wrote:
  B (fixed maps):     results/fullpipe_paper/{current,larger}/<map>_n{N}m{M}.csv   (Exp 12)
  C (random-diverse): results/fullpipe_random/{current,larger}/<map>_n{N}m{M}.csv (Exp 14)

Prints, per model size (current=h128, larger=h256): per-config B-vs-C cost ratio /
exact-match / speedup, and the mean over the shared configs, with a verdict line.
Reuses agg_paper_maps.load_group for parsing.

Usage:
    python evaluation/agg_compare.py
    python evaluation/agg_compare.py --fixed_base results/fullpipe_paper \\
                                     --random_base results/fullpipe_random
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_paper_maps import load_group  # noqa: E402


def compare_group(fixed_base, random_base, group):
    b = load_group(fixed_base, group)
    c = load_group(random_base, group)
    common = sorted(set(b) & set(c))
    size = "h128" if group == "current" else "h256" if group == "larger" else group
    print(f"\n===== {group} ({size}) — {len(common)} shared configs =====")
    if not common:
        print("  (no shared configs found — check the base dirs)")
        return
    print(f"{'config':<26}{'B ratio':>9}{'C ratio':>9}{'Δratio':>8}"
          f"{'B exact':>9}{'C exact':>9}{'B spd':>8}{'C spd':>8}")
    db, dc = [], []
    for name in common:
        rb, rc = b[name], c[name]
        dratio = rc["ratio"] - rb["ratio"]
        db.append(rb)
        dc.append(rc)
        print(f"{name:<26}{rb['ratio']:>9.3f}{rc['ratio']:>9.3f}{dratio:>+8.3f}"
              f"{rb['exact']:>9.1%}{rc['exact']:>9.1%}{rb['speedup']:>7.1f}x{rc['speedup']:>7.1f}x")
    mb_ratio = np.mean([x["ratio"] for x in db])
    mc_ratio = np.mean([x["ratio"] for x in dc])
    mb_exact = np.mean([x["exact"] for x in db])
    mc_exact = np.mean([x["exact"] for x in dc])
    print(f"{'MEAN':<26}{mb_ratio:>9.3f}{mc_ratio:>9.3f}{mc_ratio - mb_ratio:>+8.3f}"
          f"{mb_exact:>9.1%}{mc_exact:>9.1%}")
    verdict = "C (random) better" if mc_ratio < mb_ratio else "B (fixed) better"
    print(f"  verdict: {verdict} on mean cost ratio "
          f"(lower is better; B={mb_ratio:.4f}, C={mc_ratio:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_base", default="results/fullpipe_paper")
    ap.add_argument("--random_base", default="results/fullpipe_random")
    args = ap.parse_args()
    for group in ("current", "larger"):
        compare_group(args.fixed_base, args.random_base, group)


if __name__ == "__main__":
    main()
