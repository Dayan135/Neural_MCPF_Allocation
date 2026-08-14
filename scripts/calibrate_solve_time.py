"""
Calibrate LKH solve time vs (N, M) on the 4 real benchmark maps.

Standalone (no torch dependency) — used to pick the N/M grids for the two new
training tiers: Tier A (~60s/solve) and Tier B (~10s/solve). See the plan for
context: main.py argparse below controls maps / (N,M) sweep / sample count.

Usage:
    python scripts/calibrate_solve_time.py \\
        --pairs 10,20 20,40 30,60 40,80 50,100 \\
        --n_instances 5 --seed 987654321 --out results/calibration.csv

Each (map, N, M) cell runs `n_instances` fresh solves and reports mean/median/
max wall time. A per-instance SIGALRM safety timeout aborts pathological
solves (mirrors evaluation/full_pipeline_eval.py's --instance_timeout); the
underlying LKH subprocess may be left running in the background when that
fires — harmless for a calibration run, but `pkill LKH` afterwards if you
want to be tidy.
"""

import argparse
import csv
import os
import signal
import statistics
import sys
import time

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_DATA_GEN = os.path.join(_ROOT, "dataset_generation")
for p in (_ROOT, _DATA_GEN):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_gen import load_map_file, sample_agents_goals
from solver_wrapper import run_basic_mapf

BENCHMARK_MAPS = ["empty-32-32.map", "random-32-32-20.map", "maze-32-32-2.map", "room-32-32-4.map"]


class _TimedOut(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimedOut()


def time_one_solve(map_dims, N, M, seed, cbs_node_budget, timeout_s, config_str):
    agents, goals = sample_agents_goals(map_dims, N, M, seed=seed)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_s))
    start = time.perf_counter()
    try:
        result = run_basic_mapf(map_dims, agents, goals,
                                 config_str=config_str, cbs_node_budget=cbs_node_budget)
        elapsed = time.perf_counter() - start
        return elapsed, result is not None, False
    except _TimedOut:
        elapsed = time.perf_counter() - start
        return elapsed, False, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=BENCHMARK_MAPS,
                     help="Map files (resolved against RobustMCPF/Maps/)")
    ap.add_argument("--pairs", nargs="+", required=True,
                     help="N,M pairs to test, e.g. 10,20 20,40 30,60")
    ap.add_argument("--n_instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=987654321)
    ap.add_argument("--cbs_node_budget", type=int, default=50_000,
                     help="Matches oracle.py's training-data-gen default")
    ap.add_argument("--timeout", type=float, default=180.0,
                     help="Per-instance SIGALRM safety cap (seconds)")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    pairs = []
    for p in args.pairs:
        n, m = p.split(",")
        pairs.append((int(n), int(m)))

    # Warm-up: first solver call pays one-time module-import / cold-start cost
    # (~seconds, not representative) — see full_pipeline_eval.py's warm_ref.
    warm_map = load_map_file(args.maps[0])
    time_one_solve(warm_map, 2, 2, args.seed, args.cbs_node_budget, args.timeout,
                    f"calib_warm_{os.getpid()}")

    rows = []
    for map_file in args.maps:
        map_dims = load_map_file(map_file)
        map_name = map_file.replace(".map", "")
        for (N, M) in pairs:
            times, n_fail, n_timeout = [], 0, 0
            for i in range(args.n_instances):
                config_str = f"calib_{map_name}_{N}_{M}_{i}_{os.getpid()}"
                elapsed, ok, timed_out = time_one_solve(
                    map_dims, N, M, args.seed + i, args.cbs_node_budget, args.timeout, config_str)
                times.append(elapsed)
                if timed_out:
                    n_timeout += 1
                elif not ok:
                    n_fail += 1
            mean_t = statistics.mean(times)
            median_t = statistics.median(times)
            max_t = max(times)
            print(f"{map_name:20s} N={N:3d} M={M:3d}  mean={mean_t:7.2f}s  "
                  f"median={median_t:7.2f}s  max={max_t:7.2f}s  "
                  f"fail={n_fail}/{args.n_instances}  timeout={n_timeout}/{args.n_instances}")
            rows.append({
                "map": map_name, "N": N, "M": M,
                "mean_s": round(mean_t, 3), "median_s": round(median_t, 3),
                "max_s": round(max_t, 3), "n_instances": args.n_instances,
                "n_fail": n_fail, "n_timeout": n_timeout,
            })

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
