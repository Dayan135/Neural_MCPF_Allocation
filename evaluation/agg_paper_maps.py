"""
Aggregate Exp 12 full-pipeline CSVs (results/fullpipe_paper/{current,larger}/*.csv)
into the comparison tables: per-config and per-model execution-cost stats for the
h128/L6 vs h256/L8 paper-maps models.

Columns in each CSV (written by full_pipeline_eval.py):
  inst_seed, cost_nn, cost_solver, nn_k, solver_k,
  conflicts_nn, conflicts_solver, alloc_ms, nn_plan_ms, solver_ms

Usage:
    python agg_paper_maps.py [--base results/fullpipe_paper]
"""

import argparse
import csv
import glob
import os

import numpy as np

MAPS = ["empty-32-32", "random-32-32-20", "maze-32-32-2", "room-32-32-4"]
NS = [5, 10, 15]
MS = [10, 20, 30]


def load_group(base, group):
    out = {}
    for f in sorted(glob.glob(os.path.join(base, group, "*.csv"))):
        name = os.path.basename(f)[:-4]
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        cn = np.array([float(r["cost_nn"]) for r in rows])
        cs = np.array([float(r["cost_solver"]) for r in rows])
        nn_ms = np.array([float(r["alloc_ms"]) + float(r["nn_plan_ms"]) for r in rows])
        sv_ms = np.array([float(r["solver_ms"]) for r in rows])
        diff = cn - cs
        ratio = np.where(cs > 0, cn / cs, 1.0)
        out[name] = dict(
            n=len(rows), ratio=ratio.mean(), exact=(cn == cs).mean(),
            dmean=diff.mean(), dmax=diff.max(), dstd=diff.std(),
            speedup=sv_ms.mean() / nn_ms.mean(),
        )
    return out


def agg(o):
    keys = ["ratio", "exact", "dmean", "dstd", "speedup"]
    m = {k: np.mean([v[k] for v in o.values()]) for k in keys}
    m["dmax"] = np.max([v["dmax"] for v in o.values()])
    m["N"] = sum(v["n"] for v in o.values())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/fullpipe_paper")
    ap.add_argument("--ns", default="5,10,15", help="comma-separated N values")
    ap.add_argument("--ms", default="10,20,30", help="comma-separated M values")
    ap.add_argument("--markdown", action="store_true",
                    help="emit per-config tables as GitHub-flavored markdown "
                         "(for pasting into RESULTS.md)")
    args = ap.parse_args()

    global NS, MS
    NS = [int(x) for x in args.ns.split(",")]
    MS = [int(x) for x in args.ms.split(",")]

    cur = load_group(args.base, "current")
    lrg = load_group(args.base, "larger")

    if args.markdown:
        for label, o in [("h128/L6 (current)", cur), ("h256/L8 (larger)", lrg)]:
            print(f"\n**{label} — per-config ({len(o)} configs):**\n")
            print("| Config | n | Cost ratio | Exact | Diff mean | Diff max | Diff std | Speedup |")
            print("|--------|---|-----------|-------|-----------|----------|----------|---------|")
            for mp in MAPS:
                for N in NS:
                    for M in MS:
                        k = f"{mp}_n{N}m{M}"
                        if k not in o:
                            continue
                        v = o[k]
                        cfg = f"{mp} n{N}m{M}"
                        print(f"| {cfg} | {v['n']} | {v['ratio']:.4f} | {v['exact']:.3f} "
                              f"| {v['dmean']:.2f} | {v['dmax']:.0f} | {v['dstd']:.2f} "
                              f"| {v['speedup']:.2f}× |")
        return

    for label, o in [("h128/L6 (current)", cur), ("h256/L8 (larger)", lrg)]:
        print(f"\n=== {label}: per-config ===")
        print(f"{'config':<26}{'n':>5}{'ratio':>9}{'exact':>8}{'dmean':>8}"
              f"{'dmax':>7}{'dstd':>7}{'speedup':>9}")
        for mp in MAPS:
            for N in NS:
                for M in MS:
                    k = f"{mp}_n{N}m{M}"
                    if k not in o:
                        continue
                    v = o[k]
                    print(f"{k:<26}{v['n']:>5}{v['ratio']:>9.4f}{v['exact']:>8.3f}"
                          f"{v['dmean']:>8.2f}{v['dmax']:>7.0f}{v['dstd']:>7.2f}"
                          f"{v['speedup']:>8.2f}x")

    print("\n=== AGGREGATE (mean over configs) ===")
    print(f"{'model':<20}{'ratio':>9}{'exact':>8}{'dmean':>8}{'dmax':>7}"
          f"{'dstd':>7}{'speedup':>9}{'Ninst':>9}")
    for label, o in [("h128/L6 current", cur), ("h256/L8 larger", lrg)]:
        m = agg(o)
        print(f"{label:<20}{m['ratio']:>9.4f}{m['exact']:>8.3f}{m['dmean']:>8.3f}"
              f"{m['dmax']:>7.0f}{m['dstd']:>7.3f}{m['speedup']:>8.2f}x{m['N']:>9}")

    print("\n=== cost ratio by map ===")
    for mp in MAPS:
        rc = np.mean([v["ratio"] for k, v in cur.items() if k.startswith(mp + "_")])
        rl = np.mean([v["ratio"] for k, v in lrg.items() if k.startswith(mp + "_")])
        print(f"  {mp:<18} current={rc:.4f}  larger={rl:.4f}")

    print("\n=== cost ratio / exact by (N,M) ===")
    for N in NS:
        for M in MS:
            suf = f"n{N}m{M}"
            rc = np.mean([v["ratio"] for k, v in cur.items() if k.endswith(suf)])
            rl = np.mean([v["ratio"] for k, v in lrg.items() if k.endswith(suf)])
            ec = np.mean([v["exact"] for k, v in cur.items() if k.endswith(suf)])
            el = np.mean([v["exact"] for k, v in lrg.items() if k.endswith(suf)])
            print(f"  N={N:<2} M={M:<2}  ratio cur={rc:.4f} lrg={rl:.4f}"
                  f"  | exact cur={ec:.3f} lrg={el:.3f}")


if __name__ == "__main__":
    main()
