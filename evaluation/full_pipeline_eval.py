"""
Full MAPF execution-cost evaluation: NN allocation vs combinatorial solver.

Per instance (K×K grid, N agents, M goals):
  1. NN forward → hard allocation → goal visit order (brute-force on BFS dists)
  2. Inject allocation into RobustMCPF CBS (no LKH) → collision-free paths → cost_nn
  3. Full solver (LKH + CBS) → collision-free paths → cost_solver
  4. Compare true execution costs and wall times.

Unlike pipeline_eval.py (BFS tour-cost approximation), both costs here are the
solver's Solution[5]: total collision-free path length after CBS resolution.

Usage:
    python full_pipeline_eval.py --checkpoint ../checkpoints/universal_s0/best.pt \\
        --grid_w 8 --grid_h 8 --num_agents 3 --num_goals 3 \\
        --n_instances 200 --seed 987654321
"""

import argparse
import heapq
import itertools
import os
import signal
import sys
import time

import numpy as np
import torch


class InstanceTimeout(Exception):
    """Raised by the SIGALRM handler to abort a single over-long instance."""


def _raise_timeout(signum, frame):
    raise InstanceTimeout()

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for p in (_ROOT, os.path.join(_ROOT, "dataset_generation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_gen import generate_random_map, load_map_file, sample_agents_goals
from distance import compute_distance_matrix, compute_goal_distance_matrix, normalize_D
from solver_wrapper import run_basic_mapf, run_basic_mapf_with_allocation
from evaluate import load_model


def _tour_cost(agent_idx: int, order: list[int], D: np.ndarray, G: np.ndarray) -> float:
    cost = D[agent_idx, order[0]]
    for k in range(1, len(order)):
        cost += G[order[k - 1], order[k]]
    return float(cost)


def _nn_2opt_order(agent_idx: int, goal_indices: list[int],
                   D: np.ndarray, G: np.ndarray) -> list[int]:
    """Nearest-neighbor construction + 2-opt refinement — a near-optimal tour in
    O(k^2) per pass, for agents carrying too many goals for exact enumeration."""
    remaining = list(goal_indices)
    first = min(remaining, key=lambda g: D[agent_idx, g])
    order = [first]
    remaining.remove(first)
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda g: G[last, g])
        order.append(nxt)
        remaining.remove(nxt)

    improved = True
    while improved:
        improved = False
        base = _tour_cost(agent_idx, order, D, G)
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = _tour_cost(agent_idx, cand, D, G)
                if c + 1e-9 < base:
                    order, base, improved = cand, c, True
    return order


def order_goals(agent_idx: int, goal_indices: list[int], D: np.ndarray, G: np.ndarray,
                exact_max: int = 8) -> list[int]:
    """Visit order minimizing D[agent, first] + Σ G[g_k, g_k+1].

    Exact brute force for ≤ exact_max goals (optimal, identical to the original);
    nearest-neighbor + 2-opt beyond it — M≫N can pile >10 goals on one agent,
    where k! enumeration is intractable (12! ≈ 5e8)."""
    if len(goal_indices) <= 1:
        return list(goal_indices)
    if len(goal_indices) > exact_max:
        return _nn_2opt_order(agent_idx, goal_indices, D, G)
    best_perm, best_cost = None, float("inf")
    for perm in itertools.permutations(goal_indices):
        cost = D[agent_idx, perm[0]]
        for k in range(1, len(perm)):
            cost += G[perm[k - 1], perm[k]]
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return list(best_perm)


def kbest_allocations(P: np.ndarray, max_k: int):
    """
    Yield hard allocation matrices Y ∈ {0,1}^(N×M) in decreasing joint
    probability Π_j P[a_j, j] — the NN-side analogue of the solver's k-best
    allocation escape.  Lazy best-first enumeration over the product of
    per-goal categorical distributions.

    k=1 is the column-wise argmax (identical to decode_assignment).
    """
    N, M = P.shape
    # Per goal: agent indices sorted by descending probability.
    order = np.argsort(-P, axis=0)              # (N, M)
    logp = np.log(np.clip(P, 1e-12, None))

    def state_logp(ranks):
        return sum(logp[order[r, j], j] for j, r in enumerate(ranks))

    start = (0,) * M
    heap = [(-state_logp(start), start)]
    seen = {start}
    yielded = 0
    while heap and yielded < max_k:
        neg, ranks = heapq.heappop(heap)
        Y = np.zeros((N, M), dtype=float)
        for j, r in enumerate(ranks):
            Y[order[r, j], j] = 1.0
        yield Y
        yielded += 1
        for j in range(M):
            if ranks[j] + 1 < N:
                child = ranks[:j] + (ranks[j] + 1,) + ranks[j + 1:]
                if child not in seen:
                    seen.add(child)
                    heapq.heappush(heap, (-state_logp(child), child))


def generate_instance(grid_w, grid_h, num_agents, num_goals, obstacle_prob, seed,
                      fixed_map=None):
    """One solvable instance, or None if rejected (unreachable goal / too full)."""
    try:
        map_dims = fixed_map if fixed_map is not None else generate_random_map(
            grid_w, grid_h, obstacle_prob, seed=seed)
        agents, goals = sample_agents_goals(map_dims, num_agents, M=num_goals, seed=seed)
    except ValueError:
        return None
    D = compute_distance_matrix(map_dims, agents, goals)
    if np.any(D == float("inf")):
        return None
    G = compute_goal_distance_matrix(map_dims, goals)
    if np.any(G == float("inf")):
        return None
    return map_dims, agents, goals, D, G


def main():
    parser = argparse.ArgumentParser(description="Full MAPF execution-cost NN vs solver")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map_file", type=str, default=None,
                        help="MovingAI .map file; when set, grid_w/h/obstacle_prob are ignored "
                             "and every instance uses this fixed map")
    parser.add_argument("--grid_w", type=int, default=8)
    parser.add_argument("--grid_h", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_goals", type=int, default=None)
    parser.add_argument("--obstacle_prob", type=float, default=0.1)
    parser.add_argument("--grid_w_max", type=int, default=None,
                        help="if set, grid width is sampled per instance from "
                             "[grid_w, grid_w_max]")
    parser.add_argument("--grid_h_max", type=int, default=None,
                        help="if set, grid height is sampled per instance from "
                             "[grid_h, grid_h_max]")
    parser.add_argument("--obstacle_prob_max", type=float, default=None,
                        help="if set, obstacle prob is sampled per instance from "
                             "[obstacle_prob, obstacle_prob_max]")
    parser.add_argument("--n_instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=987654321,
                        help="instance-stream seed (distinct from training seeds)")
    parser.add_argument("--max_fallbacks", type=int, default=3,
                        help="NN allocation candidates to try (ranked by joint "
                             "probability) before declaring the instance infeasible")
    parser.add_argument("--solver_node_budget", type=int, default=50_000,
                        help="CBS node budget for the full solver; instances the "
                             "solver cannot finish within it are skipped (dense "
                             "mazes can otherwise hang CBS)")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path for a per-instance CSV dump, so new "
                             "aggregate stats can be computed without re-running")
    parser.add_argument("--tag", type=str, default=None,
                        help="unique prefix for this run's LKH temp files (configStr). "
                             "Required when many evals run concurrently in the shared "
                             "RobustMCPF dir — pass e.g. ${SLURM_ARRAY_JOB_ID}_"
                             "${SLURM_ARRAY_TASK_ID}. Defaults to the pid (only safe "
                             "for a single host).")
    parser.add_argument("--max_seconds", type=float, default=0.0,
                        help="soft wall budget for the instance loop (0 = unlimited). "
                             "When exceeded, stop early and write the partial CSV — so a "
                             "slow CBS-bound config still yields data instead of a Slurm "
                             "TIMEOUT that loses everything. Set below the Slurm --time.")
    parser.add_argument("--instance_timeout", type=float, default=0.0,
                        help="hard per-instance wall cap in seconds (0 = off). A single "
                             "pathological CBS/LKH solve at large N,M can outrun the "
                             "between-instance --max_seconds check and blow the Slurm wall; "
                             "this SIGALRM-bounds each instance and skips it on timeout.")
    args = parser.parse_args()

    num_goals = args.num_goals if args.num_goals is not None else args.num_agents

    # All LKH temp files are named ATSP_runtime_files/{configStr}_Mtsp.* in the shared
    # RobustMCPF dir, so configStr must be unique across concurrent processes.
    run_tag = args.tag if args.tag else str(os.getpid())

    fixed_map = load_map_file(args.map_file) if args.map_file is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    ckpt_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    use_goal_dists = ckpt_args.get("use_goal_dists", False)

    rng = np.random.default_rng([args.seed, args.num_agents, num_goals])

    def draw_dims():
        """Per-instance (w, h, obstacle_prob): a single point when no _max flag
        is set, otherwise sampled uniformly over the requested ranges."""
        w = int(rng.integers(args.grid_w, args.grid_w_max + 1)) if args.grid_w_max else args.grid_w
        h = int(rng.integers(args.grid_h, args.grid_h_max + 1)) if args.grid_h_max else args.grid_h
        p = float(rng.uniform(args.obstacle_prob, args.obstacle_prob_max)) if args.obstacle_prob_max else args.obstacle_prob
        return w, h, p

    # Warm-up: first solver calls pay one-time module imports (~0.5 s),
    # which would otherwise pollute the per-instance timing of instance 0.
    while True:
        w, h, p = draw_dims()
        warm = generate_instance(w, h, args.num_agents, num_goals, p,
                                 int(rng.integers(0, 2**31)), fixed_map=fixed_map)
        if warm is None:
            continue
        w_map, w_agents, w_goals, _, _ = warm
        warm_ref = run_basic_mapf(w_map, w_agents, w_goals,
                                  config_str=f"fp_{run_tag}_warm_sv",
                                  cbs_node_budget=args.solver_node_budget)
        if warm_ref is not None:
            break
    warm_alloc = {a: [w_goals[g] for g in gi] for a, gi in warm_ref["allocation"].items()}
    run_basic_mapf_with_allocation(w_map, w_agents, w_goals, warm_alloc,
                                   config_str=f"fp_{run_tag}_warm_nn")

    cost_nn_list, cost_solver_list = [], []
    alloc_ms_list, nn_plan_ms_list, solver_ms_list = [], [], []
    fallback_counts, solver_k_list = [], []
    nn_conflicts_list, solver_conflicts_list = [], []
    csv_rows = []
    n_done, attempts, n_infeasible, n_solver_skip, n_timeout = 0, 0, 0, 0, 0
    loop_start = time.perf_counter()
    if args.instance_timeout:
        signal.signal(signal.SIGALRM, _raise_timeout)

    while n_done < args.n_instances:
        if args.max_seconds and time.perf_counter() - loop_start > args.max_seconds:
            print(f"\n[time budget] {args.max_seconds:.0f}s exceeded at {n_done} "
                  f"instances — stopping early and writing partial results.", flush=True)
            break
        inst_seed = int(rng.integers(0, 2**31))
        attempts += 1
        w, h, p = draw_dims()
        inst = generate_instance(w, h, args.num_agents, num_goals, p, inst_seed,
                                 fixed_map=fixed_map)
        if inst is None:
            continue
        map_dims, agents, goals, D_raw, G_raw = inst
        # Normalize by this instance's own dimensions (they vary when ranges are set).
        inst_w, inst_h = map_dims["Cols"], map_dims["Rows"]

        if args.instance_timeout:
            signal.alarm(int(args.instance_timeout))
        try:
            try:
                # --- NN allocation (+ goal ordering) ---
                t0 = time.perf_counter()
                D_norm = normalize_D(D_raw, inst_w, inst_h)
                D_t = torch.from_numpy(D_norm).float()[None].to(device)
                G_t = None
                if use_goal_dists:
                    G_norm = normalize_D(G_raw, inst_w, inst_h)
                    G_t = torch.from_numpy(G_norm).float()[None].to(device)
                with torch.no_grad():
                    P = (model(D_t, G=G_t) if G_t is not None else model(D_t))[0].cpu().numpy()
                alloc_ms = (time.perf_counter() - t0) * 1000.0

                # --- NN path planning with probability-ranked fallbacks ---
                # If the top allocation admits no collision-free plan, try the next
                # candidates in decreasing joint probability — the NN-side analogue
                # of the solver's k=2,3,… LKH escape.
                nn_result = None
                fallbacks_used = 0
                t0 = time.perf_counter()
                for cand_idx, Y_pred in enumerate(kbest_allocations(P, args.max_fallbacks)):
                    ordered_allocation = {}
                    for agent_idx in range(args.num_agents):
                        assigned = list(np.where(Y_pred[agent_idx] > 0.5)[0])
                        order = order_goals(agent_idx, assigned, D_raw, G_raw)
                        ordered_allocation[agent_idx] = [goals[g] for g in order]
                    nn_result = run_basic_mapf_with_allocation(
                        map_dims, agents, goals, ordered_allocation,
                        config_str=f"fp_{run_tag}_nn_{n_done}_{cand_idx}",
                    )
                    if nn_result is not None:
                        fallbacks_used = cand_idx
                        break
                nn_plan_ms = (time.perf_counter() - t0) * 1000.0

                if nn_result is None:
                    n_infeasible += 1
                    n_done += 1
                    continue

                # --- Full solver (LKH + CBS) ---
                t0 = time.perf_counter()
                solver_result = run_basic_mapf(
                    map_dims, agents, goals, config_str=f"fp_{run_tag}_sv_{n_done}",
                    cbs_node_budget=args.solver_node_budget,
                )
                solver_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                if args.instance_timeout:
                    signal.alarm(0)
        except InstanceTimeout:
            # Also catches the race where SIGALRM fires during the finally's own
            # alarm(0) call, right as the solver call finishes — that raised
            # InstanceTimeout previously escaped uncaught (outside the inner
            # try) and crashed the whole eval run.
            n_timeout += 1
            continue

        # Solver couldn't finish within budget — degenerate instance, no ground
        # truth to compare against, so discard it (don't count toward n_done).
        if solver_result is None:
            n_solver_skip += 1
            continue

        fallback_counts.append(fallbacks_used)
        cost_nn_list.append(nn_result["cost"])
        cost_solver_list.append(solver_result["cost"])
        alloc_ms_list.append(alloc_ms)
        nn_plan_ms_list.append(nn_plan_ms)
        solver_ms_list.append(solver_ms)
        solver_k_list.append(solver_result["k_roots"])
        nn_conflicts_list.append(nn_result["resolved_conflicts"])
        solver_conflicts_list.append(solver_result["resolved_conflicts"])
        csv_rows.append((inst_seed, nn_result["cost"], solver_result["cost"],
                         fallbacks_used + 1, solver_result["k_roots"],
                         nn_result["resolved_conflicts"], solver_result["resolved_conflicts"],
                         alloc_ms, nn_plan_ms, solver_ms))
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{args.n_instances} instances done", flush=True)

    if n_done == 0:
        print(f"\n[no instances completed] {attempts} attempts, "
              f"{n_solver_skip} solver-skipped, {n_infeasible} infeasible — "
              f"nothing to write. Raise --max_seconds / --time or lower N,M.")
        return

    cost_nn = np.array(cost_nn_list, dtype=float)
    cost_solver = np.array(cost_solver_list, dtype=float)
    ratios = np.where(cost_solver > 0, cost_nn / cost_solver, 1.0)
    nn_total_ms = np.array(alloc_ms_list) + np.array(nn_plan_ms_list)
    solver_ms_arr = np.array(solver_ms_list)

    grid_desc = (args.map_file if args.map_file is not None
                 else f"{args.grid_w}x{args.grid_h}")
    print(f"\n--- Full MAPF execution cost: NN vs solver "
          f"({grid_desc}, N={args.num_agents}, M={num_goals}, "
          f"{n_done} instances, {attempts} attempts) ---")
    print(f"  infeasible_rate       : {n_infeasible / n_done:.4f} "
          f"({n_infeasible} of {n_done}, after {args.max_fallbacks} candidates)")
    print(f"  solver_skipped        : {n_solver_skip} (solver exceeded "
          f"{args.solver_node_budget}-node budget; excluded)")
    print(f"  instance_timeouts     : {n_timeout} (exceeded "
          f"{args.instance_timeout:.0f}s per-instance cap; excluded)")
    print(f"  fallback_rate         : {np.mean(np.array(fallback_counts) > 0):.4f} "
          f"(mean fallbacks {np.mean(fallback_counts):.3f})")

    def k_hist(values):
        vals = np.array(values)
        parts = [f"k={k}: {np.mean(vals == k):.1%}" for k in sorted(set(vals))]
        return "  ".join(parts)

    nn_k = [f + 1 for f in fallback_counts]
    print(f"  nn_k_chosen           : {k_hist(nn_k)} (mean {np.mean(nn_k):.3f})")
    print(f"  solver_k_roots        : {k_hist(solver_k_list)} (mean {np.mean(solver_k_list):.3f})")
    print(f"  mean_conflicts_nn     : {np.mean(nn_conflicts_list):.3f}")
    print(f"  mean_conflicts_solver : {np.mean(solver_conflicts_list):.3f}")
    print(f"  mean_cost_nn          : {cost_nn.mean():.3f}")
    print(f"  mean_cost_solver      : {cost_solver.mean():.3f}")
    print(f"  mean_exec_cost_ratio  : {ratios.mean():.4f}")
    print(f"  max_exec_cost_ratio   : {ratios.max():.4f}")
    diff = cost_nn - cost_solver
    print(f"  mean_cost_diff_steps  : {diff.mean():.3f}")
    print(f"  std_cost_diff_steps   : {diff.std():.3f}")
    print(f"  max_cost_diff_steps   : {diff.max():.3f}")
    print(f"  exact_cost_match_rate : {(cost_nn == cost_solver).mean():.4f}")
    print(f"  mean_nn_alloc_ms      : {np.mean(alloc_ms_list):.3f}")
    print(f"  mean_nn_plan_ms       : {np.mean(nn_plan_ms_list):.3f}")
    print(f"  mean_nn_total_ms      : {nn_total_ms.mean():.3f}")
    print(f"  mean_solver_ms        : {solver_ms_arr.mean():.3f}")
    print(f"  speedup               : {solver_ms_arr.mean() / nn_total_ms.mean():.1f}x")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w") as f:
            f.write("inst_seed,cost_nn,cost_solver,nn_k,solver_k,"
                    "conflicts_nn,conflicts_solver,alloc_ms,nn_plan_ms,solver_ms\n")
            for row in csv_rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                                 for v in row) + "\n")
        print(f"  per-instance CSV      : {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
