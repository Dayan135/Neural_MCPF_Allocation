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
import itertools
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for p in (_ROOT, os.path.join(_ROOT, "dataset_generation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_gen import generate_random_map, sample_agents_goals
from distance import compute_distance_matrix, compute_goal_distance_matrix, normalize_D
from solver_wrapper import run_basic_mapf, run_basic_mapf_with_allocation
from evaluate import load_model, decode_assignment


def order_goals(agent_idx: int, goal_indices: list[int], D: np.ndarray, G: np.ndarray) -> list[int]:
    """Visit order minimizing D[agent, first] + Σ G[g_k, g_k+1] (brute force, ≤6 goals)."""
    if len(goal_indices) <= 1:
        return list(goal_indices)
    best_perm, best_cost = None, float("inf")
    for perm in itertools.permutations(goal_indices):
        cost = D[agent_idx, perm[0]]
        for k in range(1, len(perm)):
            cost += G[perm[k - 1], perm[k]]
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return list(best_perm)


def generate_instance(grid_w, grid_h, num_agents, num_goals, obstacle_prob, seed):
    """One solvable instance, or None if rejected (unreachable goal / too full)."""
    try:
        map_dims = generate_random_map(grid_w, grid_h, obstacle_prob, seed=seed)
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
    parser.add_argument("--grid_w", type=int, default=8)
    parser.add_argument("--grid_h", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_goals", type=int, default=None)
    parser.add_argument("--obstacle_prob", type=float, default=0.1)
    parser.add_argument("--n_instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=987654321,
                        help="instance-stream seed (distinct from training seeds)")
    args = parser.parse_args()

    num_goals = args.num_goals if args.num_goals is not None else args.num_agents

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    ckpt_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    use_goal_dists = ckpt_args.get("use_goal_dists", False)

    rng = np.random.default_rng([args.seed, args.num_agents, num_goals])

    # Warm-up: first solver calls pay one-time module imports (~0.5 s),
    # which would otherwise pollute the per-instance timing of instance 0.
    while True:
        warm_seed = int(rng.integers(0, 2**31))
        warm = generate_instance(args.grid_w, args.grid_h, args.num_agents,
                                 num_goals, args.obstacle_prob, warm_seed)
        if warm is not None:
            break
    w_map, w_agents, w_goals, _, _ = warm
    warm_ref = run_basic_mapf(w_map, w_agents, w_goals, config_str="fp_warm_sv")
    warm_alloc = {a: [w_goals[g] for g in gi] for a, gi in warm_ref["allocation"].items()}
    run_basic_mapf_with_allocation(w_map, w_agents, w_goals, warm_alloc,
                                   config_str="fp_warm_nn")

    cost_nn_list, cost_solver_list = [], []
    alloc_ms_list, nn_plan_ms_list, solver_ms_list = [], [], []
    n_done, attempts = 0, 0

    while n_done < args.n_instances:
        inst_seed = int(rng.integers(0, 2**31))
        attempts += 1
        inst = generate_instance(
            args.grid_w, args.grid_h, args.num_agents, num_goals,
            args.obstacle_prob, inst_seed,
        )
        if inst is None:
            continue
        map_dims, agents, goals, D_raw, G_raw = inst

        # --- NN allocation (+ goal ordering) ---
        t0 = time.perf_counter()
        D_norm = normalize_D(D_raw, args.grid_w, args.grid_h)
        D_t = torch.from_numpy(D_norm).float()[None].to(device)
        G_t = None
        if use_goal_dists:
            G_norm = normalize_D(G_raw, args.grid_w, args.grid_h)
            G_t = torch.from_numpy(G_norm).float()[None].to(device)
        with torch.no_grad():
            P = (model(D_t, G=G_t) if G_t is not None else model(D_t))[0].cpu().numpy()
        Y_pred = decode_assignment(P)
        ordered_allocation = {}
        for agent_idx in range(args.num_agents):
            assigned = list(np.where(Y_pred[agent_idx] > 0.5)[0])
            order = order_goals(agent_idx, assigned, D_raw, G_raw)
            ordered_allocation[agent_idx] = [goals[g] for g in order]
        alloc_ms = (time.perf_counter() - t0) * 1000.0

        # --- NN path planning (CBS on fixed allocation, no LKH) ---
        t0 = time.perf_counter()
        nn_result = run_basic_mapf_with_allocation(
            map_dims, agents, goals, ordered_allocation,
            config_str=f"fp_nn_{n_done}",
        )
        nn_plan_ms = (time.perf_counter() - t0) * 1000.0

        # --- Full solver (LKH + CBS) ---
        t0 = time.perf_counter()
        solver_result = run_basic_mapf(
            map_dims, agents, goals, config_str=f"fp_sv_{n_done}",
        )
        solver_ms = (time.perf_counter() - t0) * 1000.0

        cost_nn_list.append(nn_result["cost"])
        cost_solver_list.append(solver_result["cost"])
        alloc_ms_list.append(alloc_ms)
        nn_plan_ms_list.append(nn_plan_ms)
        solver_ms_list.append(solver_ms)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{args.n_instances} instances done", flush=True)

    cost_nn = np.array(cost_nn_list, dtype=float)
    cost_solver = np.array(cost_solver_list, dtype=float)
    ratios = np.where(cost_solver > 0, cost_nn / cost_solver, 1.0)
    nn_total_ms = np.array(alloc_ms_list) + np.array(nn_plan_ms_list)
    solver_ms_arr = np.array(solver_ms_list)

    print(f"\n--- Full MAPF execution cost: NN vs solver "
          f"({args.grid_w}x{args.grid_h}, N={args.num_agents}, M={num_goals}, "
          f"{n_done} instances, {attempts} attempts) ---")
    print(f"  mean_cost_nn          : {cost_nn.mean():.3f}")
    print(f"  mean_cost_solver      : {cost_solver.mean():.3f}")
    print(f"  mean_exec_cost_ratio  : {ratios.mean():.4f}")
    print(f"  max_exec_cost_ratio   : {ratios.max():.4f}")
    print(f"  mean_cost_diff_steps  : {(cost_nn - cost_solver).mean():.3f}")
    print(f"  exact_cost_match_rate : {(cost_nn == cost_solver).mean():.4f}")
    print(f"  mean_nn_alloc_ms      : {np.mean(alloc_ms_list):.3f}")
    print(f"  mean_nn_plan_ms       : {np.mean(nn_plan_ms_list):.3f}")
    print(f"  mean_nn_total_ms      : {nn_total_ms.mean():.3f}")
    print(f"  mean_solver_ms        : {solver_ms_arr.mean():.3f}")
    print(f"  speedup               : {solver_ms_arr.mean() / nn_total_ms.mean():.1f}x")


if __name__ == "__main__":
    main()
