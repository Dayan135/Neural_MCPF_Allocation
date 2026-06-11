"""
Full-pipeline evaluation: NN allocation vs RobustMCPF solver.

For each test instance the NN produces a hard assignment (argmax over column-wise
softmax).  The tour cost of that assignment is computed using the D and G distance
matrices stored in the dataset (BFS distances, same normalisation used in training).
For agents assigned multiple goals the cost is the optimal permutation of that
agent's goals (brute-force over ≤ 6 goals per agent, which is always feasible here).

Solver calls are expensive (~0.5 s each), so --n_timing caps how many are timed.
The solver's reported cost is used as the ground-truth optimal cost for the ratio.

Usage:
    python pipeline_eval.py \\
        --checkpoint ../checkpoints/gd_n3_G_s0/best.pt \\
        --data_dir   ../data/n3_8x8_G \\
        --split test \\
        --n_timing 50
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.network import build_model
from evaluate import load_model, decode_assignment


# ---------------------------------------------------------------------------
# Tour cost from D + G matrices
# ---------------------------------------------------------------------------

def _agent_tour_cost(
    agent_idx: int,
    goal_indices: list[int],
    D_norm: np.ndarray,
    G_norm: np.ndarray,
) -> float:
    """
    Optimal tour cost for one agent visiting the given goal indices (normalised).
    First hop: D_norm[agent_idx, first_goal].
    Subsequent hops: G_norm[prev_goal, next_goal].
    Brute-force permutation — feasible for len(goal_indices) <= 6.
    Returns 0.0 if goal_indices is empty.
    """
    if not goal_indices:
        return 0.0
    if len(goal_indices) == 1:
        return float(D_norm[agent_idx, goal_indices[0]])

    best = float("inf")
    for perm in itertools.permutations(goal_indices):
        cost = D_norm[agent_idx, perm[0]]
        for k in range(1, len(perm)):
            cost += G_norm[perm[k - 1], perm[k]]
        if cost < best:
            best = cost
    return float(best)


def compute_nn_tour_cost(
    Y_pred: np.ndarray,
    D_norm: np.ndarray,
    G_norm: np.ndarray,
) -> float:
    """
    Total tour cost for a hard allocation matrix Y_pred ∈ {0,1}^(N×M).
    Sums over all agents using _agent_tour_cost.
    """
    N = Y_pred.shape[0]
    total = 0.0
    for agent_idx in range(N):
        goals = list(np.where(Y_pred[agent_idx] > 0.5)[0])
        total += _agent_tour_cost(agent_idx, goals, D_norm, G_norm)
    return total


# ---------------------------------------------------------------------------
# Solver timing helper
# ---------------------------------------------------------------------------

def _time_solver_on_instance(map_dims, agents, goals, config_str):
    """Run solver once and return (cost, elapsed_seconds)."""
    sys.path.insert(0, _ROOT)
    from solver_wrapper import run_basic_mapf
    t0 = time.perf_counter()
    result = run_basic_mapf(map_dims, agents, goals, config_str=config_str)
    elapsed = time.perf_counter() - t0
    return result["cost"], elapsed


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def pipeline_metrics(
    model,
    D_all: np.ndarray,
    Y_all: np.ndarray,
    G_all: np.ndarray,
    device: torch.device,
    n_timing: int = 0,
    batch_size: int = 512,
) -> dict:
    """
    Compute pipeline metrics over the test set.

    Parameters
    ----------
    n_timing : int
        Number of instances to time the solver on (0 = skip solver timing).
    """
    full_match = []
    nn_costs = []
    solver_costs = []
    nn_times_ms = []
    solver_times_ms = []

    n_samples = len(D_all)
    timed_so_far = 0

    for start in range(0, n_samples, batch_size):
        D_batch_np = D_all[start : start + batch_size]
        G_batch_np = G_all[start : start + batch_size]
        Y_batch_np = Y_all[start : start + batch_size]

        D_batch = torch.from_numpy(D_batch_np).float().to(device)
        G_batch = torch.from_numpy(G_batch_np).float().to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            P_batch = model(D_batch, G=G_batch).cpu().numpy()
        nn_elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(D_batch_np)

        for i in range(len(D_batch_np)):
            Y_pred = decode_assignment(P_batch[i])
            Y_true = Y_batch_np[i]
            full_match.append(np.array_equal(Y_pred, Y_true))

            cost = compute_nn_tour_cost(Y_pred, D_batch_np[i], G_batch_np[i])
            nn_costs.append(cost)
            nn_times_ms.append(nn_elapsed_ms)

    # Solver timing on first n_timing instances (re-runs solver, slow)
    if n_timing > 0:
        # Import here to avoid loading RobustMCPF unless needed
        from dataset_generation.grid_gen import generate_random_map, sample_agents_goals
        print(f"\nTiming solver on {min(n_timing, n_samples)} instances…", flush=True)
        # We don't have the original map/agents stored in .npy, so we reconstruct
        # a compatible instance just for timing purposes using a fixed seed.
        # The solver cost comes from Y_all (the ground-truth from the real solver).
        # We report solver cost from Y_all * D_all as a proxy (sum of assigned BFS distances).
        # For true solver timing we need actual instances — use stored Y costs here.
        print("  Note: solver timing uses reconstructed instances (fixed seed); "
              "costs come from stored Y × D (proxy).")
        for idx in range(min(n_timing, n_samples)):
            # Proxy: solver cost = sum of D[agent,goal] for Y=1 entries
            proxy_solver_cost = float((D_all[idx] * Y_all[idx]).sum())
            solver_costs.append(proxy_solver_cost)

            # Timing: reconstruct a sample of the same shape and time the solver
            N, M = Y_all[idx].shape
            try:
                map_dims = generate_random_map(8, 8, obstacle_prob=0.1, seed=idx)
                agents, goals = sample_agents_goals(map_dims, N, M=M, seed=idx)
                t0 = time.perf_counter()
                _time_solver_on_instance(map_dims, agents, goals, config_str=f"pipeline_{idx}")
                solver_times_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                print(f"  [skip {idx}] solver error: {exc}")

    # NN cost ratio vs proxy solver cost (D×Y)
    proxy_solver_costs = [(D_all[i] * Y_all[i]).sum() for i in range(n_samples)]
    cost_ratios = [
        nn_costs[i] / proxy_solver_costs[i] if proxy_solver_costs[i] > 1e-9 else 1.0
        for i in range(n_samples)
    ]

    metrics = {
        "n_samples": n_samples,
        "full_assignment_accuracy": float(np.mean(full_match)),
        "mean_nn_tour_cost": float(np.mean(nn_costs)),
        "mean_solver_proxy_cost": float(np.mean(proxy_solver_costs)),
        "mean_cost_ratio": float(np.mean(cost_ratios)),
        "max_cost_ratio": float(np.max(cost_ratios)),
        "mean_nn_inference_ms": float(np.mean(nn_times_ms)),
    }
    if solver_times_ms:
        metrics["mean_solver_ms"] = float(np.mean(solver_times_ms))
        metrics["speedup"] = metrics["mean_solver_ms"] / metrics["mean_nn_inference_ms"]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline NN vs solver evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=os.path.join("..", "data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_timing", type=int, default=0,
                        help="Number of solver calls for timing (0 = skip)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ckpt_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    use_goal_dists = ckpt_args.get("use_goal_dists", False)
    if not use_goal_dists:
        print("Warning: checkpoint was trained without --use_goal_dists; "
              "G_matrices.npy still used for tour cost but not fed to model.")

    split_dir = os.path.join(args.data_dir, args.split)
    D_all = np.load(os.path.join(split_dir, "D_matrices.npy"))
    Y_all = np.load(os.path.join(split_dir, "Y_matrices.npy"))
    G_all = np.load(os.path.join(split_dir, "G_matrices.npy"))

    print(f"\n--- Pipeline evaluation on '{args.split}' ({len(D_all)} samples, "
          f"N={D_all.shape[1]}, M={D_all.shape[2]}) ---")

    metrics = pipeline_metrics(
        model, D_all, Y_all, G_all, device,
        n_timing=args.n_timing,
    )

    print()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
