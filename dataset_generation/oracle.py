"""
Ground-truth label extraction via the RobustMCPF solver in BasicMAPF mode.

The solver — not Hungarian — is the oracle so that the NN learns the same
policy the solver implements, including collision-aware path costs.
"""

import sys
import os
import numpy as np

# Allow importing solver_wrapper from the project root
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solver_wrapper import run_basic_mapf


def get_ground_truth(
    map_dims: dict,
    agents: list,
    goals: list,
    config_str: str | None = None,
) -> tuple[np.ndarray, int] | None:
    """
    Run RobustMCPF in BasicMAPF mode and extract the optimal assignment matrix Y.

    Y[i][j] = 1.0  iff the solver assigned agent i to goal j (0-indexed into `goals`).

    Returns (Y, total_cost) on success, or None if the solver fails / raises.
    """
    try:
        result = run_basic_mapf(map_dims, agents, goals, config_str=config_str)
    except Exception:
        return None

    N = len(agents)
    M = len(goals)
    Y = np.zeros((N, M), dtype=float)
    for agent_id, goal_indices in result["allocation"].items():
        for goal_idx in goal_indices:
            Y[agent_id, goal_idx] = 1.0

    # Sanity: every goal must be assigned to exactly one agent (column sums == 1)
    if not np.allclose(Y.sum(axis=0), 1.0):
        return None

    return Y, result["cost"]
