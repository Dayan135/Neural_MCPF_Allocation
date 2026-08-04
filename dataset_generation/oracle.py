"""
Ground-truth label extraction via the RobustMCPF solver in BasicMAPF mode.

The solver — not Hungarian — is the oracle so that the NN learns the same
policy the solver implements, including collision-aware path costs.
"""

import signal
import sys
import os
import numpy as np

# Allow importing solver_wrapper from the project root
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solver_wrapper import run_basic_mapf


class _InstanceTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _InstanceTimeout()


def get_ground_truth(
    map_dims: dict,
    agents: list,
    goals: list,
    config_str: str | None = None,
    cbs_node_budget: int | None = 50_000,
    instance_timeout: float = 0.0,
) -> tuple[np.ndarray, int] | None:
    """
    Run RobustMCPF in BasicMAPF mode and extract the optimal assignment matrix Y.

    Y[i][j] = 1.0  iff the solver assigned agent i to goal j (0-indexed into `goals`).

    cbs_node_budget bounds CBS work per instance: dense-wall instances can be
    BFS-reachable yet so constraint-heavy that CBS never terminates.  Budget
    exhaustion rejects the instance (same as unreachable goals).

    instance_timeout (seconds, 0 = disabled) is a wall-clock safety net on top
    of cbs_node_budget: at N/M scale (~100+), individual CBS node expansions
    get slow enough that 50k of them can take minutes even though the budget
    itself is "reached" — the node count alone doesn't bound wall time. Mirrors
    evaluation/full_pipeline_eval.py's --instance_timeout SIGALRM pattern.

    Returns (Y, total_cost) on success, or None if the solver fails / raises /
    times out.
    """
    try:
        old_handler = None
        if instance_timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(int(instance_timeout))
        try:
            result = run_basic_mapf(map_dims, agents, goals, config_str=config_str,
                                    cbs_node_budget=cbs_node_budget)
        finally:
            if instance_timeout > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    except (Exception, _InstanceTimeout):
        # Also catches the race where SIGALRM fires during the finally's own
        # alarm(0) call, right as run_basic_mapf finishes — that raised
        # _InstanceTimeout previously escaped uncaught (it's outside the inner
        # try) and crashed the whole worker process, killing the Slurm task.
        return None
    if result is None:
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
