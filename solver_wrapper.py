"""
Thin wrapper around RobustMCPF in BasicMAPF mode (no orientation, no delays).

BasicMAPF routes through:
  Run_Robust_Cbss_Framework -> else  -> kBestSequencing (LKH-TSP)
  LowLevelPlan              -> else  -> runLowLevelPlanWithoutRotations
  Verify                    -> else  -> verify_without_delays
  FindConflict              -> else  -> findConflictWithoutDelays
  line 89 guard             -> skips positive-constraint nodes

kBestSequencing writes ATSP files and calls the LKH binary using os.getcwd(),
so every call must be made with CWD = REPO_DIR.
"""

import os
import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RobustMCPF")


def run_basic_mapf(
    map_dims: dict,
    agents: list,
    goals: list,
    config_str: str | None = None,
) -> dict:
    """
    Run the RobustMCPF solver in BasicMAPF (basic, no-delay, no-orientation) mode.

    Parameters
    ----------
    map_dims : {"Rows": int, "Cols": int, "Map": list[int]}
        Grid in RobustMCPF's flat format (0=free, 1=obstacle).
    agents : list of (flat_idx, direction)
        Agent starting positions.  Direction is ignored in basic mode.
    goals : list of flat_idx
        Goal locations.
    config_str : str, optional
        Unique identifier for LKH temp files.  Defaults to PID-based string.

    Returns
    -------
    dict with keys:
        "allocation" : {agent_id (int): goal_index (int)}
            goal_index is the position of the assigned goal in `goals`.
        "paths"      : {agent_id: [(loc, dir), ...]}  collision-free paths
        "cost"       : int  total sum-of-path-lengths
    """
    if config_str is None:
        config_str = f"basic_{os.getpid()}"

    orig_dir = os.getcwd()
    os.chdir(REPO_DIR)
    os.makedirs("ATSP_runtime_files", exist_ok=True)

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        from Run_Robust_Cbss_Framework import Robust_Cbss_framework

        N = len(agents)
        solver = Robust_Cbss_framework(
            Positions=agents,
            GoalLocations=goals,
            no_collision_prob=1.0,
            delaysProb={i: 0.0 for i in range(N)},
            MapAndDims=map_dims,
            verifyAlpha=0.05,
            algorithm="BasicMAPF",
            configStr=config_str,
        )
    finally:
        os.chdir(orig_dir)

    alloc_raw = solver.K_optimal_sequences[1]["Allocations"]
    # alloc_raw[agent_id] = [start_flat_loc, goal_loc1, goal_loc2, ...]
    # An agent may have 0, 1, or multiple goals (general mTSP).
    allocation: dict[int, list[int]] = {}
    for agent_id, locs in alloc_raw.items():
        allocation[agent_id] = [
            goals.index(loc) for loc in locs[1:] if loc in goals
        ]

    return {
        "allocation": allocation,   # {agent_id: [goal_index, ...]}
        "paths": solver.Solution[0],
        "cost": solver.Solution[5],
    }
