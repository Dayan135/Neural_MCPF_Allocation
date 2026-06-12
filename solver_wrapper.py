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

    def factory(framework_cls):
        N = len(agents)
        return framework_cls(
            Positions=agents,
            GoalLocations=goals,
            no_collision_prob=1.0,
            delaysProb={i: 0.0 for i in range(N)},
            MapAndDims=map_dims,
            verifyAlpha=0.05,
            algorithm="BasicMAPF",
            configStr=config_str,
        )

    solver = _in_solver_cwd(lambda: factory(_framework_cls()))
    return _build_result(solver, goals, solver.K_optimal_sequences[1]["Allocations"])


def run_basic_mapf_with_allocation(
    map_dims: dict,
    agents: list,
    goals: list,
    ordered_allocation: dict[int, list[int]],
    config_str: str | None = None,
) -> dict:
    """
    Run only the MAPF path-planning part of RobustMCPF (CBS + low-level A*)
    on a FIXED, externally supplied allocation — no LKH/TSP call is made.

    ordered_allocation : {agent_id: [goal_flat_loc, ...] in visit order}
        Every goal must appear exactly once across all agents.  Agents with
        no goals may be omitted or map to [].

    Returns the same dict shape as run_basic_mapf.  "cost" is the true
    execution cost: total collision-free path length under this allocation.
    """
    if config_str is None:
        config_str = f"fixed_{os.getpid()}"

    def build():
        base = _framework_cls()

        class _FixedAllocCbss(base):
            """Pins K_optimal_sequences[1] to the supplied allocation.

            Cost is a sentinel high value so CheckNewRoot never escapes to a
            k=2 re-allocation: CBS must resolve conflicts within this one."""

            def __init__(self, fixed_sequences, **kwargs):
                self._fixed_seq = fixed_sequences  # before super().__init__ triggers run()
                super().__init__(**kwargs)

            def run(self):
                seq = {
                    "Allocations": self._fixed_seq,
                    "Alloc_edges": [
                        (s[i], s[i + 1])
                        for s in self._fixed_seq.values()
                        for i in range(1, len(s) - 1)
                    ],
                    "Cost": 10**9,
                }
                self.K_Best_Seq_Solver.find_k_best_solution = lambda k: seq
                return super().run()

        N = len(agents)
        fixed_sequences = {
            i: [agents[i][0]] + [int(g) for g in ordered_allocation.get(i, [])]
            for i in range(N)
        }
        return _FixedAllocCbss(
            fixed_sequences,
            Positions=agents,
            GoalLocations=goals,
            no_collision_prob=1.0,
            delaysProb={i: 0.0 for i in range(N)},
            MapAndDims=map_dims,
            verifyAlpha=0.05,
            algorithm="BasicMAPF",
            configStr=config_str,
        )

    solver = _in_solver_cwd(build)
    return _build_result(solver, goals, solver.K_optimal_sequences[1]["Allocations"])


def _in_solver_cwd(build):
    """Run `build()` with CWD set to the RobustMCPF repo (LKH and ATSP temp
    files are resolved via os.getcwd()), restoring CWD afterwards."""
    orig_dir = os.getcwd()
    os.chdir(REPO_DIR)
    os.makedirs("ATSP_runtime_files", exist_ok=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    try:
        return build()
    finally:
        os.chdir(orig_dir)


def _framework_cls():
    """Import Robust_Cbss_framework lazily (requires REPO_DIR on sys.path)."""
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    from Run_Robust_Cbss_Framework import Robust_Cbss_framework
    return Robust_Cbss_framework


def _build_result(solver, goals: list, alloc_raw: dict) -> dict:
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
