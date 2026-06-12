"""
Integration tests for run_basic_mapf_with_allocation (fixed-allocation CBS).

These invoke the CBS planner (and, for the reference run, the LKH binary),
so they are marked `slow`.
"""

import os

import numpy as np
import pytest

from solver_wrapper import run_basic_mapf, run_basic_mapf_with_allocation

pytestmark = pytest.mark.slow


AGENTS = [(0, 0), (1, 0)]
GOALS = [24, 23]


@pytest.fixture
def open_5x5():
    return {"Rows": 5, "Cols": 5, "Map": [0] * 25}


def _ordered_allocation_from_result(result, goals):
    """Convert run_basic_mapf's goal-index allocation to flat-loc visit order."""
    return {
        agent: [goals[g] for g in goal_indices]
        for agent, goal_indices in result["allocation"].items()
    }


def test_roundtrip_solver_allocation_same_cost(open_5x5):
    # Injecting the solver's own allocation must reproduce the solver's cost.
    ref = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_rt_ref")
    ordered = _ordered_allocation_from_result(ref, GOALS)

    fixed = run_basic_mapf_with_allocation(
        open_5x5, AGENTS, GOALS, ordered, config_str="test_rt_fix"
    )
    assert fixed["cost"] == ref["cost"]
    assert fixed["allocation"] == ref["allocation"]


def test_bad_allocation_costs_at_least_optimal(open_5x5):
    # Cross assignment (each agent sent to the far-swapped goal) can never
    # beat the solver's optimum.
    ref = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_bad_ref")
    ordered = _ordered_allocation_from_result(ref, GOALS)
    # Swap the two agents' goal lists to force a (weakly) worse allocation.
    swapped = {0: ordered.get(1, []), 1: ordered.get(0, [])}

    fixed = run_basic_mapf_with_allocation(
        open_5x5, AGENTS, GOALS, swapped, config_str="test_bad_fix"
    )
    assert fixed["cost"] >= ref["cost"]


def test_multi_goal_single_agent(open_5x5):
    # All goals on one agent: agent 0 visits both goals in order; agent 1 idles.
    ordered = {0: [23, 24], 1: []}
    fixed = run_basic_mapf_with_allocation(
        open_5x5, AGENTS, GOALS, ordered, config_str="test_multi_fix"
    )
    assert np.isfinite(fixed["cost"])
    assert fixed["cost"] > 0
    assert sorted(g for gs in fixed["allocation"].values() for g in gs) == [0, 1]


def test_fixed_alloc_restores_cwd(open_5x5):
    before = os.getcwd()
    run_basic_mapf_with_allocation(
        open_5x5, AGENTS, GOALS, {0: [24], 1: [23]}, config_str="test_fix_cwd"
    )
    assert os.getcwd() == before
