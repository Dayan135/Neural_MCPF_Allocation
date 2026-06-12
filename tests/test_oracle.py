"""
Integration tests for the RobustMCPF boundary (solver_wrapper + oracle).

These invoke the LKH binary and are marked `slow`. Run the fast suite with:
    pytest -m "not slow"
"""

import numpy as np
import pytest

from solver_wrapper import run_basic_mapf
from oracle import get_ground_truth

pytestmark = pytest.mark.slow


# Two agents in the top-left corner, two goals in the bottom-right of an open 5x5.
AGENTS = [(0, 0), (1, 0)]
GOALS = [24, 23]


@pytest.fixture
def open_5x5():
    return {"Rows": 5, "Cols": 5, "Map": [0] * 25}


def test_run_basic_mapf_allocation_format(open_5x5):
    result = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_format")

    assert set(result) == {"allocation", "paths", "cost", "k_roots", "resolved_conflicts"}
    # allocation maps each agent to a (possibly empty) list of goal indices
    for agent_id, goal_indices in result["allocation"].items():
        assert isinstance(goal_indices, list)
        assert all(0 <= g < len(GOALS) for g in goal_indices)


def test_run_basic_mapf_every_goal_assigned_once(open_5x5):
    result = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_once")
    assigned = [g for goals in result["allocation"].values() for g in goals]
    assert sorted(assigned) == list(range(len(GOALS)))


def test_run_basic_mapf_budget_exhaustion_returns_none(open_5x5):
    result = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_budget",
                            cbs_node_budget=0)
    assert result is None


def test_run_basic_mapf_no_budget_unchanged(open_5x5):
    result = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_nobudget")
    assert result is not None and result["cost"] > 0


def test_run_basic_mapf_positive_cost(open_5x5):
    result = run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_cost")
    assert np.isfinite(result["cost"])
    assert result["cost"] > 0


def test_run_basic_mapf_restores_cwd(open_5x5):
    import os

    before = os.getcwd()
    run_basic_mapf(open_5x5, AGENTS, GOALS, config_str="test_cwd")
    assert os.getcwd() == before


def test_ground_truth_matrix_shape_and_columns(open_5x5):
    out = get_ground_truth(open_5x5, AGENTS, GOALS, config_str="test_gt")
    assert out is not None
    Y, cost = out

    assert Y.shape == (len(AGENTS), len(GOALS))
    # Every goal (column) is assigned to exactly one agent.
    np.testing.assert_allclose(Y.sum(axis=0), np.ones(len(GOALS)))
    assert cost > 0
