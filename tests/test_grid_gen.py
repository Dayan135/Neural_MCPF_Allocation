"""Tests for random grid generation and agent/goal placement."""

import pytest

from grid_gen import generate_random_map, sample_agents_goals


@pytest.mark.parametrize("w, h", [(5, 5), (10, 8), (3, 12)])
def test_generate_random_map_dimensions(w, h):
    m = generate_random_map(w, h, obstacle_prob=0.2, seed=0)
    assert m["Cols"] == w
    assert m["Rows"] == h
    assert len(m["Map"]) == w * h
    assert set(m["Map"]) <= {0, 1}


def test_generate_random_map_no_obstacles_when_prob_zero():
    m = generate_random_map(6, 6, obstacle_prob=0.0, seed=123)
    assert sum(m["Map"]) == 0


def test_generate_random_map_is_seed_deterministic():
    a = generate_random_map(7, 7, obstacle_prob=0.3, seed=42)
    b = generate_random_map(7, 7, obstacle_prob=0.3, seed=42)
    assert a == b


def test_generate_random_map_different_seeds_differ():
    a = generate_random_map(8, 8, obstacle_prob=0.3, seed=1)
    b = generate_random_map(8, 8, obstacle_prob=0.3, seed=2)
    assert a != b


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sample_agents_goals_counts(open_5x5, n):
    agents, goals = sample_agents_goals(open_5x5, n, seed=0)
    assert len(agents) == n
    assert len(goals) == n


def test_sample_agents_goals_formats(open_5x5):
    agents, goals = sample_agents_goals(open_5x5, 3, seed=0)
    # agents: (flat_idx, direction in 0..3); goals: plain flat indices
    for loc, direction in agents:
        assert isinstance(loc, int)
        assert 0 <= direction <= 3
    assert all(isinstance(g, int) for g in goals)


def test_sample_agents_goals_on_free_cells_and_distinct(detour_grid):
    agents, goals = sample_agents_goals(detour_grid, 2, seed=7)
    free = {i for i, v in enumerate(detour_grid["Map"]) if v == 0}
    agent_locs = [loc for loc, _ in agents]

    assert all(loc in free for loc in agent_locs)
    assert all(g in free for g in goals)
    # agents and goals occupy distinct cells
    assert set(agent_locs).isdisjoint(goals)
    assert len(set(agent_locs)) == len(agent_locs)
    assert len(set(goals)) == len(goals)


def test_sample_agents_goals_is_seed_deterministic(open_5x5):
    a = sample_agents_goals(open_5x5, 3, seed=99)
    b = sample_agents_goals(open_5x5, 3, seed=99)
    assert a == b


def test_sample_agents_goals_raises_when_grid_too_full(full_grid):
    # 1 free cell cannot host 2 agents + 2 goals.
    with pytest.raises(ValueError):
        sample_agents_goals(full_grid, 2, seed=0)
