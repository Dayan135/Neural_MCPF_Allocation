"""Tests for the BFS distance matrix and normalization."""

import math

import numpy as np
import pytest

from distance import bfs_distance, compute_distance_matrix, normalize_D


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (0, 0, 0),   # same cell
        (0, 1, 1),   # adjacent horizontally
        (0, 3, 1),   # adjacent vertically
        (0, 2, 2),   # along a row
        (0, 8, 4),   # corner to corner (Manhattan on an open grid)
        (4, 0, 2),   # centre to a corner
    ],
)
def test_bfs_open_grid(open_3x3, src, dst, expected):
    assert bfs_distance(open_3x3, src, dst) == expected


def test_bfs_is_symmetric(open_3x3):
    assert bfs_distance(open_3x3, 2, 6) == bfs_distance(open_3x3, 6, 2)


def test_bfs_respects_walls_with_detour(detour_grid):
    # Direct Manhattan distance is 2, but the wall row forces a 6-step detour.
    assert bfs_distance(detour_grid, 0, 6) == 6


def test_bfs_unreachable_returns_inf(isolated_grid):
    assert bfs_distance(isolated_grid, 0, 8) == math.inf


def test_bfs_unreachable_same_cell_is_zero(isolated_grid):
    # The same-cell short circuit applies even when the cell is walled off.
    assert bfs_distance(isolated_grid, 0, 0) == 0


def test_compute_distance_matrix_shape_and_values(open_3x3):
    agents = [(0, 0), (8, 2)]   # direction is ignored
    goals = [8, 0]
    D = compute_distance_matrix(open_3x3, agents, goals)

    assert D.shape == (2, 2)
    # D[i, j] = bfs(agent_i, goal_j)
    np.testing.assert_array_equal(D, np.array([[4.0, 0.0], [0.0, 4.0]]))


def test_compute_distance_matrix_ignores_agent_direction(open_3x3):
    d0 = compute_distance_matrix(open_3x3, [(0, 0)], [8])
    d3 = compute_distance_matrix(open_3x3, [(0, 3)], [8])
    np.testing.assert_array_equal(d0, d3)


@pytest.mark.parametrize(
    "w, h, divisor",
    [
        (5, 5, 8),     # (5-1) + (5-1)
        (10, 10, 18),
        (3, 7, 8),     # (3-1) + (7-1)
    ],
)
def test_normalize_divides_by_max_bfs_distance(w, h, divisor):
    D = np.full((2, 2), float(divisor))
    np.testing.assert_allclose(normalize_D(D, w, h), np.ones((2, 2)))


def test_normalize_keeps_values_in_unit_range(open_5x5):
    agents = [(0, 0), (24, 0)]
    goals = [24, 0]
    D = compute_distance_matrix(open_5x5, agents, goals)
    D_norm = normalize_D(D, 5, 5)
    assert D_norm.min() >= 0.0
    assert D_norm.max() <= 1.0
