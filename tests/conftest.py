"""
Shared pytest fixtures and import-path setup.

The project modules use flat imports (`import distance`, `from solver_wrapper import ...`),
so both the project root and dataset_generation/ must be on sys.path.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "dataset_generation")):
    if path not in sys.path:
        sys.path.insert(0, path)


def _open_map(w: int, h: int) -> dict:
    """An obstacle-free w x h grid in MapAndDims format."""
    return {"Rows": h, "Cols": w, "Map": [0] * (w * h)}


@pytest.fixture
def open_3x3() -> dict:
    """
    3x3 obstacle-free grid. Flat indices:
        0 1 2
        3 4 5
        6 7 8
    """
    return _open_map(3, 3)


@pytest.fixture
def open_5x5() -> dict:
    """5x5 obstacle-free grid (matches the default dataset config)."""
    return _open_map(5, 5)


@pytest.fixture
def detour_grid() -> dict:
    """
    3x3 grid with a wall row forcing a detour. Layout (. free, @ wall):
        . . .      0 1 2
        @ @ .      3 4 5   (3, 4 are walls)
        . . .      6 7 8
    Shortest path 0 -> 6 must go 0->1->2->5->8->7->6 = 6 steps (Manhattan would be 2).
    """
    return {"Rows": 3, "Cols": 3, "Map": [0, 0, 0, 1, 1, 0, 0, 0, 0]}


@pytest.fixture
def isolated_grid() -> dict:
    """
    3x3 grid where cell 0 is free but walled off from the rest:
        . @ .      0 1 2   (1 is wall)
        @ . .      3 4 5   (3 is wall)
        . . .      6 7 8
    Cell 0's only neighbours (1, 3) are walls -> everything else is unreachable from 0.
    """
    return {"Rows": 3, "Cols": 3, "Map": [0, 1, 0, 1, 0, 0, 0, 0, 0]}


@pytest.fixture
def full_grid() -> dict:
    """3x3 grid with a single free cell -> too few free cells to place agents+goals."""
    return {"Rows": 3, "Cols": 3, "Map": [0, 1, 1, 1, 1, 1, 1, 1, 1]}
