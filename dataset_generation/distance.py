"""BFS-based distance matrix computation and normalization."""

from collections import deque
import numpy as np


def bfs_distance(map_dims: dict, src_flat: int, dst_flat: int) -> float:
    """
    Shortest path length from src to dst on the grid, respecting obstacles.
    Returns float('inf') if dst is unreachable from src.
    """
    if src_flat == dst_flat:
        return 0

    rows = map_dims["Rows"]
    cols = map_dims["Cols"]
    grid = map_dims["Map"]

    queue = deque([(src_flat, 0)])
    visited = {src_flat}

    while queue:
        loc, dist = queue.popleft()
        r, c = divmod(loc, cols)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = nr * cols + nc
                if nb == dst_flat:
                    return dist + 1
                if nb not in visited and grid[nb] == 0:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

    return float("inf")


def compute_distance_matrix(map_dims: dict, agents: list, goals: list) -> np.ndarray:
    """
    Compute the N×N distance matrix D where D[i,j] = BFS distance from agent i to goal j.

    agents : list of (flat_idx, direction) — direction is ignored
    goals  : list of flat_idx
    """
    N = len(agents)
    D = np.zeros((N, N), dtype=float)
    for i, (agent_flat, _) in enumerate(agents):
        for j, goal_flat in enumerate(goals):
            D[i, j] = bfs_distance(map_dims, agent_flat, goal_flat)
    return D


def normalize_D(D: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Normalize D to [0, 1] using the maximum possible BFS distance on an
    obstacle-free grid: (W-1) + (H-1).
    """
    max_dist = (W - 1) + (H - 1)
    return D / max_dist
