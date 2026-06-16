"""
Random grid generation in RobustMCPF's MapAndDims format and agent/goal placement.

The placement logic mirrors GenerateInstances.create_positions_for_agents_And_Locs_For_Goals
but is inlined here because GenerateInstances.py runs map-loading code at module level,
making it impossible to import from without side effects.
"""

import os
import random
import numpy as np

# Benchmark maps vendored with RobustMCPF (MovingAI octile format).
MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RobustMCPF", "Maps")


def load_map_file(name_or_path: str) -> dict:
    """
    Load a MovingAI octile .map file into a MapAndDims dict.

    Mirrors RobustMCPF/GenerateInstances.read_map_file: skips the header up to
    the "map" line, then '.' -> 0 (free), any other char (e.g. '@', 'T') -> 1
    (obstacle). A bare filename is resolved against the vendored Maps/ dir;
    anything containing a path separator is used as-is.

    Returns {"Rows": H, "Cols": W, "Map": [0/1, ...]}, the same format as
    generate_random_map.
    """
    path = name_or_path if os.sep in name_or_path else os.path.join(MAPS_DIR, name_or_path)
    with open(path, "r") as f:
        lines = f.readlines()

    map_start = lines.index("map\n") + 1
    map_lines = [ln.rstrip("\n") for ln in lines[map_start:] if ln.strip() != ""]

    flat = []
    rows = 0
    cols = 0
    for line in map_lines:
        cols = len(line)
        rows += 1
        flat.extend(0 if ch == "." else 1 for ch in line)
    return {"Rows": rows, "Cols": cols, "Map": flat}


def generate_random_map(W: int, H: int, obstacle_prob: float = 0.1, seed=None) -> dict:
    """
    Generate a random grid and return it as a MapAndDims dict.

    Returns {"Rows": H, "Cols": W, "Map": [0/1, ...]} where 0=free, 1=obstacle.
    """
    rng = np.random.default_rng(seed)
    flat = rng.choice([0, 1], size=H * W, p=[1 - obstacle_prob, obstacle_prob]).tolist()
    return {"Rows": H, "Cols": W, "Map": flat}


def sample_agents_goals(map_dims: dict, N: int, M: int | None = None, seed=None) -> tuple[list, list]:
    """
    Randomly place N agents and M goals on distinct free cells.

    M defaults to N (the N=M case used in all current experiments).

    Uses the same algorithm as GenerateInstances.create_positions_for_agents_And_Locs_For_Goals:
    - Agents get a random direction in {0, 1, 2, 3} (ignored in BasicMAPF mode)
    - Goals are plain flat indices

    Returns
    -------
    agents : list of (flat_idx, direction)
    goals  : list of flat_idx
    """
    if M is None:
        M = N
    if seed is not None:
        random.seed(seed)

    free = [i for i, v in enumerate(map_dims["Map"]) if v == 0]
    if len(free) < N + M:
        raise ValueError(
            f"Grid has only {len(free)} free cells; need at least {N + M} for {N} agents + {M} goals."
        )

    agent_locs = random.sample(free, N)
    agents = [(loc, random.randint(0, 3)) for loc in agent_locs]

    remaining = [idx for idx in free if idx not in set(agent_locs)]
    goals = random.sample(remaining, M)

    return agents, goals
