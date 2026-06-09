"""
Hybrid loss for mTSP goal-allocation training.

Total_Loss = L_CE + λ * L_MinSum

L_CE     : per-goal cross-entropy — for each goal j, which agent i visits it?
           Y[:, j] is a one-hot over agents (exactly one agent per goal).
L_MinSum : expected assignment cost — penalises high-distance allocations.
"""

import torch
from torch import Tensor


def mTSP_loss(
    P: Tensor,
    Y: Tensor,
    D: Tensor,
    lam: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Parameters
    ----------
    P   : (B, N, M)  column-wise softmax output from the model
    Y   : (B, N, M)  binary ground-truth assignment (columns sum to 1)
    D   : (B, N, M)  normalized distance matrix
    lam : weight on the MinSum auxiliary loss

    Returns
    -------
    total, l_ce, l_minsum  — individual terms for logging
    """
    eps = 1e-7
    # Per-goal cross-entropy: sum over agents, mean over batch & goals
    l_ce = -(Y * P.clamp(eps).log()).sum(dim=1).mean()

    M = Y.shape[2]
    l_minsum = (P * D).sum(dim=(1, 2)).mean() / M

    return l_ce + lam * l_minsum, l_ce, l_minsum
