"""
MLP-based mTSP goal allocation network.

For each goal j, predicts which agent i should visit it (column-wise softmax).
Columns of the output sum to 1; rows are unconstrained (general mTSP assignment).

Input  : D (B, N, M)  normalized distance matrix — agent i to goal j
Output : P (B, N, M)  assignment probabilities;  P[:, :, j].sum(dim=1) == 1
"""

import torch
import torch.nn as nn
from torch import Tensor


class GoalAllocMLP(nn.Module):
    """
    MLP: flatten D -> hidden -> hidden -> N*M logits -> column softmax.

    Parameters
    ----------
    N      : number of agents
    M      : number of goals (defaults to N when None)
    hidden : hidden layer width
    """

    def __init__(self, N: int, M: int | None = None, hidden: int = 64):
        super().__init__()
        self.N = N
        self.M = M if M is not None else N

        self.net = nn.Sequential(
            nn.Linear(self.N * self.M, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.N * self.M),
        )

    def forward(self, D: Tensor) -> Tensor:
        """
        D : (B, N, M) normalized distance matrix
        Returns P : (B, N, M) assignment probabilities (columns sum to 1)
        """
        B = D.shape[0]
        logits = self.net(D.view(B, -1)).view(B, self.N, self.M)
        return torch.softmax(logits, dim=1)   # softmax over agents for each goal
