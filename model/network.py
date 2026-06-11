"""
Goal allocation networks for mTSP assignment.

All models share the same interface:
  Input  : D (B, N, M)  normalized distance matrix — agent i to goal j
  Output : P (B, N, M)  assignment probabilities;  P[:, :, j].sum(dim=1) == 1

Three architectures (order of increasing inductive bias):
  GoalAllocMLP        — flat MLP, baseline, fixed N×M input size
  GoalAllocDeepSets   — per-goal MLP with shared weights; no cross-goal communication
  GoalAllocTransformer — row-column attention on D; captures agent-goal and goal-goal interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        B = D.shape[0]
        logits = self.net(D.view(B, -1)).view(B, self.N, self.M)
        return torch.softmax(logits, dim=1)


class GoalAllocDeepSets(nn.Module):
    """
    Per-goal MLP with weights shared across goals.

    For each goal j, the input is D[:, :, j] ∈ R^(B,N) — the distance from
    every agent to that goal.  A shared MLP maps this to N logits; column
    softmax gives the assignment distribution for that goal.

    This is strictly stronger than GoalAllocMLP (parameter sharing forces
    permutation equivariance over goals) but weaker than the Transformer
    (no communication across goals).

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

        # shared MLP: R^N -> R^N
        self.net = nn.Sequential(
            nn.Linear(N, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N),
        )

    def forward(self, D: Tensor) -> Tensor:
        B, N, M = D.shape
        # Reshape so each goal is an independent "sample": (B*M, N)
        x = D.permute(0, 2, 1).reshape(B * M, N)   # (B*M, N)
        logits = self.net(x).view(B, M, N)           # (B, M, N)
        logits = logits.permute(0, 2, 1)             # (B, N, M)
        return torch.softmax(logits, dim=1)


class _RowColBlock(nn.Module):
    """One block of row-attention + column-attention + FFN."""

    def __init__(self, d: int, num_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d),
            nn.ReLU(),
            nn.Linear(ffn_mult * d, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, E: Tensor) -> Tensor:
        """E: (B, N, M, d)"""
        B, N, M, d = E.shape

        # Row attention: each agent attends over M goal embeddings
        Er = E.view(B * N, M, d)
        attn_out, _ = self.row_attn(Er, Er, Er)
        E = E + self.norm1(attn_out.view(B, N, M, d))

        # Column attention: each goal attends over N agent embeddings
        Ec = E.permute(0, 2, 1, 3).reshape(B * M, N, d)
        attn_out, _ = self.col_attn(Ec, Ec, Ec)
        E = E + self.norm2(attn_out.view(B, M, N, d).permute(0, 2, 1, 3))

        # FFN
        E = E + self.norm3(self.ffn(E))
        return E


class GoalAllocTransformer(nn.Module):
    """
    Row-column attention transformer on the distance matrix D.

    Architecture:
      D (B,N,M) → embed each scalar D[b,i,j] to R^d
      + learned agent/goal positional embeddings
      [+ goal-goal context from G (B,M,M) if use_goal_dists=True]
      → E (B,N,M,d)
      → L × [RowAttn + ColAttn + FFN + LayerNorm]
      → Linear(d→1) → logits (B,N,M)
      → column softmax → P (B,N,M)

    Parameters
    ----------
    N              : number of agents
    M              : number of goals (defaults to N when None)
    hidden         : embedding dimension d
    num_heads      : attention heads per block
    num_layers     : number of row-column blocks
    use_goal_dists : if True, accept G (B,M,M) and inject goal-neighbourhood
                     context via a Linear(M→d) projection before the blocks
    """

    def __init__(
        self,
        N: int,
        M: int | None = None,
        hidden: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        use_goal_dists: bool = False,
    ):
        super().__init__()
        self.N = N
        self.M = M if M is not None else N
        d = hidden
        self.use_goal_dists = use_goal_dists

        self.input_proj = nn.Linear(1, d)
        self.agent_emb = nn.Embedding(self.N, d)
        self.goal_emb = nn.Embedding(self.M, d)

        # Optional goal-neighbourhood context: for each goal j, G[b,j,:] ∈ R^M
        # encodes distances to all other goals.  Linear(M→d) projects it into
        # the same space as the agent-goal embeddings.
        if use_goal_dists:
            self.goal_ctx_proj = nn.Linear(self.M, d)
        else:
            self.goal_ctx_proj = None

        self.blocks = nn.ModuleList([
            _RowColBlock(d, num_heads) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d, 1)

    def forward(self, D: Tensor, G: Tensor | None = None) -> Tensor:
        B, N, M = D.shape
        device = D.device

        E = self.input_proj(D.unsqueeze(-1))

        agent_ids = torch.arange(N, device=device)
        goal_ids = torch.arange(M, device=device)
        E = E + self.agent_emb(agent_ids)[None, :, None, :]
        E = E + self.goal_emb(goal_ids)[None, None, :, :]

        if self.goal_ctx_proj is not None and G is not None:
            # G: (B, M, M); goal_ctx_proj maps R^M → R^d per goal row
            goal_ctx = self.goal_ctx_proj(G)   # (B, M, d)
            E = E + goal_ctx[:, None, :, :]    # broadcast (B, 1, M, d) → (B, N, M, d)

        for block in self.blocks:
            E = block(E)

        logits = self.out_proj(E).squeeze(-1)
        return torch.softmax(logits, dim=1)


def build_model(
    model_type: str,
    N: int,
    M: int,
    hidden: int,
    num_heads: int = 4,
    num_layers: int = 3,
    use_goal_dists: bool = False,
) -> nn.Module:
    """Factory used by train.py and evaluate.py."""
    if model_type == "mlp":
        return GoalAllocMLP(N=N, M=M, hidden=hidden)
    elif model_type == "deepsets":
        return GoalAllocDeepSets(N=N, M=M, hidden=hidden)
    elif model_type == "transformer":
        return GoalAllocTransformer(
            N=N, M=M, hidden=hidden,
            num_heads=num_heads, num_layers=num_layers,
            use_goal_dists=use_goal_dists,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
