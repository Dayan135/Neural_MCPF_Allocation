"""Tests for goal-allocation network architectures and the column-softmax invariant."""

import pytest
import torch

from model.network import GoalAllocMLP, GoalAllocTransformer, build_model


@pytest.mark.parametrize("N, M", [(2, 2), (3, 3), (2, 4), (5, 3)])
def test_output_shape(N, M):
    model = GoalAllocMLP(N=N, M=M, hidden=32)
    D = torch.rand(8, N, M)
    P = model(D)
    assert P.shape == (8, N, M)


@pytest.mark.parametrize("N, M", [(2, 2), (3, 3), (2, 4), (5, 3)])
def test_columns_sum_to_one(N, M):
    # Each goal's distribution over agents must sum to 1 (mTSP column constraint).
    model = GoalAllocMLP(N=N, M=M, hidden=32)
    D = torch.rand(8, N, M)
    P = model(D)
    col_sums = P.sum(dim=1)   # (B, M)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums))


def test_output_is_a_probability():
    model = GoalAllocMLP(N=3, M=3, hidden=32)
    P = model(torch.rand(4, 3, 3))
    assert (P >= 0).all()
    assert (P <= 1).all()


def test_m_defaults_to_n():
    model = GoalAllocMLP(N=3, hidden=16)
    assert model.M == 3
    P = model(torch.rand(2, 3, 3))
    assert P.shape == (2, 3, 3)


def test_forward_is_differentiable():
    model = GoalAllocMLP(N=2, M=2, hidden=16)
    D = torch.rand(4, 2, 2, requires_grad=True)
    P = model(D)
    P.sum().backward()
    assert D.grad is not None
    assert torch.isfinite(D.grad).all()


def test_batch_independence():
    # Row b of the output depends only on row b of the input.
    torch.manual_seed(0)
    model = GoalAllocMLP(N=2, M=2, hidden=16).eval()
    d0 = torch.rand(1, 2, 2)
    d1 = torch.rand(1, 2, 2)
    stacked = model(torch.cat([d0, d1], dim=0))
    solo0 = model(d0)
    torch.testing.assert_close(stacked[0:1], solo0)


class TestTransformerGoalDists:
    """GoalAllocTransformer with use_goal_dists=True."""

    @pytest.mark.parametrize("N, M", [(2, 2), (3, 3), (4, 3)])
    def test_shape_and_column_sum(self, N, M):
        model = GoalAllocTransformer(N=N, M=M, hidden=32, num_heads=4,
                                     num_layers=2, use_goal_dists=True)
        D = torch.rand(4, N, M)
        G = torch.rand(4, M, M)
        P = model(D, G=G)
        assert P.shape == (4, N, M)
        col_sums = P.sum(dim=1)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums))

    def test_backward(self):
        model = GoalAllocTransformer(N=3, M=3, hidden=32, num_heads=4,
                                     num_layers=2, use_goal_dists=True)
        D = torch.rand(4, 3, 3, requires_grad=True)
        G = torch.rand(4, 3, 3)
        P = model(D, G=G)
        P.sum().backward()
        assert D.grad is not None
        assert torch.isfinite(D.grad).all()

    def test_g_ignored_when_flag_off(self):
        # Without use_goal_dists, passing G must be harmless (G is ignored).
        model = GoalAllocTransformer(N=3, M=3, hidden=32, use_goal_dists=False)
        D = torch.rand(4, 3, 3)
        G = torch.rand(4, 3, 3)
        P_with = model(D, G=G)
        P_without = model(D)
        torch.testing.assert_close(P_with, P_without)

    def test_build_model_factory(self):
        m = build_model("transformer", N=3, M=3, hidden=32, use_goal_dists=True)
        assert isinstance(m, GoalAllocTransformer)
        assert m.use_goal_dists is True
        assert m.goal_ctx_proj is not None
