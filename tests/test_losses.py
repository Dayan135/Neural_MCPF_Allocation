"""Tests for the mTSP hybrid loss."""

import pytest
import torch

from model.losses import mTSP_loss


@pytest.fixture
def perfect_case():
    """Ground truth and a matching one-hot prediction (columns sum to 1)."""
    Y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])   # agent0->goal0, agent1->goal1
    D = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    return Y, D


def test_returns_three_terms(perfect_case):
    Y, D = perfect_case
    out = mTSP_loss(Y, Y, D, lam=0.1)
    assert len(out) == 3
    total, l_ce, l_minsum = out
    assert torch.isclose(total, l_ce + 0.1 * l_minsum)


def test_perfect_prediction_has_zero_ce(perfect_case):
    Y, D = perfect_case
    _, l_ce, _ = mTSP_loss(Y, Y, D, lam=0.0)
    assert torch.isclose(l_ce, torch.tensor(0.0), atol=1e-6)


def test_minsum_matches_manual_computation(perfect_case):
    Y, D = perfect_case
    # With P == Y, expected cost = sum of D at assigned entries / M
    # assigned D = 0.1 + 0.2 = 0.3 ; M = 2 -> 0.15
    _, _, l_minsum = mTSP_loss(Y, Y, D, lam=1.0)
    assert torch.isclose(l_minsum, torch.tensor(0.15), atol=1e-6)


def test_ce_increases_for_wrong_prediction(perfect_case):
    Y, D = perfect_case
    wrong = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])   # swapped assignment
    _, ce_right, _ = mTSP_loss(Y, Y, D, lam=0.0)
    _, ce_wrong, _ = mTSP_loss(wrong, Y, D, lam=0.0)
    assert ce_wrong > ce_right


def test_loss_is_finite_with_zero_probabilities(perfect_case):
    # P containing exact 0s must not produce -inf/nan thanks to the clamp guard.
    Y, D = perfect_case
    total, l_ce, l_minsum = mTSP_loss(Y, Y, D, lam=0.1)
    assert torch.isfinite(total)
    assert torch.isfinite(l_ce)
    assert torch.isfinite(l_minsum)


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1, 1.0])
def test_lambda_weighting(perfect_case, lam):
    Y, D = perfect_case
    total, l_ce, l_minsum = mTSP_loss(Y, Y, D, lam=lam)
    assert torch.isclose(total, l_ce + lam * l_minsum)


def test_gradients_flow_through_prediction(perfect_case):
    Y, D = perfect_case
    P = torch.full_like(Y, 0.5, requires_grad=True)
    total, _, _ = mTSP_loss(P, Y, D, lam=0.1)
    total.backward()
    assert P.grad is not None
    assert torch.isfinite(P.grad).all()


@pytest.mark.parametrize("N, M", [(2, 2), (3, 3), (2, 5), (4, 2)])
def test_handles_rectangular_shapes(N, M):
    P = torch.softmax(torch.randn(4, N, M), dim=1)
    Y = torch.zeros(4, N, M)
    Y[:, 0, :] = 1.0   # all goals to agent 0
    D = torch.rand(4, N, M)
    total, l_ce, l_minsum = mTSP_loss(P, Y, D, lam=0.1)
    assert total.ndim == 0   # scalar
    assert torch.isfinite(total)
