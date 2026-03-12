"""Tests for influence function (vs autograd) and weight function (r->0 limit).

(A) Influence vs autograd: verify that loss_fn.influence(r) matches
    torch.autograd.grad(rho.sum(), r) for all losses.
(B) Weight at r->0: verify that weight(r) near zero is finite and matches
    the expected analytical limit.
"""

from __future__ import annotations

import pytest
import torch

from robust_loss.torch import L1, L2, Cauchy, Charbonnier, Huber, Tukey

# =========================================================================== #
# (A) Influence vs autograd
# =========================================================================== #

# Smooth losses can use random inputs including values near zero.
SMOOTH_LOSSES_SCALE1 = [
    pytest.param(L2(scale=1.0, reduction="none"), id="L2-s1"),
    pytest.param(Huber(delta=1.0, scale=1.0, reduction="none"), id="Huber-s1"),
    pytest.param(
        Charbonnier(eps=1e-3, scale=1.0, reduction="none"), id="Charbonnier-s1"
    ),
    pytest.param(Cauchy(scale=1.0, reduction="none"), id="Cauchy-s1"),
    pytest.param(Tukey(c=4.685, scale=1.0, reduction="none"), id="Tukey-s1"),
]

SMOOTH_LOSSES_SCALE2 = [
    pytest.param(L2(scale=2.0, reduction="none"), id="L2-s2"),
    pytest.param(Huber(delta=1.0, scale=2.0, reduction="none"), id="Huber-s2"),
    pytest.param(
        Charbonnier(eps=1e-3, scale=2.0, reduction="none"), id="Charbonnier-s2"
    ),
    pytest.param(Cauchy(scale=2.0, reduction="none"), id="Cauchy-s2"),
    pytest.param(Tukey(c=4.685, scale=2.0, reduction="none"), id="Tukey-s2"),
]


@pytest.mark.parametrize("loss_fn", SMOOTH_LOSSES_SCALE1)
def test_influence_vs_autograd_scale1(loss_fn: torch.nn.Module) -> None:
    """Influence must match autograd gradient for smooth losses (scale=1)."""
    r = torch.randn(16, dtype=torch.float64, requires_grad=True)
    rho = loss_fn.rho(r)
    (grad,) = torch.autograd.grad(rho.sum(), r)
    psi = loss_fn.influence(r)
    assert torch.allclose(grad, psi, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("loss_fn", SMOOTH_LOSSES_SCALE2)
def test_influence_vs_autograd_scale2(loss_fn: torch.nn.Module) -> None:
    """Influence must match autograd gradient for smooth losses (scale=2)."""
    r = torch.randn(16, dtype=torch.float64, requires_grad=True)
    rho = loss_fn.rho(r)
    (grad,) = torch.autograd.grad(rho.sum(), r)
    psi = loss_fn.influence(r)
    assert torch.allclose(grad, psi, atol=1e-8, rtol=1e-6)


def test_influence_vs_autograd_l1_scale1() -> None:
    """L1 influence vs autograd — use inputs that avoid zero (non-smooth)."""
    loss_fn = L1(scale=1.0, reduction="none")
    r = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
    )
    rho = loss_fn.rho(r)
    (grad,) = torch.autograd.grad(rho.sum(), r)
    psi = loss_fn.influence(r)
    assert torch.allclose(grad, psi, atol=1e-8, rtol=1e-6)


def test_influence_vs_autograd_l1_scale2() -> None:
    """L1 influence vs autograd with scale=2 — inputs avoid zero."""
    loss_fn = L1(scale=2.0, reduction="none")
    r = torch.tensor(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True
    )
    rho = loss_fn.rho(r)
    (grad,) = torch.autograd.grad(rho.sum(), r)
    psi = loss_fn.influence(r)
    assert torch.allclose(grad, psi, atol=1e-8, rtol=1e-6)


# =========================================================================== #
# (B) Weight at r -> 0
# =========================================================================== #

WEIGHT_LIMIT_CASES = [
    pytest.param(L2(scale=1.0, reduction="none"), 1.0, id="L2"),
    pytest.param(L1(scale=1.0, reduction="none"), 0.0, id="L1"),
    pytest.param(Huber(delta=1.0, scale=1.0, reduction="none"), 1.0, id="Huber"),
    pytest.param(
        Charbonnier(eps=1e-3, scale=1.0, reduction="none"), 1000.0, id="Charbonnier"
    ),
    pytest.param(Cauchy(scale=1.0, reduction="none"), 1.0, id="Cauchy"),
    pytest.param(Tukey(c=4.685, scale=1.0, reduction="none"), 1.0, id="Tukey"),
]


@pytest.mark.parametrize("loss_fn, expected_limit", WEIGHT_LIMIT_CASES)
def test_weight_near_zero_is_finite(
    loss_fn: torch.nn.Module, expected_limit: float
) -> None:
    """Weight at r~0 must be finite and match the analytical limit."""
    r_small = torch.tensor([1e-15, -1e-15, 0.0])
    w = loss_fn.weight(r_small)
    assert torch.all(torch.isfinite(w)), f"Non-finite weight: {w}"
    expected = torch.tensor(expected_limit).expand_as(w)
    assert torch.allclose(w, expected, atol=1e-6), (
        f"Weight near zero: got {w}, expected {expected}"
    )


@pytest.mark.parametrize("loss_fn, expected_limit", WEIGHT_LIMIT_CASES)
def test_weight_at_exact_zero(
    loss_fn: torch.nn.Module, expected_limit: float
) -> None:
    """Weight at exactly r=0 must return the limit value."""
    r_zero = torch.tensor([0.0])
    w = loss_fn.weight(r_zero)
    assert torch.isfinite(w).all()
    assert torch.allclose(
        w, torch.tensor([expected_limit]), atol=1e-6
    ), f"Weight at 0: got {w.item()}, expected {expected_limit}"


# =========================================================================== #
# Additional: weight limit with scale=2.0
# =========================================================================== #

WEIGHT_LIMIT_SCALED_CASES = [
    pytest.param(L2(scale=2.0, reduction="none"), 1.0, id="L2-s2"),
    pytest.param(L1(scale=2.0, reduction="none"), 0.0, id="L1-s2"),
    pytest.param(Huber(delta=1.0, scale=2.0, reduction="none"), 1.0, id="Huber-s2"),
    pytest.param(
        Charbonnier(eps=1e-3, scale=2.0, reduction="none"),
        1000.0,
        id="Charbonnier-s2",
    ),
    pytest.param(Cauchy(scale=2.0, reduction="none"), 1.0, id="Cauchy-s2"),
    pytest.param(Tukey(c=4.685, scale=2.0, reduction="none"), 1.0, id="Tukey-s2"),
]


@pytest.mark.parametrize("loss_fn, expected_limit", WEIGHT_LIMIT_SCALED_CASES)
def test_weight_near_zero_scaled(
    loss_fn: torch.nn.Module, expected_limit: float
) -> None:
    """Weight limit at r->0 should hold regardless of scale."""
    r_small = torch.tensor([1e-15, -1e-15, 0.0])
    w = loss_fn.weight(r_small)
    assert torch.all(torch.isfinite(w)), f"Non-finite weight: {w}"
    expected = torch.tensor(expected_limit).expand_as(w)
    assert torch.allclose(w, expected, atol=1e-6), (
        f"Weight near zero (scaled): got {w}, expected {expected}"
    )
