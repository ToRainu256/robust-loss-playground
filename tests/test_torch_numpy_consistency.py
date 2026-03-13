"""Cross-backend consistency tests: PyTorch vs NumPy.

For each of the 6 losses, create both PyTorch and NumPy instances with
identical parameters and verify that rho, influence, and weight produce
matching outputs to within floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.testing import assert_allclose

import robust_loss.numpy as np_losses
import robust_loss.torch as pt_losses

# Dense residual grid covering negative, zero, and positive regions.
R_NP = np.linspace(-5.0, 5.0, 101)
R_PT = torch.from_numpy(R_NP)

# Tolerance for cross-backend agreement.
ATOL = 1e-6
RTOL = 1e-6


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and convert a PyTorch tensor to NumPy array."""
    return t.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Helper: generic consistency check for a (np_loss, pt_loss) pair
# --------------------------------------------------------------------------- #
def _check_consistency(
    np_loss: np_losses.BaseRobustLoss,
    pt_loss: pt_losses.BaseRobustLoss,
    label: str,
) -> None:
    """Assert rho, influence, and weight match across backends."""
    # rho
    rho_np = np_loss.rho(R_NP)
    rho_pt = _to_numpy(pt_loss.rho(R_PT))
    assert_allclose(
        rho_pt, rho_np, atol=ATOL, rtol=RTOL,
        err_msg=f"{label} rho mismatch",
    )

    # influence
    psi_np = np_loss.influence(R_NP)
    psi_pt = _to_numpy(pt_loss.influence(R_PT))
    assert_allclose(
        psi_pt, psi_np, atol=ATOL, rtol=RTOL,
        err_msg=f"{label} influence mismatch",
    )

    # weight
    w_np = np_loss.weight(R_NP)
    w_pt = _to_numpy(pt_loss.weight(R_PT))
    assert_allclose(
        w_pt, w_np, atol=ATOL, rtol=RTOL,
        err_msg=f"{label} weight mismatch",
    )


# --------------------------------------------------------------------------- #
# Tests with default scale (scale=1.0)
# --------------------------------------------------------------------------- #
class TestL2Consistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.L2(scale=1.0, reduction="none")
        pt_loss = pt_losses.L2(scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "L2(scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.L2(scale=2.0, reduction="none")
        pt_loss = pt_losses.L2(scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "L2(scale=2)")


class TestL1Consistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.L1(scale=1.0, reduction="none")
        pt_loss = pt_losses.L1(scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "L1(scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.L1(scale=2.0, reduction="none")
        pt_loss = pt_losses.L1(scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "L1(scale=2)")


class TestHuberConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.Huber(delta=1.0, scale=1.0, reduction="none")
        pt_loss = pt_losses.Huber(delta=1.0, scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Huber(delta=1,scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.Huber(delta=1.0, scale=2.0, reduction="none")
        pt_loss = pt_losses.Huber(delta=1.0, scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Huber(delta=1,scale=2)")


class TestCharbonnierConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.Charbonnier(eps=1e-3, scale=1.0, reduction="none")
        pt_loss = pt_losses.Charbonnier(eps=1e-3, scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Charbonnier(eps=1e-3,scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.Charbonnier(eps=1e-3, scale=2.0, reduction="none")
        pt_loss = pt_losses.Charbonnier(eps=1e-3, scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Charbonnier(eps=1e-3,scale=2)")


class TestCauchyConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.Cauchy(scale=1.0, reduction="none")
        pt_loss = pt_losses.Cauchy(scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Cauchy(scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.Cauchy(scale=2.0, reduction="none")
        pt_loss = pt_losses.Cauchy(scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Cauchy(scale=2)")


class TestTukeyConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.Tukey(c=4.685, scale=1.0, reduction="none")
        pt_loss = pt_losses.Tukey(c=4.685, scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Tukey(c=4.685,scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.Tukey(c=4.685, scale=2.0, reduction="none")
        pt_loss = pt_losses.Tukey(c=4.685, scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Tukey(c=4.685,scale=2)")


class TestGemanMcClureConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.GemanMcClure(scale=1.0, reduction="none")
        pt_loss = pt_losses.GemanMcClure(scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "GemanMcClure(scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.GemanMcClure(scale=2.0, reduction="none")
        pt_loss = pt_losses.GemanMcClure(scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "GemanMcClure(scale=2)")


class TestWelschConsistency:
    def test_default_scale(self) -> None:
        np_loss = np_losses.Welsch(scale=1.0, reduction="none")
        pt_loss = pt_losses.Welsch(scale=1.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Welsch(scale=1)")

    def test_scale_2(self) -> None:
        np_loss = np_losses.Welsch(scale=2.0, reduction="none")
        pt_loss = pt_losses.Welsch(scale=2.0, reduction="none")
        _check_consistency(np_loss, pt_loss, "Welsch(scale=2)")
