"""Known-value correctness tests for all 6 PyTorch loss functions.

Each test verifies rho, influence, and weight against hand-computed values
using r = [-2, -1, 0, 1, 2] with scale=1.0, reduction="none".
"""

from __future__ import annotations

import math

import torch
from torch.testing import assert_close

from robust_loss.torch import L1, L2, Cauchy, Charbonnier, Huber, Tukey

# Shared residual tensor used across all tests.
R = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])


# --------------------------------------------------------------------------- #
# L2: rho(r) = 1/2 * r^2
# --------------------------------------------------------------------------- #
class TestL2:
    def setup_method(self) -> None:
        self.loss = L2(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = torch.tensor([2.0, 0.5, 0.0, 0.5, 2.0])
        assert_close(self.loss.rho(R), expected, atol=1e-12, rtol=0.0)

    def test_influence(self) -> None:
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert_close(self.loss.influence(R), expected, atol=1e-12, rtol=0.0)

    def test_weight(self) -> None:
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        assert_close(self.loss.weight(R), expected, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# L1: rho(r) = |r|
# --------------------------------------------------------------------------- #
class TestL1:
    def setup_method(self) -> None:
        self.loss = L1(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
        assert_close(self.loss.rho(R), expected, atol=1e-12, rtol=0.0)

    def test_influence(self) -> None:
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert_close(self.loss.influence(R), expected, atol=1e-12, rtol=0.0)

    def test_weight(self) -> None:
        # w(r) = sign(r)/r; limit at 0 returns 0.0
        expected = torch.tensor([0.5, 1.0, 0.0, 1.0, 0.5])
        assert_close(self.loss.weight(R), expected, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# Huber(delta=1)
# --------------------------------------------------------------------------- #
class TestHuber:
    def setup_method(self) -> None:
        self.loss = Huber(delta=1.0, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = torch.tensor([1.5, 0.5, 0.0, 0.5, 1.5])
        assert_close(self.loss.rho(R), expected, atol=1e-12, rtol=0.0)

    def test_influence(self) -> None:
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert_close(self.loss.influence(R), expected, atol=1e-12, rtol=0.0)

    def test_weight(self) -> None:
        expected = torch.tensor([0.5, 1.0, 1.0, 1.0, 0.5])
        assert_close(self.loss.weight(R), expected, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# Charbonnier(eps=1e-3)
# --------------------------------------------------------------------------- #
class TestCharbonnier:
    def setup_method(self) -> None:
        self.eps = 1e-3
        self.loss = Charbonnier(eps=self.eps, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        eps = self.eps
        u = R  # scale=1
        expected = torch.sqrt(u**2 + eps**2) - eps
        assert_close(self.loss.rho(R), expected, rtol=1e-6, atol=0.0)

    def test_influence(self) -> None:
        eps = self.eps
        u = R
        expected = R / torch.sqrt(u**2 + eps**2)
        assert_close(self.loss.influence(R), expected, rtol=1e-6, atol=0.0)

    def test_weight(self) -> None:
        eps = self.eps
        u = R
        result = self.loss.weight(R)
        # Nonzero residuals: w = 1/sqrt(u^2 + eps^2)
        mask = R.abs() > 1e-12
        expected_nonzero = 1.0 / torch.sqrt(u[mask] ** 2 + eps**2)
        assert_close(result[mask], expected_nonzero, rtol=1e-6, atol=0.0)
        # r=0 limit is 1/eps
        assert_close(result[~mask], torch.tensor([1.0 / eps]), rtol=1e-6, atol=0.0)

    def test_rho_at_zero(self) -> None:
        result = self.loss.rho(torch.tensor([0.0]))
        assert_close(result, torch.tensor([0.0]), atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# Cauchy
# --------------------------------------------------------------------------- #
class TestCauchy:
    def setup_method(self) -> None:
        self.loss = Cauchy(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = 0.5 * torch.log1p(R**2)
        assert_close(self.loss.rho(R), expected, rtol=1e-6, atol=0.0)

    def test_influence(self) -> None:
        expected = R / (1.0 + R**2)
        assert_close(self.loss.influence(R), expected, rtol=1e-6, atol=0.0)

    def test_weight(self) -> None:
        result = self.loss.weight(R)
        mask = R.abs() > 1e-12
        expected_nonzero = 1.0 / (1.0 + R[mask] ** 2)
        assert_close(result[mask], expected_nonzero, rtol=1e-6, atol=0.0)
        assert_close(result[~mask], torch.tensor([1.0]), atol=1e-12, rtol=0.0)

    def test_rho_known_values(self) -> None:
        r = torch.tensor([0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.5 * math.log(2.0), 0.5 * math.log(5.0)])
        assert_close(self.loss.rho(r), expected, rtol=1e-10, atol=0.0)


# --------------------------------------------------------------------------- #
# Tukey(c=4.685)
# --------------------------------------------------------------------------- #
class TestTukey:
    def setup_method(self) -> None:
        self.c = 4.685
        self.loss = Tukey(c=self.c, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        c = self.c
        u = R
        c2_over_6 = c**2 / 6.0
        ratio2 = (u / c) ** 2
        expected = torch.where(
            u.abs() <= c,
            c2_over_6 * (1.0 - (1.0 - ratio2) ** 3),
            torch.tensor(c2_over_6),
        )
        assert_close(self.loss.rho(R), expected, rtol=1e-6, atol=0.0)

    def test_influence(self) -> None:
        c = self.c
        u = R
        ratio2 = (u / c) ** 2
        expected = torch.where(
            u.abs() <= c, R * (1.0 - ratio2) ** 2, torch.tensor(0.0)
        )
        assert_close(self.loss.influence(R), expected, rtol=1e-6, atol=0.0)

    def test_weight(self) -> None:
        c = self.c
        u = R
        ratio2 = (u / c) ** 2
        expected = torch.where(u.abs() <= c, (1.0 - ratio2) ** 2, torch.tensor(0.0))
        # At r=0, weight returns 1.0 via limit
        expected[R == 0.0] = 1.0
        assert_close(self.loss.weight(R), expected, rtol=1e-6, atol=0.0)

    def test_rho_outside_c(self) -> None:
        r_far = torch.tensor([-10.0, -5.0, 5.0, 10.0])
        expected = torch.full_like(r_far, self.c**2 / 6.0)
        assert_close(self.loss.rho(r_far), expected, rtol=1e-12, atol=0.0)

    def test_influence_outside_c(self) -> None:
        r_far = torch.tensor([-10.0, -5.0, 5.0, 10.0])
        expected = torch.zeros_like(r_far)
        assert_close(self.loss.influence(r_far), expected, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# Scale=2.0 tests
# --------------------------------------------------------------------------- #
class TestScaledHuber:
    """Verify that the s^2 * phi(r/s) scaling convention works correctly."""

    def setup_method(self) -> None:
        self.loss = Huber(delta=1.0, scale=2.0, reduction="none")

    def test_rho_with_scale(self) -> None:
        # u = r/2 = [-1, -0.5, 0, 0.5, 1]; all |u| <= 1
        # rho = s^2 * 0.5 * u^2 = 4 * 0.5 * (r/2)^2 = 0.5 * r^2
        expected = torch.tensor([2.0, 0.5, 0.0, 0.5, 2.0])
        assert_close(self.loss.rho(R), expected, atol=1e-12, rtol=0.0)

    def test_influence_with_scale(self) -> None:
        # All |u| <= 1, so psi = r
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert_close(self.loss.influence(R), expected, atol=1e-12, rtol=0.0)


class TestScaledCauchy:
    """Verify Cauchy scaling: rho(r;s) = s^2/2 * log(1 + (r/s)^2)."""

    def setup_method(self) -> None:
        self.scale = 2.0
        self.loss = Cauchy(scale=self.scale, reduction="none")

    def test_rho_with_scale(self) -> None:
        s = self.scale
        u = R / s
        expected = 0.5 * s**2 * torch.log1p(u**2)
        assert_close(self.loss.rho(R), expected, rtol=1e-6, atol=0.0)

    def test_influence_with_scale(self) -> None:
        s = self.scale
        u = R / s
        expected = R / (1.0 + u**2)
        assert_close(self.loss.influence(R), expected, rtol=1e-6, atol=0.0)
