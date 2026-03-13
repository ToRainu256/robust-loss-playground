"""Known-value correctness tests for all 6 NumPy loss functions.

Each test verifies rho, influence, and weight against hand-computed values
using r = [-2, -1, 0, 1, 2] with scale=1.0, reduction="none".
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from robust_loss.numpy import L1, L2, Cauchy, Charbonnier, GemanMcClure, Huber, Tukey, Welsch

# Shared residual vector used across all tests.
R = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


# --------------------------------------------------------------------------- #
# L2: rho(r) = 1/2 * r^2
# --------------------------------------------------------------------------- #
class TestL2:
    def setup_method(self) -> None:
        self.loss = L2(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = np.array([2.0, 0.5, 0.0, 0.5, 2.0])
        assert_allclose(self.loss.rho(R), expected, atol=1e-12)

    def test_influence(self) -> None:
        # psi(r) = r
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert_allclose(self.loss.influence(R), expected, atol=1e-12)

    def test_weight(self) -> None:
        # w(r) = psi(r)/r = 1 everywhere (limit at r=0 is also 1)
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert_allclose(self.loss.weight(R), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# L1: rho(r) = |r|  (scale-independent exception)
# --------------------------------------------------------------------------- #
class TestL1:
    def setup_method(self) -> None:
        self.loss = L1(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        assert_allclose(self.loss.rho(R), expected, atol=1e-12)

    def test_influence(self) -> None:
        # psi(r) = sign(r), with sign(0) = 0
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert_allclose(self.loss.influence(R), expected, atol=1e-12)

    def test_weight(self) -> None:
        # w(r) = sign(r)/r; limit at r=0 returns 0.0 (safe fallback)
        expected = np.array([0.5, 1.0, 0.0, 1.0, 0.5])
        assert_allclose(self.loss.weight(R), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Huber(delta=1): phi(u) = 1/2*u^2 if |u|<=1, else delta*(|u|-delta/2)
# --------------------------------------------------------------------------- #
class TestHuber:
    def setup_method(self) -> None:
        self.loss = Huber(delta=1.0, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        # |u|<=1: 0.5*u^2 -> [_, 0.5, 0.0, 0.5, _]
        # |u|>1:  1*(|u|-0.5) -> [1.5, _, _, _, 1.5]
        expected = np.array([1.5, 0.5, 0.0, 0.5, 1.5])
        assert_allclose(self.loss.rho(R), expected, atol=1e-12)

    def test_influence(self) -> None:
        # |u|<=1: psi=r, |u|>1: psi=sign(u)*delta*scale = sign(r)
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert_allclose(self.loss.influence(R), expected, atol=1e-12)

    def test_weight(self) -> None:
        # |u|<=1: w=1, |u|>1: w=sign(r)/r = 1/|r|
        expected = np.array([0.5, 1.0, 1.0, 1.0, 0.5])
        assert_allclose(self.loss.weight(R), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Charbonnier(eps=1e-3): phi(u) = sqrt(u^2 + eps^2) - eps
# --------------------------------------------------------------------------- #
class TestCharbonnier:
    def setup_method(self) -> None:
        self.eps = 1e-3
        self.loss = Charbonnier(eps=self.eps, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        eps = self.eps
        u = R  # scale=1
        expected = np.sqrt(u**2 + eps**2) - eps
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence(self) -> None:
        eps = self.eps
        u = R
        expected = R / np.sqrt(u**2 + eps**2)
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)

    def test_weight(self) -> None:
        eps = self.eps
        u = R
        # w(r) = 1/sqrt(u^2 + eps^2), limit at 0 is 1/eps = 1000
        expected_nonzero = 1.0 / np.sqrt(u**2 + eps**2)
        result = self.loss.weight(R)
        # Check nonzero residuals
        mask = np.abs(R) > 1e-12
        assert_allclose(result[mask], expected_nonzero[mask], rtol=1e-6)
        # Check r=0 limit
        assert_allclose(result[~mask], 1.0 / eps, atol=1e-6)

    def test_rho_at_zero_is_zero(self) -> None:
        # phi(0) = sqrt(0 + eps^2) - eps = eps - eps = 0
        assert_allclose(self.loss.rho(np.array([0.0])), [0.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# Cauchy: phi(u) = 1/2 * log(1 + u^2)
# --------------------------------------------------------------------------- #
class TestCauchy:
    def setup_method(self) -> None:
        self.loss = Cauchy(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = 0.5 * np.log1p(R**2)
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence(self) -> None:
        # psi(r) = r / (1 + u^2) with u = r (scale=1)
        expected = R / (1.0 + R**2)
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)

    def test_weight(self) -> None:
        # w(r) = 1/(1 + u^2), limit at 0 is 1.0
        expected = np.where(np.abs(R) > 1e-12, 1.0 / (1.0 + R**2), 1.0)
        assert_allclose(self.loss.weight(R), expected, rtol=1e-6)

    def test_rho_known_values(self) -> None:
        # Spot-check specific values
        r_single = np.array([0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.5 * np.log(2.0), 0.5 * np.log(5.0)])
        assert_allclose(self.loss.rho(r_single), expected, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Tukey(c=4.685): phi(u) = c^2/6*(1-(1-(u/c)^2)^3) if |u|<=c, else c^2/6
# --------------------------------------------------------------------------- #
class TestTukey:
    def setup_method(self) -> None:
        self.c = 4.685
        self.loss = Tukey(c=self.c, scale=1.0, reduction="none")

    def test_rho(self) -> None:
        c = self.c
        u = R  # scale=1
        c2_over_6 = c**2 / 6.0
        ratio2 = (u / c) ** 2
        expected = np.where(
            np.abs(u) <= c,
            c2_over_6 * (1.0 - (1.0 - ratio2) ** 3),
            c2_over_6,
        )
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence(self) -> None:
        c = self.c
        u = R
        ratio2 = (u / c) ** 2
        expected = np.where(np.abs(u) <= c, R * (1.0 - ratio2) ** 2, 0.0)
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)

    def test_weight(self) -> None:
        c = self.c
        u = R
        ratio2 = (u / c) ** 2
        # w(r) = (1-(u/c)^2)^2 for |u|<=c, 0 otherwise; limit at 0 is 1.0
        expected = np.where(np.abs(u) <= c, (1.0 - ratio2) ** 2, 0.0)
        # At r=0, weight should return 1.0 from the limit
        expected[R == 0.0] = 1.0
        assert_allclose(self.loss.weight(R), expected, rtol=1e-6)

    def test_rho_outside_c(self) -> None:
        # For |r| > c, rho should be constant = c^2/6
        r_far = np.array([-10.0, -5.0, 5.0, 10.0])
        expected = np.full_like(r_far, self.c**2 / 6.0)
        assert_allclose(self.loss.rho(r_far), expected, rtol=1e-12)

    def test_influence_outside_c(self) -> None:
        # For |r| > c, influence should be exactly 0
        r_far = np.array([-10.0, -5.0, 5.0, 10.0])
        expected = np.zeros_like(r_far)
        assert_allclose(self.loss.influence(r_far), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Geman-McClure
# --------------------------------------------------------------------------- #
class TestGemanMcClure:
    def setup_method(self) -> None:
        self.loss = GemanMcClure(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        u2 = R**2
        expected = u2 / (1.0 + u2)
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence(self) -> None:
        u2 = R**2
        expected = 2.0 * R / (1.0 + u2) ** 2
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)

    def test_weight(self) -> None:
        u2 = R**2
        expected = np.where(np.abs(R) > 1e-12, 2.0 / (1.0 + u2) ** 2, 2.0)
        assert_allclose(self.loss.weight(R), expected, rtol=1e-6)

    def test_rho_known_values(self) -> None:
        r = np.array([0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.5, 0.8])
        assert_allclose(self.loss.rho(r), expected, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Welsch
# --------------------------------------------------------------------------- #
class TestWelsch:
    def setup_method(self) -> None:
        self.loss = Welsch(scale=1.0, reduction="none")

    def test_rho(self) -> None:
        expected = 1.0 - np.exp(-0.5 * R**2)
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence(self) -> None:
        expected = R * np.exp(-0.5 * R**2)
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)

    def test_weight(self) -> None:
        expected = np.where(np.abs(R) > 1e-12, np.exp(-0.5 * R**2), 1.0)
        assert_allclose(self.loss.weight(R), expected, rtol=1e-6)

    def test_rho_known_values(self) -> None:
        r = np.array([0.0, 1.0])
        expected = np.array([0.0, 1.0 - np.exp(-0.5)])
        assert_allclose(self.loss.rho(r), expected, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Scale=2.0 test (Huber used as representative)
# --------------------------------------------------------------------------- #
class TestScaledHuber:
    """Verify that the s^2 * phi(r/s) scaling convention works correctly."""

    def setup_method(self) -> None:
        self.loss = Huber(delta=1.0, scale=2.0, reduction="none")

    def test_rho_with_scale(self) -> None:
        # u = r/s = [-1, -0.5, 0, 0.5, 1]. All |u| <= delta=1.
        # rho = s^2 * 0.5 * u^2 = 4 * 0.5 * (r/2)^2 = 0.5 * r^2
        expected = np.array([2.0, 0.5, 0.0, 0.5, 2.0])
        assert_allclose(self.loss.rho(R), expected, atol=1e-12)

    def test_influence_with_scale(self) -> None:
        # All |u| <= 1, so psi = r
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert_allclose(self.loss.influence(R), expected, atol=1e-12)


class TestScaledCauchy:
    """Verify Cauchy scaling: rho(r;s) = s^2/2 * log(1 + (r/s)^2)."""

    def setup_method(self) -> None:
        self.scale = 2.0
        self.loss = Cauchy(scale=self.scale, reduction="none")

    def test_rho_with_scale(self) -> None:
        s = self.scale
        u = R / s
        expected = 0.5 * s**2 * np.log1p(u**2)
        assert_allclose(self.loss.rho(R), expected, rtol=1e-6)

    def test_influence_with_scale(self) -> None:
        s = self.scale
        u = R / s
        expected = R / (1.0 + u**2)
        assert_allclose(self.loss.influence(R), expected, rtol=1e-6)
