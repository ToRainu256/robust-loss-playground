"""L2 (squared) loss for NumPy."""

from __future__ import annotations

from robust_loss.numpy.base import BaseRobustLoss
from robust_loss.registry import register
from robust_loss.types import NDArray


class L2(BaseRobustLoss):
    """L2 loss: rho(r) = 1/2 * r^2.

    Since phi(u) = 1/2 * u^2, the scale cancels:
        rho(r; s) = s^2 * 1/2 * (r/s)^2 = 1/2 * r^2.
    """

    def rho(self, residual: NDArray) -> NDArray:
        """Element-wise loss: 1/2 * r^2."""
        return 0.5 * residual**2

    def influence(self, residual: NDArray) -> NDArray:
        """Influence function: psi(r) = r."""
        return residual

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = 1."""
        return 1.0


register("numpy_l2", L2)
