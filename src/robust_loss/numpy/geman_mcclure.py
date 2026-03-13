"""Geman-McClure loss for NumPy."""

from __future__ import annotations

from robust_loss.numpy.base import BaseRobustLoss
from robust_loss.registry import register
from robust_loss.types import NDArray


class GemanMcClure(BaseRobustLoss):
    """Geman-McClure loss.

    phi(u) = u^2 / (1 + u^2)
    rho(r; s) = s^2 * u^2 / (1 + u^2),  u = r/s
    """

    def rho(self, residual: NDArray) -> NDArray:
        """Element-wise Geman-McClure loss."""
        u = self.normalized_residual(residual)
        u2 = u**2
        s2 = self.scale**2
        return s2 * u2 / (1.0 + u2)

    def influence(self, residual: NDArray) -> NDArray:
        """Influence function: psi(r; s) = 2r / (1 + u^2)^2."""
        u = self.normalized_residual(residual)
        denom = (1.0 + u**2) ** 2
        return 2.0 * residual / denom

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = 2 / (1+0)^2 = 2."""
        return 2.0


register("numpy_geman_mcclure", GemanMcClure)
