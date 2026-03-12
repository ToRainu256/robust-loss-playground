"""Tukey biweight loss for NumPy."""

from __future__ import annotations

import numpy as np

from robust_loss.numpy.base import BaseRobustLoss
from robust_loss.registry import register
from robust_loss.types import NDArray


class Tukey(BaseRobustLoss):
    """Tukey biweight loss.

    phi(u) = c^2/6 * [1 - (1 - (u/c)^2)^3]   if |u| <= c
             c^2/6                               otherwise

    rho(r; s) = s^2 * phi(r/s)
    """

    def __init__(
        self,
        c: float = 4.685,
        scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)
        self.c = c

    def rho(self, residual: NDArray) -> NDArray:
        """Element-wise Tukey biweight loss."""
        u = self.normalized_residual(residual)
        abs_u = np.abs(u)
        s2 = self.scale**2
        c2_over_6 = self.c**2 / 6.0
        ratio2 = (u / self.c) ** 2
        inner = s2 * c2_over_6 * (1.0 - (1.0 - ratio2) ** 3)
        outer = s2 * c2_over_6
        return np.where(abs_u <= self.c, inner, outer)

    def influence(self, residual: NDArray) -> NDArray:
        """Influence function: psi(r; s) = r * (1 - (u/c)^2)^2 if |u| <= c, else 0."""
        u = self.normalized_residual(residual)
        abs_u = np.abs(u)
        ratio2 = (u / self.c) ** 2
        psi = residual * (1.0 - ratio2) ** 2
        return np.where(abs_u <= self.c, psi, 0.0)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = (1 - 0)^2 = 1."""
        return 1.0


register("numpy_tukey", Tukey)
