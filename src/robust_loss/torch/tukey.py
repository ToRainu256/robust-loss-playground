"""Tukey biweight loss for PyTorch."""

from __future__ import annotations

import torch

from robust_loss.registry import register
from robust_loss.torch.base import BaseRobustLoss
from robust_loss.types import Tensor


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

    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise Tukey biweight loss."""
        u = self.normalized_residual(residual)
        abs_u = u.abs()
        s2 = self.scale**2
        c2_over_6 = self.c**2 / 6.0
        ratio2 = (u / self.c) ** 2
        inner = s2 * c2_over_6 * (1.0 - (1.0 - ratio2) ** 3)
        outer = s2 * c2_over_6
        return torch.where(abs_u <= self.c, inner, outer)

    def influence(self, residual: Tensor) -> Tensor:
        """Influence function: psi(r; s) = r * (1 - (u/c)^2)^2 if |u| <= c, else 0."""
        u = self.normalized_residual(residual)
        abs_u = u.abs()
        ratio2 = (u / self.c) ** 2
        psi = residual * (1.0 - ratio2) ** 2
        zero = torch.zeros_like(residual)
        return torch.where(abs_u <= self.c, psi, zero)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = (1 - 0)^2 = 1."""
        return 1.0


register("tukey", Tukey)
