"""Welsch loss for PyTorch."""

from __future__ import annotations

import torch

from robust_loss.registry import register
from robust_loss.torch.base import BaseRobustLoss
from robust_loss.types import Tensor


class Welsch(BaseRobustLoss):
    """Welsch loss.

    phi(u) = 1 - exp(-u^2 / 2)
    rho(r; s) = s^2 * (1 - exp(-u^2 / 2)),  u = r/s
    """

    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise Welsch loss."""
        u = self.normalized_residual(residual)
        s2 = self.scale**2
        return s2 * (1.0 - torch.exp(-0.5 * u**2))

    def influence(self, residual: Tensor) -> Tensor:
        """Influence function: psi(r; s) = r * exp(-u^2 / 2)."""
        u = self.normalized_residual(residual)
        return residual * torch.exp(-0.5 * u**2)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = exp(0) = 1."""
        return 1.0


register("welsch", Welsch)
