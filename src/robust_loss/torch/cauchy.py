"""Cauchy (Lorentzian) loss for PyTorch."""

from __future__ import annotations

import torch

from robust_loss.registry import register
from robust_loss.torch.base import BaseRobustLoss
from robust_loss.types import Tensor


class Cauchy(BaseRobustLoss):
    """Cauchy loss.

    phi(u) = 1/2 * log(1 + u^2)
    rho(r; s) = s^2/2 * log(1 + (r/s)^2)
    """

    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise Cauchy loss."""
        u = self.normalized_residual(residual)
        s2 = self.scale**2
        return 0.5 * s2 * torch.log1p(u**2)

    def influence(self, residual: Tensor) -> Tensor:
        """Influence function: psi(r; s) = r / (1 + (r/s)^2)."""
        u = self.normalized_residual(residual)
        return residual / (1.0 + u**2)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = 1/(1+0) = 1."""
        return 1.0


register("cauchy", Cauchy)
