"""Charbonnier (pseudo-Huber / L1-L2) loss for PyTorch."""

from __future__ import annotations

import torch

from robust_loss.registry import register
from robust_loss.torch.base import BaseRobustLoss
from robust_loss.types import Tensor


class Charbonnier(BaseRobustLoss):
    """Charbonnier loss.

    phi(u) = sqrt(u^2 + eps^2) - eps
    rho(r; s) = s^2 * (sqrt((r/s)^2 + eps^2) - eps)
    """

    def __init__(
        self,
        eps: float = 1e-3,
        scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)
        self.eps = eps

    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise Charbonnier loss."""
        u = self.normalized_residual(residual)
        s2 = self.scale**2
        return s2 * (torch.sqrt(u**2 + self.eps**2) - self.eps)

    def influence(self, residual: Tensor) -> Tensor:
        """Influence function: psi(r; s) = r / sqrt(u^2 + eps^2) where u = r/s."""
        u = self.normalized_residual(residual)
        return residual / torch.sqrt(u**2 + self.eps**2)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = 1 / sqrt(0 + eps^2) = 1/eps."""
        return 1.0 / self.eps


register("charbonnier", Charbonnier)
