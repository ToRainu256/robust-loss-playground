"""Huber loss for PyTorch."""

from __future__ import annotations

import torch

from robust_loss.registry import register
from robust_loss.torch.base import BaseRobustLoss
from robust_loss.types import Tensor


class Huber(BaseRobustLoss):
    """Huber loss with transition at |u| = delta (u = r/s).

    phi(u) = 1/2 * u^2            if |u| <= delta
             delta * (|u| - delta/2)  otherwise

    rho(r; s) = s^2 * phi(r/s).
    """

    def __init__(
        self,
        delta: float = 1.0,
        scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)
        self.delta = delta

    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise Huber loss."""
        u = self.normalized_residual(residual)
        abs_u = u.abs()
        s2 = self.scale**2
        quadratic = s2 * 0.5 * u**2
        linear = s2 * self.delta * (abs_u - 0.5 * self.delta)
        return torch.where(abs_u <= self.delta, quadratic, linear)

    def influence(self, residual: Tensor) -> Tensor:
        """Influence function: psi(r; s) = d rho / d r."""
        u = self.normalized_residual(residual)
        abs_u = u.abs()
        # Quadratic region: psi = s * u = r (since u = r/s)
        # Linear region: psi = s * delta * sign(u)
        psi_quad = residual
        psi_linear = self.scale * self.delta * torch.sign(u)
        return torch.where(abs_u <= self.delta, psi_quad, psi_linear)

    def _weight_limit_at_zero(self) -> float:
        """lim_{r->0} psi(r)/r = 1 (quadratic region near zero)."""
        return 1.0


register("huber", Huber)
