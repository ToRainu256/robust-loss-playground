"""PyTorch base class for robust loss functions."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

from robust_loss.types import Reduction, Tensor
from robust_loss.utils import validate_reduction, validate_scale


class BaseRobustLoss(nn.Module, abc.ABC):
    """Base class for all PyTorch robust loss functions.

    All losses follow rho(r; s) = s^2 * phi(r/s) where phi is the
    normalized base function, except L1 which is a documented exception.
    """

    def __init__(self, scale: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.scale: float = validate_scale(scale)
        self.reduction: Reduction = validate_reduction(reduction)

    def normalized_residual(self, residual: Tensor) -> Tensor:
        """Compute u = r / s."""
        return residual / self.scale

    def forward(self, residual: Tensor) -> Tensor:
        """Compute rho(residual) with reduction applied."""
        loss = self.rho(residual)
        return self._reduce(loss)

    @abc.abstractmethod
    def rho(self, residual: Tensor) -> Tensor:
        """Element-wise loss value."""
        ...

    @abc.abstractmethod
    def influence(self, residual: Tensor) -> Tensor:
        """Influence function psi(r) = d rho / d r."""
        ...

    def _weight_limit_at_zero(self) -> float:
        """Return lim_{r->0} psi(r)/r. Override in subclasses."""
        return 1.0

    def weight(self, residual: Tensor, eps: float = 1e-12) -> Tensor:
        """Weight function w(r) = psi(r) / r with safe r->0 handling."""
        psi = self.influence(residual)
        abs_r = residual.abs()
        safe = abs_r > eps
        w = torch.where(
            safe,
            psi / residual,
            torch.as_tensor(
                self._weight_limit_at_zero(),
                dtype=residual.dtype,
                device=residual.device,
            ),
        )
        return w

    def _reduce(self, x: Tensor) -> Tensor:
        """Apply reduction to tensor."""
        if self.reduction == "none":
            return x
        elif self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
