"""NumPy base class for robust loss functions (reference implementation)."""

from __future__ import annotations

import abc

import numpy as np

from robust_loss.types import NDArray, Reduction
from robust_loss.utils import validate_reduction, validate_scale


class BaseRobustLoss(abc.ABC):
    """Base class for all NumPy robust loss functions.

    All losses follow rho(r; s) = s^2 * phi(r/s) where phi is the
    normalized base function, except L1 which is a documented exception.
    """

    def __init__(self, scale: float = 1.0, reduction: str = "mean") -> None:
        self.scale: float = validate_scale(scale)
        self.reduction: Reduction = validate_reduction(reduction)

    def normalized_residual(self, residual: NDArray) -> NDArray:
        """Compute u = r / s."""
        return residual / self.scale

    def __call__(self, residual: NDArray) -> NDArray:
        """Compute rho(residual) with reduction applied."""
        loss = self.rho(residual)
        return self._reduce(loss)

    @abc.abstractmethod
    def rho(self, residual: NDArray) -> NDArray:
        """Element-wise loss value."""
        ...

    @abc.abstractmethod
    def influence(self, residual: NDArray) -> NDArray:
        """Influence function psi(r) = d rho / d r."""
        ...

    def _weight_limit_at_zero(self) -> float:
        """Return lim_{r->0} psi(r)/r. Override in subclasses."""
        return 1.0

    def weight(self, residual: NDArray, eps: float = 1e-12) -> NDArray:
        """Weight function w(r) = psi(r) / r with safe r->0 handling."""
        psi = self.influence(residual)
        abs_r = np.abs(residual)
        safe = abs_r > eps
        safe_residual = np.where(safe, residual, np.ones_like(residual))
        w = np.where(safe, psi / safe_residual, self._weight_limit_at_zero())
        return w

    def _reduce(self, x: NDArray) -> NDArray:
        """Apply reduction to array."""
        if self.reduction == "none":
            return x
        elif self.reduction == "mean":
            return np.mean(x)
        elif self.reduction == "sum":
            return np.sum(x)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
