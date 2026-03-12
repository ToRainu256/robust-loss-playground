"""L1 (absolute) loss for NumPy."""

from __future__ import annotations

import numpy as np

from robust_loss.numpy.base import BaseRobustLoss
from robust_loss.registry import register
from robust_loss.types import NDArray


class L1(BaseRobustLoss):
    """L1 loss: rho(r) = |r|.

    L1 is a documented exception to the s^2 * phi(r/s) convention —
    rho does NOT depend on scale.
    """

    def rho(self, residual: NDArray) -> NDArray:
        """Element-wise loss: |r|."""
        return np.abs(residual)

    def influence(self, residual: NDArray) -> NDArray:
        """Influence function: psi(r) = sign(r), with psi(0) = 0."""
        return np.sign(residual)

    def _weight_limit_at_zero(self) -> float:
        """Safe fallback for lim_{r->0} sign(r)/r.

        The true limit diverges, but since influence(0) = 0 we return 0.0
        to avoid NaN in downstream IRLS usage.
        """
        return 0.0


register("numpy_l1", L1)
