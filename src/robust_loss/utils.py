"""Shared utilities."""

from __future__ import annotations

from robust_loss.types import Reduction

VALID_REDUCTIONS: frozenset[str] = frozenset({"none", "mean", "sum"})


def validate_reduction(reduction: str) -> Reduction:
    """Validate and return a reduction string."""
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of {sorted(VALID_REDUCTIONS)}."
        )
    return reduction  # type: ignore[return-value]


def validate_scale(scale: float) -> float:
    """Validate that scale is positive."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return scale
