"""Plotting utilities for robust loss functions using the NumPy backend."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from robust_loss.numpy.base import BaseRobustLoss

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e


# Acceptable input types: list of losses or dict mapping label -> loss
LossesInput = Union[Dict[str, BaseRobustLoss], List[BaseRobustLoss], Sequence[BaseRobustLoss]]


def _normalize_losses(losses: LossesInput) -> Dict[str, BaseRobustLoss]:
    """Convert losses input to a dict of label -> loss instance."""
    if isinstance(losses, dict):
        return losses
    return {type(loss).__name__: loss for loss in losses}


def _get_or_create_ax(ax: Axes | None) -> Tuple[Figure, Axes]:
    """Return (fig, ax), creating them if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_rho(
    losses: LossesInput,
    xlim: Tuple[float, float] = (-5, 5),
    num: int = 1000,
    ax: Axes | None = None,
) -> Tuple[Figure, Axes]:
    """Plot rho(r) for each loss.

    Parameters
    ----------
    losses : dict or list of BaseRobustLoss
        NumPy backend loss instances. If dict, keys are used as labels.
        If list, class names are used as labels.
    xlim : tuple of float
        Range of residual values to plot.
    num : int
        Number of points in the linspace.
    ax : matplotlib Axes or None
        If None, a new figure and axes are created.

    Returns
    -------
    (fig, ax) : tuple of (Figure, Axes)
    """
    labeled_losses = _normalize_losses(losses)
    fig, ax = _get_or_create_ax(ax)

    r = np.linspace(xlim[0], xlim[1], num)

    for label, loss in labeled_losses.items():
        y = loss.rho(r)
        ax.plot(r, y, label=label)

    ax.set_xlabel("r")
    ax.set_ylabel("ρ(r)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_influence(
    losses: LossesInput,
    xlim: Tuple[float, float] = (-5, 5),
    num: int = 1000,
    ax: Axes | None = None,
) -> Tuple[Figure, Axes]:
    """Plot influence function psi(r) for each loss.

    Parameters
    ----------
    losses : dict or list of BaseRobustLoss
        NumPy backend loss instances. If dict, keys are used as labels.
        If list, class names are used as labels.
    xlim : tuple of float
        Range of residual values to plot.
    num : int
        Number of points in the linspace.
    ax : matplotlib Axes or None
        If None, a new figure and axes are created.

    Returns
    -------
    (fig, ax) : tuple of (Figure, Axes)
    """
    labeled_losses = _normalize_losses(losses)
    fig, ax = _get_or_create_ax(ax)

    r = np.linspace(xlim[0], xlim[1], num)

    for label, loss in labeled_losses.items():
        y = loss.influence(r)
        ax.plot(r, y, label=label)

    ax.set_xlabel("r")
    ax.set_ylabel("ψ(r)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_weight(
    losses: LossesInput,
    xlim: Tuple[float, float] = (-5, 5),
    num: int = 1000,
    ax: Axes | None = None,
) -> Tuple[Figure, Axes]:
    """Plot weight function w(r) = psi(r)/r for each loss.

    Values are clipped to [-10, 10] to avoid display issues near r=0.

    Parameters
    ----------
    losses : dict or list of BaseRobustLoss
        NumPy backend loss instances. If dict, keys are used as labels.
        If list, class names are used as labels.
    xlim : tuple of float
        Range of residual values to plot.
    num : int
        Number of points in the linspace.
    ax : matplotlib Axes or None
        If None, a new figure and axes are created.

    Returns
    -------
    (fig, ax) : tuple of (Figure, Axes)
    """
    labeled_losses = _normalize_losses(losses)
    fig, ax = _get_or_create_ax(ax)

    r = np.linspace(xlim[0], xlim[1], num)

    for label, loss in labeled_losses.items():
        y = loss.weight(r)
        y = np.clip(y, -10, 10)
        ax.plot(r, y, label=label)

    ax.set_xlabel("r")
    ax.set_ylabel("w(r)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
