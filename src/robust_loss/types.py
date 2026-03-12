"""Shared type aliases."""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import torch
    Tensor = torch.Tensor
except ImportError:
    Tensor = None  # type: ignore[assignment,misc]

NDArray = np.ndarray
Reduction = Literal["none", "mean", "sum"]
