"""Shared type aliases."""

from typing import Literal

import numpy as np
import torch

Tensor = torch.Tensor
NDArray = np.ndarray
Reduction = Literal["none", "mean", "sum"]
