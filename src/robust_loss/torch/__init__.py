"""PyTorch backend for robust loss functions."""

from robust_loss.torch.base import BaseRobustLoss
from robust_loss.torch.cauchy import Cauchy
from robust_loss.torch.charbonnier import Charbonnier
from robust_loss.torch.huber import Huber
from robust_loss.torch.l1 import L1
from robust_loss.torch.l2 import L2
from robust_loss.torch.tukey import Tukey

__all__ = [
    "BaseRobustLoss",
    "L2",
    "L1",
    "Huber",
    "Charbonnier",
    "Cauchy",
    "Tukey",
]
