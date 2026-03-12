"""Tests for reduction modes and constructor validation.

Verifies that:
1. reduction="none" preserves input shape
2. reduction="mean" returns rho.mean()
3. reduction="sum" returns rho.sum()
4. Invalid reduction raises ValueError
5. scale <= 0 raises ValueError

Tested for both PyTorch (forward()) and NumPy (__call__) backends.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close

from robust_loss.numpy import (
    L1 as NL1,
)
from robust_loss.numpy import (
    L2 as NL2,
)
from robust_loss.numpy import (
    Cauchy as NCauchy,
)
from robust_loss.numpy import (
    Charbonnier as NCharbonnier,
)
from robust_loss.numpy import (
    Huber as NHuber,
)
from robust_loss.numpy import (
    Tukey as NTukey,
)
from robust_loss.torch import L1, L2, Cauchy, Charbonnier, Huber, Tukey

# Shared random residuals (fixed seed for reproducibility).
_gen = torch.Generator().manual_seed(42)
R_TORCH = torch.randn(32, dtype=torch.float64, generator=_gen)
R_NUMPY = R_TORCH.numpy().copy()


# =========================================================================== #
# Helper: build (torch_class, numpy_class, kwargs) tuples
# =========================================================================== #

LOSS_PAIRS = [
    pytest.param(L2, NL2, {}, id="L2"),
    pytest.param(L1, NL1, {}, id="L1"),
    pytest.param(Huber, NHuber, {"delta": 1.0}, id="Huber"),
    pytest.param(Charbonnier, NCharbonnier, {"eps": 1e-3}, id="Charbonnier"),
    pytest.param(Cauchy, NCauchy, {}, id="Cauchy"),
    pytest.param(Tukey, NTukey, {"c": 4.685}, id="Tukey"),
]


# =========================================================================== #
# PyTorch reduction tests
# =========================================================================== #


@pytest.mark.parametrize("torch_cls, numpy_cls, kwargs", LOSS_PAIRS)
class TestTorchReduction:
    """Reduction modes for PyTorch losses via forward()."""

    def test_none_preserves_shape(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = torch_cls(scale=1.0, reduction="none", **kwargs)
        result = loss_fn(R_TORCH)
        assert result.shape == R_TORCH.shape

    def test_mean_matches_manual(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_none = torch_cls(scale=1.0, reduction="none", **kwargs)
        loss_mean = torch_cls(scale=1.0, reduction="mean", **kwargs)
        rho = loss_none.rho(R_TORCH)
        expected = rho.mean()
        result = loss_mean(R_TORCH)
        assert_close(result, expected, atol=1e-12, rtol=0.0)

    def test_sum_matches_manual(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_none = torch_cls(scale=1.0, reduction="none", **kwargs)
        loss_sum = torch_cls(scale=1.0, reduction="sum", **kwargs)
        rho = loss_none.rho(R_TORCH)
        expected = rho.sum()
        result = loss_sum(R_TORCH)
        assert_close(result, expected, atol=1e-12, rtol=0.0)

    def test_mean_returns_scalar(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = torch_cls(scale=1.0, reduction="mean", **kwargs)
        result = loss_fn(R_TORCH)
        assert result.ndim == 0

    def test_sum_returns_scalar(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = torch_cls(scale=1.0, reduction="sum", **kwargs)
        result = loss_fn(R_TORCH)
        assert result.ndim == 0


# =========================================================================== #
# NumPy reduction tests
# =========================================================================== #


@pytest.mark.parametrize("torch_cls, numpy_cls, kwargs", LOSS_PAIRS)
class TestNumpyReduction:
    """Reduction modes for NumPy losses via __call__."""

    def test_none_preserves_shape(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = numpy_cls(scale=1.0, reduction="none", **kwargs)
        result = loss_fn(R_NUMPY)
        assert result.shape == R_NUMPY.shape

    def test_mean_matches_manual(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_none = numpy_cls(scale=1.0, reduction="none", **kwargs)
        loss_mean = numpy_cls(scale=1.0, reduction="mean", **kwargs)
        rho = loss_none.rho(R_NUMPY)
        expected = rho.mean()
        result = loss_mean(R_NUMPY)
        assert_allclose(result, expected, atol=1e-12)

    def test_sum_matches_manual(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_none = numpy_cls(scale=1.0, reduction="none", **kwargs)
        loss_sum = numpy_cls(scale=1.0, reduction="sum", **kwargs)
        rho = loss_none.rho(R_NUMPY)
        expected = rho.sum()
        result = loss_sum(R_NUMPY)
        assert_allclose(result, expected, atol=1e-12)

    def test_mean_returns_scalar(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = numpy_cls(scale=1.0, reduction="mean", **kwargs)
        result = loss_fn(R_NUMPY)
        assert np.ndim(result) == 0

    def test_sum_returns_scalar(
        self, torch_cls: type, numpy_cls: type, kwargs: dict
    ) -> None:
        loss_fn = numpy_cls(scale=1.0, reduction="sum", **kwargs)
        result = loss_fn(R_NUMPY)
        assert np.ndim(result) == 0


# =========================================================================== #
# Validation: invalid reduction
# =========================================================================== #

TORCH_CLASSES = [
    pytest.param(L2, {}, id="L2"),
    pytest.param(L1, {}, id="L1"),
    pytest.param(Huber, {"delta": 1.0}, id="Huber"),
    pytest.param(Charbonnier, {"eps": 1e-3}, id="Charbonnier"),
    pytest.param(Cauchy, {}, id="Cauchy"),
    pytest.param(Tukey, {"c": 4.685}, id="Tukey"),
]

NUMPY_CLASSES = [
    pytest.param(NL2, {}, id="NL2"),
    pytest.param(NL1, {}, id="NL1"),
    pytest.param(NHuber, {"delta": 1.0}, id="NHuber"),
    pytest.param(NCharbonnier, {"eps": 1e-3}, id="NCharbonnier"),
    pytest.param(NCauchy, {}, id="NCauchy"),
    pytest.param(NTukey, {"c": 4.685}, id="NTukey"),
]


@pytest.mark.parametrize("cls, kwargs", TORCH_CLASSES)
def test_torch_invalid_reduction_raises(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="[Ii]nvalid reduction"):
        cls(scale=1.0, reduction="invalid", **kwargs)


@pytest.mark.parametrize("cls, kwargs", NUMPY_CLASSES)
def test_numpy_invalid_reduction_raises(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="[Ii]nvalid reduction"):
        cls(scale=1.0, reduction="invalid", **kwargs)


# =========================================================================== #
# Validation: scale <= 0
# =========================================================================== #


@pytest.mark.parametrize("cls, kwargs", TORCH_CLASSES)
def test_torch_nonpositive_scale_raises(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="scale"):
        cls(scale=0.0, reduction="mean", **kwargs)
    with pytest.raises(ValueError, match="scale"):
        cls(scale=-1.0, reduction="mean", **kwargs)


@pytest.mark.parametrize("cls, kwargs", NUMPY_CLASSES)
def test_numpy_nonpositive_scale_raises(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="scale"):
        cls(scale=0.0, reduction="mean", **kwargs)
    with pytest.raises(ValueError, match="scale"):
        cls(scale=-1.0, reduction="mean", **kwargs)
