"""Tests for dtype preservation and cross-dtype consistency.

For each PyTorch loss:
1. float32 input produces float32 output (rho, influence, weight)
2. float64 input produces float64 output
3. Results between float32 and float64 are close (rtol=1e-5)
4. CPU device works (basic sanity)
"""

from __future__ import annotations

import pytest
import torch

from robust_loss.torch import L1, L2, Cauchy, Charbonnier, GemanMcClure, Huber, Tukey, Welsch

# =========================================================================== #
# Test fixtures: build loss instances with reduction="none"
# =========================================================================== #

LOSS_INSTANCES = [
    pytest.param(L2(scale=1.0, reduction="none"), id="L2"),
    pytest.param(L1(scale=1.0, reduction="none"), id="L1"),
    pytest.param(Huber(delta=1.0, scale=1.0, reduction="none"), id="Huber"),
    pytest.param(
        Charbonnier(eps=1e-3, scale=1.0, reduction="none"), id="Charbonnier"
    ),
    pytest.param(Cauchy(scale=1.0, reduction="none"), id="Cauchy"),
    pytest.param(Tukey(c=4.685, scale=1.0, reduction="none"), id="Tukey"),
    pytest.param(GemanMcClure(scale=1.0, reduction="none"), id="GemanMcClure"),
    pytest.param(Welsch(scale=1.0, reduction="none"), id="Welsch"),
]

DTYPES = [
    pytest.param(torch.float32, id="float32"),
    pytest.param(torch.float64, id="float64"),
]


# =========================================================================== #
# dtype preservation: rho, influence, weight
# =========================================================================== #


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rho_preserves_dtype(loss_fn: torch.nn.Module, dtype: torch.dtype) -> None:
    r = torch.randn(16, dtype=dtype)
    result = loss_fn.rho(r)
    assert result.dtype == dtype, f"rho dtype: expected {dtype}, got {result.dtype}"


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_influence_preserves_dtype(
    loss_fn: torch.nn.Module, dtype: torch.dtype
) -> None:
    r = torch.randn(16, dtype=dtype)
    result = loss_fn.influence(r)
    assert result.dtype == dtype, (
        f"influence dtype: expected {dtype}, got {result.dtype}"
    )


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_weight_preserves_dtype(
    loss_fn: torch.nn.Module, dtype: torch.dtype
) -> None:
    r = torch.randn(16, dtype=dtype)
    result = loss_fn.weight(r)
    assert result.dtype == dtype, (
        f"weight dtype: expected {dtype}, got {result.dtype}"
    )


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_preserves_dtype(
    loss_fn: torch.nn.Module, dtype: torch.dtype
) -> None:
    """forward() with reduction='none' should also preserve dtype."""
    r = torch.randn(16, dtype=dtype)
    result = loss_fn(r)
    assert result.dtype == dtype, (
        f"forward dtype: expected {dtype}, got {result.dtype}"
    )


# =========================================================================== #
# Cross-dtype consistency: float32 vs float64 results are close
# =========================================================================== #


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
def test_rho_cross_dtype_consistency(loss_fn: torch.nn.Module) -> None:
    """float32 and float64 rho values should be close."""
    gen = torch.Generator().manual_seed(123)
    r64 = torch.randn(32, dtype=torch.float64, generator=gen)
    r32 = r64.float()

    rho64 = loss_fn.rho(r64)
    rho32 = loss_fn.rho(r32)
    assert torch.allclose(rho32.double(), rho64, rtol=1e-5, atol=1e-6), (
        f"rho cross-dtype mismatch: max diff = {(rho32.double() - rho64).abs().max()}"
    )


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
def test_influence_cross_dtype_consistency(loss_fn: torch.nn.Module) -> None:
    """float32 and float64 influence values should be close."""
    gen = torch.Generator().manual_seed(123)
    r64 = torch.randn(32, dtype=torch.float64, generator=gen)
    r32 = r64.float()

    psi64 = loss_fn.influence(r64)
    psi32 = loss_fn.influence(r32)
    assert torch.allclose(psi32.double(), psi64, rtol=1e-5, atol=1e-6), (
        f"influence cross-dtype mismatch: max diff = "
        f"{(psi32.double() - psi64).abs().max()}"
    )


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
def test_weight_cross_dtype_consistency(loss_fn: torch.nn.Module) -> None:
    """float32 and float64 weight values should be close (away from zero)."""
    gen = torch.Generator().manual_seed(123)
    r64 = torch.randn(32, dtype=torch.float64, generator=gen)
    # Avoid near-zero residuals where weight computation is sensitive
    r64 = r64 + torch.sign(r64) * 0.1
    r32 = r64.float()

    w64 = loss_fn.weight(r64)
    w32 = loss_fn.weight(r32)
    assert torch.allclose(w32.double(), w64, rtol=1e-5, atol=1e-6), (
        f"weight cross-dtype mismatch: max diff = {(w32.double() - w64).abs().max()}"
    )


# =========================================================================== #
# CPU device sanity
# =========================================================================== #


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
def test_cpu_device_output(loss_fn: torch.nn.Module) -> None:
    """All outputs should reside on CPU when input is on CPU."""
    r = torch.randn(8, dtype=torch.float64, device="cpu")
    assert loss_fn.rho(r).device.type == "cpu"
    assert loss_fn.influence(r).device.type == "cpu"
    assert loss_fn.weight(r).device.type == "cpu"
    assert loss_fn(r).device.type == "cpu"


# =========================================================================== #
# Shape preservation across dtypes
# =========================================================================== #


@pytest.mark.parametrize("loss_fn", LOSS_INSTANCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_output_shape_matches_input(
    loss_fn: torch.nn.Module, dtype: torch.dtype
) -> None:
    """rho, influence, weight should all preserve input shape."""
    for shape in [(8,), (4, 4), (2, 3, 4)]:
        r = torch.randn(shape, dtype=dtype)
        assert loss_fn.rho(r).shape == shape
        assert loss_fn.influence(r).shape == shape
        assert loss_fn.weight(r).shape == shape
