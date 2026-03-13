# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`robust-loss-playground` is a research-oriented robust loss library providing unified APIs for PyTorch and NumPy. It implements M-estimation loss functions (rho, influence, weight) with a residual-first design. The authoritative design spec is `specification.md` — always follow it over assumptions or conventions.

## Build & Development Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_torch_value.py

# Run a specific test
pytest tests/test_torch_value.py::test_huber_values -v

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Architecture

The library lives under `src/robust_loss/` with two parallel backends:

- **`torch/`** — PyTorch implementations (`nn.Module` subclasses). `base.py` defines `BaseRobustLoss` which handles `scale`, `reduction`, `forward()`, and `weight()`. Each loss file (l2, l1, huber, charbonnier, cauchy, tukey) implements `rho()` and `influence()`.
- **`numpy/`** — NumPy reference implementations mirroring the same API (`__call__` = `rho` + reduction). Same file structure as `torch/`.

Key design pattern: all losses follow `ρ(r; s) = s² · φ(r/s)` where φ is the normalized base function. L1 is an explicit exception (`ρ(r) = |r|`).

### Core API (every loss class)

- `forward(residual)` / `__call__` — rho + reduction
- `rho(residual)` — element-wise loss
- `influence(residual)` — ψ(r) = ∂ρ/∂r (analytic)
- `weight(residual, eps=...)` — ψ(r)/r with safe r→0 handling

### Supporting modules

- `plotting.py` — `plot_rho()`, `plot_influence()`, `plot_weight()` (matplotlib)
- `registry.py` — loss registry
- `types.py`, `utils.py` — shared types and utilities

## Key Constraints

- **device/dtype safety**: Never create bare `torch.tensor(...)` — use Python floats or `torch.as_tensor(value, dtype=residual.dtype, device=residual.device)`.
- **Reduction**: Only `"none"`, `"mean"`, `"sum"`. Implemented once in base class, never duplicated.
- **Math fidelity**: Formulas in code must match `specification.md` §8 and `docs/formulas.md`. Tests are executable specs.
- **No premature optimization**: No custom CUDA, C++ extensions, JAX, or Barron loss in v0.1.

## Test Structure

Tests in `tests/` are organized by concern:
- `test_numpy_value.py`, `test_torch_value.py` — known-value correctness
- `test_torch_numpy_consistency.py` — cross-backend agreement
- `test_influence_weight.py` — influence vs autograd, weight near zero
- `test_reduction.py` — reduction modes
- `test_device_dtype.py` — float32/float64, CPU/CUDA portability
