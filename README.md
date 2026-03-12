# robust-loss-playground

Research-oriented robust loss library with PyTorch and NumPy backends.

## Features

- **Residual-first API** -- pass raw residuals, not predictions and targets
- **Unified M-estimation interface** -- `rho`, `influence`, `weight` on every loss
- **6 loss functions** -- L2, L1, Huber, Charbonnier, Cauchy, Tukey
- **Two backends** -- PyTorch (`nn.Module`) for training, NumPy for analysis
- **Built-in plotting** -- `plot_rho`, `plot_influence`, `plot_weight`
- **Scale convention** -- all losses follow `rho(r; s) = s^2 * phi(r/s)` (L1 excepted)

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from robust_loss.torch import Charbonnier

loss_fn = Charbonnier(scale=1.0, eps=0.01, reduction="mean")

residual = torch.randn(64)

# rho -- element-wise loss value
rho = loss_fn.rho(residual)

# influence -- psi(r) = d rho / d r
psi = loss_fn.influence(residual)

# weight -- w(r) = psi(r) / r, safe at r -> 0
w = loss_fn.weight(residual)

# forward / __call__ -- rho + reduction
scalar_loss = loss_fn(residual)
```

## Supported Losses

| Loss | Key parameter | Description |
|-------------|---------------|----------------------------------------------|
| `L2` | -- | Standard squared loss, `rho = 1/2 r^2` |
| `L1` | -- | Absolute loss, `rho = \|r\|` |
| `Huber` | `delta` | Quadratic near zero, linear in the tails |
| `Charbonnier`| `eps` | Smooth L1 approximation via `sqrt(r^2+eps^2)-eps` |
| `Cauchy` | -- | Heavy-tailed, `rho ~ log(1+r^2)` |
| `Tukey` | `c` | Redescending; zero influence beyond `\|r\|>c` |

## NumPy Backend

Every loss is mirrored in the NumPy backend with the same API:

```python
import numpy as np
from robust_loss.numpy import Cauchy

loss_fn = Cauchy(scale=1.0, reduction="none")

r = np.linspace(-5, 5, 200)
rho = loss_fn.rho(r)
psi = loss_fn.influence(r)
w = loss_fn.weight(r)
```

## Examples

See the `examples/` directory:

- `basic_usage.py` -- minimal example of each loss
- `compare_losses.py` -- side-by-side comparison across all 6 losses
- `irls_demo.py` -- iteratively reweighted least squares using `weight()`
- `plot_gallery.py` -- generate rho / influence / weight plots

## Mathematical Details

Full formulas for every loss (rho, influence, weight, and limits) are documented in [docs/formulas.md](docs/formulas.md).

Design rationale is in [docs/design.md](docs/design.md).

## License

MIT -- see [LICENSE](LICENSE).
