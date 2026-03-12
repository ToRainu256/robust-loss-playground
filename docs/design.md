# Design Notes

## Residual-first design

Every loss function takes a single argument: the residual `r = y_pred - y_true`. The caller is responsible for computing residuals before passing them in. This keeps the loss layer minimal and composable -- it does not need to know about the structure of predictions or targets.

```python
residual = model(x) - y
loss = loss_fn(residual)
```

## M-estimation language

The library uses the classical M-estimation vocabulary:

- **rho(r)** -- the objective function (the "loss" in the usual sense).
- **influence(r)** -- `psi(r) = d rho / d r`. Determines how each residual influences the gradient.
- **weight(r)** -- `w(r) = psi(r) / r`. Used in iteratively reweighted least squares (IRLS). Includes safe handling at `r -> 0`.

All three are first-class methods on every loss object. This makes it straightforward to plug the losses into IRLS solvers or to visualise their behaviour.

## Scale convention

All losses (except L1) follow the form:

```
rho(r; s) = s^2 * phi(r / s)
```

where `phi` is a normalised base function and `s` is a scale parameter. Consequences:

- At `s = 1` the loss equals `phi(r)` directly.
- Changing `s` rescales the residual space without altering the shape of the loss.
- Derivatives transform cleanly: `psi(r; s) = s * phi'(r / s)`.

L1 is an explicit exception because `|r|` has no meaningful scale parameterisation under this convention.

## Two-backend architecture

- **`robust_loss.torch`** -- PyTorch `nn.Module` subclasses. Suitable for gradient-based training. `forward()` returns reduced loss. Device and dtype safety are enforced (no bare `torch.tensor`).
- **`robust_loss.numpy`** -- NumPy implementations with identical API (`__call__`, `rho`, `influence`, `weight`, `reduction`). Useful for reference computations, testing, and plotting.

Both backends share the same formulas (documented in `docs/formulas.md`) and are tested for cross-backend numerical agreement.

## Extension points

The architecture is designed for easy addition of new losses:

1. Implement `rho()` and `influence()` in a new file under `torch/` and `numpy/`.
2. `weight()` and `forward()` are inherited from the base class.
3. Register the loss in `registry.py`.
4. Add tests following the existing pattern.

Possible future directions (not in v0.1):

- Barron's general adaptive loss (parameterised shape).
- Custom CUDA kernels for fused rho + influence.
- JAX backend.
