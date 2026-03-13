# Mathematical Reference

All losses expose three functions of a residual `r` with scale parameter `s`:

| Symbol | Name | Definition |
|--------|-----------|-------------------------------|
| rho | loss | `rho(r; s)` |
| psi | influence | `psi(r; s) = d rho / d r` |
| w | weight | `w(r; s) = psi(r; s) / r` |

## Unified convention

Unless noted otherwise, every loss follows:

```
rho(r; s) = s^2 * phi(r / s)
```

where `phi(u)` is the normalised base function. Derivatives transform as:

```
psi(r; s) = s * phi'(r / s)
w(r; s)   = phi'(r / s) / (r / s)   = phi'(u) / u   where u = r / s
```

---

## L2

| | Formula |
|---|---------|
| phi(u) | `1/2 u^2` |
| rho(r; s) | `1/2 r^2` |
| psi(r; s) | `r` |
| w(r; s) | `1` |
| lim w(r->0) | `1` |

Note: scale cancels -- `s^2 * 1/2 (r/s)^2 = 1/2 r^2`.

---

## L1

**Exception to the unified convention.** L1 does not use the `s^2 phi(r/s)` form.

| | Formula |
|---|---------|
| rho(r) | `|r|` |
| psi(r) | `sign(r)` |
| w(r) | `1 / |r|` |
| lim w(r->0) | `+inf` (clamped in code via `eps`) |

---

## Huber

Parameter: `delta > 0` (transition point in normalised space).

| | Formula |
|---|---------|
| phi(u) | `1/2 u^2` if `|u| <= delta`, else `delta (|u| - 1/2 delta)` |
| rho(r; s) | `s^2 phi(r / s)` |
| psi(r; s) | `r` if `|r| <= delta * s`, else `delta * s * sign(r)` |
| w(r; s) | `1` if `|r| <= delta * s`, else `delta * s / |r|` |
| lim w(r->0) | `1` |

---

## Charbonnier

Parameter: `eps > 0` (smoothing).

| | Formula |
|---|---------|
| phi(u) | `sqrt(u^2 + eps^2) - eps` |
| rho(r; s) | `s^2 (sqrt((r/s)^2 + eps^2) - eps)` |
| psi(r; s) | `r / sqrt(r^2 + (eps * s)^2)` |
| w(r; s) | `1 / sqrt(r^2 + (eps * s)^2)` |
| lim w(r->0) | `1 / (eps * s)` |

---

## Cauchy

| | Formula |
|---|---------|
| phi(u) | `1/2 log(1 + u^2)` |
| rho(r; s) | `s^2 / 2 * log(1 + (r/s)^2)` |
| psi(r; s) | `r / (s^2 + r^2) * s^2` = `r / (1 + (r/s)^2)` |
| w(r; s) | `s^2 / (s^2 + r^2)` = `1 / (1 + (r/s)^2)` |
| lim w(r->0) | `1` |

---

## Tukey (bisquare)

Parameter: `c > 0` (rejection threshold in normalised space).

| | Formula |
|---|---------|
| phi(u) | `c^2/6 [1 - (1 - (u/c)^2)^3]` if `|u| <= c`, else `c^2/6` |
| rho(r; s) | `s^2 phi(r / s)` |
| psi(r; s) | `r (1 - (r / (c*s))^2)^2` if `|r| <= c*s`, else `0` |
| w(r; s) | `(1 - (r / (c*s))^2)^2` if `|r| <= c*s`, else `0` |
| lim w(r->0) | `1` |

Tukey is a **redescending** influence function: `psi` returns to zero for large residuals, making the loss completely reject outliers beyond `|r| > c * s`.

---

## Geman-McClure

| | Formula |
|---|---------|
| phi(u) | `u^2 / (1 + u^2)` |
| rho(r; s) | `s^2 * u^2 / (1 + u^2)`, `u = r/s` |
| psi(r; s) | `2r / (1 + u^2)^2` |
| w(r; s) | `2 / (1 + u^2)^2` |
| lim w(r->0) | `2` |

Geman-McClure is a **redescending** influence function with a smooth, bounded loss. Unlike Tukey, there is no hard cutoff — outlier rejection increases gradually.

---

## Welsch

| | Formula |
|---|---------|
| phi(u) | `1 - exp(-u^2 / 2)` |
| rho(r; s) | `s^2 (1 - exp(-u^2 / 2))`, `u = r/s` |
| psi(r; s) | `r exp(-u^2 / 2)` |
| w(r; s) | `exp(-u^2 / 2)` |
| lim w(r->0) | `1` |

Welsch is a **redescending** influence function with exponential suppression of outliers. The loss saturates at `s^2` for large residuals.
