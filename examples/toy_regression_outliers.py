"""Toy regression: L2 vs robust loss with outliers."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_loss.torch import L2, Cauchy

# Generate data: y = 2x + 1 + noise, plus outliers
np.random.seed(42)
n_inliers = 50
n_outliers = 10
x_in = np.random.uniform(-3, 3, n_inliers)
y_in = 2 * x_in + 1 + np.random.randn(n_inliers) * 0.3
x_out = np.random.uniform(-3, 3, n_outliers)
y_out = 2 * x_out + 1 + np.random.randn(n_outliers) * 0.3 + 10  # large offset
x_all = np.concatenate([x_in, x_out])
y_all = np.concatenate([y_in, y_out])


def fit_line(x, y, loss_fn, lr=0.01, steps=2000):
    """Fit y = a*x + b using gradient descent with the given loss function."""
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    a = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for _ in range(steps):
        residual = y_t - (a * x_t + b)
        loss = loss_fn(residual)
        loss.backward()
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
            a.grad.zero_()
            b.grad.zero_()
    return a.item(), b.item()


a_l2, b_l2 = fit_line(x_all, y_all, L2(reduction="mean"))
a_robust, b_robust = fit_line(x_all, y_all, Cauchy(reduction="mean"))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_in, y_in, c="blue", label="Inliers", alpha=0.7)
ax.scatter(x_out, y_out, c="red", label="Outliers", alpha=0.7, marker="x", s=80)
x_line = np.linspace(-3.5, 3.5, 100)
ax.plot(x_line, 2 * x_line + 1, "k--", label="Ground truth", linewidth=2)
ax.plot(
    x_line,
    a_l2 * x_line + b_l2,
    "orange",
    label=f"L2: y={a_l2:.2f}x+{b_l2:.2f}",
    linewidth=2,
)
ax.plot(
    x_line,
    a_robust * x_line + b_robust,
    "green",
    label=f"Cauchy: y={a_robust:.2f}x+{b_robust:.2f}",
    linewidth=2,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Toy Regression: L2 vs Cauchy Loss with Outliers")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig("toy_regression.png", dpi=150, bbox_inches="tight")
print("Saved toy_regression.png")
