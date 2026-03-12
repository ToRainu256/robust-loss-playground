"""MLP regression on GPU: comparing L2 vs robust losses on data with outliers.

Requires CUDA. Usage:
    python examples/mlp_regression_gpu.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from robust_loss.torch import L1, L2, Cauchy, Charbonnier, Huber, Tukey

# --- Device setup -----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("WARNING: CUDA not available, falling back to CPU")
print(f"Using device: {device}")


# --- Data generation ---------------------------------------------------------

def make_data(
    n_inliers: int = 500,
    n_outliers: int = 50,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D nonlinear regression data: y = sin(x) + 0.1*x^2, with outliers."""
    rng = np.random.RandomState(seed)

    # Inliers
    x_in = rng.uniform(-4, 4, n_inliers)
    y_in = np.sin(x_in) + 0.1 * x_in**2 + rng.randn(n_inliers) * 0.2

    # Outliers: large y offset
    x_out = rng.uniform(-4, 4, n_outliers)
    y_out = np.sin(x_out) + 0.1 * x_out**2 + rng.randn(n_outliers) * 0.2 + 5.0

    x = np.concatenate([x_in, x_out]).astype(np.float32)
    y = np.concatenate([y_in, y_out]).astype(np.float32)

    return (
        torch.from_numpy(x).unsqueeze(1).to(device),
        torch.from_numpy(y).unsqueeze(1).to(device),
    )


# --- Model -------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Training ----------------------------------------------------------------

def train(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    lr: float = 1e-3,
    epochs: int = 2000,
) -> tuple[MLP, list[float]]:
    """Train an MLP with the given loss function. Returns (model, loss_history)."""
    torch.manual_seed(0)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        pred = model(x)
        residual = y - pred
        loss = loss_fn(residual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history


# --- Main --------------------------------------------------------------------

def main() -> None:
    x, y = make_data()

    losses = {
        "L2": L2(reduction="mean"),
        "L1": L1(reduction="mean"),
        "Huber": Huber(scale=1.0, delta=1.0, reduction="mean"),
        "Charbonnier": Charbonnier(scale=1.0, eps=1e-3, reduction="mean"),
        "Cauchy": Cauchy(scale=1.0, reduction="mean"),
        "Tukey": Tukey(scale=1.0, c=4.685, reduction="mean"),
    }

    results: dict[str, tuple[MLP, list[float]]] = {}
    for name, loss_fn in losses.items():
        model, history = train(x, y, loss_fn)
        results[name] = (model, history)
        print(f"{name:>8s}  final loss = {history[-1]:.4f}")

    # --- Plot: fitted curves -------------------------------------------------
    x_plot = torch.linspace(-4.5, 4.5, 300, device=device).unsqueeze(1)
    x_np = x_plot.cpu().numpy().ravel()
    y_true = np.sin(x_np) + 0.1 * x_np**2

    x_data = x.cpu().numpy().ravel()
    y_data = y.cpu().numpy().ravel()

    # Separate inliers/outliers for visualization (first 500 = inliers)
    n_in = 500
    colors = {
        "L2": "orange",
        "L1": "purple",
        "Huber": "blue",
        "Charbonnier": "cyan",
        "Cauchy": "green",
        "Tukey": "magenta",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(x_data[:n_in], y_data[:n_in], s=8, alpha=0.4, c="gray", label="Inliers")
    ax1.scatter(
        x_data[n_in:], y_data[n_in:], s=30, alpha=0.7, c="red", marker="x", label="Outliers",
    )
    ax1.plot(x_np, y_true, "k--", linewidth=2, label="Ground truth")

    for name, (model, _) in results.items():
        with torch.no_grad():
            y_pred = model(x_plot).cpu().numpy().ravel()
        ax1.plot(x_np, y_pred, color=colors[name], linewidth=2, label=name)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("MLP Regression: L2 vs Robust Losses")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot: training loss curves ------------------------------------------
    for name, (_, history) in results.items():
        ax2.plot(history, color=colors[name], label=name, alpha=0.8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("mlp_regression_gpu.png", dpi=150, bbox_inches="tight")
    print("Saved mlp_regression_gpu.png")


if __name__ == "__main__":
    main()
