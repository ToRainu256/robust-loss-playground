"""Compare rho(r) curves for all loss functions."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from robust_loss.numpy import L2, Huber, Charbonnier, Cauchy, Tukey
from robust_loss.plotting import plot_rho

losses = {
    "L2": L2(reduction="none"),
    "Huber": Huber(reduction="none"),
    "Charbonnier": Charbonnier(reduction="none"),
    "Cauchy": Cauchy(reduction="none"),
    "Tukey": Tukey(reduction="none"),
}
fig, ax = plot_rho(losses)
ax.set_title("Robust Loss Functions: ρ(r)")
fig.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
print("Saved loss_curves.png")
