"""Compare influence functions psi(r) for all loss functions."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from robust_loss.numpy import L2, Huber, Charbonnier, Cauchy, Tukey
from robust_loss.plotting import plot_influence

losses = {
    "L2": L2(reduction="none"),
    "Huber": Huber(reduction="none"),
    "Charbonnier": Charbonnier(reduction="none"),
    "Cauchy": Cauchy(reduction="none"),
    "Tukey": Tukey(reduction="none"),
}
fig, ax = plot_influence(losses)
ax.set_title("Influence Functions: ψ(r)")
fig.savefig("influence_functions.png", dpi=150, bbox_inches="tight")
print("Saved influence_functions.png")
