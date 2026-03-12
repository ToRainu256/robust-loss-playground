"""Compare weight functions w(r) for all loss functions."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from robust_loss.numpy import L2, Huber, Charbonnier, Cauchy, Tukey
from robust_loss.plotting import plot_weight

losses = {
    "L2": L2(reduction="none"),
    "Huber": Huber(reduction="none"),
    "Charbonnier": Charbonnier(reduction="none"),
    "Cauchy": Cauchy(reduction="none"),
    "Tukey": Tukey(reduction="none"),
}
fig, ax = plot_weight(losses)
ax.set_title("Weight Functions: w(r)")
fig.savefig("weight_functions.png", dpi=150, bbox_inches="tight")
print("Saved weight_functions.png")
