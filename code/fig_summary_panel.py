"""
fig_summary_panel.py -- assemble a compact panel of the key result figures
for the 2-page Chinese briefing.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
OUT = os.path.join(FIG, "summary_panel.png")

PANELS = [
    ("fig1_global_mode.png", "A  global mode dominates"),
    ("fig6_strong.png", "B  full vs global vs specific"),
    ("fig12_random_column.png", "C  matched random column"),
    ("fig13_effects.png", "D  per-dataset effects / Pearson"),
    ("fig14_specificity_control.png", "E  specificity test power"),
    ("fig15_residual_decomposition.png", "F  residual decomposition"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (fname, title) in zip(axes.ravel(), PANELS):
    p = os.path.join(FIG, fname)
    if os.path.exists(p):
        ax.imshow(mpimg.imread(p))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
