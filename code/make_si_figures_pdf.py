"""Combine figures/fig1..fig29 into a single labelled supplementary-figures PDF
(Fig. S1 .. Fig. S29), for submission as Supporting Information.

Usage:  python make_si_figures_pdf.py
Output: ../submission/S1_Figures.pdf
"""

import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "submission", "S1_Figures.pdf")

FIGS = [
    ("fig1_global_mode.png", "A single global mode dominates the control covariance"),
    ("fig2_specific_signal.png", "Perturbation-specific signal is approximately zero (raw)"),
    ("fig3_full_r2.png", "CIPHER full-response R-squared reproduction"),
    ("fig4_scatter_fullR2.png", "Full-response R-squared versus global-mode dominance"),
    ("fig5_scatter_specificity.png", "Specificity versus global-mode dominance"),
    ("fig6_strong.png", "Full-response R-squared reproduced by a single global mode (raw)"),
    ("fig7_cpm_strong.png", "CPM: full versus global-only versus specific"),
    ("fig8_global_mode_identity.png", "The global mode is a cell-complexity axis"),
    ("fig9_identity_cpm.png", "CPM normalisation removes the cell-complexity axis"),
    ("fig10_deconfound.png", "Removing the cell-complexity confound recovers the specific signal"),
    ("fig11_robustness.png", "A single global direction matches the full model"),
    ("fig12_random_column.png", "Matched random covariance column versus the perturbed column"),
    ("fig13_effects.png", "Forward effect sizes"),
    ("fig14_specificity_control.png", "Specificity positive control"),
    ("fig15_residual_decomposition.png", "Residual decomposition (self / shared / residual)"),
    ("fig16_cross_dataset_transfer.png", "Cross-dataset forward transfer"),
    ("fig17_forward_indistinguishability.png",
     "Forward predictions are indistinguishable across perturbations"),
    ("fig18_inverse_driver.png", "Inverse driver recovery"),
    ("fig19_spectrum_rank.png", "Covariance spectrum and rank"),
    ("fig20_inverse_decomposition.png", "Inverse ablation decomposition"),
    ("fig21_axis_origin_specific.png", "Origin of the shared axis and the specific structure"),
    ("fig22_exploration_summary.png", "Operator blindness across spaces and estimators"),
    ("fig23_crosslab_network.png", "Cross-laboratory transfer and network organisation"),
    ("fig24_structure_meta.png", "Structure strength versus dataset properties"),
    ("fig25_meta_epistasis.png", "Meta-analysis and combinatorial perturbations"),
    ("fig26_grand_summary.png", "Grand summary"),
    ("fig27_sota_decompose.png", "State-of-the-art model decomposition"),
    ("fig28_theory_simulation.png", "Theory simulation"),
    ("fig29_guide_target.png", "Guide / target decoupling"),
]

PAGE_W, PAGE_H = 1700, 2200
MARGIN = 70
CAP_H = 100


def _font(size):
    for p in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    font = _font(36)
    pages = []
    for i, (fn, title) in enumerate(FIGS, 1):
        path = os.path.join(FIG, fn)
        if not os.path.exists(path):
            print("MISSING", fn)
            continue
        im = Image.open(path).convert("RGB")
        maxw = PAGE_W - 2 * MARGIN
        maxh = PAGE_H - 2 * MARGIN - CAP_H
        scale = min(maxw / im.width, maxh / im.height)
        im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.LANCZOS)
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(page)
        d.text((MARGIN, MARGIN), "Fig. S%d. %s" % (i, title), fill="black", font=font)
        page.paste(im2, ((PAGE_W - im2.width) // 2, MARGIN + CAP_H))
        pages.append(page)
    pages[0].save(OUT, "PDF", save_all=True, append_images=pages[1:], resolution=200)
    print("wrote", OUT, "with", len(pages), "pages")


if __name__ == "__main__":
    main()
