"""
make_main_figures.py -- assemble the eight main figures of the paper.

Each main figure is a single, tightly-laid-out canvas built from validated analysis
panels: white margins are auto-cropped, panels are scaled to a common height and packed
horizontally with panel letters and a title.  The panel-to-source map mirrors
paper_v13.md and figures/MAIN_FIGURES.md.

Output: ../figures/main/main_fig1.png ... main_fig8.png
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(FIG, "main")
os.makedirs(OUT, exist_ok=True)

PANEL_H = 1000          # common panel height (px)
GAP = 40                # gap between panels (px)
MARGIN = 40
TITLE_H = 90

MAIN = [
    ("main_fig1_capability_boundary",
     "Fig. 1  A capability boundary for covariance-driven forward models",
     ["fig26_grand_summary.png", "fig17_forward_indistinguishability.png"]),
    ("main_fig2_depth_axis",
     "Fig. 2  The leading covariance axis is a sequencing-depth artefact",
     ["fig21_axis_origin_specific.png", "fig19_spectrum_rank.png"]),
    ("main_fig3_reverse_detector",
     "Fig. 3  The reverse recovers the driver through the target's own coordinate",
     ["fig18_inverse_driver.png", "fig20_inverse_decomposition.png"]),
    ("main_fig4_specific_structure",
     "Fig. 4  A real, reproducible specific structure beneath the axis",
     ["fig24_structure_meta.png", "fig29_guide_target.png"]),
    ("main_fig5_crosslab_time",
     "Fig. 5  Cross-lab / cell-line / library / time transfer",
     ["fig23_crosslab_network.png", "fig16_cross_dataset_transfer.png"]),
    ("main_fig6_operator_blind",
     "Fig. 6  The covariance operator is blind to the specific structure",
     ["fig22_exploration_summary.png"]),
    ("main_fig7_organisation_epistasis",
     "Fig. 7  Organisation by PPI / pathways / regulons; combinatorial perturbations",
     ["fig24_structure_meta.png", "fig25_meta_epistasis.png"]),
    ("main_fig8_theory_sota",
     "Fig. 8  Theory and model comparison",
     ["fig27_sota_decompose.png", "fig28_theory_simulation.png"]),
]


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def crop_white(im, thresh=248):
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < thresh).any(axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def main():
    font = load_font(30)
    tfont = load_font(40)
    for name, title, srcs in MAIN:
        panels = []
        for s in srcs:
            p = os.path.join(FIG, s)
            if not os.path.exists(p):
                print("SKIP", name, "missing", s)
                panels = None
                break
            im = crop_white(Image.open(p))
            w = max(1, int(im.width * PANEL_H / im.height))
            panels.append(im.resize((w, PANEL_H), Image.LANCZOS))
        if not panels:
            continue
        width = MARGIN * 2 + sum(p.width for p in panels) + GAP * (len(panels) - 1)
        height = MARGIN + TITLE_H + PANEL_H + MARGIN
        canvas = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(canvas)
        d.text((MARGIN, 25), title, fill="black", font=tfont)
        x = MARGIN
        for i, p in enumerate(panels):
            canvas.paste(p, (x, MARGIN + TITLE_H))
            d.text((x + 6, MARGIN + TITLE_H + 6), "(%s)" % chr(ord("A") + i),
                   fill="black", font=font)
            x += p.width + GAP
        outpath = os.path.join(OUT, name + ".png")
        canvas.save(outpath, dpi=(300, 300))
        print("wrote", outpath, canvas.size)


if __name__ == "__main__":
    main()
