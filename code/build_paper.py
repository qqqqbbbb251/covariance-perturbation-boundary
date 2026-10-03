"""
build_paper.py -- build an illustrated manuscript and a self-contained HTML preview.

Reads paper_v13.md, inserts the eight main figures (figures/main/*.png) at the end of
the appropriate Results sections, and writes:
  * paper_v13_illustrated.md  (Markdown with inline images)
  * paper_v13.html            (single file; images embedded as base64)

The original paper_v13.md is never modified (it may be open in an editor).
"""

import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "paper_v13.md")
MD_OUT = os.path.join(ROOT, "paper_v13_illustrated.md")
HTML_OUT = os.path.join(ROOT, "paper_v13.html")

# insert each figure immediately BEFORE these headings (i.e. at the end of the section)
FIGURES = [
    ("figures/main/main_fig1_capability_boundary.png",
     "Fig. 1. A capability boundary for covariance-driven forward models. (A) Concept; "
     "(B–E) forward predictions are collinear and RSA ≈ 0, the inverse is accurate, and a "
     "real specific structure lies beneath the axis.",
     "### Reproduction with CIPHER's own code"),
    ("figures/main/main_fig2_depth_axis.png",
     "Fig. 2. The leading covariance axis is a technical sequencing-depth artefact. "
     "Equal-depth resampling halves the top-1 share (0.68 → 0.26); PC1 loadings equal the "
     "gene-mean vector (|r| ≈ 1); a permutation spectral null finds one real mode.",
     "### Forward predictions are not perturbation-specific"),
    ("figures/main/main_fig3_reverse_detector.png",
     "Fig. 3. The reverse recovers the driver through the target's own coordinate. "
     "Official inverse AUC 0.76–0.99; dropping the covariance leaves AUC unchanged "
     "(p = 0.74), dropping the target coordinate collapses 0.93 → 0.37.",
     "### A real, reproducible perturbation-specific structure lies beneath"),
    ("figures/main/main_fig4_specific_structure.png",
     "Fig. 4. A real, reproducible specific structure beneath the axis. Split-half RSA "
     "reliability (0.49 [0.31, 0.67]); broad-spectrum robustness; guide/target decoupling.",
     "### The structure is organised by PPI"),
    ("figures/main/main_fig5_crosslab_time.png",
     "Fig. 5. Cross-laboratory / cell-line / library / time transfer of the specific "
     "structure (same-target cos 0.38–0.49 across labs; Marson D1–D4 0.28–0.48).",
     "### Individual guides are noisy reporters"),
    ("figures/main/main_fig6_operator_blind.png",
     "Fig. 6. The covariance operator is blind to the specific structure (off-axis "
     "predictability ≈ random, p = 0.52, across spaces and estimators); programme/direction "
     "decomposition and second-moment signals.",
     "### Combinatorial perturbations"),
    ("figures/main/main_fig7_organisation_epistasis.png",
     "Fig. 7. Organisation by PPI / pathways / regulons (edge − non-edge +0.098, p = 0.023); "
     "85% additivity of double perturbations with 66% of the interaction off-axis.",
     "### A capability boundary: theory and model comparison"),
    ("figures/main/main_fig8_theory_sota.png",
     "Fig. 8. Theory and model comparison. Numerical verification that rank-one forward "
     "R² = shared_frac and the operator is ε-blind; SOTA decomposition (CIPHER/linear-mean "
     "forward cos² 0.39/0.57 but RSA ≈ 0).",
     "## Discussion"),
]


def build_markdown(text):
    for img, cap, anchor in FIGURES:
        if img in text:
            continue  # already embedded (idempotent)
        block = "![%s](%s)\n\n" % (cap, img)
        if anchor not in text:
            print("WARN anchor not found:", anchor)
            continue
        text = text.replace(anchor, block + anchor, 1)
    return text


def main():
    with open(SRC, encoding="utf-8") as fh:
        text = fh.read()
    illustrated = build_markdown(text)
    # make embedded figures fit the page in DOCX/PDF/HTML outputs
    illustrated = re.sub(r"(!\[\]\([^)]+\.png\))", r"\1{width=100%}", illustrated)
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write(illustrated)
    print("wrote", MD_OUT)

    try:
        import markdown
    except Exception:
        print("python-markdown not installed; skipping HTML (pip install markdown)")
        return
    html = markdown.markdown(illustrated, extensions=["tables", "fenced_code", "toc"])

    def embed(m):
        src = m.group(1)
        p = os.path.join(ROOT, src.replace("/", os.sep))
        if not os.path.exists(p):
            return m.group(0)
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return 'src="data:image/png;base64,%s"' % b64

    html = re.sub(r'src="([^"]+\.png)"', embed, html)
    css = ("body{max-width:900px;margin:2rem auto;font-family:Georgia,serif;"
           "line-height:1.5;padding:0 1rem}img{max-width:100%;border:1px solid #ddd;"
           "border-radius:4px;margin:1rem 0}table{border-collapse:collapse}"
           "td,th{border:1px solid #ccc;padding:4px 8px;font-size:.9em}"
           "h1{font-size:1.6em}h2{font-size:1.3em;border-bottom:1px solid #eee;padding-top:1em}"
           "code{background:#f6f8fa;padding:2px 4px}")
    doc = ("<!doctype html><html><head><meta charset='utf-8'><style>" + css
           + "</style></head><body>" + html + "</body></html>")
    with open(HTML_OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("wrote", HTML_OUT)


if __name__ == "__main__":
    main()
