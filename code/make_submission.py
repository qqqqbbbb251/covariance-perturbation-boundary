"""
make_submission.py -- assemble a PLOS-submission package from the manuscript and figures.

Creates ./submission/ containing:
  * manuscript.md            main text (figures removed; figure legends kept)
  * manuscript.pdf           text-only reference PDF
  * figures/Fig1..8.pdf/.tif main figures exported for submission (<=2250 px wide, 300 dpi)
  * figures/main_figN.png    full-resolution source panels (for reference)
  * S1_File_code_results.zip code, result tables and figures (the archived repository)
  * Cover_Letter.md
  * Submission_Checklist.md
The manuscript source (paper_v13.md) is never modified.
"""

import os
import re
import shutil
import subprocess
import zipfile

import pypandoc
import typst
from PIL import Image


def md_to_pdf(src_md, out_pdf, resource):
    """pandoc -> typst source -> typst Python compiler (no typst CLI needed)."""
    typ = out_pdf + ".typ"
    pypandoc.convert_file(src_md, "typst", outputfile=typ,
                          extra_args=["--standalone", "--toc", "--toc-depth=2",
                                      "--resource-path=" + resource])
    typst.compile(typ, output=out_pdf)
    if os.path.exists(typ):
        os.remove(typ)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "paper_v13.md")
SUB = os.path.join(ROOT, "submission")
FIG = os.path.join(ROOT, "figures", "main")
FIG_OUT = os.path.join(SUB, "figures")
ZIP_SRC = os.path.join(os.path.expanduser("~"), "Desktop", "covariance-perturbation-boundary.zip")

MAIN = [
    "main_fig1_capability_boundary",
    "main_fig2_depth_axis",
    "main_fig3_reverse_detector",
    "main_fig4_specific_structure",
    "main_fig5_crosslab_time",
    "main_fig6_operator_blind",
    "main_fig7_organisation_epistasis",
    "main_fig8_theory_sota",
]

COVER = """\
Dear Editors of PLOS ONE,

We submit our manuscript, "A capability boundary for covariance-driven single-perturbation
models: the perturbed gene is detected through its own coordinate, not predicted by
covariance", for consideration as a Research Article.

Covariance-driven forward models such as CIPHER (Delta X = Sigma u) derive perturbation
responses from the control gene-gene covariance and are used as zero-shot "virtual cell"
predictors. Reproducing these models with the authors' own code across sixteen single-cell
perturbation datasets and matched baselines, we show that the forward prediction is
dominated by a shared technical sequencing-depth axis and carries no detectable
gene-specific signal: a covariance column from a random gene predicts as well as the
perturbed gene's column, and a single global mode is the strongest predictor. We further
show that the reverse (driver-identification) task is accurate only because of the
perturbed gene's own coordinate, not the covariance. Crucially, a real, reproducible,
network-organised target-specific structure does exist beneath the depth axis - it
transfers across laboratories, cell lines, guide libraries and stimulation times and is
visible in several moments of the response - but the covariance/precision operator is
blind to it in every data space and estimator we tested. The work thus (i) locates the
mechanism of a widely used class of models, (ii) resolves the apparent tension between
"high accuracy" and "no biological specificity", and (iii) provides a reusable
specificity-audit protocol and concrete reporting recommendations (matched-random-column,
equal-depth and global-mode baselines).

Relationship to previously published work (PLOS ONE disputing-published-work policy).
This manuscript directly evaluates and disputes the predictive claims of a specific
published framework, CIPHER (Kuznets-Speck et al., bioRxiv 2025,
doi:10.1101/2025.06.27.661814), which we reproduce with its authors' released code. We
wish to be explicit that the manuscript disputes that prior work. We have read and accept
the PLOS ONE policy on manuscripts disputing published work, including the invitation of a
signed review by the disputed authors during peer review, and we would welcome such a
review.

The manuscript is original, is not under consideration elsewhere, and all authors have
approved its submission. The author declares no competing interests. The analyses use only
publicly available data; all analysis code, result tables and figures are provided as
Supporting Information (S1 File), and a citable Zenodo DOI will be added upon acceptance.
We have no opposed reviewers, and we would be glad to suggest appropriate Academic Editors
if helpful.

Thank you for your consideration.

Sincerely,
Yichen Xie
School of Life Sciences, Peking University
2500012266@stu.pku.edu.cn
ORCID 0009-0004-7218-4838
"""

CHECKLIST = """\
# PLOS submission checklist

## Files to upload (Editorial Manager -> Attach Files)
- [ ] Manuscript: **manuscript.docx** (PLOS accepts DOC/DOCX/RTF; .pdf is also provided)
- [ ] Figure 1 (figures/Fig1.tif; Fig1.pdf also provided) ... Figure 8
- [ ] Supporting Information S1 File: **S1_File_code_results.zip** (kept < 10 MB)
- [ ] Cover letter: Cover_Letter.pdf (or .docx)

## Required metadata
- [ ] Title, Abstract, Author summary
- [ ] Author: Yichen Xie, School of Life Sciences, Peking University, Beijing 100871, China
- [ ] Corresponding author e-mail: 2500012266@stu.pku.edu.cn
- [ ] ORCID: 0009-0004-7218-4838 (linked to the Editorial Manager account)
- [ ] Author contributions (CRediT): Y.X. (all roles)
- [ ] Competing interests: none
- [ ] Funding: none specific
- [ ] Data availability: public datasets (accessions) + code/results as S1 File

## Figure requirements
- [ ] TIFF/EPS/PDF, 300-600 dpi, width 789-2250 px
- [ ] Figure legends listed in the manuscript (end)
- [ ] NOTE: current main figures are wide multi-panel composites; if the submission system
      flags the aspect ratio, use the full-resolution PNGs (submission/figures/*.png) or
      redraw each main figure as a standard-width single figure.

## Before hitting Submit
- [ ] Confirm title page and author details
- [ ] Confirm all 8 figures + S1 File are attached with correct file types
- [ ] Confirm the built PDF renders figures and tables correctly
- [ ] (Optional) add Zenodo DOI once available
"""


def main():
    os.makedirs(SUB, exist_ok=True)
    os.makedirs(FIG_OUT, exist_ok=True)

    # 1) manuscript text (strip inline image lines, keep captions/legends)
    text = open(SRC, encoding="utf-8").read()
    text = re.sub(r"^!\[\]\([^)]*\.png\)\s*\n", "", text, flags=re.M)
    text = text.replace("{width=100%}", "")
    with open(os.path.join(SUB, "manuscript.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    md_to_pdf(os.path.join(SUB, "manuscript.md"), os.path.join(SUB, "manuscript.pdf"), SUB)
    pypandoc.convert_file(os.path.join(SUB, "manuscript.md"), "docx",
                          outputfile=os.path.join(SUB, "manuscript.docx"),
                          extra_args=["--resource-path=" + SUB])
    from plos_format_docx import plos_format_docx
    plos_format_docx(os.path.join(SUB, "manuscript.docx"))
    print("wrote submission/manuscript.md, .pdf and .docx (PLOS formatting applied)")

    # 2) figures
    for i, name in enumerate(MAIN, 1):
        src = os.path.join(FIG, name + ".png")
        if not os.path.exists(src):
            print("missing", src); continue
        im = Image.open(src).convert("RGB")
        if im.width > 2250:
            h = round(im.height * 2250 / im.width)
            im2 = im.resize((2250, h), Image.LANCZOS)
        else:
            im2 = im
        im2.save(os.path.join(FIG_OUT, "Fig%d.pdf" % i))
        im2.save(os.path.join(FIG_OUT, "Fig%d.tif" % i), dpi=(300, 300))
        shutil.copyfile(src, os.path.join(FIG_OUT, "Fig%d_fullres.png" % i))
        print("figure", i, name, im.size, "->", im2.size)

    # 2b) supplementary figures, collected into one labelled PDF (Fig. S1-S29)
    from make_si_figures_pdf import main as _make_si_figures
    _make_si_figures()

    # 3) S1 file: lean archive (code + result tables + README/LICENSE), kept < 10 MB
    s1 = os.path.join(SUB, "S1_File_code_results.zip")
    skip_ext = (".pyc", ".log", ".err", ".out", ".bak", ".tmp")
    with zipfile.ZipFile(s1, "w", zipfile.ZIP_DEFLATED) as z:
        for base in ("code", "results", "figures"):
            for root, dirs, files in os.walk(os.path.join(ROOT, base)):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(skip_ext):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, ROOT))
        for f in ("README.md", "LICENSE", "requirements.txt",
                  "REPRODUCIBILITY.md", "paper_v13.md"):
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                z.write(p, f)
    print("wrote submission/S1_File_code_results.zip  %.1f MB" % (os.path.getsize(s1) / 1e6))

    # 4) cover letter + checklist
    open(os.path.join(SUB, "Cover_Letter.md"), "w", encoding="utf-8").write(COVER)
    open(os.path.join(SUB, "Submission_Checklist.md"), "w", encoding="utf-8").write(CHECKLIST)
    md_to_pdf(os.path.join(SUB, "Cover_Letter.md"), os.path.join(SUB, "Cover_Letter.pdf"), SUB)
    print("wrote submission/Cover_Letter.* and submission/Submission_Checklist.md")


if __name__ == "__main__":
    main()
