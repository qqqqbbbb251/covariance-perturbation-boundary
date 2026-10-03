import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
strengthen_figs.py

Two extra figures:
  fig7 : CPM version of fig6 (full vs global-only vs specific)
  fig8 : the global mode is a cell-complexity axis
         (|corr| of the PC1 score with ncounts / ngenes)
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return list(csv.DictReader(open(path)))


def fig7_cpm():
    rows = read("results/strong_r2.csv") if __import__("os").path.exists("results/strong_r2.csv") \
        else read("strong_r2.csv")
    rows = [r for r in rows if r["full_cpm"] not in ("", "nan")]
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows))
    full = [float(r["full_cpm"]) for r in rows]
    glob = [float(r["global_cpm"]) for r in rows]
    spec = [float(r["spec_cpm"]) for r in rows]
    w = 0.28
    plt.figure(figsize=(12, 5))
    plt.bar(x - w, full, w, label="full (covariance)")
    plt.bar(x, glob, w, label="global mode only")
    plt.bar(x + w, spec, w, label="specific (orthogonal)")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("mean R2 (CPM)")
    plt.title("CPM space: full vs global-only vs specific")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/fig7_cpm_strong.png", dpi=150); plt.close()
    print("wrote figures/fig7_cpm_strong.png")


def fig8_identity():
    rows = read("global_mode_identity.csv")
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows))
    nc = [abs(float(r["ncounts_raw"])) for r in rows]
    ng = [abs(float(r["ngenes_raw"])) for r in rows]
    pm = [abs(float(r["percent_mito_raw"])) if r["percent_mito_raw"] not in ("", "nan") else np.nan
          for r in rows]
    pr = [abs(float(r["percent_ribo_raw"])) if r["percent_ribo_raw"] not in ("", "nan") else np.nan
          for r in rows]
    w = 0.2
    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5*w, nc, w, label="|corr| ncounts")
    plt.bar(x - 0.5*w, ng, w, label="|corr| ngenes")
    plt.bar(x + 0.5*w, pm, w, label="|corr| %mito")
    plt.bar(x + 1.5*w, pr, w, label="|corr| %ribo")
    plt.ylim(0, 1.05)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("|correlation| with global-mode score")
    plt.title("The global mode is a cell-complexity axis (library size / genes detected)")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/fig8_global_mode_identity.png", dpi=150); plt.close()
    print("wrote figures/fig8_global_mode_identity.png")


def fig9_identity_cpm():
    rows = read("global_mode_identity.csv")
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows))
    raw = [abs(float(r["ncounts_raw"])) for r in rows]
    cpm = [abs(float(r["ncounts_cpm"])) for r in rows]
    w = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, raw, w, label="raw counts")
    plt.bar(x + w/2, cpm, w, label="CPM")
    plt.ylim(0, 1.05)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("|corr(global-mode score, total counts)|")
    plt.title("CPM normalisation removes the cell-complexity axis from the global mode")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/fig9_identity_cpm.png", dpi=150); plt.close()
    print("wrote figures/fig9_identity_cpm.png")


if __name__ == "__main__":
    import os
    os.chdir(_PROJECT_ROOT)
    fig7_cpm()
    fig8_identity()
    fig9_identity_cpm()
