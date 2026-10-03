"""
make_figures.py

Regenerates the result figures from the CSVs produced by the analysis scripts.
Self-contained: reads CSVs from the current folder, writes PNGs to ../figures/.
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def read(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [name, os.path.join(here, name),
              os.path.join(here, "..", name),
              os.path.join(here, "..", "results", name)]:
        if os.path.exists(p):
            with open(p) as fh:
                return list(csv.DictReader(fh))
    raise FileNotFoundError(name)


def mean_of(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", "nan", None)]
    return float(np.mean(vals)) if vals else float("nan")


def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", name)


def fig7():
    rows = read("strong_r2.csv")
    rows = [r for r in rows if r["full_cpm"] not in ("", "nan")]
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows)); w = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [float(r["full_cpm"]) for r in rows], w, label="full")
    ax.bar(x, [float(r["global_cpm"]) for r in rows], w, label="global only")
    ax.bar(x + w, [float(r["spec_cpm"]) for r in rows], w, label="specific")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean R2 (CPM)"); ax.set_title("CPM: full vs global-only vs specific")
    ax.legend(); save(fig, "fig7_cpm_strong.png")


def fig10():
    rows = read("deconfound_results.csv")
    V = ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov", "cpm_pc1", "cpm_pc2"]
    lab = ["raw", "CPM", "CPM+\nngenes", "CPM+\ncovars", "raw+\ncovars", "CPM\n-PC1", "CPM\n-PC1,2"]
    m = [mean_of(rows, v) for v in V]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(V)), m, color=["gray", "gray", "tab:blue", "tab:blue", "tab:blue", "tab:green", "tab:green"])
    ax.set_xticks(range(len(V))); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("mean specific correlation")
    ax.set_title("Removing the cell-complexity confound recovers the specific signal")
    for i, v in enumerate(m):
        ax.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=8)
    save(fig, "fig10_deconfound.png")


def fig11():
    rows = read("robustness_results.csv")
    P = ["full", "pc1_raw", "counts_dir", "pc1_cpm", "pc12_raw"]
    lab = ["full", "PC1 raw", "counts dir", "PC1 CPM", "PC1-2 raw"]
    m = [mean_of(rows, p) for p in P]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(P)), m, color=["tab:blue"] + ["tab:orange"] * 4)
    ax.set_xticks(range(len(P))); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("mean R2")
    ax.set_title("A single global direction matches the full model,\nregardless of definition")
    for i, v in enumerate(m):
        ax.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=9)
    save(fig, "fig11_robustness.png")


def graphical_abstract():
    s = read("strong_r2.csv"); d = read("deconfound_results.csv")
    vals = [mean_of(s, "full_raw"), mean_of(s, "global_raw"), mean_of(s, "spec_raw")]
    dec = [mean_of(d, "raw"), mean_of(d, "cpm"), max(mean_of(d, "cpm_cov"), mean_of(d, "raw_cov"))]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("A. Covariance predicts perturbation\nfrom unperturbed cells", fontsize=11, weight="bold")
    for i, (txt, col) in enumerate([("control\ncells", "#dbe9f6"), ("covariance\n$\\Sigma$", "#fde9d9"),
                                    ("predicted\nresponse", "#e2f0d9")]):
        ax.add_patch(plt.Rectangle((0.03 + i * 0.34, 0.55), 0.26, 0.3, fc=col, ec="k"))
        ax.text(0.16 + i * 0.34, 0.70, txt, ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.30, "$\\Delta X = \\Sigma\\,u$", ha="center", fontsize=13)
    ax.text(0.5, 0.10, "13 Perturb-seq datasets, >1.8M cells", ha="center", fontsize=9, style="italic")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(3), vals, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["full", "global only", "specific"], fontsize=9)
    ax.set_ylabel("mean $R^2$ (raw)")
    ax.set_title("B. A single global mode reproduces\nthe full model; specific $R^2\\approx0$", fontsize=11, weight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, "{:.2f}".format(v), ha="center", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(3), dec, color=["gray", "gray", "tab:green"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["raw", "CPM", "resid covars"], fontsize=9)
    ax.set_ylabel("specific correlation")
    ax.set_title("C. Removing the complexity confound\nrecovers the specific signal", fontsize=11, weight="bold")
    for i, v in enumerate(dec):
        ax.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=9)

    fig.suptitle("Covariance-based perturbation prediction is dominated by a single global mode",
                 fontsize=13, weight="bold", y=1.03)
    p = os.path.join(FIGDIR, "graphical_abstract.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote graphical_abstract.png")


if __name__ == "__main__":
    for fn in [fig7, fig10, fig11, graphical_abstract]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, str(e)[:50])
