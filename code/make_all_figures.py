"""
make_all_figures.py

Regenerates every paper figure (fig1-fig13 plus the graphical abstract) from the
canonical result CSVs in ../results/ (falling back to the project root / code
folder for a few inputs).  Writes PNGs to ../figures/.

Run after the analysis pipeline:
    python make_all_figures.py
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def read(name):
    for p in [os.path.join(RES, name), os.path.join(ROOT, name),
              os.path.join(HERE, name)]:
        if os.path.exists(p):
            with open(p) as fh:
                return list(csv.DictReader(fh))
    raise FileNotFoundError(name)


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def col(rows, k):
    return np.array([num(r.get(k, "nan")) for r in rows])


def mean_col(rows, k):
    v = col(rows, k)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def save(name):
    p = os.path.join(FIG, name)
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    print("wrote", name)


def bar_two(rows, k1, k2, l1, l2, ylab, title, fname, zero=False):
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, col(rows, k1), w, label=l1)
    plt.bar(x + w / 2, col(rows, k2), w, label=l2)
    if zero:
        plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel(ylab); plt.title(title); plt.legend()
    save(fname)


def bar_three(rows, k1, k2, k3, l1, l2, l3, ylab, title, fname):
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows)); w = 0.28
    plt.figure(figsize=(12, 5))
    plt.bar(x - w, col(rows, k1), w, label=l1)
    plt.bar(x, col(rows, k2), w, label=l2)
    plt.bar(x + w, col(rows, k3), w, label=l3)
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel(ylab); plt.title(title); plt.legend()
    save(fname)


def scatter(x, y, names, xlab, ylab, title, fname, hline=None):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=60, c="tab:blue")
    for i, n in enumerate(names):
        plt.annotate(n, (x[i], y[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if hline is not None:
        plt.axhline(hline, color="k", lw=0.8)
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    save(fname)


def fig1_3():
    rows = [r for r in read("scan_results.csv") if num(r["global_frac_raw"]) == num(r["global_frac_raw"])]
    bar_two(rows, "global_frac_raw", "global_frac_cpm", "raw", "CPM",
            "global-mode variance fraction",
            "Single global mode dominates the covariance", "fig1_global_mode.png")
    bar_two(rows, "orth_corr_raw", "orth_corr_cpm", "raw", "CPM",
            "corr with orthogonal specific response",
            "Perturbation-specific signal is ~0 (raw)", "fig2_specific_signal.png", zero=True)
    bar_two(rows, "full_r2_raw", "full_r2_cpm", "raw", "CPM",
            "full-response $R^2$", "CIPHER full-response $R^2$ reproduction",
            "fig3_full_r2.png")
    names = [r["dataset"][:18] for r in rows]
    scatter(col(rows, "global_frac_raw"), col(rows, "full_r2_raw"), names,
            "global-mode variance fraction (raw)", "full-response $R^2$ (raw)",
            "Is full-response $R^2$ explained by the global mode?",
            "fig4_scatter_fullR2.png")
    scatter(col(rows, "global_frac_raw"), col(rows, "orth_corr_raw"), names,
            "global-mode variance fraction (raw)", "orthogonal specific corr (raw)",
            "Specificity vs global-mode dominance", "fig5_scatter_specificity.png", hline=0)


def fig6_7():
    rows = [r for r in read("strong_r2.csv") if num(r["full_raw"]) == num(r["full_raw"])]
    bar_three(rows, "full_raw", "global_raw", "spec_raw",
              "full (covariance)", "global mode only", "specific (orthogonal)",
              "mean $R^2$ (raw)",
              "Full-response $R^2$ is reproduced by a single global mode",
              "fig6_strong.png")
    rows = [r for r in read("strong_r2.csv") if num(r["full_cpm"]) == num(r["full_cpm"])]
    bar_three(rows, "full_cpm", "global_cpm", "spec_cpm",
              "full (covariance)", "global mode only", "specific (orthogonal)",
              "mean $R^2$ (CPM)", "CPM: full vs global-only vs specific",
              "fig7_cpm_strong.png")


def fig8_9():
    rows = read("global_mode_identity.csv")
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows)); w = 0.2
    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5 * w, np.abs(col(rows, "ncounts_raw")), w, label="|corr| ncounts")
    plt.bar(x - 0.5 * w, np.abs(col(rows, "ngenes_raw")), w, label="|corr| ngenes")
    plt.bar(x + 0.5 * w, np.abs(col(rows, "percent_mito_raw")), w, label="|corr| %mito")
    plt.bar(x + 1.5 * w, np.abs(col(rows, "percent_ribo_raw")), w, label="|corr| %ribo")
    plt.ylim(0, 1.05)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("|correlation| with global-mode score")
    plt.title("The global mode is a cell-complexity axis")
    plt.legend(); save("fig8_global_mode_identity.png")

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, np.abs(col(rows, "ncounts_raw")), w, label="raw counts")
    plt.bar(x + w / 2, np.abs(col(rows, "ncounts_cpm")), w, label="CPM")
    plt.ylim(0, 1.05)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("|corr(global-mode score, total counts)|")
    plt.title("CPM normalisation removes the cell-complexity axis")
    plt.legend(); save("fig9_identity_cpm.png")


def fig10_11():
    rows = read("deconfound_results.csv")
    V = ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov", "cpm_pc1", "cpm_pc2"]
    lab = ["raw", "CPM", "CPM+\nngenes", "CPM+\ncovars", "raw+\ncovars", "CPM\n-PC1", "CPM\n-PC1,2"]
    m = [mean_col(rows, v) for v in V]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(V)), m, color=["gray", "gray", "tab:blue", "tab:blue",
                                     "tab:blue", "tab:green", "tab:green"])
    plt.xticks(range(len(V)), lab, fontsize=8)
    plt.ylabel("mean specific correlation")
    plt.title("Removing the cell-complexity confound recovers the specific signal")
    for i, v in enumerate(m):
        if np.isfinite(v):
            plt.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=8)
    save("fig10_deconfound.png")

    rows = read("robustness_results.csv")
    P = ["full", "pc1_raw", "counts_dir", "pc1_cpm", "pc12_raw"]
    lab = ["full", "PC1 raw", "counts dir", "PC1 CPM", "PC1-2 raw"]
    m = [mean_col(rows, p) for p in P]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(P)), m, color=["tab:blue"] + ["tab:orange"] * 4)
    plt.xticks(range(len(P)), lab, fontsize=9)
    plt.ylabel("mean $R^2$")
    plt.title("A single global direction matches the full model,\nregardless of definition")
    for i, v in enumerate(m):
        if np.isfinite(v):
            plt.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=9)
    save("fig11_robustness.png")


def graphical_abstract():
    strong = read("strong_r2.csv")
    decon = read("deconfound_results.csv")
    vals = [mean_col(strong, "full_raw"), mean_col(strong, "global_raw"),
            mean_col(strong, "spec_raw")]
    dec = [mean_col(decon, "raw"), mean_col(decon, "cpm"),
           max(mean_col(decon, "cpm_cov"), mean_col(decon, "raw_cov"))]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("A. Covariance predicts perturbation\nfrom unperturbed cells",
                 fontsize=11, weight="bold")
    for i, (txt, c) in enumerate([("control\ncells", "#dbe9f6"), ("covariance\n$\\Sigma$", "#fde9d9"),
                                  ("predicted\nresponse", "#e2f0d9")]):
        ax.add_patch(Rectangle((0.03 + i * 0.34, 0.55), 0.26, 0.3, fc=c, ec="k"))
        ax.text(0.16 + i * 0.34, 0.70, txt, ha="center", va="center", fontsize=9)
    for x0, x1 in [(0.30, 0.36), (0.64, 0.70)]:
        ax.add_patch(FancyArrowPatch((x0, 0.70), (x1, 0.70), arrowstyle="-|>",
                                     mutation_scale=14, color="k"))
    ax.text(0.5, 0.32, "$\\Delta X = \\Sigma\\,u$", ha="center", fontsize=13)
    ax.text(0.5, 0.10, "16 Perturb-seq datasets, >2M cells", ha="center",
            fontsize=9, style="italic")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(3), vals, color=["#2b6cb0", "#dd6b20", "#276749"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["full", "global only", "specific"], fontsize=9)
    ax.set_ylabel("mean $R^2$ (raw)")
    ax.set_title("B. A single global mode reproduces\nthe full model; specific $R^2\\approx 0$",
                 fontsize=11, weight="bold")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.01, "{:.2f}".format(v), ha="center", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(3), dec, color=["gray", "gray", "#276749"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["raw", "CPM", "resid covars"], fontsize=9)
    ax.set_ylabel("specific correlation")
    ax.set_title("C. Removing the complexity confound\nrecovers the specific signal",
                 fontsize=11, weight="bold")
    for i, v in enumerate(dec):
        if np.isfinite(v):
            ax.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=9)

    fig.suptitle("Covariance-based perturbation prediction is dominated by a single global mode",
                 fontsize=13, weight="bold", y=1.03)
    p = os.path.join(FIG, "graphical_abstract.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote graphical_abstract.png")


def main():
    for fn in [fig1_3, fig6_7, fig8_9, fig10_11, graphical_abstract]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, ":", str(e)[:60])
    try:
        import fig12_13
        fig12_13.fig12()
        fig12_13.fig13()
    except Exception as e:
        print("skip fig12_13:", str(e)[:60])


if __name__ == "__main__":
    main()
