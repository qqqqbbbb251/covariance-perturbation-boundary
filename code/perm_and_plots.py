import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
perm_and_plots.py

Two things for the paper:
  --plots : figures from scan_results.csv
  --perm  : permutation test for the "orthogonal-specific correlation ~ 0"
            claim (does the gene-specific pairing carry any signal?)

Usage:
  python perm_and_plots.py --plots
  python perm_and_plots.py --perm --dir <folder>
"""

import argparse
import csv
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.sparse as sp
import anndata as ad


CONTROL_LIKE = {"control", "ctrl", "ntc", "non-targeting", "nontargeting", "safe"}

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def to_space(X, ncounts, space):
    if space == "raw":
        return X
    inv = (1e4 / np.maximum(ncounts, 1.0)).astype(np.float32)
    return (sp.diags(inv) @ X).tocsr()


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


# --------------------------------------------------------------------------- #
def plots(csv_path, out_prefix="fig"):
    rows = list(csv.DictReader(open(csv_path)))
    names = [r["dataset"][:22] for r in rows]
    x = np.arange(len(rows))

    def col(k):
        return np.array([float(r[k]) for r in rows])

    for key, ylab, title, fname in [
        (("global_frac_raw", "global_frac_cpm"),
         "global-mode variance fraction",
         "Single global mode dominates the covariance",
         "fig1_global_mode.png"),
        (("orth_corr_raw", "orth_corr_cpm"),
         "corr with global-mode-orthogonalised specific response",
         "Perturbation-specific signal is ~0 (raw) and modest (CPM)",
         "fig2_specific_signal.png"),
        (("full_r2_raw", "full_r2_cpm"),
         "R2 of full perturbation response",
         "CIPHER full-response R2 reproduction",
         "fig3_full_r2.png"),
    ]:
        a, b = key
        w = 0.38
        plt.figure(figsize=(11, 5))
        plt.bar(x - w/2, col(a), w, label=a.replace("_", " "))
        plt.bar(x + w/2, col(b), w, label=b.replace("_", " "))
        if "orth_corr" in a:
            plt.axhline(0, color="k", lw=0.8)
        plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylab); plt.title(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, fname), dpi=150)
        plt.close()
        print("wrote", fname)


# --------------------------------------------------------------------------- #
def perm_dataset(path, n_top=2000, n_perm=1000, max_pert=150, seed=0):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pert = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts = obs["ncounts"].values.astype(float)
    var_ncounts = a.var["ncounts"].values.astype(float)
    var_names = np.asarray(a.var_names)
    ctrl = np.array([p.lower() in CONTROL_LIKE for p in pert]) | (nperts == 0)
    singles = [g for g in np.unique(pert[nperts == 1]) if g.lower() not in CONTROL_LIKE]
    counts = {g: int(np.sum(pert == g)) for g in singles}
    singles = [g for g in singles if counts[g] >= 30]
    rng = np.random.default_rng(seed)
    if len(singles) > max_pert:
        singles = list(rng.choice(singles, max_pert, replace=False))
    selected = set(np.argsort(-var_ncounts)[:n_top].tolist())
    pos = {g: i for i, g in enumerate(var_names)}
    for g in singles:
        if g in pos:
            selected.add(pos[g])
    gi = np.array(sorted(selected))
    genes = var_names[gi]
    Xm = a[:, gi].to_memory().X
    X_raw = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    out = {}
    for space in ["raw", "cpm"]:
        X = to_space(X_raw, ncounts, space)
        Xc = np.asarray(X[ctrl].todense(), dtype=np.float64)
        cmean = Xc.mean(0)
        cov = np.cov(Xc, rowvar=False)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)]; v = v / np.linalg.norm(v)
        del Xc
        deltas = np.atleast_2d(np.asarray(
            [np.asarray(X[pert == g].todense(), dtype=np.float64).mean(0) - cmean
             for g in singles]))
        n = len(singles)
        common = (deltas.sum(0)[None, :] - deltas) / max(n - 1, 1)
        specific = deltas - common
        sorth = specific - np.outer(specific @ v, v)
        sigs = np.array([cov[:, idx[g]] for g in singles])
        sn = sigs / (np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-12)
        on = sorth / (np.linalg.norm(sorth, axis=1, keepdims=True) + 1e-12)
        C = sn @ on.T                       # n x n correlation matrix
        obs = float(np.mean(np.diag(C)))
        ar = np.arange(n)
        null = np.empty(n_perm)
        for k in range(n_perm):
            null[k] = C[ar, rng.permutation(n)].mean()
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        out[space] = (obs, p, float(null.std()))
        del X, deltas, specific, sorth, sigs
    return out


def perm(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "*.h5ad")))
    print("{:<44}{:>10}{:>10}{:>10}{:>10}".format(
        "dataset", "obs_raw", "p_raw", "obs_cpm", "p_cpm"))
    print("-" * 84)
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            r = perm_dataset(f)
            print("{:<44}{:>10.3f}{:>10.3g}{:>10.3f}{:>10.3g}".format(
                name[:43], r["raw"][0], r["raw"][1], r["cpm"][0], r["cpm"][1]))
        except Exception as e:
            print("{:<44} SKIP: {}".format(name[:43], str(e)[:30]))


def scatter(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    def col(k):
        return np.array([float(r[k]) for r in rows])
    names = [r["dataset"][:18] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.scatter(col("global_frac_raw"), col("full_r2_raw"), s=60, c="tab:blue")
    for i, n in enumerate(names):
        plt.annotate(n, (col("global_frac_raw")[i], col("full_r2_raw")[i]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("global-mode variance fraction (raw)")
    plt.ylabel("full-response R2 (raw)")
    plt.title("Is full-response R2 explained by the global mode?")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig4_scatter_fullR2.png"), dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(col("global_frac_raw"), col("orth_corr_raw"), s=60, c="tab:red")
    for i, n in enumerate(names):
        plt.annotate(n, (col("global_frac_raw")[i], col("orth_corr_raw")[i]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("global-mode variance fraction (raw)")
    plt.ylabel("orthogonal specific corr (raw)")
    plt.title("Specificity vs global-mode dominance")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig5_scatter_specificity.png"), dpi=150); plt.close()
    print("wrote fig4_scatter_fullR2.png, fig5_scatter_specificity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--csv", default=os.path.join(RESULTS, "scan_results.csv"))
    ap.add_argument("--dir", default=_PERTURB_DATA)
    args = ap.parse_args()
    if args.plots:
        plots(args.csv)
    if args.scatter:
        scatter(args.csv)
    if args.perm:
        perm(args.dir)


if __name__ == "__main__":
    main()
