import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
composition_control.py  (reviewer point 5, strengthened)

Proper confounding controls for the global-mode (size) axis, on control cells.

Findings that motivate this script:
  * every scPerturb file here annotates a SINGLE cell type, so cell-type
    composition cannot be controlled directly; we report that explicitly.
  * batch and guide identity are available and are controlled with ONE-HOT
    design matrices (not integer codes, which are meaningless for >2 levels).
  * the top principal component IS the global mode, so "controlling for the top
    PCs" only removes the size axis if PC1 is included.  We report both:
    controlling PC2-5 (does not remove size) and controlling PC1-5 / PC1-10
    (removes it), making the interpretation unambiguous.

For each dataset we report the correlation of the global-mode score with an
independent median-of-ratios (MOR) size factor, and partial correlations
controlling for batch, guide, PC2-5, PC1-5 and PC1-10, plus the variance of the
global-mode score explained by batch / cell type (eta^2).

Output: ../results/composition_control.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "composition_control.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, N_TOP = 3000, 2000
MAX_GUIDE_LEVELS = 200


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(x, y, Z):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if Z is None or np.asarray(Z).size == 0:
        return corr(x, y)
    Z = np.atleast_2d(np.asarray(Z, float))
    if Z.shape[0] != len(x):
        Z = Z.T
    Z = np.column_stack([np.ones(len(x)), Z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return corr(x - Z @ bx, y - Z @ by)


def onehot(codes):
    u, inv = np.unique(codes, return_inverse=True)
    if len(u) <= 1:
        return None
    M = np.zeros((len(codes), len(u)), dtype=np.float32)
    M[np.arange(len(codes)), inv] = 1.0
    return M[:, 1:]          # drop first level (intercept)


def eta2(values, groups):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_between += m.sum() * (values[m].mean() - grand) ** 2
    return float(ss_between / ss_tot)


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    if len(cidx) < 100:
        raise ValueError("few controls")

    var_ncounts = a.var["ncounts"].values.astype(float)
    gi = np.array(sorted(set(np.argsort(-var_ncounts)[:N_TOP].tolist())))
    genes = list(np.asarray(a.var_names)[gi])
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()

    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c = obs["ncounts"].values.astype(np.float32)[cidx]
    Z = Xc - Xc.mean(0)
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    g = Z @ V[:, order[0]]
    pcs = Z @ V[:, order]        # all PCs, descending
    del V, w
    gc.collect()

    logX = np.log1p(Xc)
    ref = logX.mean(0)
    keep = ref > 0
    ratio = Xc[:, keep] / np.expm1(ref[keep])[None, :]
    mor = np.median(ratio, axis=1)

    out = {"n_ctrl": int(len(cidx)), "corr_total": corr(g, nc_c), "corr_mor": corr(g, mor)}

    if "batch" in obs.columns:
        b = obs["batch"].values[cidx]
        B = onehot(b)
        out["n_batch"] = int(len(np.unique(b)))
        out["partial_mor_given_batch"] = partial_corr(g, mor, B)
        out["eta2_batch"] = eta2(g, b)
    else:
        out["n_batch"] = 0
        out["partial_mor_given_batch"] = float("nan")
        out["eta2_batch"] = float("nan")

    if "guide_id" in obs.columns:
        gd = obs["guide_id"].values[cidx]
        nlev = len(np.unique(gd))
        out["n_guide"] = int(nlev)
        if nlev <= MAX_GUIDE_LEVELS:
            out["partial_mor_given_guide"] = partial_corr(g, mor, onehot(gd))
        else:
            out["partial_mor_given_guide"] = float("nan")
    else:
        out["n_guide"] = 0
        out["partial_mor_given_guide"] = float("nan")

    out["partial_mor_given_pc2_5"] = partial_corr(g, mor, pcs[:, 1:5])
    out["partial_mor_given_pc1_5"] = partial_corr(g, mor, pcs[:, 0:5])
    out["partial_mor_given_pc1_10"] = partial_corr(g, mor, pcs[:, 0:10])

    if "celltype" in obs.columns:
        ct = obs["celltype"].values[cidx]
        out["n_celltype"] = int(len(np.unique(ct)))
        out["eta2_celltype"] = eta2(g, ct) if len(np.unique(ct)) > 1 else float("nan")
    else:
        out["n_celltype"] = 0
        out["eta2_celltype"] = float("nan")

    del X, Xc, Z, pcs, cov
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["n_ctrl", "n_batch", "n_guide", "n_celltype", "corr_total", "corr_mor",
            "partial_mor_given_batch", "partial_mor_given_guide",
            "partial_mor_given_pc2_5", "partial_mor_given_pc1_5",
            "partial_mor_given_pc1_10", "eta2_batch", "eta2_celltype"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    acc = {k: [] for k in keys}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open(OUT, "a") as fh:
                fh.write(name + "," + ",".join(
                    str(o[k]) if k.startswith("n_") else "{:.4f}".format(o[k]) for k in keys) + "\n")
            for k in keys:
                if np.isfinite(o[k]):
                    acc[k].append(o[k])
            print("OK   {:<28} MOR={:.3f} batch={:.3f} guide={:.3f} pc2_5={:.3f} pc1_5={:.3f}".format(
                name[:27], o["corr_mor"], o["partial_mor_given_batch"],
                o["partial_mor_given_guide"], o["partial_mor_given_pc2_5"],
                o["partial_mor_given_pc1_5"]), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:40]), flush=True)
    print("\nMEANS (|corr|):", flush=True)
    for k in keys:
        if k.startswith("n_"):
            continue
        v = [abs(x) for x in acc[k]]
        print("  {:<26} {:.3f} (n={})".format(k, np.mean(v) if v else float("nan"), len(v)), flush=True)


if __name__ == "__main__":
    main()
