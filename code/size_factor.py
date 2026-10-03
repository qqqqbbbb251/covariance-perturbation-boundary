import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
size_factor.py  (reviewer point 4)

Tests whether the global mode tracks an INDEPENDENT size factor (not the CPM
tautology). We build a size factor from the 500 least-variable genes across
control cells, then correlate the per-cell global-mode score with it and with
total counts, in raw and CPM space.

Output: size_factor_results.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, N_TOP = 3000, 2000


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    if len(cidx) < 100:
        raise ValueError("few controls")

    # all genes for the control cells, to pick stable genes
    Xall = a[cidx, :].to_memory().X
    Xall = (Xall.tocsr() if sp.issparse(Xall) else sp.csr_matrix(Xall)).astype(np.float32)
    mean = np.asarray(Xall.mean(0)).ravel()
    var = np.asarray(Xall.power(2).mean(0)).ravel() - mean ** 2
    cv = np.sqrt(np.maximum(var, 0)) / (mean + 1e-9)
    expressed = mean > 0.5
    cv_masked = np.where(expressed, cv, np.inf)
    stable = np.argsort(cv_masked)[:500]
    stable = stable[np.isfinite(cv_masked[stable])]
    if len(stable) < 50:
        stable = np.argsort(-mean)[:500]
    sf_indep = np.asarray(Xall[:, stable].sum(1)).ravel()   # independent size factor
    del Xall
    gc.collect()

    # selected genes for the covariance
    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    gi = np.array(sorted(sel))
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    del X
    gc.collect()

    out = {}
    for space in ["raw", "cpm"]:
        if space == "raw":
            Y = Xc.copy()
        else:
            Y = Xc * (1e4 / np.maximum(nc_all[cidx], 1.0))[:, None]
        Y = Y - Y.mean(0)
        cov = (Y.T @ Y) / (Y.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)]
        score = Y @ v
        out[space + "_corr_total"] = corr(score, nc_all[cidx])
        out[space + "_corr_indep"] = corr(score, sf_indep)
        del Y, cov, V, w
        gc.collect()
    del Xc
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    keys = ["raw_corr_total", "raw_corr_indep", "cpm_corr_total", "cpm_corr_indep"]
    with open("size_factor_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    acc = {k: [] for k in keys}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("size_factor_results.csv", "a") as fh:
                fh.write(name + "," + ",".join("{:.4f}".format(o[k]) for k in keys) + "\n")
            for k in keys:
                if not np.isnan(o[k]):
                    acc[k].append(o[k])
            print("OK   {:<32} raw total={:.3f} indep={:.3f} | cpm total={:.3f} indep={:.3f}".format(
                name[:31], o["raw_corr_total"], o["raw_corr_indep"],
                o["cpm_corr_total"], o["cpm_corr_indep"]), flush=True)
        except Exception as e:
            print("SKIP {:<32} {}".format(name[:31], str(e)[:30]), flush=True)
    print("\nMEANS:", flush=True)
    for k in keys:
        print("  {:<16} {:.3f}".format(k, np.mean(acc[k]) if acc[k] else float("nan")), flush=True)


if __name__ == "__main__":
    main()
