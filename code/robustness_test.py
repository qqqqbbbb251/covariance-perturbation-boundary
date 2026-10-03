import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
robustness_test.py  (clean)

Robustness of the central claim to the DEFINITION of the global mode.
Output: robustness_results.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
PREDICTORS = ["full", "pc1_raw", "pc12_raw", "counts_dir", "pc1_cpm"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def r2(y, pred):
    ss = float(y @ y)
    return 1.0 - float((y - pred) @ (y - pred)) / ss if ss else float("nan")


def fit_r2(B, y):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return r2(y, B @ coef)


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
    base = raw[(~ctrl) & (nperts == 1)]
    if len(np.unique(base)) < 10:
        base = raw[~ctrl]
    uq, cn = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uq, cn) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    if len(singles) < 10 or len(cidx) < 100:
        raise ValueError("too few")

    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    pos = {g: i for i, g in enumerate(a.var_names)}
    for g in singles:
        if g in pos:
            sel.add(pos[g])
    gi = np.array(sorted(sel))
    genes = np.asarray(a.var_names)[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    Xc_raw = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c = nc_all[cidx]
    cmean = Xc_raw.mean(0)
    Xc = Xc_raw - cmean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]; V = V[:, o]
    v1 = V[:, 0].astype(np.float32)
    V2 = np.ascontiguousarray(V[:, :2].astype(np.float32))
    cd = (Xc.T @ (nc_c - nc_c.mean())).astype(np.float32)
    cd = cd / (np.linalg.norm(cd) + 1e-9)
    Yc = Xc_raw * (1e4 / np.maximum(nc_c, 1.0))[:, None]
    Yc = Yc - Yc.mean(0)
    cov_cpm = (Yc.T @ Yc) / (Yc.shape[0] - 1)
    w2, Vc = np.linalg.eigh(cov_cpm)
    v1c = Vc[:, np.argmax(w2)].astype(np.float32)
    del V, Vc, w, w2, Yc, cov_cpm
    gc.collect()

    bases = {"full": None, "pc1_raw": v1[:, None], "pc12_raw": V2,
             "counts_dir": cd[:, None], "pc1_cpm": v1c[:, None]}
    acc = {p: [] for p in PREDICTORS}
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        Eg = np.asarray(X[rows].todense(), dtype=np.float32)
        delta = Eg.mean(0) - cmean
        del Eg
        if delta @ delta == 0:
            continue
        acc["full"].append(fit_r2(cov[:, idx[g]][:, None], delta))
        for p in PREDICTORS[1:]:
            acc[p].append(fit_r2(bases[p], delta))
        del delta
    del X, Xc, cov
    gc.collect()
    return {p: float(np.nanmean(acc[p])) if acc[p] else float("nan") for p in PREDICTORS}


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    with open("robustness_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(PREDICTORS) + "\n")
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("robustness_results.csv", "a") as fh:
                fh.write(name + "," + ",".join("{:.4f}".format(o[p]) for p in PREDICTORS) + "\n")
            print("OK   {:<36} ".format(name[:35]) + " ".join(
                "{}={:.3f}".format(p, o[p]) for p in PREDICTORS), flush=True)
        except Exception as e:
            print("SKIP {:<36} {}".format(name[:35], str(e)[:30]), flush=True)


if __name__ == "__main__":
    main()
