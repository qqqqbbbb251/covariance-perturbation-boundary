import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
spaces_diagnostic.py  (reviewer point 2/3)

Runs the key diagnostic in four expression spaces:
  raw            raw counts
  cpm            counts per 10k
  logcpm         log1p(CPM)
  pearson        Pearson residuals under a size-factor Poisson model

For each space reports (means over datasets):
  full, global, shufcol, meandir, dR2, dR2_shuf

Output: spaces_results.csv
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
SPACES = ["raw", "cpm", "logcpm", "pearson"]
METRICS = ["full", "global", "shufcol", "meandir", "dR2", "dR2_shuf"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def fit_r2(B, y):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    pred = B @ coef
    ss = float(y @ y)
    return 1.0 - float((y - pred) @ (y - pred)) / ss if ss else float("nan")


def transform(E, nc, space, mu=None):
    """E: cells x genes raw counts (float32)."""
    if space == "raw":
        return E
    sf = nc / (nc.mean() + 1e-9)
    if space == "cpm":
        return E * (1e4 / np.maximum(nc, 1.0))[:, None]
    if space == "logcpm":
        return np.log1p(E * (1e4 / np.maximum(nc, 1.0))[:, None])
    if space == "pearson":
        muij = sf[:, None] * mu[None, :]
        return (E - muij) / np.sqrt(muij + 1e-6)
    raise ValueError(space)


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
    Xraw = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]
    gidx = {}
    for g in singles:
        r = np.where(raw == g)[0]
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        gidx[g] = r

    # size-factor gene means (for pearson)
    Xc_raw = np.asarray(Xraw[cidx].todense(), dtype=np.float32)
    nc_c = nc_all[cidx]
    sf_c = nc_c / (nc_c.mean() + 1e-9)
    mu = (Xc_raw / sf_c[:, None]).mean(0)          # size-normalised gene mean

    out = {}
    for space in SPACES:
        Xc = transform(Xc_raw, nc_c, space, mu)
        cmean = Xc.mean(0)
        Xc = Xc - cmean
        cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
        v = V[:, 0].astype(np.float32); v = v / np.linalg.norm(v)
        meandir = cmean / (np.linalg.norm(cmean) + 1e-9)
        del Xc, V, w
        gc.collect()
        M = {k: [] for k in METRICS}
        for g in singles:
            Eg = np.asarray(Xraw[gidx[g]].todense(), dtype=np.float32)
            d = transform(Eg, nc_all[gidx[g]], space, mu).mean(0) - cmean
            del Eg
            if d @ d == 0:
                continue
            sig = cov[:, idx[g]]
            g2 = singles[rng.integers(len(singles))]
            sig_shuf = cov[:, idx[g2]]
            r2g = float((d @ v) ** 2) / float(d @ d)
            sig_perp = sig - float(sig @ v) * v
            shuf_perp = sig_shuf - float(sig_shuf @ v) * v
            M["full"].append(cos(d, sig) ** 2)
            M["global"].append(r2g)
            M["shufcol"].append(cos(d, sig_shuf) ** 2)
            M["meandir"].append(cos(d, meandir) ** 2)
            M["dR2"].append(fit_r2(np.stack([v, sig_perp], 1), d) - r2g)
            M["dR2_shuf"].append(fit_r2(np.stack([v, shuf_perp], 1), d) - r2g)
            del d, sig, sig_shuf, sig_perp, shuf_perp
        del cov
        gc.collect()
        out[space] = {k: float(np.nanmean(vals)) if vals else float("nan") for k, vals in M.items()}
    del Xraw, Xc_raw
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    cols = [s + "_" + k for s in SPACES for k in METRICS]
    with open("spaces_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(cols) + "\n")
    acc = {s: {k: [] for k in METRICS} for s in SPACES}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("spaces_results.csv", "a") as fh:
                fh.write(name + "," + ",".join(
                    "{:.4f}".format(o[s][k]) for s in SPACES for k in METRICS) + "\n")
            for s in SPACES:
                for k in METRICS:
                    if not np.isnan(o[s][k]):
                        acc[s][k].append(o[s][k])
            print("OK   {:<28} " + " | ".join(
                "{} full={:.2f} glob={:.2f} shuf={:.2f} dR2={:.2f}/{:.2f}".format(
                    s, o[s]["full"], o[s]["global"], o[s]["shufcol"], o[s]["dR2"], o[s]["dR2_shuf"])
                for s in SPACES), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<28} {}".format(name[:27], repr(e)), flush=True)
            traceback.print_exc()
    print("\nMEANS:", flush=True)
    for s in SPACES:
        print(" [{}]".format(s), flush=True)
        for k in METRICS:
            v = acc[s][k]
            print("   {:<9} {:.4f}".format(k, np.mean(v) if v else float("nan")), flush=True)


if __name__ == "__main__":
    main()
