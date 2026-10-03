import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
diagnostic.py  (v2 - raw and CPM)

Per space (raw, cpm), per dataset:
  r2_full      optimal-scale fit of the correct gene's covariance column
  r2_global    variance along the global mode v
  r2_shufcol   same fit but with a RANDOM gene's covariance column
  r2_meandir   fit of the mean-expression direction
  r2_randdir   fit of a random direction
  r2_pc5       fit of the 5th principal component
  dR2          nested [v, Sigma_perp[:,g]] minus v-only
  dR2_shuf     same with a RANDOM gene's perpendicular column
  cos_spec     correlation of the orthogonalised response with the
               orthogonalised correct column

Writes diagnostic_results.csv.
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
METRICS = ["r2_full", "r2_global", "r2_shufcol", "r2_meandir", "r2_randdir",
           "r2_pc5", "dR2", "dR2_shuf", "cos_spec"]


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


def to_space(X, nc, space):
    if space == "raw":
        return X
    inv = (1e4 / np.maximum(nc, 1.0)).astype(np.float32)
    return (sp.diags(inv) @ X).tocsr()


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

    out = {}
    for space in ["raw", "cpm"]:
        X = to_space(Xraw, nc_all, space)
        Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
        cmean = Xc.mean(0)
        Xc = Xc - cmean
        cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
        v = V[:, 0].astype(np.float32); v = v / np.linalg.norm(v)
        pc5 = V[:, 4].astype(np.float32); pc5 = pc5 / np.linalg.norm(pc5)
        meandir = cmean / (np.linalg.norm(cmean) + 1e-9)
        randdir = rng.normal(size=len(genes)).astype(np.float32)
        randdir = randdir / np.linalg.norm(randdir)
        del Xc, V, w
        gc.collect()

        M = {k: [] for k in METRICS}
        for g in singles:
            Eg = np.asarray(X[gidx[g]].todense(), dtype=np.float32)
            if space != "raw":
                Eg = Eg * (1e4 / np.maximum(nc_all[gidx[g]], 1.0))[:, None]
            d = Eg.mean(0) - cmean
            del Eg
            if d @ d == 0:
                continue
            sig = cov[:, idx[g]]
            g2 = singles[rng.integers(len(singles))]
            sig_shuf = cov[:, idx[g2]]
            r2g = float((d @ v) ** 2) / float(d @ d)
            sig_perp = sig - float(sig @ v) * v
            shuf_perp = sig_shuf - float(sig_shuf @ v) * v
            M["r2_full"].append(cos(d, sig) ** 2)
            M["r2_global"].append(r2g)
            M["r2_shufcol"].append(cos(d, sig_shuf) ** 2)
            M["r2_meandir"].append(cos(d, meandir) ** 2)
            M["r2_randdir"].append(cos(d, randdir) ** 2)
            M["r2_pc5"].append(cos(d, pc5) ** 2)
            M["dR2"].append(fit_r2(np.stack([v, sig_perp], 1), d) - r2g)
            M["dR2_shuf"].append(fit_r2(np.stack([v, shuf_perp], 1), d) - r2g)
            d_orth = d - float(d @ v) * v
            M["cos_spec"].append(cos(d_orth, sig_perp))
            del d, sig, sig_shuf, sig_perp, shuf_perp, d_orth
        del X, cov
        gc.collect()
        out[space] = {k: float(np.nanmean(vals)) if vals else float("nan") for k, vals in M.items()}
    del Xraw
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    cols = [s + "_" + k for s in ["raw", "cpm"] for k in METRICS]
    with open("diagnostic_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(cols) + "\n")
    acc = {s: {k: [] for k in METRICS} for s in ["raw", "cpm"]}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("diagnostic_results.csv", "a") as fh:
                fh.write(name + "," + ",".join(
                    "{:.4f}".format(o[s][k]) for s in ["raw", "cpm"] for k in METRICS) + "\n")
            for s in ["raw", "cpm"]:
                for k in METRICS:
                    if not np.isnan(o[s][k]):
                        acc[s][k].append(o[s][k])
            print("OK   {:<30} raw full={:.3f} glob={:.3f} shuf={:.3f} dR2={:.3f}/{:.3f} | "
                  "cpm full={:.3f} glob={:.3f} shuf={:.3f} dR2={:.3f}/{:.3f}".format(
                      name[:29], o["raw"]["r2_full"], o["raw"]["r2_global"], o["raw"]["r2_shufcol"],
                      o["raw"]["dR2"], o["raw"]["dR2_shuf"],
                      o["cpm"]["r2_full"], o["cpm"]["r2_global"], o["cpm"]["r2_shufcol"],
                      o["cpm"]["dR2"], o["cpm"]["dR2_shuf"]), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:30]), flush=True)
    print("\nMEANS:", flush=True)
    for s in ["raw", "cpm"]:
        print(" [{}]".format(s), flush=True)
        for k in METRICS:
            v = acc[s][k]
            print("   {:<11} {:.4f}".format(k, np.mean(v) if v else float("nan")), flush=True)


if __name__ == "__main__":
    main()
