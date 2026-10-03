import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
fig_strong.py

The decisive figure: for every dataset, decompose the perturbation response and
compare three R2 values (raw space):

  full      : fit  a * Sigma[:,g]        -> the CIPHER prediction
  global    : fit  b * v                 -> using ONLY the global mode
  specific  : fit  a * Sigma[:,g] on the global-mode-orthogonalised response

If full ~ global and specific ~ 0, the covariance model adds nothing beyond a
single global mode. Writes strong_r2.csv and fig6_strong.png.
"""

import glob
import os

import numpy as np
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def to_space(X, ncounts, space):
    if space == "raw":
        return X
    inv = (1e4 / np.maximum(ncounts, 1.0)).astype(np.float32)
    return (sp.diags(inv) @ X).tocsr()


def fit_r2(sig, y):
    d = float(np.dot(sig, sig))
    if d == 0:
        return float("nan")
    a = float(np.dot(sig, y) / d)
    ss = float(np.sum(y ** 2))
    return 1.0 - float(np.sum((y - a * sig) ** 2)) / ss if ss else float("nan")


def run(path, n_top=2000, max_pert=150):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts = obs["ncounts"].values.astype(float)
    var_ncounts = a.var["ncounts"].values.astype(float)
    var_names = np.asarray(a.var_names)
    ctrl = np.array([is_control(p) for p in raw])
    gene = np.array(["control" if c else p for p, c in zip(raw, ctrl)])

    n1 = (~ctrl) & (nperts == 1)
    base = gene[n1] if len(np.unique(gene[n1])) >= 10 else gene[~ctrl]
    uniq, cnt = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uniq, cnt) if c >= 30 and g != "control"]
    if len(singles) > max_pert:
        singles = list(np.random.default_rng(0).choice(singles, max_pert, replace=False))
    if len(singles) < 10 or int(ctrl.sum()) < 100:
        raise ValueError("too few")

    selected = set(np.argsort(-var_ncounts)[:n_top].tolist())
    pos = {g: i for i, g in enumerate(var_names)}
    for g in singles:
        if g in pos:
            selected.add(pos[g])
    gi = np.array(sorted(selected))
    genes = var_names[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    out = {}
    for space in ["raw", "cpm"]:
        Y = to_space(X, ncounts, space)
        Yc = np.asarray(Y[ctrl].todense(), dtype=np.float64)
        cmean = Yc.mean(0)
        cov = np.cov(Yc, rowvar=False)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)]; v = v / np.linalg.norm(v)
        del Yc
        full, glob, spec = [], [], []
        for g in singles:
            d = np.asarray(Y[gene == g].todense(), dtype=np.float64).mean(0) - cmean
            if np.sum(d ** 2) == 0:
                continue
            sig = cov[:, idx[g]]
            full.append(fit_r2(sig, d))
            glob.append(fit_r2(v, d))
            d_orth = d - (d @ v) * v
            spec.append(fit_r2(sig, d_orth))
        out[space] = (np.nanmean(full), np.nanmean(glob), np.nanmean(spec))
        del Y
    return out


def main():
    d = _PERTURB_DATA
    files = sorted(glob.glob(os.path.join(d, "*.h5ad")))
    rows = []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            r = run(f)
            rows.append((name, r["raw"], r["cpm"]))
            print("OK   {:<44} raw full={:.3f} glob={:.3f} spec={:.3f}".format(
                name[:43], *r["raw"]))
        except Exception as e:
            print("SKIP {:<44} {}".format(name[:43], str(e)[:35]))

    with open(os.path.join(RESULTS, "strong_r2.csv"), "w") as fh:
        fh.write("dataset,full_raw,global_raw,spec_raw,full_cpm,global_cpm,spec_cpm\n")
        for n, rr, cc in rows:
            fh.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(n, *rr, *cc))

    names = [r[0][:16] for r in rows]
    x = np.arange(len(rows))
    full = [r[1][0] for r in rows]
    globr = [r[1][1] for r in rows]
    spec = [r[1][2] for r in rows]
    w = 0.28
    plt.figure(figsize=(12, 5))
    plt.bar(x - w, full, w, label="full (covariance)")
    plt.bar(x, globr, w, label="global mode only")
    plt.bar(x + w, spec, w, label="specific (orthogonal)")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("mean R2 (raw)")
    plt.title("Full-response R2 is reproduced by a single global mode; specific R2 is ~0")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig6_strong.png"), dpi=150)
    print("wrote fig6_strong.png, strong_r2.csv")


if __name__ == "__main__":
    main()
