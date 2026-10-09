import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
global_mode_identity.py  (v2)

What IS the global mode, in raw-count and in CPM space?

For each dataset we project control cells onto the leading eigenvector v of the
control covariance (the "global mode") and correlate that per-cell score with
total counts, genes detected, %mito and %ribo.

In CPM space library size is normalised away, so this reveals what the global
mode becomes once the technical size factor is removed.

Writes global_mode_identity.csv (raw and cpm columns).
"""

import glob
import os

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(path, n_top=2000):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts = obs["ncounts"].values.astype(float)
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    if ctrl.sum() < 100:
        raise ValueError("few controls")
    gi = np.array(sorted(set(np.argsort(-var_ncounts)[:n_top].tolist())))
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)

    covs = {}
    for c in ["ncounts", "ngenes", "percent_mito", "percent_ribo"]:
        if c in obs.columns:
            covs[c] = obs[c].values.astype(float)[ctrl]
    del a, Xm

    out = {}
    for space in ["raw", "cpm"]:
        if space == "raw":
            Y = X
        else:
            inv = (1e4 / np.maximum(ncounts, 1.0)).astype(np.float32)
            Y = (sp.diags(inv) @ X).tocsr()
        Yc = np.asarray(Y[ctrl].todense(), dtype=np.float64)
        Yc = Yc - Yc.mean(0)
        cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)]
        pc1 = Yc @ v
        for c, vals in covs.items():
            out[c + "_" + space] = corr(pc1, vals)
        del Yc, cov
    return out


def main():
    d = _PERTURB_DATA
    cols = [c + "_" + s for s in ("raw", "cpm")
            for c in ("ncounts", "ngenes", "percent_mito", "percent_ribo")]
    rows = []
    print("{:<40}".format("dataset") + "".join("{:>13}".format(c) for c in cols))
    print("-" * (40 + 13 * len(cols)))
    # 16-dataset analysis set (matches the reported table)
    names = [
        "AissaBenevolenskaya2021", "ChangYe2021", "DatlingerBock2017",
        "DatlingerBock2021", "FrangiehIzar2021_RNA", "NadigOConner2024_hepg2",
        "NadigOConner2024_jurkat", "NormanWeissman2019_filtered",
        "PapalexiSatija2021_eccite_arrayed_RNA", "PapalexiSatija2021_eccite_RNA",
        "ReplogleWeissman2022_K562_essential", "ReplogleWeissman2022_rpe1",
        "TianKampmann2019_day7neuron", "TianKampmann2019_iPSC",
        "TianKampmann2021_CRISPRa", "TianKampmann2021_CRISPRi"]
    for name in names:
        f = os.path.join(d, name + ".h5ad")
        try:
            r = run(f)
            rows.append((name, r))
            print("{:<40}".format(name[:39]) + "".join(
                "{:>13.3f}".format(r.get(c, float("nan"))) for c in cols))
        except Exception as e:
            print("{:<40} SKIP: {}".format(name[:39], str(e)[:30]))
    with open("global_mode_identity.csv", "w") as fh:
        fh.write("dataset," + ",".join(cols) + "\n")
        for name, r in rows:
            fh.write(name + "," + ",".join(
                "{:.4f}".format(r.get(c, float("nan"))) for c in cols) + "\n")
    print("wrote global_mode_identity.csv")


if __name__ == "__main__":
    main()
