import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
guide_consistency.py -- robustness: is the specific residual target-specific rather than
a guide/batch artifact?

For datasets that record guide_id: for every target with >=2 guides (each with >=MIN
cells), compute the (global-axis- and self-removed) residual response per guide and the
cosine agreement across guides.  High agreement means the specific structure is
target-driven, not guide-driven.  A target-shuffled null gives the baseline.

Output: ../results/guide_consistency.csv
"""

import csv
import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "guide_consistency.csv")
GUIDE_COLS = ["guide_id", "sgRNA", "guide_target", "guide"]
N_TOP, MAX_CTRL, MIN_CELLS = 1500, 3000, 20
CONTROL = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")


def is_control(l):
    l = str(l).strip().lower()
    return l in CONTROL or any(t in l for t in ("control", "ctrl", "non-targeting", "nontargeting"))


def analyze(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = next((c for c in ["perturbation", "gene", "target"] if c in obs.columns), "perturbation")
    gcol = next((c for c in GUIDE_COLS if c in obs.columns), None)
    if gcol is None:
        a.file.close(); raise ValueError("no guide column")
    labels = obs[pc].astype(str).values
    guides = obs[gcol].astype(str).values
    var_names = np.asarray(a.var_names).astype(str)
    vn = None
    for c in ("ncounts", "total_counts", "mean_counts", "means"):
        if c in a.var.columns:
            vn = np.asarray(a.var[c].values, dtype=float); break
    if vn is None:
        vn = np.ones(len(var_names))
    sel = np.sort(np.argsort(-np.nan_to_num(vn))[:N_TOP])
    genes = var_names[sel]
    gpos = {g: i for i, g in enumerate(genes)}
    ctrl = np.array([is_control(l) for l in labels])
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    Xm = a[:, sel].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    Ec = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Ec.mean(0)
    Z = Ec - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Ec, Z, V, w
    gc.collect()

    targets = [g for g in np.unique(labels[~ctrl]) if g in gpos]
    per_target, nulls = [], []
    for g in targets:
        rows = np.where(labels == g)[0]
        gs = guides[rows]
        ug = np.unique(gs)
        vecs = []
        for u in ug:
            rr = rows[gs == u]
            if len(rr) < MIN_CELLS:
                continue
            d = np.asarray(X[rr].todense(), dtype=np.float64).mean(0) - cmean
            if d @ d == 0:
                continue
            d = d - float(d @ v) * v
            d[gpos[g]] = 0.0
            vecs.append(d)
        if len(vecs) < 2:
            continue
        cs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                x, y = vecs[i], vecs[j]
                cs.append(float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        per_target.append(np.mean(cs))
    del X, cov
    gc.collect()
    if len(per_target) < 10:
        raise ValueError("too few multi-guide targets")

    return {"dataset": os.path.basename(path).replace(".h5ad", ""),
            "n_targets_multiguide": len(per_target),
            "guide_agreement": float(np.mean(per_target)),
            "guide_agreement_median": float(np.median(per_target))}


def main():
    d = DATA
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "ReplogleWeissman2022_K562_gwps_filtered.h5ad", "NadigOConner2024_jurkat.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_targets_multiguide", "guide_agreement", "guide_agreement_median"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<26} n_multiguide={} agreement={:.3f} (median {:.3f})".format(
                r["dataset"][:25], r["n_targets_multiguide"], r["guide_agreement"],
                r["guide_agreement_median"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
