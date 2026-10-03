import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cross_dataset_transfer.py  (reviewer: generalisation / non-specificity)

Tests whether a covariance model learned on one dataset predicts the responses of
a *different* dataset as well as that dataset's own covariance does.  If the model
captures a generic global response rather than dataset-specific gene structure,
cross-dataset transfer should match the within-dataset model.

For each pair (A, B) restricted to shared genes we compare, on B's single
perturbations (raw counts, squared cosine and uncentered R2):
    within_full   : B's own Sigma_B[:, g]
    within_global : B's own global mode v_B
    cross_full    : A's Sigma_A[:, g]
    cross_global  : A's global mode v_A
    cross_random  : A's random column

Output: ../results/cross_dataset_transfer.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "cross_dataset_transfer.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 1500, 120


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def r2u(yt, yp):
    ss = float(yt @ yt)
    return 1.0 - float((yt - yp) @ (yt - yp)) / ss if ss > 0 else float("nan")


def fit_pred(col, y):
    d = float(col @ col)
    a = float(col @ y) / d if d > 0 else 0.0
    return a * col


PERT_COLS = ["perturbation", "gene", "target", "target_gene", "gene_name",
             "guide_target", "condition", "sgRNA", "guide_id"]


def _pert_col(obs):
    for c in PERT_COLS:
        if c in obs.columns and obs[c].astype(str).nunique() > 5:
            return c
    return "perturbation"


def load_side(path, shared):
    """Return (Sigma, v, cmean, X, raw, genes, gpos, singles, nperts) restricted to `shared`.

    Robust to the perturbation column name and to labels that are not bare gene symbols
    (guide ids etc. are mapped to targets via :func:`cipher.data.infer_target_gene`).
    """
    import pandas as pd
    from cipher.data import infer_target_gene

    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = _pert_col(obs)
    labels = obs[pc].astype(str).values
    nperts = obs["nperts"].values if "nperts" in obs.columns else np.ones(len(labels), dtype=int)
    var_names = np.asarray(a.var_names)
    pos_all = {g: i for i, g in enumerate(var_names)}
    geneset = set(map(str, var_names))
    lab2gene = {l: (infer_target_gene(str(l), geneset)[0] or "") for l in pd.unique(labels)}
    mapped = np.array([lab2gene.get(l, "") for l in labels])
    shared = [g for g in shared if g in pos_all]
    ctrl = np.array([is_control(p) for p in labels]) | (nperts == 0)
    base = mapped[(~ctrl) & (nperts == 1)]
    if len(np.unique(base[base != ""])) < 10:
        base = mapped[~ctrl]
    uq, cn = np.unique(base[base != ""], return_counts=True)
    singles = [g for g, c in zip(uq, cn) if c >= 30]
    rng = np.random.default_rng(0)
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    vn = (a.var["ncounts"].values.astype(float) if "ncounts" in a.var.columns
          else np.zeros(len(var_names)))
    order = np.argsort(-vn)
    sel = []
    sset = set(shared)
    for j in order:
        if var_names[j] in sset:
            sel.append(j)
        if len(sel) >= N_TOP:
            break
    for g in singles:
        if g in pos_all:
            sel.append(pos_all[g])
    sel = sorted(set(sel))
    genes = var_names[sel]
    gpos = {g: i for i, g in enumerate(genes)}
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    Xm = a[:, sel].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    Sigma = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(Sigma)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()
    singles = [g for g in singles if g in gpos]
    return Sigma, v, cmean, X, mapped, np.asarray(genes), gpos, singles, nperts


def transfer(pair):
    pathA, pathB = pair
    # shared gene names
    aA = ad.read_h5ad(pathA, backed="r")
    aB = ad.read_h5ad(pathB, backed="r")
    shared = sorted(set(map(str, aA.var_names)) & set(map(str, aB.var_names)))
    del aA, aB
    gc.collect()
    if len(shared) < 500:
        raise ValueError("few shared genes")
    SigmaA, vA, cmeanA, XA, rawA, genesA, gposA, singlesA, npA = load_side(pathA, shared)
    SigmaB, vB, cmeanB, XB, rawB, genesB, gposB, singlesB, npB = load_side(pathB, shared)
    common = sorted(set(genesA) & set(genesB))
    ia = {g: i for i, g in enumerate(genesA)}
    ib = {g: i for i, g in enumerate(genesB)}
    n = len(common)
    if n < 500:
        raise ValueError("few common genes")
    SigA = SigmaA[np.ix_([ia[g] for g in common], [ia[g] for g in common])]
    SigB = SigmaB[np.ix_([ib[g] for g in common], [ib[g] for g in common])]
    vAa = np.array([vA[ia[g]] for g in common]); vAa /= np.linalg.norm(vAa)
    vBb = np.array([vB[ib[g]] for g in common]); vBb /= np.linalg.norm(vBb)
    cmA = np.array([cmeanA[ia[g]] for g in common])
    cmB = np.array([cmeanB[ib[g]] for g in common])
    ci = {g: i for i, g in enumerate(common)}

    rng = np.random.default_rng(0)
    singlesB = [g for g in singlesB if g in ci]
    res = {k: {"cos": [], "r2": []} for k in
           ["within_full", "within_global", "cross_full", "cross_global", "cross_random"]}
    for g in singlesB:
        rows = np.where(rawB == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        # B response on common genes
        Xg = np.asarray(XB[rows].todense(), dtype=np.float32)
        idxB = np.array([ib[c] for c in common])
        d = Xg[:, idxB].mean(0) - cmB
        del Xg
        if d @ d == 0:
            continue
        i = ci[g]
        preds = {
            "within_full": fit_pred(SigB[:, i], d),
            "within_global": fit_pred(vBb, d),
            "cross_full": fit_pred(SigA[:, i], d),
            "cross_global": fit_pred(vAa, d),
            "cross_random": fit_pred(SigA[:, int(rng.integers(len(common)))], d),
        }
        for k, p in preds.items():
            res[k]["cos"].append(cos2(d, p))
            res[k]["r2"].append(r2u(d, p))
        del d
    del XA, XB, SigmaA, SigmaB, SigA, SigB
    gc.collect()
    out = {"pair": os.path.basename(pathA).replace(".h5ad", "") + "->" +
                   os.path.basename(pathB).replace(".h5ad", ""),
           "n_shared": len(common), "n_perts": len(res["within_full"]["cos"])}
    for k in res:
        out[k + "_cos"] = float(np.nanmean(res[k]["cos"]))
        out[k + "_r2"] = float(np.nanmean(res[k]["r2"]))
    return out


def main():
    d = _PERTURB_DATA
    pairs = [
        ("NadigOConner2024_hepg2.h5ad", "NadigOConner2024_jurkat.h5ad"),
        ("ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
        ("TianKampmann2021_CRISPRa.h5ad", "TianKampmann2021_CRISPRi.h5ad"),
    ]
    keys = ["within_full", "within_global", "cross_full", "cross_global", "cross_random"]
    cols = ["pair", "n_shared", "n_perts"] + [k + "_cos" for k in keys] + [k + "_r2" for k in keys]
    with open(OUT, "w") as fh:
        fh.write(",".join(cols) + "\n")
    for pa, pb in pairs:
        try:
            o = transfer((os.path.join(d, pa), os.path.join(d, pb)))
            with open(OUT, "a") as fh:
                fh.write(o["pair"] + ",{},{},".format(o["n_shared"], o["n_perts"]) +
                         ",".join("{:.4f}".format(o[k + "_cos"]) for k in keys) + "," +
                         ",".join("{:.4f}".format(o[k + "_r2"]) for k in keys) + "\n")
            print("OK {:<40} within_full={:.3f} cross_full={:.3f} cross_global={:.3f} cross_rand={:.3f}".format(
                o["pair"][:39], o["within_full_cos"], o["cross_full_cos"],
                o["cross_global_cos"], o["cross_random_cos"]), flush=True)
        except Exception as e:
            print("SKIP {:<40} {}".format(pa[:39], str(e)[:40]), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
