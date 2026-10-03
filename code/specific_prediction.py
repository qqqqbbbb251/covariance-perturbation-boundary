import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_prediction.py -- Stage 2d: what actually predicts the specific residual?

Two questions, raw space, for K562 / rpe1 / hepg2 / jurkat:

  (A) Within dataset: how much of one target's specific residual is captured by the
      *population subspace* of the other targets (leave-one-out SVD, rank k)?  This is
      the part predictable without knowing the target.

  (B) Across datasets: for the *same* target in two datasets, how well does the other
      dataset's residual predict it?
        - identity copy  r_B ~ r_A
        - Procrustes map r_B ~ r_A Q (Q fitted on training targets, 5-fold)
        - random target baseline
      This measures the target-specific, transferable part.

Output: ../results/specific_prediction.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import differential_identity as di  # noqa: E402
di.MAX_PERT = 2000          # keep all single perturbations for cross-dataset matching
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "specific_prediction.csv")

MAX_PERT_CELLS = 500
LOO_N = 120
SEED = 0
KS = [1, 3, 5, 10, 20]


def build_R(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    genes = np.asarray(genes)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()
    R, tg = [], []
    for g in singles:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        d = d - float(d @ v) * v
        d[t] = 0.0
        R.append(d); tg.append(g)
    R = np.asarray(R)
    R = R - R.mean(0)
    del X, cov
    gc.collect()
    return os.path.basename(path).replace(".h5ad", ""), genes, np.asarray(tg), R


def loo_svd_cos2(R, ks, n_use):
    n = min(n_use, len(R))
    idxs = np.random.default_rng(SEED).choice(len(R), n, replace=False)
    acc = {k: [] for k in ks}
    for i in idxs:
        train = np.delete(R, i, axis=0)
        r = R[i]
        rr = float(r @ r)
        if rr == 0:
            continue
        # right singular vectors of train
        _, _, Vt = np.linalg.svd(train, full_matrices=False)
        for k in ks:
            Vk = Vt[:k]                       # (k, p)
            proj = Vk.T @ (Vk @ r)
            acc[k].append(float(proj @ proj) / rr)
    return {k: float(np.mean(acc[k])) for k in ks}


def cross_predictions(nameA, genesA, tgA, RA, nameB, genesB, tgB, RB):
    gA = {g: i for i, g in enumerate(genesA)}
    gB = {g: i for i, g in enumerate(genesB)}
    common_genes = sorted(set(map(str, genesA)) & set(map(str, genesB)))
    cg = {g: i for i, g in enumerate(common_genes)}
    ia = np.array([gA[g] for g in common_genes]); ib = np.array([gB[g] for g in common_genes])
    tA = {g: i for i, g in enumerate(tgA)}
    tB = {g: i for i, g in enumerate(tgB)}
    targets = sorted(set(map(str, tgA)) & set(map(str, tgB)))
    A = np.asarray([RA[tA[g]][ia] for g in targets])
    B = np.asarray([RB[tB[g]][ib] for g in targets])
    A = A - A.mean(0); B = B - B.mean(0)
    n = len(targets)
    if n < 20:
        raise ValueError("few targets")

    def cos2_rows(P, Q):
        num = np.sum(P * Q, axis=1)
        npn = np.sum(P * P, axis=1)
        nqn = np.sum(Q * Q, axis=1)
        return float(np.nanmean(num ** 2 / np.maximum(npn * nqn, 1e-24)))

    rng = np.random.default_rng(SEED)
    identity = cos2_rows(A, B)
    rand = cos2_rows(A, B[rng.permutation(n)])
    # 5-fold Procrustes: Q = U V^T from SVD(A_tr^T B_tr)
    fold = np.array_split(rng.permutation(n), 5)
    preds = np.zeros_like(B)
    for f in fold:
        tr = np.setdiff1d(np.arange(n), f)
        U, S, Vt = np.linalg.svd(A[tr].T @ B[tr], full_matrices=False)
        Qm = U @ Vt
        preds[f] = A[f] @ Qm
    proc = cos2_rows(preds, B)
    # rank-1 Procrustes (scalar cosine order) baseline
    return {"n_targets": n, "n_common_genes": len(common_genes),
            "cos2_identity": identity, "cos2_procrustes": proc, "cos2_random": rand}


def main():
    d = _PERTURB_DATA
    datasets = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
                "NadigOConner2024_hepg2.h5ad", "NadigOConner2024_jurkat.h5ad"]
    cache = {}
    for f in datasets:
        cache[f] = build_R(os.path.join(d, f))
        print("loaded", f, "n_perts", len(cache[f][3]), flush=True)

    rows = []
    for f in datasets:
        name, genes, tg, R = cache[f]
        r = loo_svd_cos2(R, KS, LOO_N)
        row = {"pair": name, "type": "within", "n_targets": len(R)}
        for k in KS:
            row["loo_svd_cos2_k%d" % k] = r[k]
        rows.append(row)
        print("within {:<28} LOO-SVD cos2: ".format(name[:27]) +
              " ".join("k%d=%.3f" % (k, r[k]) for k in KS), flush=True)

    pairs = [("ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
             ("NadigOConner2024_hepg2.h5ad", "NadigOConner2024_jurkat.h5ad"),
             ("NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
             ("NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
             ("NadigOConner2024_hepg2.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
             ("NadigOConner2024_hepg2.h5ad", "ReplogleWeissman2022_rpe1.h5ad")]
    for fa, fb in pairs:
        try:
            o = cross_predictions(*cache[fa], *cache[fb])
            o["pair"] = os.path.basename(fa).replace(".h5ad", "") + "->" + os.path.basename(fb).replace(".h5ad", "")
            o["type"] = "cross"
            rows.append(o)
            print("cross  {:<50} n={} identity={:.3f} procrustes={:.3f} random={:.3f}".format(
                o["pair"][:49], o["n_targets"], o["cos2_identity"], o["cos2_procrustes"],
                o["cos2_random"]), flush=True)
        except Exception as e:
            print("cross-skip", fa, str(e)[:50], flush=True)

    keys = ["type", "n_targets"] + ["loo_svd_cos2_k%d" % k for k in KS] + \
           ["n_common_genes", "cos2_identity", "cos2_procrustes", "cos2_random"]
    with open(OUT, "w") as fh:
        fh.write("pair," + ",".join(keys) + "\n")
        for r in rows:
            fh.write(r["pair"] + "," + ",".join(
                ("%.4f" % r[k]) if isinstance(r.get(k), float) else (str(r[k]) if k in r else "") for k in keys) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
