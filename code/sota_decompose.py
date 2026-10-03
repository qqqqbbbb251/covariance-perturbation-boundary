"""
sota_decompose.py -- put model predictions under the program/amplitude/direction
decomposition (forward accuracy, RSA, prediction collinearity, shared vs specific
residual agreement).

Input: one or more benchmark ``results.pkl`` files (dict with DELTA_X/DELTA_Y per
perturbation, as written by the CIPHER benchmark drivers).  Gene order is taken from a
reference ``.h5ad`` (var_names) to zero the target's own coordinate.

Usage:
    python sota_decompose.py <ref.h5ad> <name>:<results.pkl> [<name2>:<pkl2> ...]
Output: ../results/sota_decompose.csv
"""

import csv
import os
import pickle
import sys

import numpy as np
import anndata as ad
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "sota_decompose.csv")


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b) / (na * nb) if na and nb else np.nan


def cos2(a, b):
    c = cos(a, b)
    return c * c if np.isfinite(c) else np.nan


def main():
    ref = ad.read_h5ad(sys.argv[1], backed="r")
    genes = np.asarray(ad.read_h5ad(sys.argv[1], backed="r").var_names).astype(str)
    ref.file.close()
    gpos = {g: i for i, g in enumerate(genes)}

    entries = []
    for a in sys.argv[2:]:
        name, path = a.split(":", 1)
        with open(path, "rb") as f:
            d = pickle.load(f)
        keys = sorted([k for k in d["DELTA_X"] if not any(s in k for s in ["+", "|", "_"])])
        # field convention differs by driver: truth is DELTA_X; the prediction is
        # RAW_PREDICTION when present (CIPHER) else DELTA_Y (linear_mean).
        tf = "DELTA_X"
        pf = "RAW_PREDICTION" if "RAW_PREDICTION" in d else "DELTA_Y"
        print("   %s fields truth=%s pred=%s" % (name, tf, pf), flush=True)
        Dt = np.asarray([d[tf][k] for k in keys], dtype=np.float64)
        Dp = np.asarray([d[pf][k] for k in keys], dtype=np.float64)
        entries.append((name, keys, Dt, Dp))

    # common perturbations
    common = set(entries[0][1])
    for _, k, _, _ in entries[1:]:
        common &= set(k)
    common = sorted(common)

    rows = []
    v = None
    for name, keys, Dt, Dp in entries:
        idx = [keys.index(k) for k in common]
        T = Dt[idx]; P = Dp[idx]
        n = len(common)
        # forward accuracy
        fwd = float(np.nanmean([cos2(T[i], P[i]) for i in range(n)]))
        # self-removed residuals (global axis from TRUE shifts, computed once)
        if v is None:
            M = T - T.mean(0)
            _, _, Vt = np.linalg.svd(M, full_matrices=False)
            v = Vt[0]
        Rt = T - np.outer(T @ v, v); Rp = P - np.outer(P @ v, v)
        # zero target gene coordinate
        for i, k in enumerate(common):
            t = gpos.get(k, -1)
            if t >= 0:
                Rt[i, t] = 0.0; Rp[i, t] = 0.0
        # RSA of true vs predicted structure (pairwise 1-cos^2)
        def rsa(A, B):
            n = A.shape[0]
            DA = np.array([[1 - cos2(A[i], A[j]) for j in range(n)] for i in range(n)])
            DB = np.array([[1 - cos2(B[i], B[j]) for j in range(n)] for i in range(n)])
            iu = np.triu_indices(n, 1)
            a, b = DA[iu], DB[iu]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() <= 10 or a[m].std() == 0 or b[m].std() == 0:
                return float("nan")
            return float(spearmanr(a[m], b[m]).correlation)
        rsa_full = rsa(T, P)
        rsa_spec = rsa(Rt, Rp)
        # prediction collinearity
        def pc(A):
            vals = [abs(cos(A[i], A[j])) for i in range(n) for j in range(i + 1, n)]
            return float(np.nanmean(vals))
        # shared fraction of true / predicted
        rows.append({
            "model": name, "n_perts": n,
            "fwd_cos2": fwd,
            "rsa_full": rsa_full,
            "specific_cos2": float(np.nanmean([cos2(Rt[i], Rp[i]) for i in range(n)])),
            "rsa_specific": rsa_spec,
            "pc_true": pc(T), "pc_pred": pc(P),
            "shared_frac_true": float(np.nanmean([cos2(T[i], v) for i in range(n)])),
            "shared_frac_pred": float(np.nanmean([cos2(P[i], v) for i in range(n)])),
        })
        print("OK %-14s n=%d fwd_cos2=%.3f RSA_full=%.3f specific_cos2=%.3f RSA_spec=%.3f "
              "pc_true=%.3f pc_pred=%.3f" % (name, n, fwd, rsa_full, rows[-1]["specific_cos2"],
                                             rsa_spec, rows[-1]["pc_true"], rows[-1]["pc_pred"]), flush=True)

    keys = ["model", "n_perts", "fwd_cos2", "rsa_full", "specific_cos2", "rsa_specific",
            "pc_true", "pc_pred", "shared_frac_true", "shared_frac_pred"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
