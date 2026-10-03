"""
positive_controls_extra.py -- sensitivity controls for the new tests

Part A (differential / identity): build a synthetic covariance whose columns are
v + alpha*u_g (u_g orthogonal to v).  At alpha=0 all columns are collinear and the
metrics must show "no specificity" (pc_pred ~ 1, RSA ~ 0, diff_pred ~ 0).  As alpha
grows the test must detect specificity (pc_pred falls, RSA rises, diff_pred rises).

Part B (inverse driver recovery): with the same synthetic covariance, the matched
filter must recover the true driver above chance as alpha grows; at alpha=0 it must
be at chance (AUC ~ 0.5).

Output: ../results/positive_controls_extra.csv
"""

import csv
import math
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "positive_controls_extra.csv")

N_GENES = 2000
N_PERT = 100
SIGMA = 0.8
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
N_REPS = 20


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na and nb else float("nan")


def rank_auc(scores, i):
    s = np.abs(scores); sp = s[i]
    neg = np.delete(s, i)
    neg = neg[np.isfinite(neg)]
    if neg.size == 0 or not np.isfinite(sp):
        return float("nan")
    return float((np.count_nonzero(neg < sp) + 0.5 * np.count_nonzero(neg == sp)) / neg.size)


def main():
    from scipy.stats import spearmanr
    rows = []
    rng = np.random.default_rng(0)
    for alpha in ALPHAS:
        pc_pred, rsa_l, diff_pred, aucs = [], [], [], []
        for _ in range(N_REPS):
            v = rng.standard_normal(N_GENES); v /= np.linalg.norm(v)
            U = rng.standard_normal((N_GENES, N_PERT))
            U -= np.outer(v, v @ U)
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            S = v[:, None] + alpha * U                    # columns
            noise = rng.standard_normal((N_GENES, N_PERT))
            noise /= np.linalg.norm(noise, axis=0, keepdims=True)
            D = S + SIGMA * noise                          # responses
            # CIPHER prediction
            cn = np.einsum("ij,ij->j", S, S)
            DH = S * ((np.einsum("ij,ij->j", S, D) / np.maximum(cn, 1e-12))[None, :])
            # pairwise |cos| of predictions
            C = np.abs(np.array([[cos(DH[:, i], DH[:, j]) for j in range(N_PERT)]
                                 for i in range(N_PERT)]))
            iu = np.triu_indices(N_PERT, 1)
            pc_pred.append(float(np.nanmean(C[iu])))
            # RSA
            Ct = 1 - np.array([[cos(D[:, i], D[:, j]) ** 2 for j in range(N_PERT)]
                               for i in range(N_PERT)])
            Cp = 1 - C ** 2
            rsa_l.append(float(spearmanr(Ct[iu], Cp[iu]).correlation))
            # differential prediction
            dpr = []
            for i in range(0, N_PERT, 5):
                for j in range(i + 1, min(i + 5, N_PERT)):
                    dd = D[:, i] - D[:, j]
                    dpr.append(cos(DH[:, i] - DH[:, j], dd) ** 2)
            diff_pred.append(float(np.nanmean(dpr)))
            # inverse AUC
            aucs.append(float(np.nanmean([rank_auc((S.T @ D[:, k]) / np.maximum(cn, 1e-12), k)
                                          for k in range(N_PERT)])))
        rows.append({"alpha": alpha, "pc_pred": float(np.mean(pc_pred)),
                     "rsa": float(np.mean(rsa_l)), "diff_pred": float(np.mean(diff_pred)),
                     "auc": float(np.mean(aucs))})
        print("alpha={:<4} pc_pred={:.3f} RSA={:+.3f} diff_pred={:.3f} inverse_AUC={:.3f}".format(
            alpha, rows[-1]["pc_pred"], rows[-1]["rsa"], rows[-1]["diff_pred"], rows[-1]["auc"]), flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "pc_pred", "rsa", "diff_pred", "auc"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.6g}".format(x) if isinstance(x, float) else x) for k, x in r.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
