import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
transform_estimator_scan.py -- C8: can a better data space / covariance estimator let
the operator see the perturbation-specific residual?

For each dataset we rebuild the control covariance under several transforms
(raw / cpm / log1p / CLR / NB-Pearson) and ask whether the operator (covariance column
Sigma[:,g] or precision column Theta[:,g], each with the global-axis component removed)
predicts the target's specific residual any better than a random column.

If any transform makes the operator non-random, that is a lead; if none does, the
operator's blindness is transform-independent.

Output: ../results/transform_estimator_scan.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load, to_space  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "transform_estimator_scan.csv")

SPACES = ["raw", "cpm", "log1p", "clr", "pearson"]
MAX_PERT = 120
MAX_PERT_CELLS = 500
SEED = 0


def _dense(M):
    return np.asarray(M.todense(), dtype=np.float64)


def space_fns(X, cidx, nc_all, space):
    if space in ("raw", "cpm", "pearson"):
        return to_space(X, cidx, nc_all, space)
    if space == "log1p":
        def f(M, nc):
            return np.log1p(_dense(M))
        return f, f
    if space == "clr":
        pc = 1.0
        def f(M, nc):
            E = _dense(M) + pc
            L = np.log(E)
            return L - L.mean(1, keepdims=True)
        return f, f
    raise ValueError(space)


def _proj_cos2(r, p):
    pp = float(p @ p); rr = float(r @ r)
    return float(r @ p) ** 2 / (rr * pp) if pp > 0 and rr > 0 else 0.0


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    rows = []
    for space in SPACES:
        try:
            ctrl_fn, pert_fn = space_fns(X, cidx, nc_all, space)
            Xc = ctrl_fn(X[cidx], nc_all[cidx])
            cmean = Xc.mean(0)
            Z = Xc - cmean
            cov = (Z.T @ Z) / (Z.shape[0] - 1)
            w, V = np.linalg.eigh(cov)
            v = V[:, int(np.argmax(w))]
            eps = 1e-6 * float(np.median(w[w > 0])) if np.any(w > 0) else 1e-8
            prec = (V * (1.0 / (w + eps))[None, :]) @ V.T
            del Xc, Z, V, w
            gc.collect()

            R, tc = [], []
            for g in singles[:MAX_PERT]:
                r = np.where(raw == g)[0]
                t = idx.get(g, -1)
                if not (0 <= t < cov.shape[0]) or len(r) < 2:
                    continue
                if len(r) > MAX_PERT_CELLS:
                    r = rng.choice(r, MAX_PERT_CELLS, replace=False)
                d = pert_fn(X[r], nc_all[r]).mean(0) - cmean
                if d @ d == 0:
                    continue
                d = d - float(d @ v) * v
                d[t] = 0.0
                R.append(d); tc.append(t)
            R = np.asarray(R); tc = np.asarray(tc)
            if len(R) < 20:
                continue
            R = R - R.mean(0)
            s = np.linalg.svd(R, compute_uv=False)
            sv1 = float((s[:1] ** 2).sum() / (s ** 2).sum())

            rng2 = np.random.default_rng(SEED)
            cs_sig, cs_sig_r, cs_prec, cs_prec_r = [], [], [], []
            for i in range(len(R)):
                t = tc[i]
                sig = cov[:, t] - float(cov[:, t] @ v) * v
                prc = prec[:, t] - float(prec[:, t] @ v) * v
                rc = int(rng2.integers(cov.shape[0]))
                sigr = cov[:, rc] - float(cov[:, rc] @ v) * v
                prcr = prec[:, rc] - float(prec[:, rc] @ v) * v
                cs_sig.append(_proj_cos2(R[i], sig))
                cs_prec.append(_proj_cos2(R[i], prc))
                cs_sig_r.append(_proj_cos2(R[i], sigr))
                cs_prec_r.append(_proj_cos2(R[i], prcr))

            rows.append({"dataset": ds, "space": space, "n_perts": len(R), "sv1_frac": sv1,
                         "cos2_sigma_off": float(np.nanmean(cs_sig)),
                         "cos2_sigma_off_rand": float(np.nanmean(cs_sig_r)),
                         "cos2_prec_off": float(np.nanmean(cs_prec)),
                         "cos2_prec_off_rand": float(np.nanmean(cs_prec_r))})
            del cov, prec, R
            gc.collect()
        except Exception as e:
            print("   sub-skip", ds, space, str(e)[:50], flush=True)
        gc.collect()
    del X
    gc.collect()
    return rows


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_jurkat.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "sv1_frac", "cos2_sigma_off", "cos2_sigma_off_rand",
            "cos2_prec_off", "cos2_prec_off_rand"]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(keys) + "\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["space"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK {:<22} {:<8} sv1={:.3f} SigOff={:.3f}(r {:.3f}) PrecOff={:.3f}(r {:.3f})".format(
                    r["dataset"][:21], r["space"], r["sv1_frac"], r["cos2_sigma_off"],
                    r["cos2_sigma_off_rand"], r["cos2_prec_off"], r["cos2_prec_off_rand"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
