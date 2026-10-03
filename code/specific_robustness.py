import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_robustness.py -- Stage 2c robustness: is the reproducible specific structure
broad, or driven by a few high-expression/high-variance genes?

For K562 / rpe1 / Norman (raw): build the split-half residual responses (global axis
and self removed), then compare the centered RSA reliability under
    full        : all genes
    zscore      : each gene standardised across perturbations (correlation, not cosine)
    no_mitoRibo : drop MT-*, MALAT1, RPS*/RPL* genes
    no_topvar   : drop the 20 highest cross-perturbation-variance genes
and test whether the covariance / precision columns predict each residual *after*
removing their global-axis component (off-axis operator test).

Output: ../results/specific_robustness.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "specific_robustness.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
SEED = 0
DROP_PREFIX = ("MT-", "RPS", "RPL", "MRPS", "MRPL")
DROP_NAMES = {"MALAT1", "NEAT1"}


def _sim_upper(R):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    iu = np.triu_indices(R.shape[0], 1)
    return S[iu]


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _proj_cos2(r, p):
    pp = float(p @ p); rr = float(r @ r)
    return float(r @ p) ** 2 / (rr * pp) if pp > 0 and rr > 0 else 0.0


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    genes = np.asarray(genes)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    eps = 1e-6 * float(np.median(w[w > 0])) if np.any(w > 0) else 1e-8
    prec = (V * (1.0 / (w + eps))[None, :]) @ V.T
    del Xc, Z, V, w
    gc.collect()

    D1, D2, tcols = [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        if len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        r = rng.permutation(r)
        h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
        d1 = np.asarray(X[h1].todense(), dtype=np.float64).mean(0) - cmean
        d2 = np.asarray(X[h2].todense(), dtype=np.float64).mean(0) - cmean
        if d1 @ d1 == 0 or d2 @ d2 == 0:
            continue
        t = idx.get(g, -1)
        for d in (d1, d2):
            d -= float(d @ v) * v
            if 0 <= t < d.size:
                d[t] = 0.0
        D1.append(d1); D2.append(d2); tcols.append(t)
    D1 = np.asarray(D1); D2 = np.asarray(D2)
    tcols = np.asarray(tcols)

    # off-axis operator predictability (residualize the predictor by v)
    rng2 = np.random.default_rng(SEED)
    cs_sig_off, cs_sig_off_rand = [], []
    for i in range(len(D1)):
        t = tcols[i]
        if not (0 <= t < cov.shape[0]):
            continue
        sig = cov[:, t] - float(cov[:, t] @ v) * v
        rc = int(rng2.integers(cov.shape[0]))
        sigr = cov[:, rc] - float(cov[:, rc] @ v) * v
        cs_sig_off.append(_proj_cos2(D1[i], sig))
        cs_sig_off_rand.append(_proj_cos2(D1[i], sigr))

    # centered residual similarity reliability under different gene treatments
    def rel(cols):
        A = D1[:, cols] - D1[:, cols].mean(0)
        B = D2[:, cols] - D2[:, cols].mean(0)
        return _veccorr(_sim_upper(A), _sim_upper(B))

    allc = np.arange(len(genes))
    drop = np.array([str(g).upper().startswith(DROP_PREFIX) or g in DROP_NAMES for g in genes])
    keep_nomr = allc[~drop]
    sd = D1.std(0) + 1e-12
    z1 = D1 - D1.mean(0); z2 = D2 - D2.mean(0)
    z1 = z1 / sd; z2 = z2 / sd
    relz = _veccorr(_sim_upper(z1 - z1.mean(0)), _sim_upper(z2 - z2.mean(0)))
    # drop 20 highest cross-perturbation-variance genes
    topvar = np.argsort(-D1.var(0))[:20]
    keep_topvar = np.setdiff1d(allc, topvar)

    out = {
        "dataset": ds, "n_perts": len(D1), "n_genes": len(genes),
        "rel_full": rel(allc),
        "rel_zscore": relz,
        "rel_no_mitoRibo": rel(keep_nomr),
        "rel_no_topvar": rel(keep_topvar),
        "cos2_sigma_offaxis": float(np.nanmean(cs_sig_off)),
        "cos2_sigma_offaxis_rand": float(np.nanmean(cs_sig_off_rand)),
    }
    del X, cov, prec, D1, D2
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "n_genes", "rel_full", "rel_zscore", "rel_no_mitoRibo",
            "rel_no_topvar", "cos2_sigma_offaxis", "cos2_sigma_offaxis_rand"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<28} full={:.3f} z={:.3f} noMR={:.3f} noTopVar={:.3f} | "
                  "SigOff={:.3f} rand={:.3f}".format(
                      r["dataset"][:27], r["rel_full"], r["rel_zscore"], r["rel_no_mitoRibo"],
                      r["rel_no_topvar"], r["cos2_sigma_offaxis"],
                      r["cos2_sigma_offaxis_rand"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
