import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
spectrum_null.py -- Stage 1, E3: how many covariance eigenmodes are real?

For each dataset (raw), compare the control-covariance eigenvalue spectrum to a
per-gene permutation null (each gene shuffled independently across cells, which
preserves per-gene marginals but destroys gene-gene covariance).  A mode is called
"real" when its observed eigenvalue exceeds the 95th percentile of the null at the
same rank.  This separates genuine shared covariance structure from the
diagonal/depth-driven bulk.

Output: ../results/spectrum_null.csv
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
OUT = os.path.join(RESULTS, "spectrum_null.csv")

N_NULL = 8
SEED = 0
TOPK = 5


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    n_ctrl, p = Xc.shape
    Xc -= Xc.mean(0)
    cov = (Xc.T @ Xc) / (n_ctrl - 1)
    obs = np.linalg.eigvalsh(cov)[::-1]
    top1 = float(obs[0] / obs.sum())

    rs = np.random.default_rng(SEED)
    null = np.empty((N_NULL, p), dtype=np.float64)
    for b in range(N_NULL):
        Xp = Xc.copy()
        for j in range(p):
            Xp[:, j] = rs.permutation(Xp[:, j])
        cn = (Xp.T @ Xp) / (n_ctrl - 1)
        null[b] = np.linalg.eigvalsh(cn)[::-1]
        del Xp, cn
        gc.collect()
    null95 = np.percentile(null, 95, axis=0)
    null_max = float(null[:, 0].max())

    gt = obs > null95
    # contiguous "real" prefix (modes that clear the null at their rank)
    pref = 0
    while pref < p and gt[pref]:
        pref += 1
    n_above_nullmax = int(np.sum(obs > null_max))

    del X, cov, Xc
    gc.collect()
    r = {"dataset": ds, "n_ctrl": n_ctrl, "p": p, "top1_frac": top1,
         "n_sig_modes": int(gt.sum()), "pref_sig": int(pref),
         "n_above_nullmax": n_above_nullmax,
         "null_max": null_max}
    for k in range(TOPK):
        r["obs_%d" % (k + 1)] = float(obs[k])
        r["null95_%d" % (k + 1)] = float(null95[k])
    return r


def main():
    d = _PERTURB_DATA
    default = [
        "NormanWeissman2019_filtered.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_ctrl", "p", "top1_frac", "n_sig_modes", "pref_sig", "n_above_nullmax",
            "null_max"] + ["obs_%d" % (k + 1) for k in range(TOPK)] + \
           ["null95_%d" % (k + 1) for k in range(TOPK)]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK   {:<28} n_sig={:<5} pref={:<5} top1={:.2f} obs1={:.3g} null95_1={:.3g} "
                  "nullmax={:.3g}".format(
                      r["dataset"][:27], r["n_sig_modes"], r["pref_sig"], r["top1_frac"],
                      r["obs_1"], r["null95_1"], r["null_max"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
