import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
timecourse.py -- does the perturbation-specific structure evolve over time?

Marson2025 D1 CRISPRi T-cell stimulation time-course (Rest / 8 h / 48 h).  For each
timepoint we build the (global-axis- and self-removed) residual response for the
perturbation targets, then:
  * compare the specific residual across timepoints for the SAME target (cosine vs null);
  * report the off-axis fraction of the response at each timepoint.

Output: ../results/timecourse.csv
"""

import csv
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import differential_identity as di  # noqa: E402
di.MAX_PERT = 5000                      # keep all single perturbations for matching
from differential_identity import load  # noqa: E402

DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "timecourse.csv")
MAX_CELLS = 300
SEED = 0


def resid(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()
    R, off = {}, {}
    for g in singles:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_CELLS:
            r = rng.choice(r, MAX_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        off[g] = 1.0 - float(d @ v) ** 2 / float(d @ d)
        r = d - float(d @ v) * v
        r[t] = 0.0
        R[g] = r
    del X, cov
    gc.collect()
    return np.asarray(genes), R, off


def main():
    # args: [donor file0 file1 file2] (one group), else default D1
    if len(sys.argv) >= 5:
        donor = sys.argv[1]; fs = sys.argv[2:5]
    else:
        donor = "D1"
        fs = ["Marson2025_D1_Rest_filtered.h5ad", "Marson2025_D1_Stim8hr_filtered.h5ad",
              "Marson2025_D1_Stim48hr_filtered.h5ad"]
    labels = {"t0": fs[0], "t1": fs[1], "t2": fs[2]}
    tnames = {"D1": ("Rest", "8h", "48h"), "D2": ("Rest", "8h", "48h"),
              "D3": ("Rest", "8h", "48h"), "D4": ("Rest", "8h", "48h")}
    nm = tnames.get(donor, ("t0", "t1", "t2"))
    data = {}
    for k, f in labels.items():
        if not os.path.exists(os.path.join(DATA, f)):
            print("missing", f); return
        g, R, off = resid(os.path.join(DATA, f))
        data[k] = (g, R, off)
        print("load %-4s %-4s targets=%d offaxis=%.3f" % (donor, k, len(R), np.mean(list(off.values()))), flush=True)
    common_t = sorted(set(data["t0"][1]) & set(data["t1"][1]) & set(data["t2"][1]))
    common_g = sorted(set(data["t0"][0]) & set(data["t1"][0]) & set(data["t2"][0]))
    print("common targets", len(common_t), flush=True)

    def vec(k, target):
        g, R, off = data[k]
        gi = {gg: i for i, gg in enumerate(g)}
        return np.asarray([R[target][gi[c]] for c in common_g])

    rows = []
    rng = np.random.default_rng(SEED)
    for a, b, na, nb in [("t0", "t1", nm[0], nm[1]), ("t1", "t2", nm[1], nm[2]),
                         ("t0", "t2", nm[0], nm[2])]:
        A = np.asarray([vec(a, t) for t in common_t]); B = np.asarray([vec(b, t) for t in common_t])
        Ac = A - A.mean(0); Bc = B - B.mean(0)
        num = np.sum(Ac * Bc, axis=1); den = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
        cos = float(np.nanmean(num / np.maximum(den, 1e-12)))
        perm = rng.permutation(len(common_t))
        numn = np.sum(Ac * Bc[perm], axis=1); denn = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc[perm], axis=1)
        cosn = float(np.nanmean(numn / np.maximum(denn, 1e-12)))
        rows.append({"donor": donor, "pair": na + "->" + nb, "n_targets": len(common_t),
                     "cos_specific": cos, "cos_null": cosn,
                     "offaxis_a": float(np.mean([data[a][2].get(t, np.nan) for t in common_t])),
                     "offaxis_b": float(np.mean([data[b][2].get(t, np.nan) for t in common_t]))})
        print("OK %-3s %-12s cos=%.3f (null %.3f)" % (donor, na + "->" + nb, cos, cosn), flush=True)

    keys = ["donor", "n_targets", "cos_specific", "cos_null", "offaxis_a", "offaxis_b"]
    exists = os.path.exists(OUT) and os.path.getsize(OUT) > 0
    done = set()
    if exists:
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    p = line.split(","); done.add((p[0], p[1]))
    with open(OUT, "a" if exists else "w") as fh:
        if not exists:
            fh.write("pair," + ",".join(keys) + "\n")
        for r in rows:
            if (r["donor"], r["pair"]) in done:
                continue
            fh.write(r["pair"] + "," + ",".join(
                str(r[k]) if k == "donor" else (("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]))
                for k in keys) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
