import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
structure_meta.py -- meta-analysis: why do some datasets have reproducible specific
structure and others not?

Per dataset: structure strength (centred, self-removed split-half RSA reliability, raw),
plus candidate predictors computed on the data:
  n_cells / n_targets (from cross_lab_inventory.csv),
  median cells per perturbation,
  mean |ΔX|, mean off-axis fraction (1 - cos^2(ΔX, v)), mean self z-score.
Then Spearman-correlate strength with each predictor across datasets.

Output: ../results/structure_meta.csv
"""

import csv
import gc
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "structure_meta.csv")
MAX_PERT = 150
MAX_PERT_CELLS = 500


def read(name):
    p = os.path.join(RESULTS, name)
    with open(p) as fh:
        return list(csv.DictReader(fh))


def effect_stats(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()
    norms, offs, selfz, cells = [], [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        cells.append(len(r))
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        norms.append(float(np.sqrt(d @ d)))
        offs.append(1.0 - float(d @ v) ** 2 / (d @ d))
        selfz.append(abs(d[t]) / (np.sqrt(cov[t, t] / max(len(r), 1)) + 1e-9))
    del X, cov
    gc.collect()
    return {"n_ctrl": int(len(cidx)), "n_perts": len(singles),
            "median_cells_per_pert": float(np.median(cells)) if cells else float("nan"),
            "mean_dx_norm": float(np.mean(norms)) if norms else float("nan"),
            "mean_offaxis_frac": float(np.mean(offs)) if offs else float("nan"),
            "mean_self_z": float(np.mean(selfz)) if selfz else float("nan")}


def main():
    rr = {r["dataset"]: float(r["rsa_selfrem_spec"]) for r in read("residual_reproducibility.csv")
          if r["space"] == "raw"}
    inv = {r["dataset"]: r for r in read("cross_lab_inventory.csv")}
    files = sys.argv[1:] if len(sys.argv) > 1 else list(rr.keys())
    rows = []
    for ds in files:
        f = ds + ".h5ad"
        if not os.path.exists(os.path.join(DATA, f)):
            print("missing", f, flush=True); continue
        try:
            e = effect_stats(os.path.join(DATA, f))
        except Exception as ex:
            print("skip", ds, str(ex)[:50], flush=True); continue
        iv = inv.get(ds, {})
        e.update({"dataset": ds, "rsa_selfrem_spec": rr.get(ds, float("nan")),
                  "n_cells": int(iv.get("n_cells", 0) or 0),
                  "n_targets_annot": int(iv.get("n_targets", 0) or 0),
                  "cell_line": iv.get("cell_line", ""), "modality": iv.get("modality", "")})
        rows.append(e)
        print("OK {:<26} RSA={:.3f} n_ctrl={:<6} medCells={:<6.0f} dx={:.2f} offaxis={:.2f}".format(
            ds[:25], e["rsa_selfrem_spec"], e["n_ctrl"], e["median_cells_per_pert"],
            e["mean_dx_norm"], e["mean_offaxis_frac"]), flush=True)

    cols = ["dataset", "rsa_selfrem_spec", "n_cells", "n_targets_annot", "n_ctrl", "n_perts",
            "median_cells_per_pert", "mean_dx_norm", "mean_offaxis_frac", "mean_self_z",
            "cell_line", "modality"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    y = np.array([r["rsa_selfrem_spec"] for r in rows if np.isfinite(r["rsa_selfrem_spec"])])
    print("\nSpearman(structure, predictor) over %d datasets:" % len(y))
    for pred in ["n_cells", "n_targets_annot", "n_ctrl", "n_perts", "median_cells_per_pert",
                 "mean_dx_norm", "mean_offaxis_frac", "mean_self_z"]:
        x = np.array([r[pred] for r in rows if np.isfinite(r["rsa_selfrem_spec"])], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 5 and np.std(x[m]) > 0:
            rho, p = spearmanr(x[m], y[m])
            print("  {:<24} rho={:+.3f} p={:.3f}  (n={})".format(pred, rho, p, int(m.sum())))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
