import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
axis_origin.py -- Stage 1: what is the shared (global) axis, and is it an artifact?

The forward model's accuracy is dominated by a single covariance axis v (PC1),
which earlier work identified with the cell-size / MOR factor.  This experiment
dissects v and tests whether it is a technical scaling artifact.

E1 (nuisance decomposition): on control cells (raw), project each cell onto v and
    regress the PC1 score on log-total-counts, #genes, %mito, %ribo; also correlate
    the eigenvector v with the per-gene mean, sqrt(mean) and 1/mean (Poisson-scaling
    signatures).

E2 (equal-depth resampling): downsample every control cell to a common sequencing
    depth and recompute the covariance; compare the top-1 variance fraction and the
    fraction of each perturbation's shift on the global axis (shared_frac).  If the
    axis is a library-size artifact, both should collapse.

Output: ../results/axis_origin.csv
"""

import gc
import os
import sys

import numpy as np
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load, cos2  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "axis_origin.csv")

MAX_CTRL = 3000
DOWNSAMPLE_PCTL = 25.0
SEED = 0


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    A = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")

    a = ad.read_h5ad(path, backed="r")
    obs = a.obs

    def cov_col(name):
        if name in obs.columns:
            return np.asarray(obs[name].values[cidx], dtype=float)
        return None

    ngenes = cov_col("ngenes")
    mito = cov_col("percent_mito")
    ribo = cov_col("percent_ribo")
    a.file.close()

    logcounts = np.log10(nc_all[cidx].astype(float) + 1.0)

    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    genes = np.asarray(genes)
    n_ctrl = int(Xc.shape[0])
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    i1 = int(np.argmax(w))
    v = V[:, i1]
    top1 = float(w[i1] / w.sum())
    score = Z @ v                                  # PC1 score per control cell
    del V, w, Z
    gc.collect()

    # ---- E1: nuisance decomposition of the PC1 score ----
    feats = [logcounts]
    names = ["logcounts"]
    for nm, arr in (("ngenes", ngenes), ("mito", mito), ("ribo", ribo)):
        if arr is not None:
            feats.append(arr)
            names.append(nm)
    F = np.column_stack(feats)
    r_each = {nm: _pearson(score, f) for nm, f in zip(names, feats)}
    r2_all = _r2(F, score) if F.shape[1] else float("nan")

    gene_mean = Xc.mean(0)
    r_v_mean = _pearson(v, gene_mean)
    r_v_sqrt = _pearson(v, np.sqrt(np.maximum(gene_mean, 0)))
    r_v_inv = _pearson(v, 1.0 / (gene_mean + 1e-6))

    # ---- E2: equal-depth resampling of control cells ----
    tot = Xc.sum(1)
    D = int(np.floor(np.percentile(tot, DOWNSAMPLE_PCTL)))
    keep = tot >= max(D, 1)
    sub = Xc[keep]
    rs = np.random.default_rng(SEED)
    Xeq = np.zeros_like(sub)
    for i in range(sub.shape[0]):
        s = sub[i].sum()
        if s <= 0:
            continue
        Xeq[i] = rs.multinomial(D, sub[i] / s)
    cmean_eq = Xeq.mean(0)
    Zeq = Xeq - cmean_eq
    cov_eq = (Zeq.T @ Zeq) / max(Zeq.shape[0] - 1, 1)
    we, Ve = np.linalg.eigh(cov_eq)
    ie = int(np.argmax(we))
    v_eq = Ve[:, ie]
    top1_eq = float(we[ie] / we.sum())
    del Xeq, Zeq, cov_eq, Ve, we

    # shared_frac: fraction of each perturbation's raw shift on the global axis
    def shared(vv):
        vals = []
        for g in singles:
            rows = np.where(raw == g)[0]
            if len(rows) > 500:
                rows = rng.choice(rows, 500, replace=False)
            d = np.asarray(X[rows].todense(), dtype=np.float64).mean(0) - cmean
            if d @ d == 0:
                continue
            vals.append(cos2(d, vv))
        return float(np.nanmean(vals)) if vals else float("nan")

    shared_orig = shared(v)
    shared_eq = shared(v_eq)

    del X, cov, Xc, sub
    gc.collect()
    return {
        "dataset": ds, "n_ctrl": n_ctrl,
        "n_genes": int(len(genes)), "top1_frac": top1,
        "r_logcounts": r_each.get("logcounts", float("nan")),
        "r_ngenes": r_each.get("ngenes", float("nan")),
        "r_mito": r_each.get("mito", float("nan")),
        "r_ribo": r_each.get("ribo", float("nan")),
        "R2_nuisance": r2_all,
        "r_v_geneMean": r_v_mean, "r_v_sqrtMean": r_v_sqrt, "r_v_invMean": r_v_inv,
        "depth_D": D, "n_cells_eq": int(keep.sum()),
        "top1_frac_eq": top1_eq, "shared_frac": shared_orig, "shared_frac_eq": shared_eq,
    }


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
    keys = ["n_ctrl", "n_genes", "top1_frac", "r_logcounts", "r_ngenes", "r_mito",
            "r_ribo", "R2_nuisance", "r_v_geneMean", "r_v_sqrtMean", "r_v_invMean",
            "depth_D", "n_cells_eq", "top1_frac_eq", "shared_frac", "shared_frac_eq"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK   {:<28} top1={:.3f}->{:.3f}  shared={:.3f}->{:.3f}  R2nui={:.2f} "
                  "r_log={:.2f} r_geneMean={:.2f}".format(
                      r["dataset"][:27], r["top1_frac"], r["top1_frac_eq"],
                      r["shared_frac"], r["shared_frac_eq"], r["R2_nuisance"],
                      r["r_logcounts"], r["r_v_geneMean"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
