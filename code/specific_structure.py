import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_structure.py -- Stage 2c: what *is* the reproducible specific structure,
and can any operator predict it?

For datasets with a reproducible perturbation-specific residual (Stage 2), we:
  1. build the residual response matrix R (perturbations x genes), removing the
     global axis v, the target's own coordinate, and the common off-axis mode;
  2. characterise it by SVD (is it low-rank or diffuse?) against a per-gene
     permutation null, and check simple signature enrichment of the top axes;
  3. test whether CIPHER's operators predict r_g at all: the covariance column
     Sigma[:,g] and the precision column Theta[:,g], vs random columns.

Outputs: ../results/specific_structure.csv, ../results/specific_structure_topgenes.csv
"""

import csv
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load, to_space  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "specific_structure.csv")
OUT_TOP = os.path.join(RESULTS, "specific_structure_topgenes.csv")

SPACES = ["raw", "pearson"]
MAX_PERT = 150
MAX_PERT_CELLS = 500
N_NULL = 10
SEED = 0

CELLCYCLE = {"MKI67", "TOP2A", "CCNB1", "CCNA2", "CDK1", "BUB1", "BUB1B", "AURKA", "AURKB",
             "PLK1", "CENPA", "CENPF", "TPX2", "NDC80", "NUSAP1", "ASPM", "RRM2", "TYMS",
             "UBE2C", "BIRC5", "KIF11", "KIF23", "ANLN", "ECT2"}
STRESS = {"HSPA1A", "HSPA1B", "HSPA1L", "DNAJB1", "HSPB1", "HSPA6", "JUN", "FOS", "FOSB",
          "JUNB", "EGR1", "ATF3", "DDIT3", "HSP90AA1", "HSPH1"}


def _enrich(v, genes, gene_set):
    mask = np.array([g in gene_set for g in genes])
    if mask.sum() < 3:
        return float("nan")
    return float(np.mean(np.abs(v[mask])) / (np.mean(np.abs(v)) + 1e-12))


def _prefix_enrich(v, genes, prefixes):
    mask = np.array([str(g).upper().startswith(prefixes) for g in genes])
    if mask.sum() < 3:
        return float("nan")
    return float(np.mean(np.abs(v[mask])) / (np.mean(np.abs(v)) + 1e-12))


def _proj_cos2(r, p):
    pp = float(p @ p)
    return float(r @ p) ** 2 / (float(r @ r) * pp) if pp > 0 and float(r @ r) > 0 else 0.0


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    genes = np.asarray(genes)
    rows_out, top_rows = [], []

    for space in SPACES:
        ctrl_fn, pert_fn = to_space(X, cidx, nc_all, space)
        Xc = ctrl_fn(X[cidx], nc_all[cidx])
        cmean = Xc.mean(0)
        Z = Xc - cmean
        cov = (Z.T @ Z) / (Z.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        i1 = int(np.argmax(w))
        v = V[:, i1]
        del Xc, Z
        gc.collect()
        eps = 1e-6 * float(np.median(w[w > 0])) if np.any(w > 0) else 1e-8
        # precision columns Theta[:,g] = V diag(1/(w+eps)) V^T e_g
        prec = (V * (1.0 / (w + eps))[None, :]) @ V.T      # (p, p)

        R = []
        tcols = []
        for g in singles[:MAX_PERT]:
            r = np.where(raw == g)[0]
            if len(r) < 2:
                continue
            if len(r) > MAX_PERT_CELLS:
                r = rng.choice(r, MAX_PERT_CELLS, replace=False)
            d = pert_fn(X[r], nc_all[r]).mean(0) - cmean
            if d @ d == 0:
                continue
            resid = d - float(d @ v) * v
            t = idx.get(g, -1)
            if 0 <= t < resid.size:
                resid[t] = 0.0
            R.append(resid); tcols.append(t)
        if len(R) < 20:
            continue
        R = np.asarray(R)
        Rc = R - R.mean(0)

        # SVD and per-gene permutation null
        s = np.linalg.svd(Rc, compute_uv=False)
        var = s ** 2
        svfrac = lambda k: float(var[:k].sum() / var.sum())
        nd = np.random.default_rng(SEED)
        null = np.empty((N_NULL, min(5, len(s))))
        for b in range(N_NULL):
            Rp = Rc.copy()
            for j in range(Rp.shape[1]):
                Rp[:, j] = nd.permutation(Rp[:, j])
            sn = np.linalg.svd(Rp, compute_uv=False)
            vn = sn ** 2
            for k in range(1, null.shape[1] + 1):
                null[b, k - 1] = vn[:k].sum() / vn.sum()
        null95 = np.percentile(null, 95, axis=0)

        # predictability of each residual by Sigma column vs precision column vs random
        rng2 = np.random.default_rng(SEED)
        cs_sig, cs_sig_rand, cs_prec, cs_prec_rand = [], [], [], []
        for i in range(len(R)):
            t = tcols[i]
            if not (0 <= t < cov.shape[0]):
                continue
            randc = int(rng2.integers(cov.shape[0]))
            cs_sig.append(_proj_cos2(R[i], cov[:, t]))
            cs_sig_rand.append(_proj_cos2(R[i], cov[:, randc]))
            cs_prec.append(_proj_cos2(R[i], prec[:, t]))
            cs_prec_rand.append(_proj_cos2(R[i], prec[:, randc]))

        # top gene loadings of the first right-singular vector
        U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
        v1 = Vt[0]
        order = np.argsort(-np.abs(v1))[:15]
        top_rows.append({"dataset": ds, "space": space,
                         "top_genes": ";".join(str(genes[j]) for j in order)})

        rows_out.append({
            "dataset": ds, "space": space, "n_perts": len(R), "n_genes": R.shape[1],
            "sv1_frac": svfrac(1), "sv2_frac": svfrac(2), "sv5_frac": svfrac(5),
            "sv1_null95": float(null95[0]), "sv2_null95": float(null95[1]),
            "sv5_null95": float(null95[4]),
            "cos2_sigma": float(np.nanmean(cs_sig)) if cs_sig else float("nan"),
            "cos2_sigma_rand": float(np.nanmean(cs_sig_rand)) if cs_sig_rand else float("nan"),
            "cos2_precision": float(np.nanmean(cs_prec)) if cs_prec else float("nan"),
            "cos2_precision_rand": float(np.nanmean(cs_prec_rand)) if cs_prec_rand else float("nan"),
            "enr_cellcycle": _enrich(v1, genes, CELLCYCLE),
            "enr_ribo": _prefix_enrich(v1, genes, ("RPS", "RPL")),
            "enr_mito": _prefix_enrich(v1, genes, ("MT-",)),
            "enr_stress": _enrich(v1, genes, STRESS),
        })
        del cov, V, w, prec, R, Rc
        gc.collect()
    del X
    gc.collect()
    return rows_out, top_rows


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "n_genes", "sv1_frac", "sv2_frac", "sv5_frac", "sv1_null95",
            "sv2_null95", "sv5_null95", "cos2_sigma", "cos2_sigma_rand", "cos2_precision",
            "cos2_precision_rand", "enr_cellcycle", "enr_ribo", "enr_mito", "enr_stress"]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(keys) + "\n")
    with open(OUT_TOP, "w") as fh:
        fh.write("dataset,space,top_genes\n")
    for f in files:
        try:
            rows, tops = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["space"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK {:<22} {:<8} sv1={:.3f}(null95 {:.3f}) sv5={:.3f} | "
                      "cos2 Sig={:.3f} rand={:.3f} Prec={:.3f} rand={:.3f} | "
                      "cc={:.2f} ribo={:.2f}".format(
                          r["dataset"][:21], r["space"], r["sv1_frac"], r["sv1_null95"],
                          r["sv5_frac"], r["cos2_sigma"], r["cos2_sigma_rand"],
                          r["cos2_precision"], r["cos2_precision_rand"],
                          r["enr_cellcycle"], r["enr_ribo"]), flush=True)
            for t in tops:
                with open(OUT_TOP, "a") as fh:
                    fh.write(t["dataset"] + "," + t["space"] + "," + t["top_genes"] + "\n")
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT, "and", OUT_TOP)


if __name__ == "__main__":
    main()
