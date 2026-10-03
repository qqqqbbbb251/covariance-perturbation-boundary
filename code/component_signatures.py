import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
component_signatures.py -- A3: biological identity of the specific-structure components.

SVD of the (global-axis- and self-removed) residual response matrix, then signature
enrichment of the leading right-singular vectors using a panel of curated gene sets.

Outputs: ../results/component_signatures.csv, ../results/component_signatures_top.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "component_signatures.csv")
OUT_TOP = os.path.join(RESULTS, "component_signatures_top.csv")

SPACES = ["raw", "pearson"]
MAX_PERT = 150
MAX_PERT_CELLS = 500
NCOMP = 5
SEED = 0

SETS_NAMES = {
    "cellcycle": {"MKI67","TOP2A","CCNB1","CCNA2","CDK1","BUB1","BUB1B","AURKA","AURKB","PLK1",
                  "CENPA","CENPF","TPX2","NDC80","NUSAP1","ASPM","RRM2","TYMS","UBE2C","BIRC5",
                  "KIF11","KIF23","ANLN","ECT2"},
    "hsp": {"HSPA1A","HSPA1B","HSPA1L","DNAJB1","HSPB1","HSPA6","HSP90AA1","HSPH1","DNAJA1","HSPA8"},
    "interferon": {"ISG15","IFIT1","IFIT2","IFIT3","MX1","MX2","OAS1","OAS2","STAT1","IRF7","IFI6",
                   "IFI27","XAF1","OASL","IFIH1"},
    "apoptosis": {"BAX","BAK1","CASP3","CASP7","CASP8","FAS","PMAIP1","BBC3","TNFRSF10B","GADD45A"},
    "emt": {"VIM","FN1","CDH2","ZEB1","ZEB2","SNAI1","SNAI2","TWIST1","ACTA2","COL1A1","TAGLN"},
    "hypoxia": {"HIF1A","VEGFA","SLC2A1","LDHA","PGK1","PDK1","BNIP3","ADM","NDRG1","EGLN3"},
    "upr": {"DDIT3","ATF4","HSPA5","XBP1","ERN1","ATF3","DNAJB9","HERPUD1","SEC61A1"},
    "myc": {"MYC","NCL","NPM1","ODC1","EIF4E","SRM","LDHA","TK1"},
    "p53": {"TP53","CDKN1A","MDM2","GADD45A","BAX","BBC3","RRM2B","ZMAT3","TIGAR"},
    "translation": {"EEF1A1","EEF2","EIF4A1","EIF4E","EIF3A","RPLP0","EEF1B2","EIF3B","EIF4G1"},
}
SETS_PREFIX = {"ribo": ("RPS","RPL","MRPS","MRPL"), "mito": ("MT-",)}


def _enr(v, mask):
    if mask.sum() < 3:
        return float("nan")
    return float(np.mean(np.abs(v[mask])) / (np.mean(np.abs(v)) + 1e-12))


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    genes = np.asarray(genes)
    up = np.array([str(g).upper() for g in genes])
    masks = {}
    for name, s in SETS_NAMES.items():
        masks[name] = np.array([g in s for g in up])
    for name, pref in SETS_PREFIX.items():
        masks[name] = np.array([g.startswith(pref) for g in up])

    rows, tops = [], []
    for space in ("raw", "pearson"):
        from differential_identity import to_space
        ctrl_fn, pert_fn = to_space(X, cidx, nc_all, space)
        Xc = ctrl_fn(X[cidx], nc_all[cidx])
        cmean = Xc.mean(0)
        Z = Xc - cmean
        cov = (Z.T @ Z) / (Z.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        v = V[:, int(np.argmax(w))]
        del Xc, Z, V, w
        gc.collect()
        R = []
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
            R.append(d)
        R = np.asarray(R)
        if len(R) < 20:
            continue
        R = R - R.mean(0)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        for c in range(min(NCOMP, Vt.shape[0])):
            vc = Vt[c]
            row = {"dataset": ds, "space": space, "component": c + 1,
                   "var_frac": float((S[c] ** 2) / (S ** 2).sum())}
            for name, m in masks.items():
                row["enr_" + name] = _enr(vc, m)
            rows.append(row)
            order = np.argsort(-np.abs(vc))[:12]
            tops.append({"dataset": ds, "space": space, "component": c + 1,
                         "top_genes": ";".join(str(genes[j]) for j in order)})
        del cov, R
        gc.collect()
    del X
    gc.collect()
    return rows, tops


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["var_frac"] + ["enr_" + k for k in list(SETS_NAMES) + list(SETS_PREFIX)]
    with open(OUT, "w") as fh:
        fh.write("dataset,space,component," + ",".join(keys) + "\n")
    with open(OUT_TOP, "w") as fh:
        fh.write("dataset,space,component,top_genes\n")
    for f in files:
        try:
            rows, tops = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["space"] + "," + str(r["component"]) + "," +
                             ",".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            for t in tops:
                with open(OUT_TOP, "a") as fh:
                    fh.write("{dataset},{space},{component},{top_genes}\n".format(**t))
            print("OK", os.path.basename(f)[:28], len(rows), "rows", flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
