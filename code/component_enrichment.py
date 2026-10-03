import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
component_enrichment.py -- biological identity of the leading residual (specific)
components, via Enrichr (KEGG/Reactome/GO).

For each dataset we SVD the (global-axis- and self-removed, centred) residual response
matrix and enrich the top-loading genes of each component.

Outputs: ../results/component_enrichment.csv
"""

import csv
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "component_enrichment.csv")
LIBS = ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"]
MAX_PERT, MAX_CELLS, NCOMP, TOPG = 150, 500, 5, 200


def analyze(path, gp):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    genes = np.asarray(genes)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
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
        if len(r) > MAX_CELLS:
            r = rng.choice(r, MAX_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        d = d - float(d @ v) * v
        d[t] = 0.0
        R.append(d)
    del X, cov
    gc.collect()
    R = np.asarray(R)
    if len(R) < 20:
        raise ValueError("too few")
    R = R - R.mean(0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    rows = []
    for c in range(min(NCOMP, Vt.shape[0])):
        vc = Vt[c]
        order = np.argsort(-np.abs(vc))[:TOPG]
        glist = [str(genes[j]) for j in order]
        try:
            enr = gp.enrichr(gene_list=glist, gene_sets=LIBS, organism="human", outdir=None)
            res = enr.results.sort_values("Adjusted P-value").head(6)
            terms = "; ".join("%s(%.1e)" % (t[:40], p) for t, p in
                              zip(res["Term"], res["Adjusted P-value"]))
        except Exception as e:
            terms = "enrich-fail: " + str(e)[:40]
        rows.append({"dataset": ds, "component": c + 1, "var_frac": float(S[c] ** 2 / (S ** 2).sum()),
                     "top_genes": ";".join(glist[:15]), "top_terms": terms})
        print("OK %-22s comp%d var=%.3f  %s" % (ds[:21], c + 1, rows[-1]["var_frac"], terms[:100]), flush=True)
    return rows


def main():
    import gseapy as gp
    d = _PERTURB_DATA
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "NormanWeissman2019_filtered.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["component", "var_frac", "top_genes", "top_terms"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset"] + keys)
        w.writeheader()
        for f in files:
            try:
                for r in analyze(os.path.join(d, f), gp):
                    w.writerow(r)
            except Exception as e:
                import traceback
                print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
                traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
