import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_pathway.py -- A3c: is the specific residual organised by pathways / TF regulons?

Downloads gene sets from Enrichr (KEGG, Reactome, GO-BP, TRRUST TF regulons) and tests
whether perturbation-target genes that share a gene set have more similar specific
residuals (global-axis- and self-removed), mirroring specific_network.py:
  * mean residual cosine among co-annotated target pairs vs others (permutation p);
  * degree-controlled partial correlation;
  * "average the residuals of co-annotated targets" predictor vs random.

Output: ../results/specific_pathway.csv
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
OUT = os.path.join(RESULTS, "specific_pathway.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
N_NULL = 100
MAX_TERM_TARGETS = 80
SEED = 0

LIBS = ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023",
        "TRRUST_Transcription_Factors_2019"]


def fetch_libs():
    import gseapy
    out = {}
    for name in LIBS:
        try:
            lib = gseapy.get_library(name)
            out[name] = [set(v) for v in lib.values()]
            print("lib", name, len(lib), "terms", flush=True)
        except Exception as e:
            print("lib-fail", name, str(e)[:60], flush=True)
    return out


def adjacency_from_terms(terms, targets, tpos):
    n = len(targets)
    edges = set()
    for s in terms:
        idx = [i for i, g in enumerate(targets) if g in s]
        if len(idx) < 2 or len(idx) > MAX_TERM_TARGETS:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                edges.add((idx[a], idx[b]))
    return edges


def analyze(path, libs):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()
    R, tg = [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        d = d - float(d @ v) * v
        d[t] = 0.0
        R.append(d); tg.append(g)
    R = np.asarray(R)
    n = len(R)
    del X, cov
    gc.collect()
    if n < 20:
        raise ValueError("too few")
    tset = set(tg)
    tpos = {g: i for i, g in enumerate(tg)}
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    iu = np.triu_indices(n, 1)

    rows = []
    for name, terms in libs.items():
        edges = adjacency_from_terms(terms, tg, tpos)
        if len(edges) < 10:
            continue
        adj = np.zeros((n, n), dtype=bool)
        for a, b in edges:
            adj[a, b] = adj[b, a] = True
        sim_e = S[iu][adj[iu]].mean()
        sim_n = S[iu][~adj[iu]].mean()
        obs = sim_e - sim_n
        rs = np.random.default_rng(SEED)
        null = []
        for _ in range(N_NULL):
            p = rs.permutation(n)
            A2 = adj[np.ix_(p, p)]
            if A2[iu].sum() >= 10:
                null.append(S[iu][A2[iu]].mean() - S[iu][~A2[iu]].mean())
        null = np.asarray(null)
        pval = float(np.mean(np.abs(null) >= abs(obs))) if null.size else float("nan")
        deg = adj.sum(1)
        dv = (deg[iu[0]] + deg[iu[1]]).astype(float)
        Areg = np.column_stack([dv, np.ones(dv.size)])
        def _resid(y):
            beta, *_ = np.linalg.lstsq(Areg, y, rcond=None)
            return y - Areg @ beta
        Sr = _resid(S[iu]); Er = _resid(adj[iu].astype(float))
        partial = float(np.corrcoef(Sr, Er)[0, 1]) if Sr.std() > 0 and Er.std() > 0 else float("nan")
        pred, rpred = [], []
        for i in range(n):
            nb = np.where(adj[i])[0]
            if nb.size == 0:
                continue
            pr = R[nb].mean(0)
            pred.append(float(R[i] @ pr) ** 2 / ((R[i] @ R[i]) * (pr @ pr) + 1e-12))
            j = int(rs.integers(n))
            rpred.append(float(R[i] @ R[j]) ** 2 / ((R[i] @ R[i]) * (R[j] @ R[j]) + 1e-12))
        rows.append({"dataset": ds, "network": name, "n_targets": n,
                     "n_edges": int(adj[iu].sum()), "sim_edge": sim_e, "sim_none": sim_n,
                     "diff": obs, "p_perm": pval, "partial_edge_corr": partial,
                     "cos2_neighbor": float(np.mean(pred)) if pred else float("nan"),
                     "cos2_rand": float(np.mean(rpred)) if rpred else float("nan")})
    return rows


def main():
    libs = fetch_libs()
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "schemidt_etal_2022_crispra_perturbseq.h5ad",
        "proper_filtered.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_targets", "n_edges", "sim_edge", "sim_none", "diff", "p_perm",
            "partial_edge_corr", "cos2_neighbor", "cos2_rand"]
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    p = line.split(","); done.add((p[0], p[1]))
    else:
        with open(OUT, "w") as fh:
            fh.write("dataset,network," + ",".join(keys) + "\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f), libs)
            for r in rows:
                if (r["dataset"], r["network"]) in done:
                    print("have", r["dataset"][:20], r["network"], "(skip)", flush=True)
                    continue
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["network"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK {:<22} {:<32} edges={:<4} simE={:.3f} simN={:.3f} p={:.3f} part={:.3f}".format(
                    r["dataset"][:21], r["network"][:31], r["n_edges"], r["sim_edge"],
                    r["sim_none"], r["p_perm"], r["partial_edge_corr"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
