import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_network.py -- A3b: is the perturbation-specific residual organised by known
protein-protein (PPI) or genetic interactions?

For each dataset, build the global-axis- and self-removed residual response for the
perturbation targets, then ask whether targets that are connected in a physical (PPI) or
genetic interaction network have more similar specific residuals:
  * mean residual cosine among network-edge target pairs vs non-edge pairs (permutation p);
  * "average the residuals of my network neighbours" predictor vs random neighbours.

Networks: Human_protein_protein_interactions_collapsed.csv,
          suppl/Human_genetic_interactions_collapsed.csv.

Output: ../results/specific_network.csv
"""

import csv
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "specific_network.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
N_NULL = 200
SEED = 0


def load_network(path):
    edges = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            a, b = r.get("Interactor A", "").strip(), r.get("Interactor B", "").strip()
            if a and b and a != b:
                edges.add(frozenset((a, b)))
    return edges


def analyze(path, ppi, gen):
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
    tg = np.asarray(tg)
    del X, cov
    gc.collect()
    if n < 20:
        raise ValueError("too few")

    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    iu = np.triu_indices(n, 1)

    def evaluate(edges):
        adj = np.zeros((n, n), dtype=bool)
        tset = list(tg)
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((tset[i], tset[j])) in edges:
                    adj[i, j] = adj[j, i] = True
        nedge = int(adj[iu].sum())
        # neighbor-average prediction
        pred, true, rand_pred = [], [], []
        rs = np.random.default_rng(SEED)
        for i in range(n):
            nb = np.where(adj[i])[0]
            if nb.size == 0:
                continue
            p = R[nb].mean(0)
            pred.append(float(R[i] @ p) ** 2 / ((R[i] @ R[i]) * (p @ p) + 1e-12))
            j = int(rs.integers(n))
            pr = R[j]
            rand_pred.append(float(R[i] @ pr) ** 2 / ((R[i] @ R[i]) * (pr @ pr) + 1e-12))
            true.append(1.0)
        if nedge < 10:
            return None
        sim_e = S[iu][adj[iu]].mean()
        sim_n = S[iu][~adj[iu]].mean()
        # degree-controlled partial correlation between edge and residual similarity
        deg = adj.sum(1)
        dv = (deg[iu[0]] + deg[iu[1]]).astype(float)
        Areg = np.column_stack([dv, np.ones(dv.size)])
        def _resid(y):
            beta, *_ = np.linalg.lstsq(Areg, y, rcond=None)
            return y - Areg @ beta
        Sr = _resid(S[iu]); Er = _resid(adj[iu].astype(float))
        partial = float(np.corrcoef(Sr, Er)[0, 1]) if Sr.std() > 0 and Er.std() > 0 else float("nan")
        # permutation p-value
        null = []
        for _ in range(N_NULL):
            p = rs.permutation(n)
            A2 = adj[np.ix_(p, p)]
            if A2[iu].sum() >= 10:
                null.append(S[iu][A2[iu]].mean() - S[iu][~A2[iu]].mean())
        null = np.asarray(null)
        obs = sim_e - sim_n
        pval = float(np.mean(np.abs(null) >= abs(obs))) if null.size else float("nan")
        return {"n_edges": nedge, "sim_edge": sim_e, "sim_none": sim_n, "diff": obs,
                "p_perm": pval, "partial_edge_corr": partial,
                "cos2_neighbor": float(np.mean(pred)) if pred else float("nan"),
                "cos2_rand": float(np.mean(rand_pred)) if rand_pred else float("nan")}

    rows = []
    for name, edges in (("PPI", ppi), ("genetic", gen)):
        try:
            o = evaluate(edges)
            if o:
                o.update({"dataset": ds, "network": name, "n_targets": n})
                rows.append(o)
        except Exception as e:
            print("  sub-skip", ds, name, str(e)[:40], flush=True)
    return rows


def main():
    ppi = load_network(os.path.join(DATA, "Human_protein_protein_interactions_collapsed.csv"))
    gen = load_network(os.path.join(DATA, "suppl", "Human_genetic_interactions_collapsed.csv"))
    print("PPI edges", len(ppi), "genetic edges", len(gen), flush=True)
    d = DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
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
                    p = line.split(",")
                    done.add((p[0], p[1]))
    else:
        with open(OUT, "w") as fh:
            fh.write("dataset,network," + ",".join(keys) + "\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f), ppi, gen)
            for r in rows:
                if (r["dataset"], r["network"]) in done:
                    print("have", r["dataset"][:24], r["network"], "(skipped)", flush=True)
                    continue
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["network"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK {:<24} {:<8} edges={:<5} simE={:.3f} simN={:.3f} p={:.3f} "
                      "part={:.3f} cos2_nb={:.3f} rand={:.3f}".format(
                          r["dataset"][:23], r["network"], r["n_edges"], r["sim_edge"],
                          r["sim_none"], r["p_perm"], r["partial_edge_corr"],
                          r["cos2_neighbor"], r["cos2_rand"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
