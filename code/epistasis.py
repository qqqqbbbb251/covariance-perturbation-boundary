import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
epistasis.py -- combination-perturbation interaction under the global/specific split.

For combinatorial screens (Norman et al. CRISPRa K562), decompose each double A+B:
    I = d_AB - (d_A + d_B)
and ask how much of the response is additive, how much of the interaction is on the
global (depth) axis vs off-axis (specific), and whether interactions are stronger for
PPI-connected gene pairs.

Output: ../results/epistasis.csv
"""

import csv
import gc
import os
import re
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "epistasis.csv")

N_TOP, MAX_CTRL, MAX_CELLS, MIN_CELLS = 1500, 3000, 500, 30
CONTROL_TOKENS = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")


def is_control(l):
    l = str(l).strip().lower()
    return l in ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg") \
        or any(t in l for t in CONTROL_TOKENS)


def load_ppi():
    edges = set()
    p = os.path.join(DATA, "Human_protein_protein_interactions_collapsed.csv")
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            a, b = r.get("Interactor A", "").strip(), r.get("Interactor B", "").strip()
            if a and b and a != b:
                edges.add(frozenset((a, b)))
    return edges


def analyze(path, ppi):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    labels = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values if "nperts" in obs.columns else np.ones(len(labels), dtype=int)
    var_names = np.asarray(a.var_names).astype(str)
    geneset = set(var_names.tolist())
    vn = None
    for c in ("ncounts", "total_counts", "mean_counts", "means"):
        if c in a.var.columns:
            vn = np.asarray(a.var[c].values, dtype=float); break
    if vn is None:
        vn = np.ones(len(var_names))
    ctrl = np.array([is_control(l) for l in labels]) | (nperts == 0)
    sel = np.sort(np.argsort(-np.nan_to_num(vn))[:N_TOP])
    # ensure every perturbation-target gene is in the panel
    nameset = set(var_names.tolist())
    toks = set()
    for l in np.unique(labels[~ctrl]):
        for p in re.split(r"[+|_]", str(l)):
            if p in nameset:
                toks.add(p)
    pos_all = {g: i for i, g in enumerate(var_names)}
    extra = np.array([pos_all[g] for g in toks], dtype=int)
    sel = np.sort(np.unique(np.concatenate([sel, extra]))) if extra.size else sel
    genes = var_names[sel]
    gpos = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    Xm = a[:, sel].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    cmean = np.asarray(X[cidx].todense(), dtype=np.float64).mean(0)
    Z = np.asarray(X[cidx].todense(), dtype=np.float64) - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Z, V, w
    gc.collect()

    vc = labels[~ctrl]
    uq, cn = np.unique(vc, return_counts=True)
    labels_ok = [l for l, c in zip(uq, cn) if c >= MIN_CELLS]
    singles = {}
    doubles = {}
    for l in labels_ok:
        parts = [p for p in re.split(r"[+|_]", l) if p in gpos]
        rows = np.where(labels == l)[0]
        if len(rows) > MAX_CELLS:
            rows = rng.choice(rows, MAX_CELLS, replace=False)
        d = np.asarray(X[rows].todense(), dtype=np.float64).mean(0) - cmean
        if len(parts) == 1 and l in gpos:
            singles[l] = d
        elif len(parts) == 2:
            doubles[tuple(sorted(parts))] = d
    del X, cov
    gc.collect()

    add_cos2, I_norm, Ioff_norm, dnorm, gfrac = [], [], [], [], []
    edge_add, non_add, edge_I, non_I = [], [], [], []
    for (A, B), dAB in doubles.items():
        if A not in singles or B not in singles:
            continue
        add = singles[A] + singles[B]
        I = dAB - add
        na = float(dAB @ dAB); nn = float(add @ add); ni = float(I @ I)
        if na == 0:
            continue
        c2 = float(dAB @ add) ** 2 / (na * nn) if nn > 0 else 0.0
        Io = I - float(I @ v) * v
        add_cos2.append(c2)
        I_norm.append(np.sqrt(ni)); dnorm.append(np.sqrt(na))
        Ioff_norm.append(float(np.sqrt(Io @ Io)))
        gfrac.append(float(dAB @ v) ** 2 / na)
        if frozenset((A, B)) in ppi:
            edge_add.append(c2); edge_I.append(np.sqrt(ni))
        else:
            non_add.append(c2); non_I.append(np.sqrt(ni))
    if len(add_cos2) < 10:
        raise ValueError("too few doubles")
    out = {"dataset": os.path.basename(path).replace(".h5ad", ""), "n_doubles": len(add_cos2),
           "mean_additive_cos2": float(np.mean(add_cos2)),
           "mean_I_over_dAB": float(np.mean(np.array(I_norm) / (np.array(dnorm) + 1e-12))),
           "mean_Ioff_over_I": float(np.mean(np.array(Ioff_norm) / (np.array(I_norm) + 1e-12))),
           "mean_global_frac_dAB": float(np.mean(gfrac)),
           "n_ppi_edges": len(edge_add),
           "additive_cos2_ppi": float(np.mean(edge_add)) if edge_add else float("nan"),
           "additive_cos2_nonppi": float(np.mean(non_add)) if non_add else float("nan"),
           "Inorm_ppi": float(np.mean(edge_I)) if edge_I else float("nan"),
           "Inorm_nonppi": float(np.mean(non_I)) if non_I else float("nan")}
    return out


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "NormanWeissman2019_filtered.h5ad",
        "JoungZhang2023_combinatorial.h5ad",
    ]
    ppi = load_ppi()
    keys = ["n_doubles", "mean_additive_cos2", "mean_I_over_dAB", "mean_Ioff_over_I",
            "mean_global_frac_dAB", "n_ppi_edges", "additive_cos2_ppi", "additive_cos2_nonppi",
            "Inorm_ppi", "Inorm_nonppi"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(DATA, f), ppi)
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<28} n={} additive_cos2={:.3f} |I|/|dAB|={:.3f} Ioff/I={:.3f} "
                  "PPI-add={:.3f} nonPPI-add={:.3f}".format(
                      r["dataset"][:27], r["n_doubles"], r["mean_additive_cos2"],
                      r["mean_I_over_dAB"], r["mean_Ioff_over_I"],
                      r["additive_cos2_ppi"], r["additive_cos2_nonppi"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
