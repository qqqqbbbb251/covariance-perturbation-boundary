"""
make_bench_splits_stream.py -- streaming benchmark-split builder for very large h5ad
(e.g. XAtlas ~19 GB CSR) that cannot be materialised.

Streams the file row-chunk by row-chunk, keeps a manageable cell subset (controls +
up to MAX_PER_PERT cells per single-gene perturbation) restricted to a gene panel
(top-N HVG + all perturbation targets), and writes
<out_dir>/{filtered.h5ad, control_idx.npy, train_idx.npy, test_idx.npy} with a
train/test split over perturbations.

Usage:
    python make_bench_splits_stream.py <raw.h5ad> <out_dir> [dataset_name] [N] [max_ctrl] [max_per]
"""

import os
import re
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

CTRL = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")
CHUNK = 20000
TEST_FRAC = 0.2
SEED = 0


def is_ctrl(s):
    s = str(s).strip().lower()
    return s in CTRL or any(t in s for t in ("control", "ctrl", "non-targeting", "nontargeting"))


def map_label(lab, geneset):
    p = str(lab).strip()
    if p in geneset:
        return p
    q = re.sub(r"([_\-\s]+)(KD|KO|OE|overexpression|activation|inhibition)$", "", p, flags=re.I)
    q = re.sub(r"^(sgRNA|gRNA|sg)([_\-\s]+)?", "", q, flags=re.I)
    if q in geneset:
        return q
    for sep in ["_", "+", "|", ";", ",", " "]:
        if sep in q:
            hits = [x for x in q.split(sep) if x in geneset]
            return hits[0] if len(hits) == 1 else ""
    return ""


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(src).replace(".h5ad", "")
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    MAX_CTRL = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
    MAX_PER = int(sys.argv[6]) if len(sys.argv) > 6 else 150
    os.makedirs(out_dir, exist_ok=True)

    a0 = ad.read_h5ad(src, backed="r")
    obs = a0.obs
    labels = obs["perturbation"].astype(str).values if "perturbation" in obs.columns else \
        obs[list(obs.columns)[0]].astype(str).values
    var_names = np.asarray(a0.var_names).astype(str)
    geneset = set(var_names.tolist())
    # gene panel score
    score = None
    for c in ("highly_variable_rank", "total_counts", "mean_counts", "means", "ncounts"):
        if c in a0.var.columns:
            v = np.asarray(a0.var[c].values, dtype=float)
            score = -v if c == "highly_variable_rank" else v   # rank: smaller = better
            break
    a0.file.close()
    if score is None:
        score = np.ones(len(var_names))
    score = np.nan_to_num(score, nan=-1e18 if False else 0.0)

    mapped = np.array([map_label(l, geneset) for l in labels])
    ctrl = np.array([is_ctrl(l) for l in labels])
    singles = [g for g in pd.unique(mapped[~ctrl]) if g]
    # choose gene panel: top-N by score + all targets
    order = np.argsort(-score)
    panel = list(order[:N])
    panel += [int(np.where(var_names == g)[0][0]) for g in singles if (var_names == g).any()]
    panel = np.array(sorted(set(panel)))
    col_pos = {int(c): i for i, c in enumerate(panel)}

    rng = np.random.default_rng(SEED)
    # choose cells
    rows, tags = [], []   # tag: -1 control, else pert index
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    rows.extend(cidx.tolist()); tags.extend([-1] * len(cidx))
    for ti, g in enumerate(singles):
        ri = np.where(mapped == g)[0]
        if len(ri) == 0:
            continue
        if len(ri) > MAX_PER:
            ri = rng.choice(ri, MAX_PER, replace=False)
        rows.extend(ri.tolist()); tags.extend([ti] * len(ri))
    rows = np.asarray(rows); tags = np.asarray(tags)
    o = np.argsort(rows); rows = rows[o]; tags = tags[o]
    n_cells = len(rows)
    print("panel genes", len(panel), "cells", n_cells, "singles", len(singles), flush=True)

    out = np.zeros((n_cells, len(panel)), dtype=np.float32)
    a = ad.read_h5ad(src, backed="r")
    n = a.shape[0]
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        lo = int(np.searchsorted(rows, c0)); hi = int(np.searchsorted(rows, c1))
        if hi > lo:
            X = a[c0:c1].to_memory().X
            X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
            rloc = rows[lo:hi] - c0
            sub = X[rloc][:, panel].toarray()   # (k, P)
            out[lo:hi, :] = sub
            del X, sub
    a.file.close()

    # split perturbations train/test
    pert_arr = np.array(singles)
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(pert_arr)
    n_test = max(1, int(round(TEST_FRAC * len(pert_arr))))
    test_perts = set(pert_arr[:n_test].tolist()); train_perts = set(pert_arr[n_test:].tolist())

    condition = np.array(["control" if t == -1 else singles[t] for t in tags])
    cell_type = np.array([name.replace("_", "-")] * n_cells)
    Xcsr = sp.csr_matrix(out)
    adata = ad.AnnData(X=Xcsr,
                       obs=pd.DataFrame({"condition": condition, "cell_type": cell_type},
                                        index=[str(i) for i in rows]),
                       var=pd.DataFrame({"gene_name": var_names[panel]},
                                        index=var_names[panel]))
    adata.write_h5ad(os.path.join(out_dir, "filtered.h5ad"))
    ctrl_idx = np.where(tags == -1)[0]
    tr = np.where(np.array([(t != -1 and singles[t] in train_perts) for t in tags]))[0]
    te = np.where(np.array([(t != -1 and singles[t] in test_perts) for t in tags]))[0]
    np.save(os.path.join(out_dir, "control_idx.npy"), ctrl_idx)
    np.save(os.path.join(out_dir, "train_idx.npy"), tr)
    np.save(os.path.join(out_dir, "test_idx.npy"), te)
    print("wrote", out_dir, "shape", adata.shape, "| ctrl", len(ctrl_idx),
          "train", len(tr), "test", len(te), "| train_perts", len(train_perts),
          "test_perts", len(test_perts), flush=True)


if __name__ == "__main__":
    main()
