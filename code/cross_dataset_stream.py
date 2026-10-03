import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cross_dataset_stream.py -- cross-dataset specific-residual compare for LARGE datasets
that cannot be materialised (XAtlas HCT116 / HEK293T, ~19 GB CSR each).

Streams each h5ad row-chunk-by-row-chunk (anndata backed row slicing), accumulating:
  * a control-cell matrix (for the covariance / global axis),
  * per-target mean shifts (for the targets shared with the other dataset).
Then applies the same (global-axis- and self-removed) specific-residual comparison as
``cross_dataset_specific.py``.

Usage: cross_dataset_stream.py A.h5ad B.h5ad [N_GENES]
Output: appends to ../results/cross_dataset_specific.csv
"""

import csv
import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)
from cipher.data import infer_target_gene  # noqa: E402

DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "cross_dataset_specific.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
PERT_COLS = ["perturbation", "gene", "target", "target_gene", "gene_name", "guide_target", "condition"]
MAX_CTRL, MAX_PER_PERT, CHUNK = 3000, 100, 20000
SEED = 0


def is_control(l):
    l = str(l).lower()
    return any(t in l for t in CONTROL_TOKENS)


def read_meta(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = next((c for c in PERT_COLS if c in obs.columns and obs[c].astype(str).nunique() > 5), "perturbation")
    labels = obs[pc].astype(str).values
    var_names = np.asarray(a.var_names).astype(str)
    vtot = None
    for c in ("total_counts", "mean_counts", "ncounts", "means"):
        if c in a.var.columns:
            vtot = np.asarray(a.var[a.var.columns[a.var.columns.get_loc(c)]].values, dtype=float)
            break
    a.file.close()
    if vtot is None:
        vtot = np.ones(len(var_names))
    vtot = np.nan_to_num(vtot, nan=0.0, posinf=0.0, neginf=0.0)
    return labels, var_names, vtot


def map_targets(labels, var_names):
    gs = set(var_names.tolist())
    lab2gene = {l: (infer_target_gene(str(l), gs)[0] or "") for l in np.unique(labels)}
    mapped = np.array([lab2gene[l] for l in labels])
    ctrl = np.array([is_control(l) for l in labels])
    return mapped, ctrl


def choose_rows(mapped, ctrl, targets, rng):
    rows, tags = [], []
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    rows.extend(cidx.tolist()); tags.extend([0] * len(cidx))
    for ti, g in enumerate(targets):
        ri = np.where(mapped == g)[0]
        if len(ri) == 0:
            continue
        if len(ri) > MAX_PER_PERT:
            ri = rng.choice(ri, MAX_PER_PERT, replace=False)
        rows.extend(ri.tolist()); tags.extend([ti + 1] * len(ri))
    rows = np.asarray(rows); tags = np.asarray(tags)
    o = np.argsort(rows)
    return rows[o], tags[o]


def stream_side(path, gene_sel, targets):
    labels, var_names, vtot = read_meta(path)
    mapped, ctrl = map_targets(labels, var_names)
    rng = np.random.default_rng(SEED)
    rows, tags = choose_rows(mapped, ctrl, targets, rng)
    vpos = {g: i for i, g in enumerate(var_names.tolist())}
    col_idx = np.array([vpos[g] for g in gene_sel], dtype=np.int64)
    order = np.argsort(col_idx); col_sorted = col_idx[order]; inv = np.argsort(order)
    n, n_sel = len(labels), len(gene_sel)
    ctrl_list = []
    tsum = np.zeros((len(targets), n_sel), dtype=np.float64)
    tcnt = np.zeros(len(targets), dtype=np.int64)
    a = ad.read_h5ad(path, backed="r")
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        lo = int(np.searchsorted(rows, c0)); hi = int(np.searchsorted(rows, c1))
        if hi > lo:
            blk = a[c0:c1].to_memory()
            X = blk.X
            Xc = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
            rloc = rows[lo:hi] - c0
            sub = Xc[rloc][:, col_sorted].toarray()[:, inv].astype(np.float64)
            tg = tags[lo:hi]
            for k in range(sub.shape[0]):
                t = tg[k]
                if t == 0:
                    ctrl_list.append(sub[k])
                else:
                    tsum[t - 1] += sub[k]; tcnt[t - 1] += 1
            del blk, Xc, sub
        if (c0 // CHUNK) % 5 == 0:
            gc.collect()
    a.file.close()
    gc.collect()
    ctrl_mat = np.asarray(ctrl_list) if ctrl_list else np.zeros((0, n_sel))
    return ctrl_mat, tsum, tcnt


def residual_R(ctrl_mat, tsum, tcnt, gene_sel, targets):
    cmean = ctrl_mat.mean(0)
    Z = ctrl_mat - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    gidx = {g: i for i, g in enumerate(gene_sel)}
    R, keep = [], []
    for i, g in enumerate(targets):
        if tcnt[i] == 0 or g not in gidx:
            continue
        d = tsum[i] / tcnt[i] - cmean
        d = d - float(d @ v) * v
        d[gidx[g]] = 0.0
        R.append(d); keep.append(g)
    return np.asarray(R), keep


def main():
    if len(sys.argv) < 3:
        print("usage: cross_dataset_stream.py A.h5ad B.h5ad [N_GENES]"); return
    pa, pb = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    fa = pa if os.path.isabs(pa) else os.path.join(DATA, pa)
    fb = pb if os.path.isabs(pb) else os.path.join(DATA, pb)

    la, va, ta = read_meta(fa); lb, vb, tb = read_meta(fb)
    ma, ca = map_targets(la, va); mb, cb = map_targets(lb, vb)
    tga = {g for g in np.unique(ma) if g and not is_control(g)}
    tgb = {g for g in np.unique(mb) if g and not is_control(g)}
    targets = sorted(tga & tgb)
    common = sorted(set(va.tolist()) & set(vb.tolist()))
    if len(targets) < 15 or len(common) < 200:
        print("too few targets/common genes", len(targets), len(common)); return
    sa = {g: ta[i] for i, g in enumerate(va.tolist())}
    sb = {g: tb[i] for i, g in enumerate(vb.tolist())}
    score = np.array([sa[g] / (np.median(ta) + 1e-9) + sb[g] / (np.median(tb) + 1e-9) for g in common])
    top = [common[j] for j in np.argsort(-score)[:N]]
    gene_sel = sorted(set(top) | set(targets))
    print("targets", len(targets), "gene_sel", len(gene_sel), flush=True)

    cA, sA, nA = stream_side(fa, gene_sel, targets)
    cB, sB, nB = stream_side(fb, gene_sel, targets)
    RA, ka = residual_R(cA, sA, nA, gene_sel, targets)
    RB, kb = residual_R(cB, sB, nB, gene_sel, targets)
    common_t = [g for g in targets if g in set(ka) and g in set(kb)]
    ia = {g: i for i, g in enumerate(ka)}; ib = {g: i for i, g in enumerate(kb)}
    A = np.asarray([RA[ia[g]] for g in common_t]); B = np.asarray([RB[ib[g]] for g in common_t])

    def cos2_rows(P, Q):
        num = np.sum(P * Q, axis=1); d = np.sum(P * P, axis=1) * np.sum(Q * Q, axis=1)
        return float(np.nanmean(num ** 2 / np.maximum(d, 1e-24)))

    def gene_corr(A, B):
        out = []
        for j in range(A.shape[1]):
            x, y = A[:, j], B[:, j]
            if np.isfinite(x).sum() < 3 or x.std() == 0 or y.std() == 0:
                continue
            out.append(np.corrcoef(x, y)[0, 1])
        out = np.asarray([o for o in out if np.isfinite(o)])
        return float(out.mean()) if out.size else float("nan")

    rng = np.random.default_rng(SEED); perm = rng.permutation(len(common_t))
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    row = {"pair": os.path.basename(fa).replace(".h5ad", "") + "->" + os.path.basename(fb).replace(".h5ad", ""),
           "n_common_genes": len(gene_sel), "n_targets": len(common_t),
           "cos_resid": cos2_rows(A, B), "cos_resid_null": cos2_rows(A, B[perm]),
           "cos_specific": cos2_rows(Ac, Bc), "cos_specific_null": cos2_rows(Ac, Bc[perm]),
           "gene_corr_resid": gene_corr(A, B), "gene_corr_specific": gene_corr(Ac, Bc)}
    keys = ["n_common_genes", "n_targets", "cos_resid", "cos_resid_null", "cos_specific",
            "cos_specific_null", "gene_corr_resid", "gene_corr_specific"]
    exists = os.path.exists(OUT) and os.path.getsize(OUT) > 0
    done = False
    if exists:
        with open(OUT) as fh:
            next(fh, None)
            done = any(line.split(",", 1)[0] == row["pair"] for line in fh if line.strip())
    with open(OUT, "a" if exists else "w") as fh:
        if not exists:
            fh.write("pair," + ",".join(keys) + "\n")
        if not done:
            fh.write(row["pair"] + "," + ",".join(str(row[k]) for k in keys) + "\n")
    print("OK {:<52} n={} cos_specific={:.3f} (null {:.3f}) gene_corr={:.3f}".format(
        row["pair"][:51], row["n_targets"], row["cos_specific"], row["cos_specific_null"],
        row["gene_corr_specific"]), flush=True)


if __name__ == "__main__":
    main()
