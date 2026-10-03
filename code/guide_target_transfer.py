import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
guide_target_transfer.py -- is the cross-dataset transferable specific structure
target-common or guide-specific?

For a pair of datasets (same library or cross-lab) and their common perturbation
targets, build each target's specific residual (global depth axis + self removed) from
different estimates of the target effect:

    pooled   : mean over all cells of the target (what ``cross_dataset_specific.py`` uses)
    m1/m2/all: mean over 1 / 2 / all of the target's guide residuals

and measure the cross-dataset agreement (cosine, centred across targets) of each, with
a target-shuffled null.  If agreement rises with the number of pooled guides, the
transferable structure is target-common; if it is flat, it is guide-specific.

Output: appends to ../results/guide_target_transfer.csv
"""

import csv
import gc
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _PERTURB_DATA
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "guide_target_transfer.csv")

GUIDE_COLS = ["guide_id", "sgRNA", "guide_target", "guide", "gRNA"]
PERT_COLS = ["perturbation", "gene", "target"]
CONTROL = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")
N_TOP, MAX_CTRL, MIN_CELLS = 1500, 3000, 20
N_PERM, SEED = 50, 0


def is_control(l):
    l = str(l).strip().lower()
    return l in CONTROL or any(t in l for t in ("control", "ctrl", "non-targeting", "nontargeting"))


def cos(A, B):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return float(np.nanmean(num / np.maximum(den, 1e-12)))


def load_side(path, targets_wanted):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = next((c for c in PERT_COLS if c in obs.columns), "perturbation")
    gcol = next((c for c in GUIDE_COLS if c in obs.columns), None)
    labels = obs[pc].astype(str).values
    var_names = np.asarray(a.var_names).astype(str)
    rank = None
    for c in ("ncounts", "mean", "means", "total_counts", "ncells"):
        if c in a.var.columns:
            rank = np.nan_to_num(np.asarray(a.var[c].values, dtype=float)); break
    ctrl_mask = np.array([is_control(l) for l in labels])
    rng = np.random.default_rng(SEED)
    cidx = np.where(ctrl_mask)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    sub_idx = np.where(~ctrl_mask)[0]
    lab_to_guides = defaultdict(lambda: defaultdict(list))
    for i in sub_idx:
        if labels[i] in targets_wanted:
            lab_to_guides[labels[i]][obs[gcol].astype(str).values[i]].append(i)
    targets = [g for g in lab_to_guides if g in set(var_names.tolist())]
    if rank is None:
        cb = a[cidx].to_memory().X
        cb = cb.todense() if sp.issparse(cb) else cb
        rank = np.nan_to_num(np.asarray(cb, float).mean(0).ravel()); del cb
    top = var_names[np.argsort(-rank)[:N_TOP]]
    sel = np.array(sorted(set(top.tolist()) | set(targets)))
    gpos = {g: i for i, g in enumerate(sel)}
    need = set(int(r) for r in cidx)
    for g in targets:
        for rows in lab_to_guides[g].values():
            need.update(int(r) for r in rows)
    need = np.array(sorted(need)); pos = {int(r): i for i, r in enumerate(need)}
    sub = a[:, sel][need].to_memory()
    X = sub.X
    X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)).astype(np.float32)
    del sub
    gc.collect()
    Ec = np.asarray(X[[pos[int(r)] for r in cidx]].todense(), dtype=np.float64)
    cmean = Ec.mean(0); Z = Ec - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov); v = V[:, int(np.argmax(w))]
    del Ec, Z, V, w
    gc.collect()
    a.file.close()

    def resid(rows, g):
        d = np.asarray(X[[pos[int(r)] for r in rows]].todense(), dtype=np.float64).mean(0) - cmean
        d = d - float(d @ v) * v
        d[gpos[g]] = 0.0
        return d

    guide_mats = {}
    for g in targets:
        rows_all = []
        for gu, rows in lab_to_guides[g].items():
            if len(rows) < MIN_CELLS:
                continue
            guide_mats[(g, gu)] = resid(np.asarray(rows), g)
            rows_all.extend(rows)
        if not rows_all:
            guide_mats.pop((g, None), None)
    return sel, gpos, guide_mats


def main():
    if len(sys.argv) < 3:
        print("usage: guide_target_transfer.py A.h5ad B.h5ad [N_PERM]"); return
    fa, fb = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(SEED)

    # first pass: common targets
    def targets_of(path):
        a = ad.read_h5ad(path, backed="r")
        obs = a.obs
        pc = next((c for c in PERT_COLS if c in obs.columns), "perturbation")
        labels = obs[pc].astype(str).values
        var = set(np.asarray(a.var_names).astype(str).tolist())
        ctrl = np.array([is_control(l) for l in labels])
        out = set(labels[~ctrl]) & var
        a.file.close()
        return out

    ta = targets_of(os.path.join(DATA, fa)); tb = targets_of(os.path.join(DATA, fb))
    common = sorted(ta & tb)
    print("common targets", len(common), flush=True)
    if len(common) < 15:
        print("too few"); return

    selA, gposA, gmA = load_side(os.path.join(DATA, fa), set(common))
    selB, gposB, gmB = load_side(os.path.join(DATA, fb), set(common))
    common = [g for g in sorted(set(selA.tolist()) & set(selB.tolist()))
              if g in gposA and g in gposB]
    ng = lambda gm, g: len([1 for (gg, gu) in gm if gg == g])

    ia = {g: i for i, g in enumerate(selA)}; ib = {g: i for i, g in enumerate(selB)}
    use_genes = sorted(set(selA.tolist()) & set(selB.tolist()))
    gA = np.array([ia[g] for g in use_genes]); gB = np.array([ib[g] for g in use_genes])

    def target_est(gm, g, nguide):
        vecs = [v for (gg, gu), v in gm.items() if gg == g]
        if not vecs:
            return None
        k = min(nguide, len(vecs))
        pick = vecs if k >= len(vecs) else [vecs[j] for j in rng.choice(len(vecs), k, replace=False)]
        return np.mean(pick, axis=0)

    def eval_set(subset, tgts, rows):
        print("subset {} targets {}".format(subset, len(tgts)), flush=True)
        for tag, nguide in [("guide1", 1), ("guide2", 2), ("guideAll", 10**9)]:
            A, B = [], []
            for g in tgts:
                va = target_est(gmA, g, nguide)
                vb = target_est(gmB, g, nguide)
                if va is not None and vb is not None:
                    A.append(va[gA]); B.append(vb[gB])
            A = np.asarray(A); B = np.asarray(B)
            Ac = A - A.mean(0); Bc = B - B.mean(0)
            nulls = [cos(Ac, Bc[rng.permutation(len(A))]) for _ in range(N_PERM)]
            row = {"pair": os.path.basename(fa).replace(".h5ad", "") + "->" + os.path.basename(fb).replace(".h5ad", ""),
                   "subset": subset, "estimate": tag, "n_targets": len(A),
                   "cos_specific": cos(Ac, Bc), "cos_null": float(np.mean(nulls))}
            rows.append(row)
            print("  OK {:<9} n={:<5} cos_specific={:.3f} (null {:.3f})".format(
                tag, len(A), row["cos_specific"], row["cos_null"]), flush=True)

    rows = []
    eval_set("all", common, rows)
    eval_set("multiguide", [g for g in common if ng(gmA, g) >= 2 and ng(gmB, g) >= 2], rows)

    keys = ["pair", "subset", "estimate", "n_targets", "cos_specific", "cos_null"]
    exists = os.path.exists(OUT) and os.path.getsize(OUT) > 0
    done = set()
    if exists:
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    p = line.split(",")
                    done.add((p[0], p[1], p[2]))
    with open(OUT, "a" if exists else "w") as fh:
        if not exists:
            fh.write(",".join(keys) + "\n")
        for r in rows:
            if (r["pair"], r["subset"], r["estimate"]) in done:
                continue
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
