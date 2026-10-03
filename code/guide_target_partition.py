import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
guide_target_partition.py -- decouple the perturbation-specific residual into a
target-common component and a guide-specific component.

Motivation: ``guide_consistency.py`` showed that individual guides of the same target
agree only weakly (cos ~0.06-0.19), raising the concern that the "specific structure"
is partly a guide/capture artefact.  For every target with >= 2 guides we write

    r_{g,u} = m_g + e_{g,u},   m_g = (1/n_g) sum_u r_{g,u}   (target-common),
                               e_{g,u} = r_{g,u} - m_g        (guide-specific),

where r_{g,u} is the guide residual with the global depth axis and the target's own
coordinate removed.  We then report

  * guide_agreement      mean pairwise cos across a target's guides (the old warning),
  * within_guide_reliab  split-half (over cells) cos of a *single* guide's residual,
  * target_frac          share of total residual energy that is target-common,
  * target_frac_null     same under a guide->target permutation,
  * target_excess        target_frac - target_frac_null,
  * target_mean_reliab   split-half (over guides) cos of the target-common vector.

Thus a low guide_agreement can be either a genuine guide-level effect (within_guide
reliable) or a cell-sampling power limit (within_guide unreliably low); target_frac >
null and a sizeable target_mean_reliab mean the pooled/aggregated specific structure is
target-driven.

Output: ../results/guide_target_partition.csv
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
OUT = os.path.join(RESULTS, "guide_target_partition.csv")

GUIDE_COLS = ["guide_id", "sgRNA", "guide_target", "guide", "gRNA"]
PERT_COLS = ["perturbation", "gene", "target"]
CONTROL = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")
N_TOP, MAX_CTRL, MIN_CELLS = 1500, 3000, 25
N_PERM, SEED = 30, 0


def is_control(l):
    l = str(l).strip().lower()
    return l in CONTROL or any(t in l for t in ("control", "ctrl", "non-targeting", "nontargeting"))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b) / (na * nb)


def analyze(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = next((c for c in PERT_COLS if c in obs.columns), None)
    gcol = next((c for c in GUIDE_COLS if c in obs.columns), None)
    if pc is None or gcol is None:
        a.file.close()
        raise ValueError("no perturbation/guide column")
    labels = obs[pc].astype(str).values
    guides = obs[gcol].astype(str).values
    var_names = np.asarray(a.var_names).astype(str)
    gset = set(var_names.tolist())

    rank = None
    for c in ("ncounts", "mean", "means", "total_counts", "ncells"):
        if c in a.var.columns:
            rank = np.nan_to_num(np.asarray(a.var[c].values, dtype=float))
            break

    ctrl_mask = np.array([is_control(l) for l in labels])
    by_tg = defaultdict(dict)
    for i in np.where(~ctrl_mask)[0]:
        by_tg[labels[i]].setdefault(guides[i], []).append(i)
    tg = {}
    for lab, d in by_tg.items():
        d2 = {gu: np.asarray(rows) for gu, rows in d.items() if len(rows) >= MIN_CELLS}
        if len(d2) >= 2 and lab in gset:
            tg[lab] = d2
    if len(tg) < 10:
        a.file.close()
        raise ValueError("too few multi-guide targets")

    cidx = np.where(ctrl_mask)[0]
    rng = np.random.default_rng(SEED)
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)

    targets = sorted(tg)
    if rank is None:
        cb = a[cidx].to_memory().X
        cb = cb.todense() if sp.issparse(cb) else cb
        rank = np.nan_to_num(np.asarray(cb, dtype=float).mean(0).ravel())
        del cb
    top = var_names[np.argsort(-rank)[:N_TOP]]
    sel = np.array(sorted(set(top.tolist()) | set(targets)))
    gpos = {g: i for i, g in enumerate(sel)}

    need = set(int(r) for r in cidx)
    for g in targets:
        for rows in tg[g].values():
            need.update(int(r) for r in rows)
    need = np.array(sorted(need))
    pos = {int(r): i for i, r in enumerate(need)}
    view = a[:, sel]
    sub = view[need].to_memory()
    X = sub.X
    X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)).astype(np.float32)
    del sub, view
    gc.collect()

    Ec = np.asarray(X[[pos[int(r)] for r in cidx]].todense(), dtype=np.float64)
    cmean = Ec.mean(0)
    Z = Ec - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Ec, Z, V, w
    gc.collect()

    def resid(rows, g):
        d = np.asarray(X[[pos[int(r)] for r in rows]].todense(), dtype=np.float64).mean(0) - cmean
        d = d - float(d @ v) * v
        d[gpos[g]] = 0.0
        return d

    within, acr, mean_half = [], [], []
    R = {}
    for g_cur in targets:
        d = tg[g_cur]
        gv = []
        for gu, rows in d.items():
            r = resid(rows, g_cur)
            if r @ r == 0:
                continue
            R[(g_cur, gu)] = r
            gv.append(r)
            if len(rows) >= 2 * MIN_CELLS:
                p = rng.permutation(len(rows))
                h = len(rows) // 2
                ra, rb = resid(rows[p[:h]], g_cur), resid(rows[p[h:]], g_cur)
                c = cos(ra, rb)
                if np.isfinite(c):
                    within.append(c)
        if len(gv) >= 2:
            for i in range(len(gv)):
                for j in range(i + 1, len(gv)):
                    c = cos(gv[i], gv[j])
                    if np.isfinite(c):
                        acr.append(c)
            if len(gv) >= 2:
                p = rng.permutation(len(gv))
                h = max(1, len(gv) // 2)
                mA = np.mean([gv[k] for k in p[:h]], axis=0)
                mB = np.mean([gv[k] for k in p[h:]], axis=0)
                c = cos(mA, mB)
                if np.isfinite(c):
                    mean_half.append(c)

    # target-common energy partition (equal guide weight)
    num = den = 0.0
    for g_cur in targets:
        vs = [R[(g_cur, gu)] for gu in tg[g_cur] if (g_cur, gu) in R]
        if len(vs) < 2:
            continue
        m = np.mean(vs, axis=0)
        num += len(vs) * float(m @ m)
        den += sum(float(r @ r) for r in vs)
    target_frac = num / den if den else np.nan

    # guide->target permutation null (preserve group sizes)
    keys = list(R.keys())
    gcount = defaultdict(int)
    for k in keys:
        gcount[k[0]] += 1
    counts = [c for c in gcount.values() if c >= 1]
    vecs = [R[k] for k in keys]
    nulls = []
    for _ in range(N_PERM):
        perm = rng.permutation(len(vecs))
        idx = 0
        nn = dd = 0.0
        for c in counts:
            grp = [vecs[perm[idx + j]] for j in range(c)]
            idx += c
            m = np.mean(grp, axis=0)
            nn += c * float(m @ m)
            dd += sum(float(r @ r) for r in grp)
        if dd:
            nulls.append(nn / dd)
    target_null = float(np.mean(nulls)) if nulls else np.nan

    n_guides = sum(len(tg[g]) for g in targets)
    a.file.close()
    return {
        "dataset": os.path.basename(path).replace(".h5ad", ""),
        "n_targets": len(targets),
        "n_guides": n_guides,
        "guide_agreement": float(np.mean(acr)) if acr else np.nan,
        "within_guide_reliab": float(np.mean(within)) if within else np.nan,
        "target_mean_reliab": float(np.mean(mean_half)) if mean_half else np.nan,
        "target_frac": float(target_frac),
        "target_frac_null": target_null,
        "target_excess": float(target_frac - target_null),
    }


def main():
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "ReplogleWeissman2022_K562_gwps_filtered.h5ad", "NadigOConner2024_jurkat.h5ad",
               "NadigOConner2024_hepg2.h5ad", "NormanWeissman2019_filtered.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_targets", "n_guides", "guide_agreement", "within_guide_reliab",
            "target_mean_reliab", "target_frac", "target_frac_null", "target_excess"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(DATA, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<34} n_tg={:<4} guides={:<5} agree={:.3f} within={:.3f} mean_rel={:.3f} "
                  "tgt_frac={:.3f} (null {:.3f})".format(
                      r["dataset"][:33], r["n_targets"], r["n_guides"], r["guide_agreement"],
                      r["within_guide_reliab"], r["target_mean_reliab"], r["target_frac"],
                      r["target_frac_null"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<34} {}".format(os.path.basename(f)[:33], str(e)[:70]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
