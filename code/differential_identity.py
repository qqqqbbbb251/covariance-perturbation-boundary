import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
differential_identity.py  -- feasibility prototype

Experiment 1 (differential response): if CIPHER's forward prediction is a shared
global axis, it should predict a perturbation's full response (which contains the
shared axis) but NOT the *difference* between two perturbations' responses (which
is gene-specific).  We compute, per perturbation pair (A,B):
  - full_cos      = cos^2(d_A, d_hat_A)          CIPHER forward accuracy
  - shared_frac   = cos^2(d_A, v)                fraction of a response on the global axis
  - diff_shared   = cos^2(d_A - d_B, v)          fraction of the DIFFERENCE on the global axis
  - diff_pred_cos = cos^2(d_hat_A - d_hat_B, d_A - d_B)   can CIPHER predict the difference?

Experiment 2 (perturbation identity): are the predicted responses able to
discriminate perturbations?
  - paircos_true / paircos_pred : mean pairwise cosine among true / predicted responses
  - rsa : Spearman correlation between the pairwise-distance matrices of true and
          predicted responses (representational similarity). ~0 means CIPHER's
          predictions do not preserve perturbation-specific structure.

Output: ../results/differential_identity.csv
"""

import gc
import math
import os
import re
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "differential_identity.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
CONTROL_EXACT = {"control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg",
                 "scramble", "scrambled", "mock", "untreated", "vehicle", "empty", "safe"}
PERT_COLS = ["perturbation", "gene", "gene_target", "guide_target", "target", "target_gene",
             "gene_name", "condition", "sgRNA", "guide_id"]
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
SPACES = ["raw", "cpm", "pearson"]


def is_control(l):
    l = str(l).strip().lower()
    return l in CONTROL_EXACT or any(t in l for t in CONTROL_TOKENS)


def _pick_pert_col(obs):
    for c in PERT_COLS:
        if c in obs.columns and obs[c].astype(str).nunique() > 5:
            return c
    return "perturbation"


def _map_label(lab, geneset):
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


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na and nb else float("nan")


def cos2(a, b):
    c = cos(a, b)
    return c * c if np.isfinite(c) else float("nan")


def load(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pc = _pick_pert_col(obs)
    labels = obs[pc].astype(str).values
    n = len(labels)
    nperts = (obs["nperts"].values if "nperts" in obs.columns else np.ones(n, dtype=int))

    # gene names: prefer var_names, fall back to a symbol column if labels overlap it better
    var_names_all = np.asarray(a.var_names).astype(str)
    lab_set = set(map(str, np.unique(labels)))
    gene_names_all = var_names_all
    best = len(lab_set & set(var_names_all.tolist()))
    for c in ("gene_name", "gene_names", "gene_symbol", "symbol", "name", "genes"):
        if c in a.var.columns:
            cand = np.asarray(a.var[c].astype(str))
            ov = len(lab_set & set(cand.tolist()))
            if ov > best:
                gene_names_all, best = cand, ov
    geneset = set(gene_names_all.tolist())
    lab2gene = {l: _map_label(l, geneset) for l in lab_set}
    mapped = np.array([lab2gene.get(l, "") for l in labels])

    ctrl = np.array([is_control(p) for p in labels]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    base = mapped[(~ctrl) & (nperts == 1)]
    if len(np.unique(base[base != ""])) < 10:
        base = mapped[~ctrl]
    uq, cn = np.unique(base[base != ""], return_counts=True)
    singles = [g for g, c in zip(uq, cn) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    if len(singles) < 10 or len(cidx) < 100:
        raise ValueError("too few")

    var_ncounts = None
    for c in ("ncounts", "total_counts", "mean_counts", "means", "mean"):
        if c in a.var.columns:
            var_ncounts = np.asarray(a.var[c].values, dtype=float)
            break
    if var_ncounts is None:
        var_ncounts = np.ones(len(gene_names_all))
    var_ncounts = np.nan_to_num(var_ncounts, nan=0.0, posinf=0.0, neginf=0.0)
    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    pos = {g: i for i, g in enumerate(gene_names_all)}
    for g in singles:
        if g in pos:
            sel.add(pos[g])
    gi = np.array(sorted(sel))
    genes = gene_names_all[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()

    nc_all = None
    for c in ("ncounts", "total_counts", "UMI_count", "num_umis"):
        if c in obs.columns:
            nc_all = np.asarray(obs[c].values, dtype=np.float32)
            break
    if nc_all is None:
        nc_all = np.asarray(X.sum(1)).ravel().astype(np.float32)

    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]
    return X, genes, idx, singles, cidx, nc_all, mapped, rng


def to_space(X, cidx, nc_all, space):
    """Return (control_residual_matrix_fn, transform) for the given space."""
    if space == "raw":
        return lambda M, nc: np.asarray(M.todense(), dtype=np.float32), lambda M, nc: np.asarray(M.todense(), dtype=np.float32)
    if space == "cpm":
        def f(M, nc):
            inv = (1e4 / np.maximum(nc, 1.0)).astype(np.float32)[:, None]
            return np.asarray(M.todense(), dtype=np.float32) * inv
        return f, f
    if space == "pearson":
        Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
        nc_c = nc_all[cidx]
        sf_c = nc_c / (nc_c.mean() + 1e-9)
        mu = (Xc / sf_c[:, None]).mean(0)

        def f(M, nc):
            E = np.asarray(M.todense(), dtype=np.float32)
            sf = nc / (nc.mean() + 1e-9)
            muij = sf[:, None] * mu[None, :]
            return (E - muij) / np.sqrt(muij + 1e-6)
        return f, f
    raise ValueError(space)


def analyze(path, space):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ctrl_fn, pert_fn = to_space(X, cidx, nc_all, space)
    Xc = ctrl_fn(X[cidx], nc_all[cidx])
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    gfrac = float(w.max() / w.sum())
    del Xc, Z, V, w
    gc.collect()

    D, DH = [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        d = pert_fn(X[rows], nc_all[rows]).mean(0) - cmean
        if d @ d == 0:
            continue
        i = idx[g]
        sig = cov[:, i]
        denom = float(sig @ sig)
        a = float(sig @ d) / denom if denom > 0 else 0.0
        D.append(d)
        DH.append(a * sig)
        del d
    del X, cov
    gc.collect()
    D = np.asarray(D)
    DH = np.asarray(DH)
    n = len(D)
    if n < 10:
        raise ValueError("too few")

    full_cos = float(np.nanmean([cos2(D[i], DH[i]) for i in range(n)]))
    shared_frac = float(np.nanmean([cos2(D[i], v) for i in range(n)]))

    diff_shared, diff_pred, pc_true, pc_pred = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dd = D[i] - D[j]
            diff_shared.append(cos2(dd, v))
            diff_pred.append(cos2(DH[i] - DH[j], dd))
            pc_true.append(abs(cos(D[i], D[j])))
            pc_pred.append(abs(cos(DH[i], DH[j])))
    diff_shared = float(np.nanmean(diff_shared))
    diff_pred = float(np.nanmean(diff_pred))
    pc_true = float(np.nanmean(pc_true))
    pc_pred = float(np.nanmean(pc_pred))

    # RSA: Spearman between sign-agnostic distance matrices (1 - cos^2)
    try:
        from scipy.stats import spearmanr
        Dt = np.array([[1 - cos(D[i], D[j]) ** 2 for j in range(n)] for i in range(n)])
        Dp = np.array([[1 - cos(DH[i], DH[j]) ** 2 for j in range(n)] for i in range(n)])
        iu = np.triu_indices(n, 1)
        a_, b_ = Dt[iu], Dp[iu]
        m = np.isfinite(a_) & np.isfinite(b_)
        rsa = float(spearmanr(a_[m], b_[m]).correlation) if m.sum() > 10 else float("nan")
    except Exception:
        rsa = float("nan")

    return {"dataset": os.path.basename(path).replace(".h5ad", ""), "space": space,
            "n_perts": n, "global_frac": gfrac,
            "full_cos": full_cos, "shared_frac": shared_frac,
            "diff_shared": diff_shared, "diff_pred_cos": diff_pred,
            "paircos_true": pc_true, "paircos_pred": pc_pred, "rsa": rsa}


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.join(d, "TianKampmann2021_CRISPRa.h5ad")]
    keys = ["n_perts", "global_frac", "full_cos", "shared_frac", "diff_shared",
            "diff_pred_cos", "paircos_true", "paircos_pred", "rsa"]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(keys) + "\n")
    for f in files:
        for space in SPACES:
            try:
                r = analyze(f, space)
                with open(OUT, "a") as fh:
                    fh.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                        r["dataset"], r["space"], r["n_perts"], r["global_frac"], r["full_cos"],
                        r["shared_frac"], r["diff_shared"], r["diff_pred_cos"],
                        r["paircos_true"], r["paircos_pred"], r["rsa"]))
                print("OK   {:<26} {:<8} n={} full={:.3f} shf={:.3f} dShf={:.3f} dPred={:.3f} pcT={:.3f} pcP={:.3f} RSA={:.2f}".format(
                    r["dataset"][:25], space, r["n_perts"], r["full_cos"], r["shared_frac"],
                    r["diff_shared"], r["diff_pred_cos"], r["paircos_true"],
                    r["paircos_pred"], r["rsa"]), flush=True)
            except Exception as e:
                print("SKIP {:<28} {:<8} {}".format(os.path.basename(f)[:27], space, str(e)[:40]), flush=True)


if __name__ == "__main__":
    main()
