import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
inverse_false_positive.py  -- Experiment 3

CIPHER claims it "identified the true perturbation with high-fidelity given only
the covariance matrix and change in gene expression".  If Sigma is near rank-one,
the inverse problem is ill-posed: the matched-filter score for gene g' is

    u[g'] = <Sigma[:,g'], dx> / ||Sigma[:,g']||^2

and with Sigma[:,g'] ~= c_g' * v this becomes u[g'] ~= (v.dx) / (c_g' * ||v||^2),
i.e. the ranking is driven by the covariance COLUMN NORM, not by the perturbation
identity.  We test the true driver's rank / one-vs-rest AUC and the correlation of
|u| with 1/column_norm.

Also a "self-removed" variant (zero the target gene's own coordinate in dx and
Sigma) to check whether any recovery is just the target gene's own entry.

Output: ../results/inverse_false_positive.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "inverse_false_positive.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
SPACES = ["raw", "pearson"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def rank_auc(scores, i):
    """one-vs-rest AUC of |score| for the true driver i (rank-based)."""
    s = np.abs(scores)
    sp = s[i]
    neg = np.delete(s, i)
    neg = neg[np.isfinite(neg)]
    if neg.size == 0 or not np.isfinite(sp):
        return float("nan")
    wins = np.count_nonzero(neg < sp)
    ties = np.count_nonzero(neg == sp)
    return float((wins + 0.5 * ties) / neg.size)


def load(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    base = raw[(~ctrl) & (nperts == 1)]
    if len(np.unique(base)) < 10:
        base = raw[~ctrl]
    uq, cn = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uq, cn) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    if len(singles) < 10 or len(cidx) < 100:
        raise ValueError("too few")
    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    pos = {g: i for i, g in enumerate(a.var_names)}
    for g in singles:
        if g in pos:
            sel.add(pos[g])
    gi = np.array(sorted(sel))
    genes = np.asarray(a.var_names)[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]
    return X, idx, singles, cidx, nc_all, raw, rng


def pearson_mu(X, cidx, nc_all):
    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c = nc_all[cidx]
    sf = nc_c / (nc_c.mean() + 1e-9)
    return (Xc / sf[:, None]).mean(0)


def get_d(X, rows, nc_all, cmean, space, mu=None):
    E = np.asarray(X[rows].todense(), dtype=np.float32)
    if space == "raw":
        d = E.mean(0)
    else:
        nc = nc_all[rows]
        sf = nc / (nc.mean() + 1e-9)
        muij = sf[:, None] * mu[None, :]
        d = ((E - muij) / np.sqrt(muij + 1e-6)).mean(0)
    return d - cmean


def analyze(path, space):
    X, idx, singles, cidx, nc_all, raw, rng = load(path)
    mu = pearson_mu(X, cidx, nc_all) if space == "pearson" else None
    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    if space == "pearson":
        nc_c = nc_all[cidx]
        sf = nc_c / (nc_c.mean() + 1e-9)
        muij = sf[:, None] * mu[None, :]
        Xc = (Xc - muij) / np.sqrt(muij + 1e-6)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    gfrac = float(w.max() / w.sum())
    del Xc, Z, V, w
    gc.collect()
    col_norm = np.einsum("ij,ij->j", cov, cov)
    invnorm = 1.0 / np.maximum(col_norm, 1e-12)

    auc, top1, top10, pct, corr_inv, auc_s, pct_s, auc_mag = [], [], [], [], [], [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        d = get_d(X, rows, nc_all, cmean, space, mu)
        if d @ d == 0:
            continue
        i = idx[g]
        u = (cov.T @ d) / np.maximum(col_norm, 1e-12)      # matched filter
        order = np.argsort(-np.abs(u))
        r = int(np.where(order == i)[0][0])
        pct.append(1.0 - r / len(u))
        top1.append(1.0 if r == 0 else 0.0)
        top10.append(1.0 if r < 10 else 0.0)
        auc.append(rank_auc(u, i))
        auc_mag.append(rank_auc(d, i))                     # baseline: rank by |dx| itself
        corr_inv.append(float(np.corrcoef(np.abs(u), invnorm)[0, 1]))
        # self-removed variant: drop the target gene's own expression change (d[i]=0),
        # keeping the covariance column, so the model cannot use the self coordinate.
        d2 = d.copy(); d2[i] = 0.0
        u2 = (cov.T @ d2) / np.maximum(col_norm, 1e-12)
        order2 = np.argsort(-np.abs(u2))
        r2 = int(np.where(order2 == i)[0][0])
        pct_s.append(1.0 - r2 / len(u2))
        auc_s.append(rank_auc(u2, i))
    del X, cov
    gc.collect()
    return {"dataset": os.path.basename(path).replace(".h5ad", ""), "space": space,
            "n": len(auc), "global_frac": gfrac,
            "mean_auc": float(np.nanmean(auc)), "mean_pct": float(np.nanmean(pct)),
            "top1": float(np.nanmean(top1)), "top10": float(np.nanmean(top10)),
            "corr_absu_invnorm": float(np.nanmean(corr_inv)),
            "mean_auc_selfremoved": float(np.nanmean(auc_s)),
            "mean_pct_selfremoved": float(np.nanmean(pct_s)),
            "mean_auc_magnitude": float(np.nanmean(auc_mag))}


def main():
    d = _PERTURB_DATA
    # 15-dataset set used for the inverse-baseline results
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.join(d, x + ".h5ad") for x in [
            "DatlingerBock2017", "DatlingerBock2021", "DixitRegev2016_K562_TFs_7_days",
            "FrangiehIzar2021_RNA", "GasperiniShendure2019_lowMOI",
            "NadigOConner2024_hepg2", "NadigOConner2024_jurkat",
            "NormanWeissman2019_filtered", "PapalexiSatija2021_eccite_RNA",
            "ReplogleWeissman2022_K562_essential", "ReplogleWeissman2022_rpe1",
            "TianKampmann2019_day7neuron", "TianKampmann2019_iPSC",
            "TianKampmann2021_CRISPRa", "TianKampmann2021_CRISPRi"]]
    keys = ["n", "global_frac", "mean_auc", "mean_pct", "top1", "top10",
            "corr_absu_invnorm", "mean_auc_selfremoved", "mean_pct_selfremoved",
            "mean_auc_magnitude"]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(keys) + "\n")
    for f in files:
        for space in SPACES:
            try:
                r = analyze(f, space)
                with open(OUT, "a") as fh:
                    fh.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                        r["dataset"], r["space"], r["n"], r["global_frac"], r["mean_auc"],
                        r["mean_pct"], r["top1"], r["top10"], r["corr_absu_invnorm"],
                        r["mean_auc_selfremoved"], r["mean_pct_selfremoved"], r["mean_auc_magnitude"]))
                print("OK   {:<26} {:<8} n={} AUC={:.3f} selfRem={:.3f} mag={:.3f} top10={:.3f}".format(
                    r["dataset"][:25], space, r["n"], r["mean_auc"], r["mean_auc_selfremoved"],
                    r["mean_auc_magnitude"], r["top10"]), flush=True)
            except Exception as e:
                print("SKIP {:<26} {:<8} {}".format(os.path.basename(f)[:25], space, str(e)[:40]), flush=True)


if __name__ == "__main__":
    main()
