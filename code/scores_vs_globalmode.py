import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
scores_vs_globalmode.py  (reviewer points 6 and 7)

For control cells, correlate the per-cell global-mode score with:
  * total counts
  * an independent median-of-ratios (DESeq-style) size factor      [point 6]
  * a cell-cycle score (S and G2M marker genes)
  * stress / apoptosis / senescence scores                         [point 7]
  * batch (if present)

Also reports the partial correlation of the global mode with the MOR size factor
after regressing out the cell-cycle score, to test whether the size axis is
merely cell cycle.

Output: scores_vs_globalmode.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, N_TOP = 3000, 2000

S_GENES = ["PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "TYMS", "RRM2",
           "FEN1", "GINS2", "CDC45", "CDC6", "CDT1", "RAD51", "RPA2", "POLA1",
           "PRIM1", "DTL", "CDCA7", "HELLS", "UHRF1", "MCM10", "POLD1", "POLE",
           "RFC3", "RFC4", "RFC5", "SLBP", "GMNN", "CCNE1", "CCNE2", "CDK2", "E2F1"]
G2M_GENES = ["CDK1", "CCNB1", "CCNB2", "CDC20", "CDC25C", "AURKA", "AURKB", "PLK1",
             "BUB1", "BUB1B", "MAD2L1", "TOP2A", "MKI67", "CENPA", "CENPF", "KIF11",
             "KIF23", "ANLN", "ECT2", "TPX2", "NDC80", "NUF2", "UBE2C", "CKS1B",
             "CKS2", "CDCA8", "HJURP", "TTK", "KIF2C", "SMC4", "NCAPD2"]
STRESS_GENES = ["JUN", "FOS", "ATF3", "HSPA1A", "HSPA1B", "HSPB1", "DDIT3", "TRIB3",
                "GADD45A", "GADD45B", "HSPA5", "XBP1", "ATF4", "DNAJB9"]
APOP_GENES = ["BAX", "BAK1", "CASP3", "CASP7", "CASP8", "CASP9", "BID", "PMAIP1",
              "BBC3", "FAS", "TNFRSF10B", "APAF1", "CYCS", "DIABLO"]
SEN_GENES = ["CDKN1A", "CDKN2A", "TP53", "GLB1", "SERPINE1", "IL6", "CXCL8"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(x, y, z):
    """corr(x, y | z) with a single covariate z (linear)."""
    Z = np.column_stack([np.ones(len(z)), z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return corr(x - Z @ bx, y - Z @ by)


def score(mat, gene_names, gene_list):
    present = [i for i, g in enumerate(gene_names) if g in set(gene_list)]
    if len(present) < 5:
        return None
    sub = np.asarray(mat[:, present].todense(), dtype=np.float32)
    # mean expression of the set, after per-cell CPM
    return sub.mean(1)


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    if len(cidx) < 100:
        raise ValueError("few controls")

    var_ncounts = a.var["ncounts"].values.astype(float)
    gi = np.array(sorted(set(np.argsort(-var_ncounts)[:N_TOP].tolist())))
    genes = list(np.asarray(a.var_names)[gi])
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()

    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c = nc_all[cidx]
    # global mode
    Z = Xc - Xc.mean(0)
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)]
    gscore = Z @ v
    del Z, cov, V, w
    gc.collect()

    # median-of-ratios size factor (on the selected genes, control cells)
    logX = np.log1p(Xc)
    ref = logX.mean(0)
    keep = ref > 0
    ratio = Xc[:, keep] / np.expm1(ref[keep])[None, :]
    mor = np.median(ratio, axis=1)

    out = {"corr_total": corr(gscore, nc_c),
           "corr_mor": corr(gscore, mor)}
    for name, gl in [("s", S_GENES), ("g2m", G2M_GENES), ("stress", STRESS_GENES),
                     ("apop", APOP_GENES), ("sen", SEN_GENES)]:
        sc = score(X[cidx], genes, gl)
        out["corr_" + name] = corr(gscore, sc) if sc is not None else float("nan")
    # partial: global mode vs MOR controlling for cell cycle (S+G2M mean)
    s_sc = score(X[cidx], genes, S_GENES)
    g_sc = score(X[cidx], genes, G2M_GENES)
    if s_sc is not None and g_sc is not None:
        cc = 0.5 * (s_sc + g_sc)
        out["partial_mor_given_cc"] = partial_corr(gscore, mor, cc)
    else:
        out["partial_mor_given_cc"] = float("nan")
    # batch effect
    if "batch" in obs.columns:
        b = obs["batch"].values[cidx]
        try:
            codes = np.unique(b, return_inverse=True)[1].astype(float)
            out["corr_batch"] = corr(gscore, codes)
        except Exception:
            out["corr_batch"] = float("nan")
    else:
        out["corr_batch"] = float("nan")
    del X, Xc
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    keys = ["corr_total", "corr_mor", "corr_s", "corr_g2m", "corr_stress",
            "corr_apop", "corr_sen", "partial_mor_given_cc", "corr_batch"]
    with open("scores_vs_globalmode.csv", "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    acc = {k: [] for k in keys}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("scores_vs_globalmode.csv", "a") as fh:
                fh.write(name + "," + ",".join("{:.4f}".format(o[k]) for k in keys) + "\n")
            for k in keys:
                if not np.isnan(o[k]):
                    acc[k].append(o[k])
            print("OK   {:<30} total={:.3f} MOR={:.3f} S={:.3f} G2M={:.3f} stress={:.3f}".format(
                name[:29], o["corr_total"], o["corr_mor"], o["corr_s"],
                o["corr_g2m"], o["corr_stress"]), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:30]), flush=True)
    print("\nMEANS (|corr|):", flush=True)
    for k in keys:
        v = [abs(x) for x in acc[k]]
        print("  {:<22} {:.3f}".format(k, np.mean(v) if v else float("nan")), flush=True)


if __name__ == "__main__":
    main()
