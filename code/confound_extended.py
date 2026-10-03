import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
confound_extended.py  (reviewer points 6, 7 and cell-composition/batch/guide)

Extends scores_vs_globalmode.py.  For the control cells of each dataset we test
what the leading global mode (top eigenvector of the control covariance) tracks:

  * total counts and an independent median-of-ratios (MOR) size factor,
  * cell-cycle (S, G2M), stress, apoptosis, senescence scores,
  * additional stress programmes: DNA damage, interferon, heat shock, UPR, p53,
  * partial correlation of the global mode with the MOR size factor controlling
    for: cell cycle, the top control PCs (a proxy for cell composition/state),
    batch, guide identity, and cell cycle + PCs together.

If the size axis were merely a composition or technical artefact, controlling for
the top PCs, batch or guide should remove it.

Output: ../results/confound_extended.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "confound_extended.csv")

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
DNA_DAMAGE = ["CDKN1A", "MDM2", "GADD45A", "GADD45B", "XPC", "DDB2", "RRM2B", "BTG2",
              "TP53I3", "SESN1", "TIGAR", "ATF3", "PLK3", "RAD51", "BRCA1"]
INTERFERON = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "MX1", "MX2", "OAS1", "OAS2",
              "STAT1", "IRF7", "IFI44L", "IFITM1", "ISG20", "XAF1"]
HEAT_SHOCK = ["HSPA1A", "HSPA1B", "HSPA6", "HSPB1", "DNAJB1", "DNAJA1", "HSPH1",
              "BAG3", "HSPA8", "DNAJB4", "HSPE1", "HSPD1"]
UPR = ["DDIT3", "HSPA5", "XBP1", "ATF4", "DNAJB9", "EDEM1", "HERPUD1", "SEL1L",
       "ERN1", "ATF6", "EIF2AK3", "PPP1R15A"]
P53 = ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "ZMAT3",
       "TIGAR", "SESN1", "TP53I3", "RRM2B"]

GENE_SETS = [("s", S_GENES), ("g2m", G2M_GENES), ("stress", STRESS_GENES),
             ("apop", APOP_GENES), ("sen", SEN_GENES), ("dna_damage", DNA_DAMAGE),
             ("interferon", INTERFERON), ("heat_shock", HEAT_SHOCK),
             ("upr", UPR), ("p53", P53)]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(x, y, Z):
    """corr(x, y | columns of Z); Z may be None for simple correlation."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if Z is None or np.asarray(Z).size == 0:
        return corr(x, y)
    Z = np.atleast_2d(np.asarray(Z, float))
    if Z.shape[0] != len(x):
        Z = Z.T
    Z = np.column_stack([np.ones(len(x)), Z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return corr(x - Z @ bx, y - Z @ by)


def score(mat, gene_names, gene_list):
    gl = set(gene_list)
    present = [i for i, g in enumerate(gene_names) if g in gl]
    if len(present) < 5:
        return None
    sub = np.asarray(mat[:, present].todense(), dtype=np.float32)
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
    pos = {g: i for i, g in enumerate(a.var_names)}
    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    for _, gl in GENE_SETS:
        for g in gl:
            if g in pos:
                sel.add(pos[g])
    gi = np.array(sorted(sel))
    genes = list(np.asarray(a.var_names)[gi])
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()

    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c = nc_all[cidx]
    Z = Xc - Xc.mean(0)
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    v = V[:, order[0]]
    gscore = Z @ v
    pcs = Z @ V[:, order[1:6]]            # top 5 non-global PCs (composition/state proxy)
    del V, w, order
    gc.collect()

    logX = np.log1p(Xc)
    ref = logX.mean(0)
    keep = ref > 0
    ratio = Xc[:, keep] / np.expm1(ref[keep])[None, :]
    mor = np.median(ratio, axis=1)

    out = {"corr_total": corr(gscore, nc_c), "corr_mor": corr(gscore, mor)}
    for name, gl in GENE_SETS:
        sc = score(X[cidx], genes, gl)
        out["corr_" + name] = corr(gscore, sc) if sc is not None else float("nan")

    s_sc = score(X[cidx], genes, S_GENES)
    g_sc = score(X[cidx], genes, G2M_GENES)
    cc = 0.5 * (s_sc + g_sc) if (s_sc is not None and g_sc is not None) else None
    out["partial_mor_given_cc"] = partial_corr(gscore, mor, cc[:, None] if cc is not None else None)
    out["partial_mor_given_pcs"] = partial_corr(gscore, mor, pcs)
    if cc is not None:
        out["partial_mor_given_cc_pcs"] = partial_corr(gscore, mor, np.column_stack([cc, pcs]))
    else:
        out["partial_mor_given_cc_pcs"] = float("nan")
    out["partial_cc_given_mor"] = partial_corr(gscore, cc, mor[:, None]) if cc is not None else float("nan")

    # batch and guide
    for key, col in [("batch", "batch"), ("guide", "guide_id")]:
        if col in obs.columns:
            b = obs[col].values[cidx]
            try:
                codes = np.unique(b, return_inverse=True)[1].astype(float)
                out["corr_" + key] = corr(gscore, codes)
                out["partial_mor_given_" + key] = partial_corr(gscore, mor, codes[:, None])
            except Exception:
                out["corr_" + key] = float("nan")
                out["partial_mor_given_" + key] = float("nan")
        else:
            out["corr_" + key] = float("nan")
            out["partial_mor_given_" + key] = float("nan")
    del X, Xc, Z, pcs
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["corr_total", "corr_mor"] + ["corr_" + n for n, _ in GENE_SETS] + [
        "partial_mor_given_cc", "partial_mor_given_pcs", "partial_mor_given_cc_pcs",
        "partial_cc_given_mor", "corr_batch", "partial_mor_given_batch",
        "corr_guide", "partial_mor_given_guide"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    acc = {k: [] for k in keys}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open(OUT, "a") as fh:
                fh.write(name + "," + ",".join("{:.4f}".format(o[k]) for k in keys) + "\n")
            for k in keys:
                if np.isfinite(o[k]):
                    acc[k].append(o[k])
            print("OK   {:<30} MOR={:.3f} | cc={:.3f} pcs={:.3f} batch={:.3f} guide={:.3f}".format(
                name[:29], o["corr_mor"], o["partial_mor_given_cc"],
                o["partial_mor_given_pcs"], o["partial_mor_given_batch"],
                o["partial_mor_given_guide"]), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:40]), flush=True)
    print("\nMEANS (|corr|):", flush=True)
    for k in keys:
        v = [abs(x) for x in acc[k]]
        print("  {:<26} {:.3f} (n={})".format(k, np.mean(v) if v else float("nan"), len(v)), flush=True)


if __name__ == "__main__":
    main()
