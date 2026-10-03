import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cross_dataset_specific.py -- Stage 2b: is the reproducible specific response biological?

Stage 2 showed that in several datasets the perturbation similarity structure is
reproducible across cell halves (high split-half RSA reliability) even after removing
the global axis, the target's own coordinate, and the common off-axis mode.  That
could still be a reproducible *technical* per-perturbation signature.

This test asks whether the *same target gene* produces the same specific residual
response in two different cell lines (Replogle K562 vs rpe1).  If yes, the structure
is biological; if not, it is dataset-specific noise/technical.

For each target present in both: d = mean(pert) - control, remove the global axis v
and the target's own coordinate, then compare d_K562 vs d_rpe1 (cosine, averaged over
targets; and per-gene correlation across targets).  Null: shuffled target pairing.

Output: ../results/cross_dataset_specific.csv
"""

import gc
import os
import sys

import numpy as np
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_dataset_transfer as cdt  # noqa: E402
cdt.MAX_PERT = 5000          # keep all single perturbations (do not cap) for target matching
load_side = cdt.load_side

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "cross_dataset_specific.csv")

MAX_PERT_CELLS = 500
SEED = 0


def _residual(X, raw, g, idxcols, cmean, v, tcol):
    rows = np.where(raw == g)[0]
    if len(rows) > MAX_PERT_CELLS:
        rows = np.random.default_rng(SEED).choice(rows, MAX_PERT_CELLS, replace=False)
    d = np.asarray(X[rows].todense(), dtype=np.float64)[:, idxcols].mean(0) - cmean
    r = d - float(d @ v) * v
    if 0 <= tcol < r.size:
        r[tcol] = 0.0
    return r


def analyze(pathA, pathB):
    aA = ad.read_h5ad(pathA, backed="r")
    aB = ad.read_h5ad(pathB, backed="r")
    shared = sorted(set(map(str, aA.var_names)) & set(map(str, aB.var_names)))
    del aA, aB
    gc.collect()

    SA, vA, cmA, XA, rawA, gA, gpA, sA, _ = load_side(pathA, shared)
    SB, vB, cmB, XB, rawB, gB, gpB, sB, _ = load_side(pathB, shared)
    common = sorted(set(gA) & set(gB))
    ia = {g: i for i, g in enumerate(gA)}
    ib = {g: i for i, g in enumerate(gB)}
    ci = {g: i for i, g in enumerate(common)}
    idxA = np.array([ia[g] for g in common]); idxB = np.array([ib[g] for g in common])
    vAa = np.array([vA[ia[g]] for g in common]); vAa /= np.linalg.norm(vAa)
    vBb = np.array([vB[ib[g]] for g in common]); vBb /= np.linalg.norm(vBb)
    cmAa = np.array([cmA[ia[g]] for g in common]); cmBb = np.array([cmB[ib[g]] for g in common])
    targets = sorted(set(sA) & set(sB) & set(ci))
    if len(targets) < 15:
        raise ValueError("too few shared targets")

    RA, RB = [], []
    for g in targets:
        tcol = ci[g]
        rA = _residual(XA, rawA, g, idxA, cmAa, vAa, tcol)
        rB = _residual(XB, rawB, g, idxB, cmBb, vBb, tcol)
        if rA @ rA == 0 or rB @ rB == 0:
            continue
        RA.append(rA); RB.append(rB)
    RA = np.asarray(RA); RB = np.asarray(RB)
    n = len(RA)
    del XA, XB, SA, SB
    gc.collect()

    def cos_rows(A, B):
        num = np.sum(A * B, axis=1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        v = num / np.maximum(den, 1e-12)
        return float(np.nanmean(v))

    def gene_corr(A, B):
        out = []
        for j in range(A.shape[1]):
            x, y = A[:, j], B[:, j]
            if np.isfinite(x).sum() < 3 or x.std() == 0 or y.std() == 0:
                continue
            out.append(np.corrcoef(x, y)[0, 1])
        out = np.asarray([o for o in out if np.isfinite(o)])
        return float(out.mean()) if out.size else float("nan")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    # centered (remove common target-averaged mode) -> perturbation-specific deviation
    RAc = RA - RA.mean(0); RBc = RB - RB.mean(0)
    out = {
        "pair": os.path.basename(pathA).replace(".h5ad", "") + "->" +
                os.path.basename(pathB).replace(".h5ad", ""),
        "n_common_genes": len(common), "n_targets": n,
        "cos_resid": cos_rows(RA, RB),
        "cos_resid_null": cos_rows(RA, RB[perm]),
        "cos_specific": cos_rows(RAc, RBc),
        "cos_specific_null": cos_rows(RAc, RBc[perm]),
        "gene_corr_resid": gene_corr(RA, RB),
        "gene_corr_specific": gene_corr(RAc, RBc),
    }
    return out


def main():
    d = _PERTURB_DATA
    pairs = [
        ("ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
        ("NadigOConner2024_hepg2.h5ad", "NadigOConner2024_jurkat.h5ad"),
        ("TianKampmann2021_CRISPRa.h5ad", "TianKampmann2021_CRISPRi.h5ad"),
        ("NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
        ("NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
        ("NadigOConner2024_hepg2.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
        ("NadigOConner2024_hepg2.h5ad", "ReplogleWeissman2022_rpe1.h5ad"),
        # newly downloaded datasets
        ("XAtlas2025_HCT116_filtered.h5ad", "XAtlas2025_HEK293T_filtered.h5ad"),
        ("NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_K562_gwps_filtered.h5ad"),
        ("ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_K562_gwps_filtered.h5ad"),
        ("NadigOConner2024_hepg2.h5ad", "ReplogleWeissman2022_K562_gwps_filtered.h5ad"),
        ("ReplogleWeissman2022_K562_gwps_filtered.h5ad", "XAtlas2025_HEK293T_filtered.h5ad"),
        ("NadigOConner2024_jurkat.h5ad", "XAtlas2025_HCT116_filtered.h5ad"),
    ]
    keys = ["n_common_genes", "n_targets", "cos_resid", "cos_resid_null", "cos_specific",
            "cos_specific_null", "gene_corr_resid", "gene_corr_specific"]
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    done.add(line.split(",", 1)[0])
    else:
        with open(OUT, "w") as fh:
            fh.write("pair," + ",".join(keys) + "\n")
    for pa, pb in pairs:
        pairname = os.path.basename(pa).replace(".h5ad", "") + "->" + os.path.basename(pb).replace(".h5ad", "")
        if pairname in done:
            print("have", pairname[:55], "(skipped)", flush=True)
            continue
        try:
            o = analyze(os.path.join(d, pa), os.path.join(d, pb))
            with open(OUT, "a") as fh:
                fh.write(o["pair"] + "," + ",".join(str(o[k]) for k in keys) + "\n")
            print("OK {:<50} n={} cos_resid={:.3f} cos_specific={:.3f} (null {:.3f}) "
                  "gene_corr={:.3f}".format(
                      o["pair"][:49], o["n_targets"], o["cos_resid"], o["cos_specific"],
                      o["cos_specific_null"], o["gene_corr_specific"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<40} {}".format(pa[:39], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
