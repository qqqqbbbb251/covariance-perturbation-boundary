import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
residual_reproducibility.py -- Stage 2: is the perturbation-specific residual signal
reproducible, or is it noise?

The forward model fails (RSA~0) but the residual decomposition leaves ~86% of the
variance "unexplained".  This experiment asks whether that residual contains a
*reproducible* perturbation-specific component, using an assumption-light split-half
design: for each perturbation split its cells into two random halves, compute the
mean shift in each half, remove the global axis v (control PC1), and measure the
agreement of the residual (and of the full / shared components) across halves.

If a specific response is real, the two half-shifts agree beyond chance; if it is
noise, agreement is ~0.  Spaces: raw / cpm / pearson.  Output:
../results/residual_reproducibility.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load, to_space  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "residual_reproducibility.csv")

SPACES = ["raw", "cpm", "pearson"]
MAX_PERT = 150
MIN_CELLS = 30
MAX_PERT_CELLS = 500
SEED = 0


def _veccorr(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _gene_corr(R1, R2):
    """Per-gene Pearson correlation across perturbations."""
    out = []
    for g in range(R1.shape[1]):
        a, b = R1[:, g], R2[:, g]
        if np.isfinite(a).sum() < 3 or a.std() == 0 or b.std() == 0:
            continue
        out.append(np.corrcoef(a, b)[0, 1])
    out = np.asarray([x for x in out if np.isfinite(x)])
    if out.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(out.mean()), float(np.median(out)), float(np.mean(out > 0))


def _sim_upper(R):
    """Upper triangle of the cosine-similarity matrix between perturbation shifts."""
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    iu = np.triu_indices(R.shape[0], 1)
    return S[iu]


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    rows = []
    for space in SPACES:
        try:
            ctrl_fn, pert_fn = to_space(X, cidx, nc_all, space)
            Xc = ctrl_fn(X[cidx], nc_all[cidx])
            cmean = Xc.mean(0)
            Z = Xc - cmean
            cov = (Z.T @ Z) / (Z.shape[0] - 1)
            w, V = np.linalg.eigh(cov)
            v = V[:, int(np.argmax(w))]
            del Xc, Z, V, w
            gc.collect()

            D1, D2 = [], []
            tcols = []
            for g in singles[:MAX_PERT]:
                r = np.where(raw == g)[0]
                if len(r) < max(MIN_CELLS, 2):
                    continue
                if len(r) > MAX_PERT_CELLS:
                    r = rng.choice(r, MAX_PERT_CELLS, replace=False)
                r = rng.permutation(r)
                h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
                d1 = pert_fn(X[h1], nc_all[h1]).mean(0) - cmean
                d2 = pert_fn(X[h2], nc_all[h2]).mean(0) - cmean
                if d1 @ d1 == 0 or d2 @ d2 == 0:
                    continue
                D1.append(d1); D2.append(d2)
                tcols.append(idx.get(g, -1))
            if len(D1) < 10:
                continue
            D1 = np.asarray(D1); D2 = np.asarray(D2)
            tcols = np.asarray(tcols)

            # shared component = projection on the global axis v
            S1 = np.outer(D1 @ v, v)
            S2 = np.outer(D2 @ v, v)
            R1 = D1 - S1
            R2 = D2 - S2

            # self-removed residual: also zero the target gene's own coordinate
            R1s = R1.copy(); R2s = R2.copy()
            for i, t in enumerate(tcols):
                if 0 <= int(t) < R1.shape[1]:
                    R1s[i, int(t)] = 0.0
                    R2s[i, int(t)] = 0.0

            # null: break the perturbation pairing for the residual
            perm = rng.permutation(len(R1))
            R2n = R2[perm]
            R2sn = R2s[perm]

            rel_full = _veccorr(D1, D2)
            rel_shared = _veccorr(S1, S2)
            rel_resid = _veccorr(R1, R2)
            rel_resid_null = _veccorr(R1, R2n)
            # perturbation-specific component: remove the common off-axis mode
            rel_resid_spec = _veccorr(R1 - R1.mean(0), R2 - R2.mean(0))
            rel_selfrem = _veccorr(R1s, R2s)
            # reliability of the perturbation-similarity (RSA) structure
            rsa_full = _veccorr(_sim_upper(D1), _sim_upper(D2))
            rsa_resid = _veccorr(_sim_upper(R1), _sim_upper(R2))
            rsa_resid_null = _veccorr(_sim_upper(R1), _sim_upper(R2n))
            rsa_selfrem = _veccorr(_sim_upper(R1s), _sim_upper(R2s))
            rsa_selfrem_null = _veccorr(_sim_upper(R1s), _sim_upper(R2sn))
            # centered (common mode removed) RSA reliability = specific structure only
            rsa_resid_spec = _veccorr(_sim_upper(R1 - R1.mean(0)), _sim_upper(R2 - R2.mean(0)))
            rsa_selfrem_spec = _veccorr(_sim_upper(R1s - R1s.mean(0)),
                                        _sim_upper(R2s - R2s.mean(0)))
            mg, med, fpos = _gene_corr(R1, R2)
            _, _, fpos_null = _gene_corr(R1, R2n)

            rows.append({"dataset": ds, "space": space, "n_perts": len(D1), "n_genes": D1.shape[1],
                         "rel_full": rel_full, "rel_shared": rel_shared, "rel_resid": rel_resid,
                         "rel_resid_null": rel_resid_null, "rel_resid_spec": rel_resid_spec,
                         "rel_selfrem": rel_selfrem,
                         "rsa_full": rsa_full, "rsa_resid": rsa_resid,
                         "rsa_resid_null": rsa_resid_null,
                         "rsa_selfrem": rsa_selfrem, "rsa_selfrem_null": rsa_selfrem_null,
                         "rsa_resid_spec": rsa_resid_spec, "rsa_selfrem_spec": rsa_selfrem_spec,
                         "mean_gene_corr_resid": mg,
                         "median_gene_corr_resid": med, "frac_gene_corr_pos": fpos,
                         "frac_gene_corr_pos_null": fpos_null})
        except Exception as e:
            print("   sub-skip", ds, space, str(e)[:50], flush=True)
        gc.collect()
    del X
    gc.collect()
    return rows


def main():
    d = _PERTURB_DATA
    default = [
        "NormanWeissman2019_filtered.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "n_genes", "rel_full", "rel_shared", "rel_resid", "rel_resid_null",
            "rel_resid_spec", "rel_selfrem", "rsa_full", "rsa_resid", "rsa_resid_null",
            "rsa_selfrem", "rsa_selfrem_null", "rsa_resid_spec", "rsa_selfrem_spec",
            "mean_gene_corr_resid", "median_gene_corr_resid", "frac_gene_corr_pos",
            "frac_gene_corr_pos_null"]
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    p = line.split(",")
                    done.add((p[0], p[1]))
    else:
        with open(OUT, "w") as fh:
            fh.write("dataset,space," + ",".join(keys) + "\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f))
            for r in rows:
                if (r["dataset"], r["space"]) in done:
                    print("have", r["dataset"][:26], r["space"], "(skipped)", flush=True)
                    continue
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["space"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK   {:<22} {:<8} RSArel={:.3f} selfrem={:.3f} "
                      "centered={:.3f} centeredSelfrem={:.3f}".format(
                          r["dataset"][:21], r["space"], r["rsa_resid"], r["rsa_selfrem"],
                          r["rsa_resid_spec"], r["rsa_selfrem_spec"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
