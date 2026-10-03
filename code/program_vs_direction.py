import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
program_vs_direction.py -- A1: is the specific structure a single shared program with
per-target amplitude, or genuine per-target directions?

Build split-half residual responses (global axis + self removed), then project out the
population program W_k (top-k SVD of the residual matrix) and ask whether the *leftover*
per-target direction is still reproducible across halves.  If it collapses to 0, the
"specific structure" is just amplitude modulation of a shared program; if it survives,
there are real target-specific directions.

Also splits the residual into a predictable "population" part and a "specific" part and
reports their separate split-half reliability (centered RSA).

Output: ../results/program_vs_direction.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "program_vs_direction.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
KS = [0, 1, 5, 20, 50]
SEED = 0


def _dir_cos(e1, e2):
    n1 = np.linalg.norm(e1, axis=1); n2 = np.linalg.norm(e2, axis=1)
    ok = (n1 > 0) & (n2 > 0)
    if ok.sum() == 0:
        return float("nan")
    c = np.sum(e1[ok] * e2[ok], axis=1) / (n1[ok] * n2[ok])
    return float(np.mean(c))


def _center_rows(X):
    return X - X.mean(0)


def _sim_upper(R):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    return S[np.triu_indices(R.shape[0], 1)]


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()

    D1, D2, tg = [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        r = rng.permutation(r)
        h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
        d1 = np.asarray(X[h1].todense(), dtype=np.float64).mean(0) - cmean
        d2 = np.asarray(X[h2].todense(), dtype=np.float64).mean(0) - cmean
        if d1 @ d1 == 0 or d2 @ d2 == 0:
            continue
        for d in (d1, d2):
            d -= float(d @ v) * v
            d[int(t)] = 0.0
        D1.append(d1); D2.append(d2); tg.append(g)
    D1 = np.asarray(D1); D2 = np.asarray(D2)
    D1 = _center_rows(D1); D2 = _center_rows(D2)
    n = len(D1)

    Rmean = 0.5 * (D1 + D2)
    U, S, Vt = np.linalg.svd(Rmean, full_matrices=False)
    tot = float((D1 * D1).sum() + (D2 * D2).sum())

    rows = []
    for k in KS:
        Wk = Vt[:k].T if k > 0 else np.zeros((D1.shape[1], 0))
        if k > 0:
            E1 = D1 - (D1 @ Wk) @ Wk.T
            E2 = D2 - (D2 @ Wk) @ Wk.T
        else:
            E1, E2 = D1, D2
        explained = 1.0 - float((E1 * E1).sum() + (E2 * E2).sum()) / tot
        # per-target direction reliability (cosine of the two half directions)
        drel = _dir_cos(E1, E2)
        # RSA-style reliability of the leftover similarity structure (centered)
        rrel = _veccorr(_sim_upper(_center_rows(E1)), _sim_upper(_center_rows(E2)))
        # null: shuffle half pairing
        perm = np.random.default_rng(SEED).permutation(n)
        drel_null = _dir_cos(E1, E2[perm])
        rows.append({"dataset": ds, "k": k, "frac_var_explained": explained,
                     "dir_reliab": drel, "dir_reliab_null": drel_null,
                     "rsa_reliab_leftover": rrel})

    # amplitude reliability of the top-1 program
    W1 = Vt[:1].T
    a1 = (D1 @ W1).ravel(); a2 = (D2 @ W1).ravel()
    amp_r = _veccorr(a1, a2)
    # mean |cos(per-target direction, top program)|
    Rn = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-12)
    align = float(np.mean(np.abs(Rn @ W1.ravel())))
    return rows, {"dataset": ds, "amp_reliab_top1": amp_r, "mean_align_top1": align}


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    with open(OUT, "w") as fh:
        fh.write("dataset,k,frac_var_explained,dir_reliab,dir_reliab_null,rsa_reliab_leftover\n")
    amp_rows = []
    for f in files:
        try:
            rows, amp = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write("{dataset},{k},{frac_var_explained:.4f},{dir_reliab:.4f},"
                             "{dir_reliab_null:.4f},{rsa_reliab_leftover:.4f}\n".format(**r))
            amp_rows.append(amp)
            ks = {r["k"]: r for r in rows}
            print("OK {:<24} k0_dir={:.3f} k1_dir={:.3f} k5_dir={:.3f} k20_dir={:.3f} | "
                  "expl(k20)={:.3f} ampRel={:.3f} align={:.3f}".format(
                      amp["dataset"][:23], ks[0]["dir_reliab"], ks[1]["dir_reliab"],
                      ks[5]["dir_reliab"], ks[20]["dir_reliab"], ks[20]["frac_var_explained"],
                      amp["amp_reliab_top1"], amp["mean_align_top1"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    with open(os.path.join(RESULTS, "program_vs_direction_amp.csv"), "w") as fh:
        fh.write("dataset,amp_reliab_top1,mean_align_top1\n")
        for a in amp_rows:
            fh.write("{dataset},{amp_reliab_top1:.4f},{mean_align_top1:.4f}\n".format(**a))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
