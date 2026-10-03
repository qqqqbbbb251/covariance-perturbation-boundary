"""
specificity_positive_control.py

Power / sensitivity control for the "no gene-specific advantage" test.

The concern is "you detect no gene-specific advantage, but maybe your test is too
blunt".  This script answers that directly.  For a grid of gene-specific strengths
alpha it builds a covariance whose columns are the shared direction v plus a
tunable gene-specific component alpha * u_g, generates responses from those
columns, and runs the exact real test (full = correct column vs a matched random
column, squared cosine, paired Wilcoxon).  Power is the fraction of independent
replicates whose paired test is significant at 0.05; at alpha = 0 (all columns
collinear, the real-data regime) it must be at the nominal 0.05 level.

Outputs:
  ../results/specificity_positive_control.csv
  ../figures/fig14_specificity_control.png
"""

import csv
import math
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(RESULTS, "specificity_positive_control.csv")

N_GENES = 2000
N_PERT = 150
N_DRAW = 10
N_REPS = 100
SIGMA = 1.2          # noise level -> cos^2(full) ~ 0.4 at alpha = 0, like the real data
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def rankdata(a):
    a = np.asarray(a, float)
    s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def wilcoxon(x):
    d = np.asarray(x, float)
    d = d[~np.isnan(d)]; d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    r = rankdata(np.abs(d)); W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((W - mu) / sig) / math.sqrt(2))


def one_replicate(rng, alpha):
    v = rng.standard_normal(N_GENES)
    v /= np.linalg.norm(v)
    U = rng.standard_normal((N_GENES, N_PERT))
    U -= np.outer(v, v @ U)
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    C = v[:, None] + alpha * U
    noise = rng.standard_normal((N_GENES, N_PERT))
    noise /= np.linalg.norm(noise, axis=0, keepdims=True)
    D = C + SIGMA * noise
    norms = np.linalg.norm(C, axis=0)

    full = np.empty(N_PERT); rand = np.empty(N_PERT)
    for g in range(N_PERT):
        full[g] = cos2(D[:, g], C[:, g])
        pool = np.argsort(np.abs(norms - norms[g]))[:50]
        pool = pool[pool != g]
        draws = rng.choice(pool, size=min(N_DRAW, len(pool)), replace=False)
        rand[g] = float(np.mean([cos2(D[:, g], C[:, j]) for j in draws]))
    return full, rand


def main():
    rng = np.random.default_rng(0)
    rows = []
    for alpha in ALPHAS:
        full_m, rand_m, diff_m, ps = [], [], [], []
        for _ in range(N_REPS):
            full, rand = one_replicate(rng, alpha)
            d = full - rand
            full_m.append(float(full.mean())); rand_m.append(float(rand.mean()))
            diff_m.append(float(d.mean())); ps.append(wilcoxon(d))
        diff_m = np.asarray(diff_m)
        ps = np.asarray(ps, float)
        power = float(np.mean(ps < 0.05))
        rows.append({"alpha": alpha, "full": float(np.mean(full_m)), "random": float(np.mean(rand_m)),
                     "diff": float(diff_m.mean()),
                     "lo": float(np.percentile(diff_m, 2.5)), "hi": float(np.percentile(diff_m, 97.5)),
                     "p_median": float(np.nanmedian(ps)), "power": power})
        print("alpha={:<4} full={:.3f} random={:.3f} diff={:+.4f} [{:+.4f},{:+.4f}] p_med={:.2g} power={:.2f}".format(
            alpha, rows[-1]["full"], rows[-1]["random"], rows[-1]["diff"],
            rows[-1]["lo"], rows[-1]["hi"], rows[-1]["p_median"], power), flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "full", "random", "diff", "lo", "hi", "p_median", "power"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.6g}".format(vv) if isinstance(vv, float) else vv) for k, vv in r.items()})
    print("wrote", OUT)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        a = [r["alpha"] for r in rows]
        d = [r["diff"] for r in rows]
        lo = [r["lo"] for r in rows]; hi = [r["hi"] for r in rows]
        pw = [r["power"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].errorbar(a, d, yerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)],
                       fmt="o-", capsize=3, color="#2b6cb0")
        ax[0].axhline(0, color="k", lw=0.9)
        ax[0].set_xlabel("injected gene-specific strength $\\alpha$")
        ax[0].set_ylabel("full $-$ matched random ($\\cos^2$)")
        ax[0].set_title("A. Test recovers an injected gene-specific advantage")
        ax[1].plot(a, pw, "s-", color="#276749")
        ax[1].axhline(0.05, color="gray", ls="--", lw=0.9, label="nominal 0.05")
        ax[1].set_xlabel("injected gene-specific strength $\\alpha$")
        ax[1].set_ylabel("power (fraction of replicates, $p<0.05$)")
        ax[1].set_ylim(0, 1.02); ax[1].legend()
        ax[1].set_title("B. Power of the specificity test")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig14_specificity_control.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("wrote", p)
    except Exception as e:
        print("figure skipped:", str(e)[:60])


if __name__ == "__main__":
    main()
