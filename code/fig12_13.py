"""
fig12_13.py

fig12: matched random-column comparison
  A. per-dataset full vs matched-random-column R2 (grouped bars)
  B. per-perturbation scatter full vs matched-random (pooled) with y=x

fig13: per-dataset effects and the weak Pearson-space signal
  A. forest plot of per-dataset median (full-global) and (full-random) with IQR,
     mixed-model fixed means overlaid
  B. Pearson space: full vs matched-random vs negative control per dataset
     (BH-significant datasets marked)

Outputs: ../figures/fig12_random_column.png, ../figures/fig13_effects.png
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def fig12():
    rows = [r for r in read("random_column_stats.csv") if r["full_mean"] not in ("", "nan")]
    if not rows:
        print("fig12: no random_column_stats.csv yet")
        return
    rows.sort(key=lambda r: fnum(r["full_mean"]))
    names = [r["dataset"][:14] for r in rows]
    full = np.array([fnum(r["full_mean"]) for r in rows])
    rand = np.array([fnum(r["rand_mean"]) for r in rows])
    x = np.arange(len(rows)); w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.bar(x - w / 2, full, w, label="perturbed column", color="#2b6cb0")
    ax.bar(x + w / 2, rand, w, label="matched random column", color="#dd6b20")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean $R^2$ (raw)"); ax.legend()
    ax.set_title("A. Matched random column is as good as the perturbed column")

    ax = axes[1]
    z = None
    npz = os.path.join(RES, "random_column_dist.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        fx, ry = [], []
        for k in z.files:
            if k.endswith("_full"):
                base = k[:-5]
                if base + "_rand" in z.files:
                    fx.append(z[k]); ry.append(z[base + "_rand"])
        if fx:
            fx = np.concatenate(fx); ry = np.concatenate(ry)
            ax.scatter(fx, ry, s=10, alpha=0.4, color="#4a5568")
            lo = min(fx.min(), ry.min()); hi = max(fx.max(), ry.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            r = np.corrcoef(fx, ry)[0, 1]
            ax.set_xlabel("perturbed column $R^2$"); ax.set_ylabel("matched random column $R^2$")
            ax.set_title("B. Per-perturbation agreement (pooled, r = {:.2f})".format(r))
    fig.tight_layout()
    p = os.path.join(FIG, "fig12_random_column.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig13():
    summ = [r for r in read("per_dataset_summary.csv") if r.get("dFG_median") not in ("", "nan", None)]
    pear = [r for r in read("pearson_stats.csv") if r["full_mean"] not in ("", "nan")]
    mixed = {r["model"]: r for r in read("mixed_effects.csv")} if os.path.exists(
        os.path.join(RES, "mixed_effects.csv")) else {}

    ncol = 2 if pear else 1
    fig, axes = plt.subplots(1, ncol, figsize=(13 if ncol == 2 else 7, 5.5))
    if ncol == 1:
        axes = [axes]

    ax = axes[0]
    if summ:
        summ.sort(key=lambda r: fnum(r["dFG_median"]))
        names = [r["dataset"][:14] for r in summ]
        y = np.arange(len(summ))
        dfg = np.array([fnum(r["dFG_median"]) for r in summ])
        ifg = np.array([fnum(r["dFG_iqr"]) for r in summ])
        dfr = np.array([fnum(r["dFR_median"]) for r in summ])
        ifr = np.array([fnum(r["dFR_iqr"]) for r in summ])
        ax.errorbar(dfg, y + 0.15, xerr=ifg / 2, fmt="o", color="#2b6cb0",
                    label="full $-$ global", capsize=2, ms=4)
        ax.errorbar(dfr, y - 0.15, xerr=ifr / 2, fmt="s", color="#dd6b20",
                    label="full $-$ random", capsize=2, ms=4)
        ax.axvline(0, color="k", lw=0.9)
        if "full_minus_global" in mixed:
            ax.axvline(fnum(mixed["full_minus_global"]["estimate"]), color="#2b6cb0",
                       ls="--", lw=1)
        if "full_minus_random" in mixed:
            ax.axvline(fnum(mixed["full_minus_random"]["estimate"]), color="#dd6b20",
                       ls="--", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("median $\\Delta R^2$ (IQR/2); dashed = mixed-model mean")
        ax.set_title("A. Full model is not better than global/random")
        ax.legend(fontsize=8)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "no per_dataset_summary.csv yet", ha="center")

    if pear:
        ax = axes[1]
        pear.sort(key=lambda r: fnum(r["full_mean"]))
        names = [r["dataset"][:14] for r in pear]
        x = np.arange(len(pear)); w = 0.27
        full = np.array([fnum(r["full_mean"]) for r in pear])
        rand = np.array([fnum(r["rand_mean"]) for r in pear])
        neg = np.array([fnum(r["negctl_mean"]) for r in pear])
        sig = [fnum(r.get("p_full_rand_BH", "nan")) < 0.05 for r in pear]
        ax.bar(x - w, full, w, label="perturbed", color="#2b6cb0")
        ax.bar(x, rand, w, label="matched random", color="#dd6b20")
        ax.bar(x + w, neg, w, label="negative control", color="#a0aec0")
        for xi, s in zip(x, sig):
            if s:
                ax.text(xi - w, full[list(x).index(xi)] + 0.002, "*", ha="center", fontsize=12)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("$R^2$ (Pearson residuals)")
        ax.set_title("B. Weak, inconsistent Pearson-space signal (* BH<0.05)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(FIG, "fig13_effects.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig12()
    fig13()
