"""
fig29_guide_target.py -- guide vs target decoupling.

A. per dataset: across-guide agreement, within-guide split-half reliability and
   target-common split-half reliability (guide_target_partition.csv).
B. cross-dataset specific-residual transfer of the target estimate built from
   1 / 2 / all guides, for the multi-guide target subset (guide_target_transfer.csv).
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
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def short(s):
    for a, b in [("ReplogleWeissman2022_", "Repl."), ("NadigOConner2024_", "Nadig."),
                 ("Weissman2019_filtered", "Norman"), ("_essential", ""), ("_filtered", ""),
                 ("_gwps", "-gwps")]:
        s = s.replace(a, b)
    return s[:18]


def main():
    p = read("guide_target_partition.csv")
    p.sort(key=lambda r: float(r["guide_agreement"]))
    t = read("guide_target_transfer.csv")
    multi = [r for r in t if r["subset"] == "multiguide"]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(p)); w = 0.26
    ax[0].bar(x - w, [float(r["guide_agreement"]) for r in p], w, label="across-guide agreement", color="#c53030")
    ax[0].bar(x, [float(r["within_guide_reliab"]) for r in p], w, label="within-guide reliability", color="#2b6cb0")
    ax[0].bar(x + w, [float(r["target_mean_reliab"]) for r in p], w, label="target-common reliability", color="#38a169")
    ax[0].set_xticks(x); ax[0].set_xticklabels([short(r["dataset"]) for r in p], rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("cosine")
    ax[0].set_title("A. Per-guide residuals disagree; each guide is itself reproducible")
    ax[0].legend(fontsize=8)

    pairs = sorted(set(r["pair"] for r in multi))
    ests = ["guide1", "guide2", "guideAll"]
    x2 = np.arange(len(pairs)); w2 = 0.25
    for k, e in enumerate(ests):
        vals = []
        for pr in pairs:
            v = next((float(r["cos_specific"]) for r in multi if r["pair"] == pr and r["estimate"] == e), np.nan)
            vals.append(v)
        ax[1].bar(x2 + (k - 1) * w2, vals, w2, label=e)
    for k, pr in enumerate(pairs):
        nl = next((float(r["cos_null"]) for r in multi if r["pair"] == pr and r["estimate"] == "guideAll"), np.nan)
        ax[1].plot([k - 1.5 * w2, k + 1.5 * w2], [nl, nl], "k--", lw=1)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels([short(pr.replace("->", "\n->")) for pr in pairs], fontsize=7)
    ax[1].set_ylabel("cross-dataset cos_specific (multi-guide targets)")
    ax[1].set_title("B. Averaging guides recovers the target-level signal\n(dashed = target-shuffled null)")
    ax[1].legend(fontsize=8)

    fig.suptitle("Target/guide decoupling: the transferable structure is target-level, single guides are noisy",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "fig29_guide_target.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, "datasets", len(p), "pairs", len(pairs))


if __name__ == "__main__":
    main()
