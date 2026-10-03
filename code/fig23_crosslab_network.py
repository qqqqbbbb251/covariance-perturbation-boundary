"""
fig23_crosslab_network.py -- summary of the extended validation.

A. same-target specific-residual cosine across dataset pairs (same-lab, cross-lab,
   cross-modality), with the shuffled-target null.
B. PPI organisation: residual cosine for PPI-edge target pairs vs non-edge pairs.
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
                 ("Weissman2019_filtered", "Norman"), ("Kampmann2021_", "Tian "),
                 ("_essential", ""), ("_filtered", ""), ("2025_", ""), (".h5ad", "")]:
        s = s.replace(a, b)
    s = s.replace("Repl.K562_gwps", "Repl.K562-gwps")
    return s


def relation(pair):
    if "TianKampmann2021_CRISPRa" in pair and "CRISPRi" in pair:
        return "cross-modality"
    if "Nadig" in pair and "Replogle" in pair:
        return "cross-lab"
    if "XAtlas" in pair and "Nadig" in pair:
        return "cross-lab"
    return "same-lab"


def main():
    cds = read("cross_dataset_specific.csv")
    cds.sort(key=lambda r: float(r["cos_specific"]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    cols = {"cross-lab": "#276749", "same-lab": "#2b6cb0", "cross-modality": "#c53030"}
    names = [short(r["pair"]) for r in cds]
    vals = [float(r["cos_specific"]) for r in cds]
    nulls = [float(r["cos_specific_null"]) for r in cds]
    rels = [relation(r["pair"]) for r in cds]
    x = np.arange(len(cds)); w = 0.38
    ax[0].bar(x - w / 2, vals, w, color=[cols[k] for k in rels], label="same target")
    ax[0].bar(x + w / 2, nulls, w, color="#cbd5e0", label="shuffled targets")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n[:26] for n in names], rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("specific-residual cosine")
    ax[0].set_title("A. Specific structure transfers across labs / cell lines / libraries\n(green = cross-lab, blue = same-lab, red = cross-modality)")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=cols[k], label=k) for k in cols] +
                 [Patch(color="#cbd5e0", label="shuffled")], fontsize=8)

    sn = [r for r in read("specific_network.csv") if r["network"] == "PPI"]
    sn.sort(key=lambda r: float(r["sim_edge"]))
    names2 = [short(r["dataset"]) for r in sn]
    e = [float(r["sim_edge"]) for r in sn]
    n = [float(r["sim_none"]) for r in sn]
    p = [float(r["p_perm"]) for r in sn]
    x2 = np.arange(len(sn)); w2 = 0.38
    ax[1].bar(x2 - w2 / 2, e, w2, label="PPI edge pair", color="#276749")
    ax[1].bar(x2 + w2 / 2, n, w2, label="non-edge pair", color="#a0aec0")
    ax[1].set_xticks(x2); ax[1].set_xticklabels(names2, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("residual cosine")
    ax[1].set_title("B. PPI-connected targets have more similar specific residuals")
    for xi, (ee, pp) in enumerate(zip(e, p)):
        ax[1].text(xi, max(ee, n[xi]) + 0.01, "p=%.3f" % pp, ha="center", fontsize=7)
    ax[1].legend(fontsize=8)

    fig.suptitle("Extended validation: real, transferable, network-organised perturbation-specific structure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "fig23_crosslab_network.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, "npairs=", len(cds), "nds=", len(sn))


if __name__ == "__main__":
    main()
