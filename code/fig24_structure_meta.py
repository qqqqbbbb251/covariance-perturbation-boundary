"""
fig24_structure_meta.py -- structure across the full dataset panel.

A. centred, self-removed split-half RSA reliability per dataset (raw space).
B. organisation of the specific structure by PPI / Reactome / GO (permutation p).
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
                 ("_essential", ""), ("_filtered", ""), ("_perturbseq", ""),
                 ("akana_etal_2026_", "akana-"), ("schemidt_etal_2022_", "schemidt-"),
                 ("_RNA", "")]:
        s = s.replace(a, b)
    return s[:16]


def main():
    rr = [r for r in read("residual_reproducibility.csv") if r["space"] == "raw"]
    rr.sort(key=lambda r: float(r["rsa_selfrem_spec"]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(rr))
    ax[0].bar(x, [float(r["rsa_selfrem_spec"]) for r in rr], color="#2b6cb0")
    ax[0].axhline(0.2, color="r", ls="--", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels([short(r["dataset"]) for r in rr], rotation=60, ha="right", fontsize=8)
    ax[0].set_ylabel("centred+self-removed RSA reliability (raw)")
    ax[0].set_title("A. Reproducible specific structure across datasets\n(red line = 0.2)")

    ppi = {r["dataset"]: float(r["p_perm"]) for r in read("specific_network.csv") if r["network"] == "PPI"}
    pw = {}
    for r in read("specific_pathway.csv"):
        if "Reactome" in r["network"]:
            pw.setdefault(r["dataset"], {})["Reactome"] = float(r["p_perm"])
        if r["network"].startswith("GO_Bio"):
            pw.setdefault(r["dataset"], {})["GO"] = float(r["p_perm"])
    ds = sorted(ppi.keys())
    nets = ["PPI", "Reactome", "GO"]
    x2 = np.arange(len(ds)); w = 0.26
    for k, net in enumerate(nets):
        vals = []
        for d in ds:
            if net == "PPI":
                p = ppi.get(d, np.nan)
            else:
                p = pw.get(d, {}).get(net, np.nan)
            vals.append(-np.log10(p) if p == p and p > 0 else 0.0)
        ax[1].bar(x2 + (k - 1) * w, vals, w, label=net)
    ax[1].axhline(-np.log10(0.05), color="r", ls="--", lw=1, label="p=0.05")
    ax[1].set_xticks(x2); ax[1].set_xticklabels([short(d) for d in ds], rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("-log10 permutation p")
    ax[1].set_title("B. Specific structure is organised by PPI / pathways / GO")
    ax[1].legend(fontsize=8)

    fig.suptitle("The perturbation-specific signal is real and network/pathway-organised in the strong datasets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "fig24_structure_meta.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, "datasets", len(rr))


if __name__ == "__main__":
    main()
