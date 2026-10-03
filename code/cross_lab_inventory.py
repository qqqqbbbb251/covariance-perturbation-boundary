import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cross_lab_inventory.py -- which local datasets can serve as cross-lab / cross-modality
validation pairs?

For every .h5ad in the data dir, extract the single-gene perturbation targets (and, when
present, the cell line and perturbation modality from obs), then report which dataset
pairs share enough target genes to test whether the perturbation-specific residual is
reproducible across labs / cell lines / modalities.

Outputs: ../results/cross_lab_inventory.csv, ../results/cross_lab_pairs.csv
"""

import os
import re
import sys
import itertools

import numpy as np
import pandas as pd
import anndata as ad

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)
from cipher.data import infer_target_gene  # noqa: E402

DATA = _PERTURB_DATA
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
MIN_CELLS = 30

PERT_COLS = ["perturbation", "gene", "target", "target_gene", "gene_name", "guide_target",
             "condition", "sgRNA", "guide_id"]


def pick_pert_col(obs):
    for c in PERT_COLS:
        if c in obs.columns:
            u = obs[c].astype(str)
            if u.nunique() > 5:
                return c
    return None


def dataset_meta(name):
    lab = re.sub(r"^(.*?)(Weissman|Regev|Izar|Kampmann|Bock|Shendure|Satija|Zhang|Trapnell|"
                 r"Benevolenskaya|Ye|OConner|Adamson|Dixit|Norman|Tian|Frangieh|Nadig|Replogle|"
                 r"Datlinger|Gasperini|Papalexi|Joung|Srivatsan|Aissa|Chang).*$",
                 r"\2", name) if True else ""
    return lab


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".h5ad"))
    sets = {}
    rows = []
    for f in files:
        try:
            a = ad.read_h5ad(os.path.join(DATA, f), backed="r")
            obs = a.obs
            geneset = set(map(str, a.var_names))
            pc = pick_pert_col(obs)
            if pc is None:
                a.file.close()
                continue
            labels = obs[pc].astype(str)
            vc = labels.value_counts()
            targets, total = set(), 0
            for lab, n in vc.items():
                if n < MIN_CELLS:
                    continue
                g, _ = infer_target_gene(str(lab), geneset)
                if g:
                    targets.add(g)
                    total += int(n)
            cell_line = ""
            for c in ("cell_line", "tissue_type", "celltype"):
                if c in obs.columns:
                    cell_line = str(obs[c].astype(str).mode().iloc[0])
                    break
            modality = ""
            for c in ("perturbation_type", "perturbation_type_2", "library_preparation_protocol"):
                if c in obs.columns:
                    modality = str(obs[c].astype(str).mode().iloc[0])
                    break
            n_cells = int(a.n_obs)
            a.file.close()
            sets[f] = targets
            rows.append({"dataset": f.replace(".h5ad", ""), "pert_col": pc,
                         "n_cells": n_cells, "n_targets": len(targets),
                         "cell_line": cell_line, "modality": modality})
            print("OK  {:<46} targets={:<5} line={:<14} mod={}".format(
                f[:45], len(targets), cell_line[:14], modality[:24]), flush=True)
        except Exception as e:
            print("SKIP {:<44} {}".format(f[:43], str(e)[:50]), flush=True)

    inv = pd.DataFrame(rows).sort_values("n_targets", ascending=False)
    inv.to_csv(os.path.join(RESULTS, "cross_lab_inventory.csv"), index=False)

    names = list(sets)
    pairs = []
    for i, j in itertools.combinations(names, 2):
        sh = sets[i] & sets[j]
        if len(sh) >= 20:
            pairs.append({"a": i.replace(".h5ad", ""), "b": j.replace(".h5ad", ""),
                          "n_shared": len(sh)})
    pdf = pd.DataFrame(pairs).sort_values("n_shared", ascending=False)
    pdf.to_csv(os.path.join(RESULTS, "cross_lab_pairs.csv"), index=False)
    print("\ntop cross-dataset shared-target pairs:")
    print(pdf.head(25).to_string(index=False))
    print("\nwrote cross_lab_inventory.csv (%d datasets) and cross_lab_pairs.csv (%d pairs)"
          % (len(inv), len(pdf)))


if __name__ == "__main__":
    main()
