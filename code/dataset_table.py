import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
dataset_table.py

Per-dataset accounting for Table S1: cells, genes, control cells, single and double
perturbations, plus cohort and source.  Reads .h5ad obs/var directly with h5py (so it
does not require anndata/pandas).

The dataset manifest (which datasets are "primary" scPerturb datasets vs "additional"
datasets, and their sources/accessions) is explicit below; every dataset listed here is
included in Table S1.

Usage:
    python dataset_table.py [data_dir]
Output: ../results/dataset_table.csv
"""

import glob
import os
import sys

import h5py
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PERTURB_DATA", _PERTURB_DATA)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "dataset_table.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")

# (dataset, cohort, source) -- order defines Table S1 row order
MANIFEST = [
    ("AissaBenevolenskaya2021", "primary", "scPerturb (Zenodo 13350497)"),
    ("ChangYe2021", "primary", "scPerturb (Zenodo 13350497)"),
    ("DatlingerBock2017", "primary", "scPerturb (Zenodo 13350497)"),
    ("DatlingerBock2021", "primary", "scPerturb (Zenodo 13350497)"),
    ("FrangiehIzar2021_RNA", "primary", "scPerturb (Zenodo 13350497)"),
    ("NadigOConner2024_hepg2", "primary", "scPerturb (Zenodo 13350497)"),
    ("NadigOConner2024_jurkat", "primary", "scPerturb (Zenodo 13350497)"),
    ("NormanWeissman2019_filtered", "primary", "scPerturb (Zenodo 13350497)"),
    ("PapalexiSatija2021_eccite_RNA", "primary", "scPerturb (Zenodo 13350497)"),
    ("PapalexiSatija2021_eccite_arrayed_RNA", "primary", "scPerturb (Zenodo 13350497)"),
    ("ReplogleWeissman2022_K562_essential", "primary", "scPerturb (Zenodo 13350497)"),
    ("ReplogleWeissman2022_rpe1", "primary", "scPerturb (Zenodo 13350497)"),
    ("TianKampmann2019_day7neuron", "primary", "scPerturb (Zenodo 13350497)"),
    ("TianKampmann2019_iPSC", "primary", "scPerturb (Zenodo 13350497)"),
    ("TianKampmann2021_CRISPRa", "primary", "scPerturb (Zenodo 13350497)"),
    ("TianKampmann2021_CRISPRi", "primary", "scPerturb (Zenodo 13350497)"),
    ("ReplogleWeissman2022_K562_gwps_filtered", "additional",
     "Replogle 2022 [5]; CIPHER benchmark (Zenodo 21729034)"),
    ("XAtlas2025_HCT116_filtered", "additional",
     "X-Atlas/Orion [14]; CIPHER benchmark (Zenodo 21729034)"),
    ("XAtlas2025_HEK293T_filtered", "additional",
     "X-Atlas/Orion [14]; CIPHER benchmark (Zenodo 21729034)"),
    ("proper_filtered", "additional",
     "CIPHER benchmark, Zenodo 21729034 (lab-generated; no upstream publication)"),
    ("schemidt_etal_2022_crispra_perturbseq", "additional",
     "Schmidt 2022 [16] (GEO GSE190604); CIPHER benchmark (Zenodo 21729034)"),
    ("akana_etal_2026_crispra_perturbseq", "additional",
     "Akana 2026 [17] (figshare 10.6084/m9.figshare.31119196); CIPHER benchmark (Zenodo 21729034)"),
]


def _decode(arr):
    return np.array([v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                     for v in np.asarray(arr)], dtype=object)


def _read_col(obs, name):
    if name not in obs:
        return None
    g = obs[name]
    if isinstance(g, h5py.Group):  # categorical
        cats = _decode(g["categories"][()])
        codes = np.asarray(g["codes"][()])
        return np.array([cats[c] if c >= 0 else "" for c in codes], dtype=object)
    return _decode(g[()])


def count(path):
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group):
            shape = X.attrs.get("shape")
            if shape is None:
                shape = (len(X["indptr"]) - 1, None)
        else:
            shape = X.shape
        n_obs = int(shape[0])
        n_vars = int(shape[1]) if shape[1] is not None else int(f["var"]["_index"].shape[0])
        obs = f["obs"]
        pert = _read_col(obs, "perturbation")
        nperts = _read_col(obs, "nperts")
        if pert is None:
            raise ValueError("no 'perturbation' column")
        if nperts is None:
            nperts_i = np.ones(n_obs)
        else:
            nperts_i = np.array([float(x) if x != "" else np.nan for x in nperts])
        is_ctrl = np.array([any(t in p.lower() for t in CONTROL_TOKENS)
                            for p in pert]) | (nperts_i == 0)
        base = pert[(~is_ctrl) & (nperts_i == 1)]
        if len(np.unique(base)) < 10:
            base = pert[~is_ctrl]
        uq, cn = np.unique(base, return_counts=True)
        n_single = int(sum(1 for c in cn if c >= 30))
        n_double = int(len(np.unique(pert[nperts_i == 2])))
        return n_obs, n_vars, int(is_ctrl.sum()), n_single, n_double


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for name, cohort, source in MANIFEST:
        path = os.path.join(DATA, name + ".h5ad")
        if not os.path.exists(path):
            print("MISSING", path, flush=True)
            continue
        try:
            cells, genes, ctrl, single, double = count(path)
            rows.append((name, cohort, source, cells, genes, ctrl, single, double))
            print("{:<44}{:>9}{:>8}{:>10}{:>9}{:>9}".format(
                name[:43], cells, genes, ctrl, single, double), flush=True)
        except Exception as e:  # noqa: BLE001
            print("{:<44} ERROR {}".format(name[:43], str(e)[:40]), flush=True)
    import csv as _csv
    with open(OUT, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["dataset", "cohort", "source", "cells", "genes",
                    "control_cells", "single_perturbations", "double_perturbations"])
        for r in rows:
            w.writerow(r)
    print("\ntotal datasets: {}  wrote {}".format(len(rows), OUT))


if __name__ == "__main__":
    main()
