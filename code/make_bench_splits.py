"""
make_bench_splits.py -- build a self-consistent CIPHER-benchmark dataset locally.

The CIPHER benchmark expects <splits-dir>/<dataset>/{filtered.h5ad, control_idx.npy,
train_idx.npy, test_idx.npy}.  The repo does not ship the preprocessing that produced
them, so we regenerate a consistent set from the raw Zenodo h5ad:
  * filter genes to mean expression >= 1 (over all cells), always keeping perturbation
    targets (mirrors src/preprocess.get_data);
  * rename obs 'perturbation' -> 'condition' and add 'cell_type';
  * split *perturbations* into train/test (80/20), cells of each perturbation go to the
    matching index; control cells form control_idx;
  * X stays raw counts (the drivers copy X to layers['counts'] then normalize).

Usage:
    python make_bench_splits.py <raw.h5ad> <out_dir> [dataset_name]
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CTRL = ("control", "ctrl", "ntc", "non-targeting", "nontargeting", "negative", "neg")
TEST_FRAC = 0.2
SEED = 0


def is_ctrl(s):
    s = str(s).strip().lower()
    return s in CTRL or any(t in s for t in ("control", "ctrl", "non-targeting", "nontargeting"))


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(src).replace(".h5ad", "")
    os.makedirs(out_dir, exist_ok=True)

    a = ad.read_h5ad(src)
    a.var_names_make_unique()
    pert = a.obs["perturbation"].astype(str)
    a.obs["condition"] = pert.values
    if "cell_type" not in a.obs.columns:
        a.obs["cell_type"] = name.replace("_", "-")

    # gene filter: mean >= 1 keeping perturbation targets
    Xc = a.X
    mean = np.asarray(Xc.mean(axis=0)).ravel()
    keep = mean >= 1.0
    targets = set()
    for s in pert.unique():
        for p in str(s).replace("|", "+").replace("_", "+").split("+"):
            if p in set(a.var_names):
                targets.add(p)
    keep |= np.array([g in targets for g in a.var_names])
    a = a[:, keep].copy()
    a.var["gene_name"] = np.asarray(a.var_names).astype(str)

    ctrl_mask = np.array([is_ctrl(x) for x in pert.values])
    cidx = np.where(ctrl_mask)[0]
    perts = np.array([x for x in pert.unique() if not is_ctrl(x)])
    rng = np.random.default_rng(SEED)
    rng.shuffle(perts)
    n_test = max(1, int(round(TEST_FRAC * len(perts))))
    test_perts, train_perts = set(perts[:n_test]), set(perts[n_test:])

    train_idx = np.where(np.isin(pert.values, list(train_perts)))[0]
    test_idx = np.where(np.isin(pert.values, list(test_perts)))[0]

    o = os.path.join(out_dir, "filtered.h5ad")
    a.write_h5ad(o)
    np.save(os.path.join(out_dir, "control_idx.npy"), cidx)
    np.save(os.path.join(out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)
    print("wrote", out_dir, "shape", a.shape,
          "| ctrl", len(cidx), "train", len(train_idx), "test", len(test_idx),
          "| train_perts", len(train_perts), "test_perts", len(test_perts))


if __name__ == "__main__":
    main()
