import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
scan_final.py

Unified scan with gene-name parsing so guide-level datasets (DatlingerBock,
Papalexi, Adamson) can be included alongside the clean ones.

Same metrics as scan_all.py, written to scan_results.csv.
"""

import argparse
import glob
import os
import re

import numpy as np
import scipy.sparse as sp
import anndata as ad


CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
PREFIXES = r"^(tcrlibrary|grna|sgrna|shrna|crispr|sh|si)[_\-]"


def parse_gene(label):
    s = re.sub(PREFIXES, "", label, flags=re.I)
    parts = [p for p in re.split(r"[_\-\s]+", s) if p]
    g = parts[0] if parts else s
    g = re.sub(r"g\d+$", "", g)          # IFNGR2g1 -> IFNGR2
    return g


def is_control(label):
    l = label.lower()
    return any(t in l for t in CONTROL_TOKENS)


def to_space(X, ncounts, space):
    if space == "raw":
        return X
    inv = (1e4 / np.maximum(ncounts, 1.0)).astype(np.float32)
    return (sp.diags(inv) @ X).tocsr()


def fit_r2(sig, y):
    d = float(np.dot(sig, sig))
    if d == 0:
        return float("nan"), 0.0
    a = float(np.dot(sig, y) / d)
    pred = a * sig
    ss = float(np.sum(y ** 2))
    return (1.0 - float(np.sum((y - pred) ** 2)) / ss if ss else float("nan")), a


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def scan(path, n_top=2000, max_pert=150, min_cells=30):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts = obs["ncounts"].values.astype(float)
    var_ncounts = a.var["ncounts"].values.astype(float)
    var_names = np.asarray(a.var_names)

    ctrl = np.array([is_control(p) for p in raw])
    gene = np.array(["control" if c else parse_gene(p) for p, c in zip(raw, ctrl)])

    # singles: prefer nperts==1 non-control, else all non-control
    n1 = (~ctrl) & (nperts == 1)
    base = gene[n1] if len(np.unique(gene[n1])) >= 10 else gene[~ctrl]
    uniq, cnt = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uniq, cnt) if c >= min_cells and g != "control"]
    rng = np.random.default_rng(0)
    if len(singles) > max_pert:
        singles = list(rng.choice(singles, max_pert, replace=False))
    if len(singles) < 10 or int(ctrl.sum()) < 100:
        raise ValueError("too few usable perturbations/controls")

    selected = set(np.argsort(-var_ncounts)[:n_top].tolist())
    pos = {g: i for i, g in enumerate(var_names)}
    for g in singles:
        if g in pos:
            selected.add(pos[g])
    gi = np.array(sorted(selected))
    genes = var_names[gi]
    Xm = a[:, gi].to_memory().X
    X_raw = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    out = {"cells": X_raw.shape[0], "genes": len(genes),
           "n_ctrl": int(ctrl.sum()), "n_singles": len(singles)}
    for space in ["raw", "cpm"]:
        X = to_space(X_raw, ncounts, space)
        Xc = np.asarray(X[ctrl].todense(), dtype=np.float64)
        cmean = Xc.mean(0)
        cov = np.cov(Xc, rowvar=False)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)]; v = v / np.linalg.norm(v)
        out["global_frac_" + space] = float(w.max() / w.sum())
        del Xc
        deltas = np.atleast_2d(np.asarray(
            [np.asarray(X[gene == g].todense(), dtype=np.float64).mean(0) - cmean
             for g in singles]))
        n = len(singles)
        common = (deltas.sum(0)[None, :] - deltas) / max(n - 1, 1)
        specific = deltas - common
        sorth = specific - np.outer(specific @ v, v)
        full, spec, oc = [], [], []
        for i, g in enumerate(singles):
            sig = cov[:, idx[g]]
            r_f, _ = fit_r2(sig, deltas[i])
            r_s, a_s = fit_r2(sig, specific[i])
            full.append(r_f); spec.append(r_s)
            oc.append(pearson(sorth[i], a_s * sig))
        out["full_r2_" + space] = float(np.nanmean(full))
        out["spec_r2_" + space] = float(np.nanmean(spec))
        out["orth_corr_" + space] = float(np.nanmean(oc))
        del X, deltas, specific, sorth
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=_PERTURB_DATA)
    ap.add_argument("--out", default="scan_results.csv")
    ap.add_argument("--n-top", type=int, default=2000)
    args = ap.parse_args()
    cols = ["dataset", "cells", "genes", "n_ctrl", "n_singles",
            "global_frac_raw", "global_frac_cpm", "full_r2_raw", "full_r2_cpm",
            "spec_r2_raw", "spec_r2_cpm", "orth_corr_raw", "orth_corr_cpm"]
    with open(args.out, "w") as fh:
        fh.write(",".join(cols) + "\n")
    for f in sorted(glob.glob(os.path.join(args.dir, "*.h5ad"))):
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            r = scan(f, args.n_top)
            row = [name] + ["{:.4g}".format(r[c]) if isinstance(r[c], float) else str(r[c])
                            for c in cols[1:]]
            with open(args.out, "a") as fh:
                fh.write(",".join(row) + "\n")
            print("OK   {:<44} n_single={:<4} global_raw={:.2f} orth_raw={:.3f} orth_cpm={:.3f}".format(
                name[:43], r["n_singles"], r["global_frac_raw"],
                r["orth_corr_raw"], r["orth_corr_cpm"]))
        except Exception as e:
            print("SKIP {:<44} {}".format(name[:43], str(e)[:40]))


if __name__ == "__main__":
    main()
