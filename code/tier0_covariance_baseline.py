"""
Tier 0: CIPHER-style covariance forward model for perturbation prediction.

Model
-----
    dX = Sigma @ u

  Sigma : gene-gene covariance of UNPERTURBED (control) cells, in the SAME
          expression space as dX.
  u     : perturbation vector. For a single-gene perturbation of gene i,
          constrain u = a * e_i and fit the scalar a by least squares.

Learned from real-data validation on Norman 2019 (see norman_cipher.py):
  * use RAW counts (or CPM) -- NOT log-normalized. Log space destroys the signal.
  * do NOT divide by Sigma[i,i] and do NOT force a sign.
  * R^2 is the natural score.
  * ~250-2000 control cells are enough; more does not help (data is not the limit).
"""

import argparse
import csv

import numpy as np


# --------------------------------------------------------------------------- #
# covariance
# --------------------------------------------------------------------------- #
def center(X):
    return X - X.mean(axis=0, keepdims=True)


def compute_covariance(X, shrinkage=0.0):
    """Gene x gene covariance from a (cells x genes) matrix."""
    Xc = center(X)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    if shrinkage > 0:
        mu = np.trace(cov) / cov.shape[0]
        cov = (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(cov.shape[0])
    return cov


# --------------------------------------------------------------------------- #
# CIPHER forward model
# --------------------------------------------------------------------------- #
def cipher_signature(cov, gene_idx):
    """The predicted response DIRECTION to perturbing gene_idx: Sigma[:, i].

    Magnitude is unknown (it is the fitted scalar a); the direction is what
    matters for comparing / ranking perturbations.
    """
    return cov[:, gene_idx]


def cipher_fit(cov, gene_idx, delta):
    """Least-squares scalar a for u = a * e_i given an observed delta."""
    col = cov[:, gene_idx]
    denom = float(np.dot(col, col))
    return float(np.dot(col, delta) / denom) if denom > 0 else 0.0


def cipher_predict(cov, gene_idx, delta=None, alpha=None):
    """Predicted dX. If delta is given, fit alpha; otherwise use alpha (or 1)."""
    col = cov[:, gene_idx]
    if alpha is None:
        alpha = cipher_fit(cov, gene_idx, delta) if delta is not None else 1.0
    return alpha * col


def r2_score(delta, pred):
    ss = float(np.sum(delta ** 2))
    if ss == 0:
        return float("nan")
    return 1.0 - float(np.sum((delta - pred) ** 2)) / ss


def predict_shift(cov, gene_idx, strength=1.0):
    """DEPRECATED (pre-CIPHER). Kept only for backward compatibility."""
    denom = cov[gene_idx, gene_idx]
    if denom <= 0:
        return np.zeros(cov.shape[0])
    return -strength * cov[:, gene_idx] / denom


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))


def _topk_idx(a, k):
    return set(np.argsort(-np.abs(a))[: min(k, len(a))].tolist())


def _jaccard(s1, s2):
    return 1.0 if not s1 and not s2 else len(s1 & s2) / len(s1 | s2)


def _nanmean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def eval_metrics(true_delta, pred_delta, k=50):
    return {
        "pearson": _pearson(true_delta, pred_delta),
        "spearman": _spearman(true_delta, pred_delta),
        "topk_jaccard": _jaccard(_topk_idx(true_delta, k), _topk_idx(pred_delta, k)),
        "r2": r2_score(true_delta, pred_delta),
    }


# --------------------------------------------------------------------------- #
# synthetic smoke test
# --------------------------------------------------------------------------- #
def make_synthetic(n_cells=3000, n_genes=200, n_factors=20, n_pert=30, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_genes, n_factors)) * (
        rng.random((n_genes, n_factors)) < 0.15
    )
    cov_true = loadings @ loadings.T + 0.5 * np.eye(n_genes)
    X = rng.multivariate_normal(np.zeros(n_genes), cov_true, size=n_cells)
    pert_genes = rng.choice(n_genes, size=n_pert, replace=False)
    true_deltas = {}
    for g in pert_genes:
        base = cov_true[:, g]                       # CIPHER direction
        effect = np.tanh(3.0 * base) / 3.0 + rng.normal(scale=0.05, size=n_genes)
        true_deltas[int(g)] = effect
    return X, pert_genes, true_deltas


def run_evaluation(X, pert_genes, true_deltas, k=50, shrinkage=0.0, verbose=True):
    cov = compute_covariance(X, shrinkage=shrinkage)
    mean_delta = np.mean([true_deltas[g] for g in pert_genes], axis=0)
    zero = np.zeros(X.shape[1])

    methods = {
        "cipher": lambda g: cipher_predict(cov, g, true_deltas[g]),
        "zero": lambda g: zero,
        "mean": lambda g: mean_delta,
    }
    summary = {m: [] for m in methods}
    for g in pert_genes:
        for m, fn in methods.items():
            summary[m].append(eval_metrics(true_deltas[g], fn(g), k))

    if verbose:
        print("\nSynthetic evaluation (n_cells={}, n_genes={}, n_pert={}, k={})".format(
            X.shape[0], X.shape[1], len(pert_genes), k))
        print("{:<10}{:>9}{:>9}{:>13}{:>8}".format("method", "pearson", "spearman", "topk_jac", "R2"))
        print("-" * 50)
        for m in methods:
            rows = summary[m]
            print("{:<10}{:>9.3f}{:>9.3f}{:>13.3f}{:>8.3f}".format(
                m, _nanmean([r["pearson"] for r in rows]),
                _nanmean([r["spearman"] for r in rows]),
                _nanmean([r["topk_jaccard"] for r in rows]),
                _nanmean([r["r2"] for r in rows])))
    return summary


# --------------------------------------------------------------------------- #
# real-data loaders
# --------------------------------------------------------------------------- #
def load_matrix_csv(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        has_labels = bool(header) and header[0].strip() == ""
        gene_names = header[1:] if has_labels else header
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(v) for v in (row[1:] if has_labels else row)])
    return np.asarray(rows, dtype=float), gene_names


def load_perturbations(delta_csv, genes_txt):
    with open(genes_txt) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    deltas, genes = load_matrix_csv(delta_csv)
    return deltas, genes, names


def run_real(baseline_csv, delta_csv, genes_txt, k=50, shrinkage=0.0):
    X, base_genes = load_matrix_csv(baseline_csv)
    deltas, delta_genes, names = load_perturbations(delta_csv, genes_txt)
    if base_genes != delta_genes:
        raise ValueError("baseline and delta gene columns differ")
    gene_to_idx = {g: i for i, g in enumerate(base_genes)}
    cov = compute_covariance(X, shrinkage=shrinkage)

    print("\nReal-data evaluation (n_cells={}, n_genes={}, n_pert={}, k={})".format(
        X.shape[0], X.shape[1], len(names), k))
    print("{:<14}{:>10}{:>10}{:>14}{:>8}".format(
        "perturbed", "pearson", "spearman", "topk_jac", "R2"))
    print("-" * 58)
    agg = []
    for row, name in zip(deltas, names):
        if name not in gene_to_idx:
            continue
        pred = cipher_predict(cov, gene_to_idx[name], row)
        m = eval_metrics(row, pred, k)
        agg.append(m)
        print("{:<14}{:>10.3f}{:>10.3f}{:>14.3f}{:>8.3f}".format(
            name, m["pearson"], m["spearman"], m["topk_jaccard"], m["r2"]))
    if agg:
        print("-" * 58)
        print("{:<14}{:>10.3f}{:>10.3f}{:>14.3f}{:>8.3f}".format(
            "MEAN",
            _nanmean([m["pearson"] for m in agg]),
            _nanmean([m["spearman"] for m in agg]),
            _nanmean([m["topk_jaccard"] for m in agg]),
            _nanmean([m["r2"] for m in agg])))


def main():
    ap = argparse.ArgumentParser(description="Tier 0 CIPHER covariance baseline")
    ap.add_argument("--baseline")
    ap.add_argument("--perturbations")
    ap.add_argument("--pert-genes")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--shrinkage", type=float, default=0.0)
    args = ap.parse_args()

    if args.baseline:
        run_real(args.baseline, args.perturbations, args.pert_genes,
                 k=args.k, shrinkage=args.shrinkage)
    else:
        X, pert_genes, true_deltas = make_synthetic()
        run_evaluation(X, pert_genes, true_deltas, k=args.k, shrinkage=args.shrinkage)


if __name__ == "__main__":
    main()
