"""
tier0_discovery.py

Discovery layer on top of the Tier 0 covariance model.

Question it answers
-------------------
Given only the baseline (unperturbed) covariance of a cell line, do the
predicted perturbation signatures of two genes converge more than expected by
chance? Convergent signatures are evidence that the two genes sit in the same
functional module / pathway -- exactly the kind of data-driven signal you want
for a HELLS <-> LMNA hypothesis, without ever telling the model the answer.

What it computes
----------------
1. signature matrix   S[g] = predicted transcriptome response to perturbing g
2. convergence matrix cosine similarity between every pair of signatures
3. target report      convergence of the chosen pair + its percentile against
                      the null distribution of all pairs
4. nearest neighbours the genes whose predicted responses most resemble HELLS
                      (or whichever target), so you can see the module
5. epistasis (optional) if you supply a true double-perturbation delta, compare
                      it with the additive prediction and quantify the residual

Usage
-----
  python tier0_discovery.py                                  # synthetic demo
  python tier0_discovery.py --baseline baseline.csv --targets HELLS LMNA
  python tier0_discovery.py --baseline baseline.csv --targets HELLS LMNA \
         --double deltas.csv --double-gene HELLS_LMNA_KD

For --double, pass the same deltas.csv produced by prepare_data.py and the
condition/gene label of the double perturbation row.
"""

import argparse
import csv

import numpy as np

from tier0_covariance_baseline import (
    compute_covariance,
    load_matrix_csv,
    _pearson,
)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def signature_matrix(cov):
    """S[g] = cov[:, g]  (CIPHER response direction to perturbing g)."""
    return cov.T.copy()


def cosine_matrix(S):
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Sn = S / norms
    return Sn @ Sn.T


def _null_upper(C):
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    return C[iu]


def pair_report(C, genes, g1, g2):
    i1, i2 = genes.index(g1), genes.index(g2)
    score = float(C[i1, i2])
    null = _null_upper(C)
    pct = float((null < score).mean())
    z = (score - null.mean()) / (null.std() + 1e-12)
    return score, pct, z


def neighbours(C, genes, g, top=10):
    i = genes.index(g)
    order = np.argsort(-C[i])
    out = []
    for j in order:
        if j == i:
            continue
        out.append((genes[j], float(C[i, j])))
        if len(out) >= top:
            break
    return out


def epistasis(true_double, s1, s2):
    add = s1 + s2
    resid = true_double - add
    return {
        "corr_double_vs_additive": _pearson(true_double, add),
        "corr_double_vs_single1": _pearson(true_double, s1),
        "corr_double_vs_single2": _pearson(true_double, s2),
        "residual_fraction": float(np.linalg.norm(resid) / (np.linalg.norm(true_double) + 1e-12)),
    }


# --------------------------------------------------------------------------- #
# synthetic demo (HELLS & LMNA placed in the same module on purpose)
# --------------------------------------------------------------------------- #
def make_synthetic(n_genes=200, n_factors=15, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_genes, n_factors)) * (
        rng.random((n_genes, n_factors)) < 0.2
    )
    loadings[1] = loadings[0] + 0.15 * rng.normal(size=n_factors)  # LMNA ~ HELLS
    cov = loadings @ loadings.T + 0.5 * np.eye(n_genes)
    genes = ["HELLS", "LMNA"] + ["G{:03d}".format(i) for i in range(2, n_genes)]
    return cov, genes


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report(cov, genes, targets, double=None, double_gene=None, top=10):
    S = signature_matrix(cov)
    C = cosine_matrix(S)

    print("\n=== signature convergence ===")
    for g in targets:
        if g not in genes:
            print("  {} not in baseline genes".format(g))
            return
    if len(targets) >= 2:
        g1, g2 = targets[0], targets[1]
        score, pct, z = pair_report(C, genes, g1, g2)
        print("  pair {}-{}: cosine = {:.3f}   percentile = {:.1f}%   z = {:.2f}".format(
            g1, g2, score, 100 * pct, z))
        print("  (percentile = share of all gene pairs with lower convergence)")

    for g in targets:
        print("\n  top-{} neighbours of {}:".format(top, g))
        for name, val in neighbours(C, genes, g, top):
            print("    {:<12}{:.3f}".format(name, val))

    if double is not None:
        if len(targets) < 2:
            print("\n  epistasis needs two targets")
            return
        g1, g2 = targets[0], targets[1]
        s1 = S[genes.index(g1)]
        s2 = S[genes.index(g2)]
        rep = epistasis(double, s1, s2)
        print("\n=== epistasis (double = {}) ===".format(double_gene))
        print("  corr(double, additive)  = {:.3f}".format(rep["corr_double_vs_additive"]))
        print("  corr(double, {:<8})   = {:.3f}".format(g1, rep["corr_double_vs_single1"]))
        print("  corr(double, {:<8})   = {:.3f}".format(g2, rep["corr_double_vs_single2"]))
        print("  residual fraction       = {:.3f}".format(rep["residual_fraction"]))
        print("  (residual fraction ~0 => purely additive, no genetic interaction)")


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def load_double(delta_csv, genes, double_gene):
    deltas, delta_genes, names = load_matrix_csv(delta_csv), None, None
    # reuse prepare_data layout: header genes, rows = perturbations, labels present
    mat, gene_names = deltas
    with open(delta_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        has_labels = bool(header) and header[0].strip() == ""
        labels = [row[0].strip() for row in reader if row]
    if double_gene not in labels:
        raise SystemExit("double gene '{}' not in {}".format(double_gene, labels))
    if gene_names != genes:
        raise SystemExit("delta gene columns differ from baseline")
    return mat[labels.index(double_gene)]


def main():
    ap = argparse.ArgumentParser(description="Tier 0 discovery: convergence + epistasis")
    ap.add_argument("--baseline", help="baseline.csv (control samples x genes)")
    ap.add_argument("--cov", help="npz with 'cov' and 'genes' (build_bj_covariance.py)")
    ap.add_argument("--targets", nargs="+", default=["HELLS", "LMNA"])
    ap.add_argument("--double", help="deltas.csv containing the double perturbation")
    ap.add_argument("--double-gene", help="row label of the double perturbation")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--shrinkage", type=float, default=0.0)
    args = ap.parse_args()

    if args.cov:
        d = np.load(args.cov, allow_pickle=True)
        cov = d["cov"]
        genes = [str(g) for g in d["genes"]]
        double = None
    elif args.baseline:
        X, genes = load_matrix_csv(args.baseline)
        cov = compute_covariance(X, shrinkage=args.shrinkage)
        double = None
        if args.double and args.double_gene:
            double = load_double(args.double, genes, args.double_gene)
    else:
        print("no --baseline given: running synthetic demo "
              "(HELLS and LMNA deliberately co-modular)")
        cov, genes = make_synthetic()
        double = None

    report(cov, genes, args.targets, double=double,
           double_gene=args.double_gene, top=args.top)


if __name__ == "__main__":
    main()
