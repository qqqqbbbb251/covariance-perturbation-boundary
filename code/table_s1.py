"""
table_s1.py

Builds Table S1: per-dataset cell/gene/control/perturbation counts plus which of
the analysis subsets each dataset enters.

Dataset metadata (cells, genes, control cells, perturbations, cohort, source) comes
from results/dataset_table.csv.  Subset inclusion is *derived* from the analysis
result CSVs so the table cannot drift from the analyses:

  global_mode        datasets in cipher_reproduction_table.csv (13) -- the
                     leading-eigenvector / framework analyses.
  cipher_repro       datasets with a completed CIPHER forward reproduction in raw
                     counts (cipher_reproduction_table.csv, norm=raw, full non-null)
                     -- exactly the ten rows of Table 1.
  predictive         datasets in random_column_stats.csv (10) -- the matched-random-
                     column and mixed-effects comparisons.  This set differs from
                     cipher_repro by two datasets because the released CIPHER loader
                     does not complete on Tian day7-neuron / iPSC.
  expression_space   datasets with Pearson-residual results (spaces_results.csv,
                     pearson_full non-null) (8).
  cross_lab          the additional datasets used for the cross-laboratory / cell-line
                     / library / time analyses.

Output: ../results/table_s1.csv  (also prints a markdown table)
"""

import csv
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RESULTS, "table_s1.csv")


def read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def present(rows, key):
    return set(r["dataset"] for r in rows if r.get(key, "nan") not in ("", "nan", None))


def main():
    base = read("dataset_table.csv")

    cip_rows = read("cipher_reproduction_table.csv")
    glob_mode = present(cip_rows, "global_frac")
    cipher_repro = set(
        r["dataset"] for r in cip_rows
        if r.get("norm") == "raw" and r.get("full", "nan") not in ("", "nan"))
    predict = present(read("random_column_stats.csv"), "full_mean")
    express = present(read("spaces_results.csv"), "pearson_full")

    cols = ["dataset", "cohort", "source", "cells", "genes", "control_cells",
            "single_perturbations", "double_perturbations",
            "global_mode", "cipher_repro", "predictive", "expression_space", "cross_lab"]

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in base:
            n = r["dataset"]
            extra = int(r.get("cohort", "") == "additional")
            w.writerow([n, r.get("cohort", ""), r.get("source", ""),
                        r["cells"], r["genes"], r["control_cells"],
                        r["single_perturbations"], r["double_perturbations"],
                        int(n in glob_mode), int(n in cipher_repro),
                        int(n in predict), int(n in express), extra])

    print("| Dataset | Cohort | Cells | Genes | Control | Single | Double | Global | CIPHER | Pred | Expr | Cross-lab |")
    print("|---|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|")
    for r in base:
        n = r["dataset"]
        extra = int(r.get("cohort", "") == "additional")
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            n, r.get("cohort", ""), r["cells"], r["genes"], r["control_cells"],
            r["single_perturbations"], r["double_perturbations"],
            "+" if n in glob_mode else "", "+" if n in cipher_repro else "",
            "+" if n in predict else "", "+" if n in express else "",
            "+" if extra else ""))

    n_primary = sum(1 for r in base if r.get("cohort") == "primary")
    cells_primary = sum(int(r["cells"]) for r in base if r.get("cohort") == "primary")
    print("\nsubsets: global_mode={} cipher_repro={} predictive={} expression_space={}".format(
        len(glob_mode), len(cipher_repro), len(predict), len(express)))
    print("primary datasets={}  primary cells={:,}".format(n_primary, cells_primary))
    print("total datasets={}  total cells={:,}".format(
        len(base), sum(int(r["cells"]) for r in base)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
