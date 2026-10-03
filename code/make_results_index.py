"""
make_results_index.py -- build an index of every results CSV (file, rows, columns)
and print a compact catalogue, for writing/handover.

Output: ../results/INDEX.csv
"""

import csv
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "INDEX.csv")

# short description per file (for the catalogue)
DESC = {
    "cipher_reproduction_table.csv": "official-framework forward R2 (full/global/matched random), 10 splits",
    "random_column_stats.csv": "matched random-column full vs random (scale-only cosine)",
    "pearson_stats.csv": "Pearson-residual full vs random vs negative control; FDR",
    "hierarchical_stats.csv": "stratified bootstrap contrasts",
    "mixed_effects.csv": "mixed-model contrasts",
    "per_dataset_summary.csv": "per-dataset forward summary",
    "robustness_extra.csv": "leave-one-dataset-out / extra robustness",
    "real_power_analysis.csv": "MDE / power",
    "composition_control.csv": "global mode vs MOR/size/composition",
    "confound_extended.csv": "batch/guide/PC controls",
    "residual_decomposition.csv": "R2 self/shared/residual",
    "cross_dataset_transfer.csv": "cross-dataset forward transfer",
    "table_s1.csv": "dataset table",
    "table_s2_scoring.csv": "scoring summary",
    "double_pert_results.csv": "double perturbation forward",
    "inverse_official.csv": "official posterior inverse driver AUC",
    "inverse_false_positive.csv": "matched-filter / self-removed / magnitude",
    "inverse_decompose.csv": "reverse ablation (self vs covariance)",
    "inverse_decompose_stats.csv": "reverse ablation stats",
    "axis_origin.csv": "PC1 nuisance + equal-depth resampling",
    "spectrum_null.csv": "covariance eigen-spectrum vs permutation null",
    "residual_reproducibility.csv": "split-half RSA reliability (per dataset x space)",
    "cross_dataset_specific.csv": "cross-dataset specific-residual cosine (same vs shuffled target)",
    "specific_structure.csv": "residual SVD + operator predictability",
    "specific_structure_topgenes.csv": "top genes per residual component",
    "specific_robustness.csv": "structure robustness (z-score / gene removal)",
    "specific_organization.csv": "co-expression organization test",
    "specific_prediction.csv": "LOO-SVD + cross-dataset Procrustes transfer",
    "program_vs_direction.csv": "program vs direction decomposition",
    "program_vs_direction_amp.csv": "amplitude reliability",
    "transform_estimator_scan.csv": "transform x estimator operator predictability",
    "component_signatures.csv": "curated signature enrichment of components",
    "component_signatures_top.csv": "top genes per component",
    "pc_robustness.csv": "structure vs #control PCs removed",
    "second_moment.csv": "variance-response reproducibility",
    "second_order_cov.csv": "covariance-response reproducibility",
    "distribution_specific.csv": "mean/var/zero/W1 reliability",
    "reverse_self_relation.csv": "reverse AUC vs self z; weak-self regime",
    "cross_lab_inventory.csv": "per-dataset targets/cell line/modality",
    "cross_lab_pairs.csv": "shared-target counts between datasets",
    "specific_network.csv": "PPI/genetic network organization",
    "specific_pathway.csv": "KEGG/Reactome/GO/TF organization",
    "gene_sensitivity.csv": "structure vs gene panel size",
    "structure_meta.csv": "structure strength vs dataset properties",
    "epistasis.csv": "double-perturbation interaction decomposition",
    "covariance_estimators.csv": "operator predictability vs covariance estimator",
    "timecourse.csv": "Marson time-course specific-structure stability",
    "guide_consistency.csv": "same-target guide agreement (confound test)",
    "guide_target_partition.csv": "target-common vs guide-specific energy / reliability",
    "guide_target_transfer.csv": "cross-dataset transfer vs number of guides averaged",
    "component_enrichment.csv": "Enrichr identity of residual components",
    "exploration_stats.csv": "consolidated CIs and tests",
    "theory_simulation.csv": "synthetic validation of the identifiability sketch",
    "sota_decompose.csv": "model predictions under the decomposition (CIPHER/linear_mean)",
    "cipher_baselines.csv": "legacy per-dataset CIPHER baselines (superseded by cipher_reproduction_table)",
    "dataset_table.csv": "legacy dataset table (superseded by table_s1)",
    "diagnostic_results.csv": "legacy diagnostic table (largely NaN; not in final pipeline)",
    "unified_results.csv": "legacy merged summary (superseded by per-analysis CSVs)",
}


def main():
    files = sorted(f for f in os.listdir(RES) if f.endswith(".csv") and f != "INDEX.csv")
    rows = []
    for f in files:
        p = os.path.join(RES, f)
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.reader(fh)
            header = next(rd, [])
            n = sum(1 for _ in rd)
        rows.append({"file": f, "rows": n, "cols": len(header), "description": DESC.get(f, "")})
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "rows", "cols", "description"])
        w.writeheader()
        w.writerows(rows)
    print("indexed %d result files -> %s" % (len(rows), OUT))
    for r in rows:
        print("  %-38s rows=%-5d cols=%-3d %s" % (r["file"][:37], r["rows"], r["cols"], r["description"][:60]))


if __name__ == "__main__":
    main()
