import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
run_all.py -- one-command reproduction of every result table and figure.

Runs each analysis script as a subprocess, writing into ../results/ (and figures
into ../figures/).  Data directory is taken from the PERTURB_DATA environment
variable, or the default below.

Usage (from the code/ folder):
    python run_all.py                 # full pipeline
    python run_all.py --skip-heavy    # skip the slowest (full-dataset) steps

Every task is independent; a failure is logged and the pipeline continues.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.environ.get("PERTURB_DATA", _PERTURB_DATA)
LOG = os.path.join(HERE, "run_all.log")

CANON = [
    "AissaBenevolenskaya2021.h5ad", "ChangYe2021.h5ad", "DatlingerBock2017.h5ad",
    "DatlingerBock2021.h5ad", "FrangiehIzar2021_RNA.h5ad",
    "NadigOConner2024_hepg2.h5ad", "NadigOConner2024_jurkat.h5ad",
    "NormanWeissman2019_filtered.h5ad", "PapalexiSatija2021_eccite_RNA.h5ad",
    "PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
    "ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
    "TianKampmann2019_day7neuron.h5ad", "TianKampmann2019_iPSC.h5ad",
    "TianKampmann2021_CRISPRa.h5ad", "TianKampmann2021_CRISPRi.h5ad",
]
FILES = [os.path.join(DATA, f) for f in CANON if os.path.exists(os.path.join(DATA, f))]

# (label, script, extra_args, takes_files, heavy)
TASKS = [
    ("scan_final", "scan_final.py", ["--dir", DATA, "--out", os.path.join(RES, "scan_results.csv")], False, False),
    ("deconfound_final", "deconfound_final.py", [], True, False),
    ("robustness_test", "robustness_test.py", [], True, False),
    ("hvg_control", "hvg_control.py", [], True, False),
    ("ci_bootstrap", "ci_bootstrap.py", [], True, False),
    ("paired_stats", "paired_stats.py", [], True, False),
    ("spaces_diagnostic", "spaces_diagnostic.py", [], True, False),
    ("size_factor", "size_factor.py", [], True, False),
    ("scores_vs_globalmode", "scores_vs_globalmode.py", [], True, False),
    ("random_column", "random_column.py", [], True, False),
    ("pearson_source", "pearson_source.py", [], True, False),
    ("cv_equivalence", "cv_equivalence.py", [], True, False),
    ("global_mode_identity", "global_mode_identity.py", [], False, False),
    ("fig_strong", "fig_strong.py", [], False, False),
    ("positive_control", "positive_control.py", [], False, False),
    ("realistic_positive_control", "realistic_positive_control.py", [], False, False),
    ("double_pert", "double_pert.py", [], True, True),
    ("cipher_repro_table", "cipher_repro_table.py", [], True, True),
    ("random_column_stats", "random_column_stats.py", [], True, False),
    ("pearson_stats", "pearson_stats.py", [], True, False),
    ("confound_extended", "confound_extended.py", [], True, False),
    ("composition_control", "composition_control.py", [], True, False),
    ("residual_decomposition", "residual_decomposition.py", [], True, False),
    ("hierarchical_stats", "hierarchical_stats.py", [], True, False),
    ("specificity_positive_control", "specificity_positive_control.py", [], False, False),
    ("mixed_effects", "mixed_effects.py", [], False, False),
    ("cross_dataset_transfer", "cross_dataset_transfer.py", [], False, True),
    ("robustness_extra", "robustness_extra.py", [], False, False),
    ("real_power_analysis", "real_power_analysis.py", [], False, False),
    ("differential_identity", "differential_identity.py", [], True, True),
    ("exp12_stats", "exp12_stats.py", [], False, False),
    ("inverse_false_positive", "inverse_false_positive.py", [], True, True),
    ("inverse_official", "inverse_official.py", [], False, True),
    ("inverse_decompose", "inverse_decompose.py", [], False, True),
    ("inverse_decompose_stats", "inverse_decompose_stats.py", [], False, False),
    ("positive_controls_extra", "positive_controls_extra.py", [], False, False),
    ("axis_origin", "axis_origin.py", [], False, False),
    ("spectrum_null", "spectrum_null.py", [], False, False),
    ("residual_reproducibility", "residual_reproducibility.py", [], False, True),
    ("cross_dataset_specific", "cross_dataset_specific.py", [], False, True),
    ("specific_structure", "specific_structure.py", [], False, False),
    ("specific_robustness", "specific_robustness.py", [], False, False),
    ("specific_organization", "specific_organization.py", [], False, False),
    ("specific_prediction", "specific_prediction.py", [], False, True),
    ("cross_transfer_pairs", "cross_transfer_pairs.py", [], False, True),
    ("make_results_index", "make_results_index.py", [], False, False),
    ("program_vs_direction", "program_vs_direction.py", [], False, False),
    ("transform_estimator_scan", "transform_estimator_scan.py", [], False, False),
    ("component_signatures", "component_signatures.py", [], False, False),
    ("pc_robustness", "pc_robustness.py", [], False, False),
    ("second_moment", "second_moment.py", [], False, False),
    ("specific_network", "specific_network.py", [], False, True),
    ("specific_pathway", "specific_pathway.py", [], False, True),
    ("gene_sensitivity", "gene_sensitivity.py", [], False, True),
    ("structure_meta", "structure_meta.py", [], False, True),
    ("exploration_stats", "exploration_stats.py", [], False, False),
    ("theory_simulation", "theory_simulation.py", [], False, False),
    ("epistasis", "epistasis.py", [], False, True),
    ("covariance_estimators", "covariance_estimators.py", [], False, True),
    ("second_order_cov", "second_order_cov.py", [], False, True),
    ("distribution_specific", "distribution_specific.py", [], False, True),
    ("timecourse", "timecourse.py", [], False, True),
    ("guide_consistency", "guide_consistency.py", [], False, True),
    ("guide_target_partition", "guide_target_partition.py", [], False, True),
    ("guide_target_transfer_k562_rpe1", "guide_target_transfer.py",
     ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad", "50"], False, True),
    ("guide_target_transfer_jurkat_k562", "guide_target_transfer.py",
     ["NadigOConner2024_jurkat.h5ad", "ReplogleWeissman2022_K562_essential.h5ad", "50"], False, True),
    ("component_enrichment", "component_enrichment.py", [], False, True),
    ("reverse_self_relation", "reverse_self_relation.py", [], False, True),
    ("cross_lab_inventory", "cross_lab_inventory.py", [], False, False),
    ("cross_dataset_stream", "cross_dataset_stream.py",
     ["XAtlas2025_HCT116_filtered.h5ad", "XAtlas2025_HEK293T_filtered.h5ad"], False, True),
    ("table_s1", "table_s1.py", [], False, False),
    ("make_all_figures", "make_all_figures.py", [], False, False),
    ("fig15_16", "fig15_16.py", [], False, False),
    ("fig20_inverse_decompose", "fig20_inverse_decompose.py", [], False, False),
    ("fig21_axis_origin", "fig21_axis_origin_specific.py", [], False, False),
    ("fig22_exploration", "fig22_exploration_summary.py", [], False, False),
    ("fig23_crosslab_network", "fig23_crosslab_network.py", [], False, False),
    ("fig24_structure_meta", "fig24_structure_meta.py", [], False, False),
    ("fig25_meta_epistasis", "fig25_meta_epistasis.py", [], False, False),
    ("fig26_grand_summary", "fig26_grand_summary.py", [], False, False),
    ("fig27_sota_decompose", "fig27_sota_decompose.py", [], False, False),
    ("fig29_guide_target", "fig29_guide_target.py", [], False, False),
    ("make_main_figures", "make_main_figures.py", [], False, False),
    ("fig_summary_panel", "fig_summary_panel.py", [], False, False),
    ("verify_results", "verify_results.py", [], False, False),
]

HEAVY = {"double_pert", "cipher_repro_table", "cross_dataset_transfer", "inverse_decompose",
         "residual_reproducibility", "cross_dataset_specific", "specific_prediction"}


def log(msg):
    line = "{}  {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    skip_heavy = "--skip-heavy" in sys.argv
    if "--verify" in sys.argv:
        sys.exit(subprocess.run([sys.executable, os.path.join(HERE, "verify_results.py")],
                                cwd=RES).returncode)
    if not FILES:
        log("WARNING: no .h5ad files found under {}".format(DATA))
    open(LOG, "w", encoding="utf-8").close()
    log("run_all start; {} datasets found; data={}".format(len(FILES), DATA))
    for label, script, extra, takes_files, heavy in TASKS:
        if skip_heavy and heavy:
            log("skip  {:<28} (heavy)".format(label))
            continue
        args = [sys.executable, os.path.join(HERE, script)] + (FILES if takes_files else []) + extra
        t0 = time.time()
        try:
            r = subprocess.run(args, cwd=RES, capture_output=True, text=True, timeout=7200)
            dt = time.time() - t0
            status = "ok" if r.returncode == 0 else "FAIL({})".format(r.returncode)
            log("{:<28} {:<10} {:.0f}s".format(label, status, dt))
            if r.returncode != 0:
                with open(LOG, "a", encoding="utf-8") as fh:
                    fh.write((r.stderr or "")[-1500:] + "\n")
        except Exception as e:
            log("{:<28} ERROR      {}".format(label, str(e)[:80]))
    log("run_all done")


if __name__ == "__main__":
    main()
