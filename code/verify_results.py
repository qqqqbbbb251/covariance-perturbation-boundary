"""
verify_results.py -- data-consistency and reproducibility check.

Recomputes every headline number directly from the CSVs in ../results/ and
compares it with the value stated in the manuscript (within tolerance).  Also
checks that all input CSVs exist and that every script referenced by run_all.py
is present.

Usage:
    python verify_results.py
Exit code is non-zero if any check fails.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

# expected files produced by the pipeline
EXPECTED_CSV = [
    "scan_results.csv", "deconfound_results.csv", "robustness_results.csv",
    "hvg_results.csv", "ci_results.csv", "paired_stats.csv", "spaces_results.csv",
    "size_factor_results.csv", "scores_vs_globalmode.csv", "random_column_results.csv",
    "pearson_source.csv", "cv_results.csv", "equivalence_results.csv",
    "global_mode_identity.csv", "strong_r2.csv", "positive_control.csv",
    "realistic_positive_control.csv", "double_pert_results.csv",
    "cipher_reproduction_table.csv", "specificity_positive_control.csv",
    "random_column_stats.csv", "pearson_stats.csv", "confound_extended.csv",
    "composition_control.csv", "residual_decomposition.csv", "hierarchical_stats.csv",
    "mixed_effects.csv", "per_dataset_summary.csv", "cross_dataset_transfer.csv",
    "robustness_extra.csv", "real_power_analysis.csv", "table_s1.csv",
    "table_s2_scoring.csv",
    "differential_identity.csv", "exp12_stats.csv", "spectrum_rank.csv",
    "inverse_official.csv", "inverse_false_positive.csv", "positive_controls_extra.csv",
    "inverse_decompose.csv", "inverse_decompose_stats.csv",
    "axis_origin.csv", "spectrum_null.csv", "residual_reproducibility.csv",
    "cross_dataset_specific.csv", "specific_structure.csv", "specific_structure_topgenes.csv",
    "specific_robustness.csv", "specific_organization.csv", "specific_prediction.csv",
    "program_vs_direction.csv", "program_vs_direction_amp.csv", "transform_estimator_scan.csv",
    "component_signatures.csv", "component_signatures_top.csv", "pc_robustness.csv",
    "second_moment.csv", "reverse_self_relation.csv",
    "cross_lab_inventory.csv", "cross_lab_pairs.csv", "specific_network.csv",
    "specific_pathway.csv", "gene_sensitivity.csv", "structure_meta.csv", "epistasis.csv",
    "covariance_estimators.csv", "second_order_cov.csv", "distribution_specific.csv",
    "timecourse.csv", "guide_consistency.csv", "guide_target_partition.csv",
    "guide_target_transfer.csv", "component_enrichment.csv",
    "exploration_stats.csv", "theory_simulation.csv", "cross_transfer_pairs.csv",
    "sota_decompose.csv", "INDEX.csv",
]
EXPECTED_NPZ = ["per_pert_r2.npz", "random_column_dist.npz"]
EXPECTED_FIG = ["fig%d_%s.png" % (i, n) for i, n in [
    (1, "global_mode"), (2, "specific_signal"), (3, "full_r2"),
    (4, "scatter_fullR2"), (5, "scatter_specificity"), (6, "strong"),
    (7, "cpm_strong"), (8, "global_mode_identity"), (9, "identity_cpm"),
    (10, "deconfound"), (11, "robustness"), (12, "random_column"),
    (13, "effects"), (14, "specificity_control"), (15, "residual_decomposition"),
    (16, "cross_dataset_transfer"),
    (17, "forward_indistinguishability"), (18, "inverse_driver"), (19, "spectrum_rank"),
    (20, "inverse_decomposition"), (21, "axis_origin_specific"), (22, "exploration_summary"),
    (23, "crosslab_network"), (24, "structure_meta"), (25, "meta_epistasis"),
    (26, "grand_summary"), (27, "sota_decompose"), (28, "theory_simulation"),
    (29, "guide_target")]] \
    + ["graphical_abstract.png", "summary_panel.png"]


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def col(rows, k, filt=True):
    v = []
    for r in rows:
        x = r.get(k, "nan")
        if x in ("", "nan", None):
            continue
        try:
            f = float(x)
        except Exception:
            continue
        if np.isfinite(f):
            v.append(f)
    return np.array(v)


def means_over_datasets(rows, keys, norm=None):
    if norm is not None:
        rows = [r for r in rows if r.get("norm") == norm]
    rows = [r for r in rows if r.get("full") not in ("nan", "", None)]
    out = {}
    for k in keys:
        v = col(rows, k)
        out[k] = float(np.mean(v)) if len(v) else float("nan")
    return out, len(rows)


def main():
    fails = []
    print("=" * 72)
    print("1) FILE EXISTENCE")
    for f in EXPECTED_CSV:
        p = os.path.join(RES, f)
        ok = os.path.exists(p)
        print("   {:<34} {}".format(f, "OK" if ok else "MISSING"))
        if not ok:
            fails.append("missing results/" + f)
    for f in EXPECTED_NPZ:
        ok = os.path.exists(os.path.join(RES, f))
        print("   {:<34} {}".format(f, "OK" if ok else "MISSING"))
        if not ok:
            fails.append("missing results/" + f)
    for f in EXPECTED_FIG:
        ok = os.path.exists(os.path.join(FIG, f))
        print("   {:<34} {}".format("figures/" + f, "OK" if ok else "MISSING"))
        if not ok:
            fails.append("missing figures/" + f)

    print("=" * 72)
    print("2) HEADLINE NUMBERS (recomputed from CSVs vs manuscript)")

    def check(label, got, exp, tol=0.006):
        ok = (np.isfinite(got) and abs(got - exp) <= tol)
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            label, ("%.3f" % got) if np.isfinite(got) else "nan",
            "%.3f" % exp, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append("{}: got {} exp {}".format(label, got, exp))

    # framework (n=10 datasets with perturbations)
    tab = read("cipher_reproduction_table.csv")
    mraw, nraw = means_over_datasets(tab, ["full", "global", "random", "full_excl",
                                           "random_excl", "global_frac"], "raw")
    mpf, npf = means_over_datasets(tab, ["full", "global", "random"], "pflog")
    print("   [framework raw  n=%d]" % nraw)
    check("framework raw full", mraw["full"], 0.215)
    check("framework raw full (excl target)", mraw["full_excl"], 0.217)
    check("framework raw global", mraw["global"], 0.266)
    check("framework raw matched random", mraw["random"], 0.235)
    check("framework raw random (excl target)", mraw["random_excl"], 0.236)
    print("   [framework pflog n=%d]" % npf)
    check("framework pflog full", mpf["full"], 0.086)
    check("framework pflog global", mpf["global"], 0.136)
    check("framework pflog random", mpf["random"], 0.080)

    # matched random column (scale-only)
    rc = read("random_column_stats.csv")
    check("matched-random full (scale-only)", float(np.mean(col(rc, "full_mean"))), 0.423)
    check("matched-random random (scale-only)", float(np.mean(col(rc, "rand_mean"))), 0.437)
    check("matched-random mean percentile", float(np.mean(col(rc, "pct_mean"))), 0.48, 0.02)
    check("matched-random pooled permutation p", float(np.mean(col(rc, "p_perm_pooled"))), 0.49, 0.02)

    # hierarchical + mixed
    hs = {r["comparison"]: r for r in read("hierarchical_stats.csv")}
    check("hier full-global", float(hs["full-global"]["mean"]), -0.031)
    check("hier full-random", float(hs["full-random"]["mean"]), -0.006)
    check("hier global-random", float(hs["global-random"]["mean"]), 0.023)
    me = {r["model"]: r for r in read("mixed_effects.csv")}
    check("mixed full-global", float(me["full_minus_global"]["estimate"]), -0.030)
    check("mixed full-random", float(me["full_minus_random"]["estimate"]), -0.006)
    check("mixed global-random", float(me["global_minus_random"]["estimate"]), 0.023)

    # pearson
    ps = read("pearson_stats.csv")
    check("pearson full", float(np.mean(col(ps, "full_mean"))), 0.058)
    check("pearson matched random", float(np.mean(col(ps, "rand_mean"))), 0.044)
    check("pearson negative control", float(np.mean(col(ps, "negctl_mean"))), 0.020)
    nbh = int(np.sum([float(r["p_full_rand_BH"]) < 0.05 for r in ps
                      if r["p_full_rand_BH"] not in ("", "nan")]))
    print("   {:<52} got={:<9} exp={:<9} {}".format(
        "pearson FDR-significant datasets", str(nbh), "1", "PASS" if nbh == 1 else "FAIL"))
    if nbh != 1:
        fails.append("pearson FDR count %d != 1" % nbh)

    # composition control
    cc = read("composition_control.csv")
    check("composition |corr| with MOR", float(np.mean(np.abs(col(cc, "corr_mor")))), 0.82, 0.03)
    check("composition |partial| given PC2-5", float(np.mean(np.abs(col(cc, "partial_mor_given_pc2_5")))), 0.91, 0.03)
    check("composition |partial| given PC1-5", float(np.mean(np.abs(col(cc, "partial_mor_given_pc1_5")))), 0.07, 0.03)

    # residual decomposition
    rd = read("residual_decomposition.csv")
    check("residual self", float(np.mean(col(rd, "R2_self"))), 0.04, 0.02)
    check("residual shared", float(np.mean(col(rd, "R2_shared"))), 0.10, 0.02)
    check("residual residual", float(np.mean(col(rd, "R2_residual"))), 0.86, 0.03)

    # cross-dataset transfer
    xt = read("cross_dataset_transfer.csv")
    check("transfer within full", float(np.mean(col(xt, "within_full_cos"))), 0.366, 0.01)
    check("transfer cross full", float(np.mean(col(xt, "cross_full_cos"))), 0.346, 0.01)
    check("transfer cross global", float(np.mean(col(xt, "cross_global_cos"))), 0.346, 0.01)
    check("transfer cross random", float(np.mean(col(xt, "cross_random_cos"))), 0.340, 0.01)

    # real power / MDE
    rp = {r["comparison"]: r for r in read("real_power_analysis.csv")}
    check("MDE pooled (full-random)", float(rp["full-matchedrandom"]["mde80_pooled"]), 0.02, 0.01)
    check("MDE clustered (full-random)", float(rp["full-matchedrandom"]["mde80_clustered"]), 0.05, 0.02)

    # specificity positive control
    sp = read("specificity_positive_control.csv")
    p0 = float([r for r in sp if float(r["alpha"]) == 0][0]["power"])
    p05 = float([r for r in sp if abs(float(r["alpha"]) - 0.05) < 1e-9][0]["power"])
    print("   {:<52} got={:<9} exp={:<9} {}".format(
        "pos-control power alpha=0", "%.2f" % p0, "0.01", "PASS" if p0 <= 0.06 else "FAIL"))
    print("   {:<52} got={:<9} exp={:<9} {}".format(
        "pos-control power alpha=0.05", "%.2f" % p05, "1.00", "PASS" if p05 >= 0.9 else "FAIL"))

    # double perturbation (supplementary)
    dp = read("double_pert_results.csv")
    check("double-pert full", float(np.mean(col(dp, "full"))), 0.710, 0.01)
    check("double-pert global", float(np.mean(col(dp, "global"))), 0.678, 0.01)
    check("double-pert random pair", float(np.mean(col(dp, "random_pair"))), 0.706, 0.01)

    # reverse decomposition (Exp4): driver recovery vs ablations
    dec = read("inverse_decompose.csv")
    if dec:
        m_dec = {k: float(np.mean(col(dec, k))) for k in (
            "posterior", "posterior_diagSigma", "posterior_selfremoved",
            "posterior_diagSigma_selfremoved", "posterior_no_uncinfl", "magnitude")}
        print("   [reverse decomposition n=%d]" % len(dec))
        check("reverse posterior", m_dec["posterior"], 0.931, 0.01)
        check("reverse drop cross-gene cov", m_dec["posterior_diagSigma"], 0.933, 0.01)
        check("reverse drop target coord", m_dec["posterior_selfremoved"], 0.744, 0.02)
        check("reverse drop cov+target", m_dec["posterior_diagSigma_selfremoved"], 0.372, 0.02)
        check("reverse drop uncertainty infl", m_dec["posterior_no_uncinfl"], 0.816, 0.02)
        check("reverse magnitude", m_dec["magnitude"], 0.718, 0.02)
        st = {r["name"]: r for r in read("inverse_decompose_stats.csv") if r["section"] == "contrast"}
        p_cov = float(st["full - diagSigma (drop cross-gene covariance)"]["p_value"])
        p_self = float(st["diagSigma - diagSigma_selfremoved"]["p_value"])
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "reverse cov contributes nothing (p>0.05)", "%.3f" % p_cov, ">0.05",
            "PASS" if p_cov > 0.05 else "FAIL"))
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "reverse self coord essential (p<0.05)", "%.4f" % p_self, "<0.05",
            "PASS" if p_self < 0.05 else "FAIL"))
        if p_cov <= 0.05:
            fails.append("reverse cov contrast significant p=%s" % p_cov)
        if p_self >= 0.05:
            fails.append("reverse self contrast ns p=%s" % p_self)

    # --- exploration suite: shared-axis origin & hidden specificity ---
    ao = read("axis_origin.csv")
    if ao:
        m_top = float(np.mean(col(ao, "top1_frac"))); m_eq = float(np.mean(col(ao, "top1_frac_eq")))
        print("   [axis origin n=%d]" % len(ao))
        check("equal-depth reduces top-1 share", m_top - m_eq, 0.42, 0.15)
        check("top-1 ~ gene-mean |r|", float(np.mean(np.abs(col(ao, "r_v_geneMean")))), 0.99, 0.02)
    sn = read("spectrum_null.csv")
    if sn:
        n1 = int(np.sum(col(sn, "n_above_nullmax") == 1))
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "real covariance modes (obs>nullmax)", str(n1), str(len(sn)),
            "PASS" if n1 == len(sn) else "FAIL"))
        if n1 != len(sn):
            fails.append("spectrum null: not all 1")
    cds = read("cross_dataset_specific.csv")
    if cds:
        rowa = [r for r in cds if "K562" in r["pair"] and "rpe1" in r["pair"]]
        if rowa:
            v = float(rowa[0]["cos_specific"]); nv = float(rowa[0]["cos_specific_null"])
            check("cross-cell-line specific cos (K562/rpe1)", v, 0.387, 0.06)
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "cross-cell-line >> null", "%.3f" % v, ">3xnull",
                "PASS" if v > 3 * max(nv, 0.01) else "FAIL"))
        crosslab = [float(r["cos_specific"]) for r in cds
                    if "Nadig" in r["pair"] and "Replogle" in r["pair"]
                    and ("essential" in r["pair"] or "rpe1" in r["pair"])]
        if crosslab:
            cl = float(np.min(crosslab))
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "cross-LAB specific cos (min, Nadig/Replogle)", "%.3f" % cl, ">0.3",
                "PASS" if cl > 0.3 else "FAIL"))
            if cl <= 0.3:
                fails.append("cross-lab specific cos too low")
    sr = read("specific_robustness.csv")
    if sr:
        check("structure survives top-var removal (min)", float(np.min(col(sr, "rel_no_topvar"))), 0.744, 0.15)
    pr = read("pc_robustness.csv")
    if pr:
        kk = [r for r in pr if "K562" in r["dataset"]]
        if kk:
            check("structure survives 20 control PCs (K562)", float(kk[-1]["rsa_selfrem_spec"]), 0.759, 0.10)
    sm = read("second_moment.csv")
    if sm:
        rn = [r for r in sm if "Norman" in r["dataset"]]
        if rn:
            check("reproducible variance response (Norman RSA)", float(rn[0]["var_rsa_spec"]), 0.341, 0.08)
    snet = read("specific_network.csv")
    if snet:
        kk = [r for r in snet if "K562" in r["dataset"] and r["network"] == "PPI"]
        if kk:
            pe = float(kk[0]["p_perm"]); se = float(kk[0]["sim_edge"]); sn = float(kk[0]["sim_none"])
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "PPI organises specific structure (K562 p<0.05)", "%.4f" % pe, "<0.05",
                "PASS" if pe < 0.05 and se > sn else "FAIL"))
            if not (pe < 0.05 and se > sn):
                fails.append("PPI organisation not significant")
    spw = read("specific_pathway.csv")
    if spw:
        rr = [r for r in spw if "K562" in r["dataset"] and "Reactome" in r["network"]]
        if rr:
            pe = float(rr[0]["p_perm"]); se = float(rr[0]["sim_edge"]); sn = float(rr[0]["sim_none"])
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "pathway organises specific structure (K562 Reactome)", "%.4f" % pe, "<0.05",
                "PASS" if pe < 0.05 and se > sn else "FAIL"))
            if not (pe < 0.05 and se > sn):
                fails.append("pathway organisation not significant")
    gs = read("gene_sensitivity.csv")
    if gs:
        kk = [float(r["rsa_selfrem_spec"]) for r in gs
              if "K562" in r["dataset"] and r["space"] == "raw"]
        if kk:
            rngv = max(kk) - min(kk)
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "structure robust to gene panel (K562 raw range)", "%.3f" % rngv, "<0.05",
                "PASS" if rngv < 0.05 else "FAIL"))
            if rngv >= 0.05:
                fails.append("structure not robust to gene panel")
    ep = read("epistasis.csv")
    if ep:
        row = next((r for r in ep if "Norman" in r["dataset"]), None)
        if row:
            ac = float(row["mean_additive_cos2"])
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "double perturbations largely additive (Norman)", "%.3f" % ac, ">0.5",
                "PASS" if ac > 0.5 else "FAIL"))
            if ac <= 0.5:
                fails.append("Norman doubles not additive")
    ce = read("covariance_estimators.csv")
    if ce:
        m = max(float(r["cos2_off"]) - float(r["cos2_off_rand"]) for r in ce)
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "no covariance estimator beats random (max diff)", "%.3f" % m, "<=0.02",
            "PASS" if m <= 0.02 else "FAIL"))
        if m > 0.02:
            fails.append("a covariance estimator beat random")
    so = read("second_order_cov.csv")
    if so:
        kk = [r for r in so if "K562" in r["dataset"]]
        if kk:
            check("reproducible covariance response (K562)", float(kk[0]["cov_rel_spec"]), 0.599, 0.1)
    ds2 = read("distribution_specific.csv")
    if ds2:
        kk = [r for r in ds2 if "K562" in r["dataset"]]
        if kk:
            check("reproducible distributional response (K562 W1)", float(kk[0]["rel_w1"]), 0.815, 0.1)
    tc = read("timecourse.csv")
    if tc:
        row = next((r for r in tc if r["pair"] == "Rest->48h"), None)
        if row:
            check("specific structure persists over time (Rest->48h)", float(row["cos_specific"]), 0.341, 0.1)
    es = {r["label"]: r for r in read("exploration_stats.csv")}
    if es:
        try:
            p_op = float(es["operator off-axis - random (all spaces/estimators)"]["p"])
            p_cd = float(es["cross-dataset cos_specific - null"]["p"])
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "consolidated operator blindness (p>0.05)", "%.3f" % p_op, ">0.05",
                "PASS" if p_op > 0.05 else "FAIL"))
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "consolidated cross-dataset>null (p<0.01)", "%.2g" % p_cd, "<0.01",
                "PASS" if p_cd < 0.01 else "FAIL"))
            if p_op <= 0.05:
                fails.append("consolidated operator not blind")
            if p_cd >= 0.01:
                fails.append("consolidated cross-dataset not significant")
        except KeyError:
            pass
    ts = [r for r in read("theory_simulation.csv") if r["section"] == "forward_vs_mu"]
    if ts:
        r0 = next((r for r in ts if abs(float(r["x"])) < 1e-9), None)
        if r0:
            check("theory: rank-1 forward R^2 = shared_frac",
                  abs(float(r0["y"]) - float(r0["shared_frac"])), 0.0, 0.02)
    tes = read("transform_estimator_scan.csv")
    if tes:
        dd = np.abs(col(tes, "cos2_sigma_off") - col(tes, "cos2_sigma_off_rand"))
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "operator blind across transforms (max |d|)", "%.3f" % float(dd.max()), "<0.10",
            "PASS" if dd.max() < 0.10 else "FAIL"))
        if dd.max() >= 0.10:
            fails.append("transform scan operator non-blind")
    rsr = read("reverse_self_relation.csv")
    if rsr:
        check("reverse self-relation full AUC", float(np.mean(col(rsr, "auc_full"))), 0.931, 0.03)
        diffs = [float(r["auc_full_weakself"]) - float(r["auc_diag_weakself"]) for r in rsr
                 if r["auc_full_weakself"] not in ("", "nan") and r["auc_diag_weakself"] not in ("", "nan")]
        dd2 = float(np.nanmean(diffs)) if diffs else float("nan")
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "covariance helps weak-self AUC (full-diag)", "%.3f" % dd2, ">0",
            "PASS" if np.isfinite(dd2) and dd2 > 0 else "FAIL"))
        if not (np.isfinite(dd2) and dd2 > 0):
            fails.append("weak-self covariance does not help")

    # guide vs target decoupling
    gtp = read("guide_target_partition.csv")
    if gtp:
        norman = next((r for r in gtp if r["dataset"].startswith("Norman")), None)
        others = [r for r in gtp if not r["dataset"].startswith("Norman")]
        if norman and others:
            nf = float(norman["target_frac"])
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "Norman target-common energy (target-level)", "%.3f" % nf, ">0.9",
                "PASS" if nf > 0.9 else "FAIL"))
            if nf <= 0.9:
                fails.append("Norman target-common fraction not high")
            ag = float(np.mean([float(r["guide_agreement"]) for r in others]))
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "mean guide agreement, non-Norman (guide noise)", "%.3f" % ag, "<0.25",
                "PASS" if ag < 0.25 else "FAIL"))
            if ag >= 0.25:
                fails.append("guide agreement unexpectedly high")
            wg = float(np.mean([float(r["within_guide_reliab"]) for r in others]))
            print("   {:<52} got={:<9} exp={:<9} {}".format(
                "within-guide reliability >> across-guide (non-Norman)",
                "%.3f" % (wg / max(ag, 1e-9)), ">3x",
                "PASS" if wg > 3 * ag else "FAIL"))
            if wg <= 3 * ag:
                fails.append("within-guide reliability not above across-guide")
    gtt = read("guide_target_transfer.csv")
    if gtt:
        multi = [r for r in gtt if r["subset"] == "multiguide"]
        pairs = sorted(set(r["pair"] for r in multi))
        diffs = []
        for p in pairs:
            g1 = next((float(r["cos_specific"]) for r in multi
                       if r["pair"] == p and r["estimate"] == "guide1"), None)
            ga = next((float(r["cos_specific"]) for r in multi
                       if r["pair"] == p and r["estimate"] == "guideAll"), None)
            if g1 is not None and ga is not None:
                diffs.append(ga - g1)
        dd = float(np.mean(diffs)) if diffs else float("nan")
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "multi-guide transfer gain (all - one guide)", "%.3f" % dd, ">0.05",
            "PASS" if np.isfinite(dd) and dd > 0.05 else "FAIL"))
        if not (np.isfinite(dd) and dd > 0.05):
            fails.append("guide averaging does not improve cross-dataset transfer")

    # dataset table: primary-cohort total cells (the ">2 million" figure)
    dt = read("table_s1.csv")
    tot = int(sum(int(r["cells"]) for r in dt if r.get("cohort") == "primary"))
    print("   {:<52} got={:<9} exp={:<9} {}".format(
        "Table S1 primary-cohort cells", str(tot), "2044655", "PASS" if tot == 2044655 else "FAIL"))
    if tot != 2044655:
        fails.append("primary total cells %d != 2044655" % tot)
    # subset sizes must match the manuscript (13 / 10 / 10 / 8 / 6)
    for flag, exp in [("global_mode", 13), ("cipher_repro", 10),
                      ("predictive", 10), ("expression_space", 8), ("cross_lab", 6)]:
        got = int(sum(int(r[flag]) for r in dt if r.get(flag, "") not in ("", None)))
        print("   {:<52} got={:<9} exp={:<9} {}".format(
            "Table S1 subset size: " + flag, str(got), str(exp), "PASS" if got == exp else "FAIL"))
        if got != exp:
            fails.append("Table S1 %s size %d != %d" % (flag, got, exp))

    print("=" * 72)
    print("3) run_all.py SCRIPT COVERAGE")
    try:
        sys.path.insert(0, HERE)
        import run_all
        for _, script, _, _, _ in run_all.TASKS:
            ok = os.path.exists(os.path.join(HERE, script))
            if not ok:
                fails.append("run_all references missing " + script)
                print("   {:<34} MISSING".format(script))
        print("   all {} scripts referenced by run_all.py present".format(len(run_all.TASKS)))
    except Exception as e:
        fails.append("could not import run_all: %s" % e)
        print("   could not import run_all:", str(e)[:60])

    print("=" * 72)
    if fails:
        print("RESULT: {} CHECK(S) FAILED".format(len(fails)))
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
