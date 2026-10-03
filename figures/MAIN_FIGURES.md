# Main figures (8) — panel-to-file map and legends

Assembled by `code/make_main_figures.py` into `figures/main/`. Each main figure is a
composite of validated analysis panels (source PNGs listed below). Supplementary figures
are the remaining `figures/fig1–fig20`, `graphical_abstract.png` and `summary_panel.png`.

| Main | File | Source panels | Supports |
|---|---|---|---|
| 1 | `main/main_fig1_capability_boundary.png` | `fig26_grand_summary.png`, `fig17_forward_indistinguishability.png` | Concept + forward boundary (Results §Forward is dominated / not perturbation-specific) |
| 2 | `main/main_fig2_depth_axis.png` | `fig21_axis_origin_specific.png`, `fig19_spectrum_rank.png` | §The shared axis is a technical depth artefact |
| 3 | `main/main_fig3_reverse_detector.png` | `fig18_inverse_driver.png`, `fig20_inverse_decomposition.png` | §The reverse is a detector (self coordinate) |
| 4 | `main/main_fig4_specific_structure.png` | `fig24_structure_meta.png`, `fig29_guide_target.png` | §Real specific structure; §guide/target decoupling |
| 5 | `main/main_fig5_crosslab_time.png` | `fig23_crosslab_network.png`, `fig16_cross_dataset_transfer.png` | §Cross-lab/cell-line/library/time |
| 6 | `main/main_fig6_operator_blind.png` | `fig22_exploration_summary.png` | §Operator blind; §program/amplitude/direction; §multi-moment |
| 7 | `main/main_fig7_organisation_epistasis.png` | `fig24_structure_meta.png`, `fig25_meta_epistasis.png` | §PPI/pathway/TF organisation; §combinations |
| 8 | `main/main_fig8_theory_sota.png` | `fig27_sota_decompose.png`, `fig28_theory_simulation.png` | §Capability boundary: theory + model comparison |

## Full legends (as in `paper_v13.md`)

- **Fig. 1. A capability boundary for covariance-driven forward models.** (A) Concept: the
  same control covariance Σ identifies the perturbed gene (its own coordinate) but the
  forward map ΔX = Σu is dominated by the shared depth axis and cannot predict the
  target-specific response. (B–E) Summary: forward predictions are collinear and RSA ≈ 0;
  the inverse is accurate; a real specific structure lies beneath the axis.
- **Fig. 2. The leading covariance axis is a technical sequencing-depth artefact.**
  Equal-depth resampling halves the top-1 variance share (0.68 → 0.26); PC1 loadings equal
  the gene-mean vector (|r| ≈ 1); a per-gene permutation spectral null finds exactly one
  real mode.
- **Fig. 3. The reverse recovers the driver through the target's own coordinate.** Official
  inverse AUC 0.76–0.99; ablation (dropping the covariance leaves AUC unchanged, p = 0.74;
  dropping the target coordinate collapses 0.93 → 0.37); reverse AUC scales with the self
  effect size.
- **Fig. 4. A real, reproducible specific structure beneath the axis.** Split-half RSA
  reliability across datasets (0.49 [0.31, 0.67]); broad-spectrum robustness (removing
  MT/ribosomal/high-variance genes does not lower it); guide/target decoupling (single
  guides are noisy; multi-guide cross-dataset transfer rises from ~0.17 to ~0.38).
- **Fig. 5. Cross-laboratory / cell-line / library / time transfer of the specific
  structure.** Same-target cosine 0.38–0.49 across labs (null ≈ 0); Procrustes transfer
  cos² 0.35–0.37; Marson D1–D4 time-course/donors 0.28–0.48.
- **Fig. 6. The covariance operator is blind to the specific structure.** Off-axis
  predictability ≈ random (pooled p = 0.52) across data spaces and covariance estimators;
  the programme/direction decomposition (amplitudes reliable 0.85–0.96, leftover direction
  reliability small) and reproducible second-moment/variance signals.
- **Fig. 7. Organisation by PPI, pathways and regulons; combinatorial perturbations.** PPI
  edge − non-edge +0.098 (p = 0.023); Reactome/GO/TF significant; 85% additivity of double
  perturbations with 66% of the interaction off-axis.
- **Fig. 8. Theory and model comparison.** Numerical verification that rank-one forward
  R² = shared_frac and the operator is ε-blind at all μ; SOTA decomposition (CIPHER /
  linear-mean forward cos² 0.39/0.57 but RSA ≈ 0, predicted collinearity 0.86/1.00 ≫ true
  0.58).
