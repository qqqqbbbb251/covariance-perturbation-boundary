# Reproducibility

This folder contains the analysis code (`code/`), the result tables (`results/`),
and the figures (`figures/`) for the CIPHER-style forward-model critique.

## Environment

Python 3.12.10 on Windows.  Exact package versions in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

CIPHER is installed from the authors' repository (not on PyPI); all analyses here used
version **0.1.0** (preprint doi:10.1101/2025.06.27.661814). Pin the commit you use:

```
git clone https://github.com/GoyalLab/CIPHER
cd CIPHER && git checkout <commit-pinned-at-submission>
python -m pip install --no-deps -e .
```

Point the scripts at the clone with `CIPHER_ROOT` (default `../CIPHER`).

## Data

All 22 datasets come from the scPerturb collection:

* scPerturb, Zenodo record **13350497** — https://doi.org/10.5281/zenodo.13350497
  (Peidli et al., *Nat Methods* 2024;21:531).

Place the `.h5ad` files in one directory and point the scripts at it with the
`PERTURB_DATA` environment variable (each script also accepts explicit paths; the default
is `./data` relative to the repository root).

The canonical analysis uses the **16 datasets** listed in `results/table_s1.csv`.

## Random seeds

Every analysis script uses `numpy.random.default_rng(0)` for subsampling and all
resampling/permutation steps; bootstrap and permutation procedures use 10,000
resamples unless noted.  Per-script:

| script | seed | resamples |
|---|---|---|
| `cipher_baselines.py` | 0 | single gene-holdout split |
| `cipher_repro_table.py` | 0–9 | 10 gene-holdout splits (values averaged); matched random column |
| `specificity_positive_control.py` | 0 | 100 replicates per gene-specific strength |
| `random_column_stats.py` | 0 | 100 matched draws / perturbation |
| `pearson_stats.py` | 0 | 10,000 bootstrap; 20 negative-control draws |
| `hierarchical_stats.py` | 0 | 10,000 hierarchical bootstrap |
| `confound_extended.py` | 0 | — |
| `cv_equivalence.py` | 0 | 10,000 |

## Pipeline

One command reproduces every table and figure (from the `code/` folder):

```
python run_all.py            # full pipeline (writes into ../results and ../figures)
python run_all.py --skip-heavy
python run_all.py --verify   # data-consistency check only (fast)
```

`verify_results.py` recomputes every headline number from the CSVs in `results/`
and compares it with the manuscript, and checks that all expected result files,
figures and scripts are present; `run_all.py` runs it as the final step.

The data directory is read from the `PERTURB_DATA` environment variable (default `./data`)
and the CIPHER clone from `CIPHER_ROOT` (default `./CIPHER`).  `run_all.py` runs these scripts
in order (all outputs land in `results/`; figures in `figures/`):

1. `scan_final.py`, `deconfound_final.py`, `robustness_test.py`, `hvg_control.py`,
   `ci_bootstrap.py`, `paired_stats.py`, `spaces_diagnostic.py`, `size_factor.py`,
   `scores_vs_globalmode.py`, `random_column.py`, `pearson_source.py`,
   `cv_equivalence.py`, `global_mode_identity.py`, `fig_strong.py`,
   `positive_control.py`, `realistic_positive_control.py`, `double_pert.py`
   (the scale-only analyses)
2. `cipher_repro_table.py`  -> `results/cipher_reproduction_table.csv`
3. `specificity_positive_control.py` -> `results/specificity_positive_control.csv`, `figures/fig14_*.png`
4. `random_column_stats.py` -> `results/random_column_stats.csv`, `random_column_dist.npz`
5. `pearson_stats.py`       -> `results/pearson_stats.csv`
6. `confound_extended.py`   -> `results/confound_extended.csv`
7. `composition_control.py` -> `results/composition_control.csv`
8. `residual_decomposition.py` -> `results/residual_decomposition.csv`
9. `hierarchical_stats.py`  -> `results/per_pert_r2.npz`, `hierarchical_stats.csv`
10. `mixed_effects.py`      -> `results/per_dataset_summary.csv`, `mixed_effects.csv`
11. `cross_dataset_transfer.py` -> `results/cross_dataset_transfer.csv`
12. `robustness_extra.py`   -> `results/robustness_extra.csv` (leave-one-dataset-out; pflog significance)
13. `real_power_analysis.py` -> `results/real_power_analysis.csv` (real-data MDE / power)
14. `differential_identity.py` -> `results/differential_identity.csv` (Exp1/2: forward indistinguishability)
15. `exp12_stats.py`        -> `results/exp12_stats.csv`
16. `spectrum_rank.py`      -> `results/spectrum_rank.csv` (eigenvalue spectrum + rank-k)
17. `inverse_official.py`   -> `results/inverse_official.csv` (official posterior inverse)
18. `inverse_false_positive.py` -> `results/inverse_false_positive.csv` (inverse baselines)
19. `positive_controls_extra.py` -> `results/positive_controls_extra.csv`
20. `table_s1.py`           -> `results/table_s1.csv`
21. `make_all_figures.py` + `fig15_16.py` + `fig_new.py` -> `figures/fig1..fig19`, `graphical_abstract.png`

`code/legacy_figures/` holds superseded plotting scripts kept for provenance;
the canonical figure source is `make_all_figures.py` + `fig15_16.py`.

## Scoring conventions

Two different R² conventions are used, and every reported number states which:

* **CIPHER uncentered R²** = `1 - SSE / sum(y_true^2)` on held-out genes
  (`holdout_frac = 0.5`), computed inside the authors' own `cipher` package.
  Used for the reproduction table (`cipher_reproduction_table.csv`) and the
  full/global/random/mean-field/shuffled comparison.
* **Scale-only R²** = squared cosine between the predicted direction and the
  response (`cos^2`), used for our own raw/CPM/logCPM/Pearson analyses.

## Data and code availability

* Data: scPerturb, Zenodo record 13350497 (DOI above).
* CIPHER: https://github.com/GoyalLab/CIPHER.
* Our code + results: this repository.  A Zenodo DOI for the archive should be
  minted by the authors on upload.
