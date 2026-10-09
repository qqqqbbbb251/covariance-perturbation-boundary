# Reproducibility

This folder contains the analysis code (`code/`), the result tables (`results/`), and the
figures (`figures/`) underlying the manuscript.

## Environment

Python 3.12.10 on Windows. Exact package versions are pinned in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

CIPHER is not on PyPI; install it from the authors' repository. All analyses here used
version **0.1.0** (preprint doi:10.1101/2025.06.27.661814). Pin the commit you use:

```
git clone https://github.com/GoyalLab/CIPHER
cd CIPHER && git checkout <commit-pinned-at-submission>
python -m pip install --no-deps -e .
```

Point the scripts at the clone with `CIPHER_ROOT` (default `../CIPHER`).

## Data

Point `PERTURB_DATA` at a single folder holding the datasets listed in
`results/table_s1.csv`. The scripts read `.h5ad` objects from that folder; scripts that
enumerate the folder assume it contains **exactly** the analysis datasets (do not point it
at a larger collection).

The 22 datasets (16 primary + 6 additional), with sources:

* **16 primary** — scPerturb, Zenodo record **13350497** (v1.4),
  https://doi.org/10.5281/zenodo.13350497 (Peidli et al., *Nat Methods* 2024;21:531).
* **6 additional** — the CIPHER curated benchmark release, Zenodo record **21729034**,
  https://doi.org/10.5281/zenodo.21729034, with upstream sources: Replogle 2022 (*Cell*
  185:2559); X-Atlas/Orion, bioRxiv 2025 (doi:10.1101/2025.06.11.659105); Schmidt 2022
  (*Science* 375:eabj4008; GEO GSE190604); Akana 2026 (*Nat Genet* 58:841; Perturb-seq data
  figshare 10.6084/m9.figshare.31119196); and a re-filtered "proper" Perturb-seq object that
  has no upstream publication and is distributed only by the CIPHER benchmark record.
* **Marson 2025 time-course** (used for the stimulation-time analysis) — Arce et al.,
  *Nature* 2025;637:930 (GEO GSE271090; Perturb-CITE-seq subseries GSE278572).
* PPI / genetic-interaction tables — CIPHER supplementary record Zenodo 21728754.

## Random seeds

Analysis scripts use `numpy.random.default_rng(0)` for subsampling and for
resampling/permutation steps; bootstrap and permutation procedures use 10,000 resamples
unless noted. Per-script seeds:

| script | seed | resamples |
|---|---|---|
| `cipher_repro_table.py` | 0–9 | 10 gene-holdout splits (averaged); matched random column |
| `specificity_positive_control.py` | 0 | 100 replicates per gene-specific strength |
| `random_column_stats.py` | 0 | 100 matched draws / perturbation |
| `pearson_stats.py` | 0 | 10,000 bootstrap; 20 negative-control draws |
| `hierarchical_stats.py` | 0 | 10,000 hierarchical bootstrap |
| `cv_equivalence.py` | 0 | 10,000 |

## Pipeline

One command runs every analysis and figure (from the `code/` folder):

```
python run_all.py            # full pipeline (writes into ../results and ../figures)
python run_all.py --skip-heavy
python run_all.py --verify   # data-consistency check only (fast)
```

`run_all.py` runs 85 independent tasks in order (subprocesses); outputs land in
`results/*.csv` and `figures/*.png`. The final step is `verify_results.py`.

`verify_results.py` recomputes **every headline number** directly from the CSVs in
`results/` and compares it with the manuscript (within tolerance), and checks that all
expected result files, figures and scripts are present. It prints
`RESULT: ALL CHECKS PASSED` when the archived tables are internally consistent with the
manuscript.

## Scoring conventions

Two R² conventions are used; every reported number states which:

* **CIPHER uncentered R²** = `1 - SSE / sum(y_true^2)` on held-out genes
  (`holdout_frac = 0.5`), computed inside the authors' own `cipher` package. Used for the
  reproduction table (`cipher_reproduction_table.csv`, Table 1) and the
  full/global/random/mean-field/shuffled comparison.
* **Scale-only R²** = squared cosine between the predicted direction and the response
  (`cos^2`), used for the raw/CPM/logCPM/Pearson analyses and the matched-random-column,
  hierarchical-bootstrap and mixed-effects comparisons.

## Reproduction status

* A full `run_all.py` run was executed with this code (**85/85 tasks completed, 0 failures**;
  log in `code/run_all.log`). It reproduces the headline analyses and the CIPHER
  reproduction table (Table 1) exactly — `results/cipher_reproduction_table.csv` is
  byte-identical to the archived table.
* `verify_results.py` recomputes every reported number from the archived tables and passes.
* The archived `results/` are the final analysis outputs used in the manuscript. Because the
  pipeline is large and the scripts evolved over the course of the project, a minority of
  **secondary** result columns (e.g., some covariance-spectrum / axis diagnostics and a few
  consolidated summary statistics) can differ by small amounts on a fresh run; the reported
  headline numbers and conclusions are unaffected. `verify_results.py` is the authoritative
  consistency check.

## Data and code availability

* Data: as listed under **Data** above.
* CIPHER: https://github.com/GoyalLab/CIPHER.
* Our code + results: this repository, and
  https://github.com/qqqqbbbb251/covariance-perturbation-boundary; a Zenodo DOI for the
  archive will be minted on release.
