# A capability boundary for covariance-driven single-perturbation models

Code, result tables and figures for the manuscript (text not included; under review).

> **A capability boundary for covariance-driven single-perturbation models: the perturbed
> gene is detected through its own coordinate, not predicted by covariance.**
> Yichen Xie. School of Life Sciences, Peking University.

Main result: the CIPHER-style forward model ΔX = Σ·u is dominated by a shared technical
sequencing-depth axis; it cannot predict the target-specific response, and detection of
the perturbed gene is carried by that gene's own coordinate rather than by the covariance.
A real, reproducible, network-organised target-specific structure exists beneath the axis
but lies outside the covariance operator, in any data space or estimator. A reusable
specificity-audit protocol is provided.

## Layout

| path | contents |
|---|---|
| `requirements.txt` | exact package versions used |
| `REPRODUCIBILITY.md` | environment, seeds and pipeline notes |
| `code/` | analysis scripts (85 tasks in `run_all.py`) |
| `results/` | result tables (`results/INDEX.csv` is the catalogue) |
| `figures/` | figures (`figures/main/` = 8 main figures) |

## Reproduce

Requires Python 3.12 with the packages in `requirements.txt` (install with
`pip install -r requirements.txt`). The single-cell datasets are **not** included
(they are public; dataset provenance and accessions are in `results/table_s1.csv`).
Point `PERTURB_DATA` at the directory holding the `.h5ad` files and `CIPHER_ROOT` at the
CIPHER clone (version 0.1.0).

```powershell
# consistency check: recompute every headline number from the CSVs
python code\verify_results.py            # -> ALL CHECKS PASSED

# full pipeline (all result tables and figures)
$env:PERTURB_DATA = "path\to\h5ad\directory"
$env:CIPHER_ROOT  = "path\to\CIPHER"
python code\run_all.py
python code\run_all.py --skip-heavy      # skip the slowest steps
```

`code/make_main_figures.py` assembles the eight main figures; `code/make_table1.py`
regenerates Table 1 from `results/cipher_reproduction_table.csv`.

## License

Code: MIT. Text, figures and result tables: CC BY 4.0. See `LICENSE`.

## Citation

If you use this work, please cite the manuscript (Zenodo DOI to be added on release).

## Contact

Yichen Xie — 2500012266@stu.pku.edu.cn — ORCID 0009-0004-7218-4838
