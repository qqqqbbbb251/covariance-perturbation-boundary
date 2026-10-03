"""
make_table1.py -- regenerate the manuscript's Table 1 (per-dataset CIPHER reproduction,
raw counts) directly from results/cipher_reproduction_table.csv, so the table can never
drift from the data.  Prints Markdown; copy into the manuscript.

Usage: python make_table1.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "cipher_reproduction_table.csv")

SHORT = {
    "DatlingerBock2017": "Datlinger 2017",
    "FrangiehIzar2021_RNA": "Frangieh 2021",
    "NadigOConner2024_hepg2": "Nadig hepg2",
    "NadigOConner2024_jurkat": "Nadig jurkat",
    "NormanWeissman2019_filtered": "Norman 2019",
    "PapalexiSatija2021_eccite_arrayed_RNA": "Papalexi arrayed",
    "ReplogleWeissman2022_K562_essential": "Replogle K562",
    "ReplogleWeissman2022_rpe1": "Replogle rpe1",
    "TianKampmann2021_CRISPRa": "Tian CRISPRa",
    "TianKampmann2021_CRISPRi": "Tian CRISPRi",
}


def num(x, d=3):
    try:
        return ("%%.%df" % d) % float(x)
    except Exception:
        return ""


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["norm"] == "raw"]
    keep = [r for r in rows if r["dataset"] in SHORT and r["full"] not in ("nan", "")]
    lines = ["| dataset | n_perts | global_frac | full | full excl. target | global | random |",
             "|:---|---:|---:|---:|---:|---:|---:|"]
    np_sum = 0
    gf = []
    for r in keep:
        np_sum += int(r["n_perts"]); gf.append(float(r["global_frac"]))
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            SHORT[r["dataset"]], r["n_perts"], num(r["global_frac"]),
            num(r["full"]), num(r["full_excl"]), num(r["global"]), num(r["random"])))
    n = len(keep)
    mean = lambda k: sum(float(r[k]) for r in keep) / n
    lines.append("| **mean (n = %d)** | **%d** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | **%.3f** |" % (
        n, round(np_sum / n), sum(gf) / n, mean("full"), mean("full_excl"),
        mean("global"), mean("random")))
    print("\n".join(lines))
    print("\n# checked vs results: n = %d datasets" % n)


if __name__ == "__main__":
    main()
