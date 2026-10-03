"""Remove the hard precondition in run_scGPT.predict that aborts on any target gene
missing from the GEARS graph; the per-pert try/except then skips such perturbations.
    python patch_scgpt_go2.py run_files/run_scGPT.py
"""
import shutil
import sys

OLD = ('        for pert in pert_list:\n'
       '            for i in pert:\n'
       '                if i not in gene_list:\n'
       '                    raise ValueError(\n'
       '                        "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"\n'
       '                    )\n')
NEW = ('        # perturbations whose target gene is absent from the GEARS graph are\n'
       '        # skipped by the try/except in the prediction loop below\n')


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?)")
        return
    shutil.copyfile(path, path + ".gopatch2.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
