"""Drop a pre-existing 'cell_type' obs column before renaming 'celltype'->'cell_type',
so the benchmark drivers do not create duplicate column names (which break anndata
subsetting).  Idempotent.  Usage:
    python patch_drop_dupcols.py run_files/run_scGPT.py run_files/run_GEARS.py
"""
import shutil
import sys

OLD = ('    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)\n'
       '    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)')
NEW = ('    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)\n'
       '    full_set.obs.drop(columns=["cell_type"], errors="ignore", inplace=True)\n'
       '    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)')


def main():
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        if 'drop(columns=["cell_type"]' in src:
            print("already patched", path)
            continue
        if OLD not in src:
            print("PATTERN NOT FOUND in", path)
            continue
        shutil.copyfile(path, path + ".dcols.bak")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(OLD, NEW))
        print("patched", path)


if __name__ == "__main__":
    main()
