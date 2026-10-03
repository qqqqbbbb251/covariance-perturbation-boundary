"""Make the benchmark drivers robust to pandas StringDtype indices (pandas 3-written
h5ad read under pandas 2.x): use an object-dtype ndarray as the anndata indexer, and
coerce obs/var names after loading.  Idempotent.
    python patch_anndata_str.py <run_scGPT.py> <run_GEARS.py> <pertdata.py>
"""
import re
import shutil
import sys

PD_OLD = "        self.adata = self.adata[filter_go.index.values, :]"
PD_NEW = "        self.adata = self.adata[np.asarray(filter_go.index, dtype=object), :]"


def patch(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    orig = src
    if "filter_go.index.values" in src:
        src = src.replace(PD_OLD, PD_NEW)
    # after any "full_set = sc.read_h5ad(...)" line, coerce names to object/str
    out = []
    inserted = "full_set.obs_names = full_set.obs_names.astype(str)" in src
    for line in src.splitlines(keepends=True):
        out.append(line)
        if (not inserted) and re.match(r"\s*full_set = sc\.read_h5ad\(", line):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(indent + "full_set.obs_names = full_set.obs_names.astype(str)\n")
            out.append(indent + "full_set.var_names = full_set.var_names.astype(str)\n")
            inserted = True
    src = "".join(out)
    if src == orig:
        print("no change (already patched or pattern absent):", path)
        return
    shutil.copyfile(path, path + ".strdtype.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched", path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
