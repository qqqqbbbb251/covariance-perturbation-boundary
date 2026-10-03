"""Replace GEARS' O(n^2) custom GO-graph construction with a sparse-matrix Jaccard
computation (genes x GO-terms binary matrix, Jaccard = inter / union).
    python patch_makego3.py models/GEARS/gears/utils.py
"""
import shutil
import sys

OLD = ("    print('Creating custom GO graph, this can take a few minutes')\n"
       "    global _GENE2GO\n"
       "    _GENE2GO = gene2go\n"
       "    all_edge_list = [_go_edges_worker(g) for g in tqdm(list(gene2go.keys()))]\n"
       "    edge_list = []\n"
       "    for i in all_edge_list:\n"
       "        edge_list = edge_list + i\n")
NEW = ("    print('Creating custom GO graph (sparse Jaccard)...')\n"
       "    import scipy.sparse as _sp\n"
       "    genes = list(gene2go.keys())\n"
       "    term_idx = {}\n"
       "    rows, cols = [], []\n"
       "    for ri, g in enumerate(genes):\n"
       "        for t in gene2go[g]:\n"
       "            ti = term_idx.get(t)\n"
       "            if ti is None:\n"
       "                ti = len(term_idx); term_idx[t] = ti\n"
       "            rows.append(ri); cols.append(ti)\n"
       "    M = _sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),\n"
       "                       shape=(len(genes), max(1, len(term_idx))))\n"
       "    inter = np.asarray((M @ M.T).todense())\n"
       "    deg = np.asarray(M.sum(1)).ravel()\n"
       "    union = deg[:, None] + deg[None, :] - inter\n"
       "    jac = inter / np.maximum(union, 1.0)\n"
       "    ii, jj = np.where(np.triu(jac, 0) > 0.1)\n"
       "    edge_list = [(genes[a], genes[b], float(jac[a, b])) for a, b in zip(ii, jj)]\n")


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?)")
        return
    shutil.copyfile(path, path + ".makego3.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
