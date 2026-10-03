"""
patch_scgpt_driver.py -- make CIPHER's run_scGPT.py compatible with the vendored GEARS.

This GEARS version builds cell-graph features of shape (n_genes, 1) (expression only)
and stores perturbation indices in ``batch_data.pert_idx``; run_scGPT.py's train() reads
``x[:, 1]`` as a pert flag and crashes.  This patch makes it fall back to constructing
the per-gene perturbation flag from ``batch_data.pert_idx``.

Run on the GPU box:
    python code/patch_scgpt_driver.py /root/autodl-tmp/work/cipher/benchmarks/run_files/run_scGPT.py
(creates a .bak first)
"""

import shutil
import sys

OLD = '''            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(cur_batch_size, n_genes)
            pert_flags = x[:, 1].long().view(cur_batch_size, n_genes)'''

NEW = '''            x: torch.Tensor = batch_data.x
            ori_gene_values = x[:, 0].view(cur_batch_size, n_genes)
            if x.dim() == 2 and x.size(1) >= 2:
                pert_flags = x[:, 1].long().view(cur_batch_size, n_genes)
            else:
                # vendored GEARS stores pert info in batch_data.pert_idx, not in x[:, 1]
                pert_flags = torch.zeros((cur_batch_size, n_genes), dtype=torch.long, device=device)
                for bi, pidx in enumerate(batch_data.pert_idx):
                    seq = pidx if isinstance(pidx, (list, tuple)) else [pidx]
                    for p in seq:
                        try:
                            ip = int(p)
                        except Exception:
                            continue
                        if 0 <= ip < n_genes:
                            pert_flags[bi, ip] = 1'''

# also relax the forward pass if it indexes x[:, 1] anywhere
OLD2 = "            input_pert_flags = pert_flags[:, input_gene_ids]"


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found; already patched or different driver version")
        print("search for 'x[:, 1]' and patch manually:")
        for i, ln in enumerate(src.splitlines(), 1):
            if "x[:, 1]" in ln or "pert_flags" in ln:
                print("  %d: %s" % (i, ln.strip()))
        return
    shutil.copyfile(path, path + ".bak")
    src = src.replace(OLD, NEW)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched", path, "(backup at .bak)")


if __name__ == "__main__":
    main()
