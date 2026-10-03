"""Patch scGPT generation_model.pred_perturb to tolerate the vendored GEARS cell-graph
features (x has only the expression column; pert info is in batch_data.pert_idx).
    python patch_scgpt_pred.py models/scGPT/scgpt/model/generation_model.py
"""
import shutil
import sys

OLD = ('        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)\n'
       '        pert_flags = x[:, 1].long().view(batch_size, -1)')
NEW = ('        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)\n'
       '        if x.dim() == 2 and x.size(1) >= 2:\n'
       '            pert_flags = x[:, 1].long().view(batch_size, -1)\n'
       '        else:\n'
       '            pert_flags = torch.zeros_like(ori_gene_values, dtype=torch.long)\n'
       '            for bi, pidx in enumerate(getattr(batch_data, "pert_idx", [])):\n'
       '                seq = pidx if isinstance(pidx, (list, tuple)) else [pidx]\n'
       '                for p in seq:\n'
       '                    try:\n'
       '                        ip = int(p)\n'
       '                    except Exception:\n'
       '                        continue\n'
       '                    if 0 <= ip < pert_flags.size(1):\n'
       '                        pert_flags[bi, ip] = 1')


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?):", path)
        return
    shutil.copyfile(path, path + ".predpatch.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
