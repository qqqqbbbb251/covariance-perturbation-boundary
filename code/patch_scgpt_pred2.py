"""Handle 1-D cell-graph features in scGPT generation_model.pred_perturb (prediction
path).  Idempotent.
    python patch_scgpt_pred2.py models/scGPT/scgpt/model/generation_model.py
"""
import shutil
import sys

OLD = ('        x: torch.Tensor = batch_data.x\n'
       '        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)\n'
       '        if x.dim() == 2 and x.size(1) >= 2:')
NEW = ('        x: torch.Tensor = batch_data.x\n'
       '        if x.dim() == 1:\n'
       '            ori_gene_values = x.view(batch_size, -1)\n'
       '        else:\n'
       '            ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)\n'
       '        if x.dim() >= 2 and x.size(1) >= 2:')


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?)")
        return
    shutil.copyfile(path, path + ".predpatch2.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
