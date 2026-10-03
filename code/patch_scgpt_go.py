"""Patch run_scGPT.py to skip perturbations whose target gene is not in the GEARS
perturbation graph (GO panel), instead of crashing during prediction/scoring.
    python patch_scgpt_go.py run_files/run_scGPT.py
"""
import shutil
import sys

LOOP_OLD = ('            for pert in pert_list:\n'
            '                cell_graphs = create_cell_graph_dataset_for_prediction(\n'
            '                    pert, ctrl_adata, gene_list, model_device, num_samples=pool_size\n'
            '                )')
LOOP_NEW = ('            for pert in pert_list:\n'
            '                try:\n'
            '                    cell_graphs = create_cell_graph_dataset_for_prediction(\n'
            '                        pert, ctrl_adata, gene_list, model_device, num_samples=pool_size\n'
            '                    )\n'
            '                except ValueError:\n'
            '                    continue')

SCORE_OLD = ('    for cond, pert_genes in zip(id_test_conditions, id_test_pert_list):\n'
             '        pred_centered = predicted_means_dict["_".join(pert_genes)] - mean_expression')
SCORE_NEW = ('    for cond, pert_genes in zip(id_test_conditions, id_test_pert_list):\n'
             '        _key = "_".join(pert_genes)\n'
             '        if _key not in predicted_means_dict:\n'
             '            skipped_perts.append(cond)\n'
             '            continue\n'
             '        pred_centered = predicted_means_dict[_key] - mean_expression')


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    orig = src
    if LOOP_OLD in src:
        src = src.replace(LOOP_OLD, LOOP_NEW)
    else:
        print("predict-loop pattern not found")
    if SCORE_OLD in src:
        src = src.replace(SCORE_OLD, SCORE_NEW)
    else:
        print("scoring pattern not found")
    if src == orig:
        print("no change")
        return
    shutil.copyfile(path, path + ".gopatch.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched", path)


if __name__ == "__main__":
    main()
