"""run_GEARS scores only perturbations that appear in the TRAIN conditions, which yields
zero predictions for our identity/OOD split (test perturbations are disjoint from train).
Predict instead for the TEST perturbation genes that are in the GEARS perturbation graph.
    python patch_gears_testperts.py run_files/run_GEARS.py
"""
import shutil
import sys

OLD = ("    train_pert_genes = {c.replace('ctrl+', '') for c in train_conditions if c != 'ctrl'}\n"
       "    perturbation_names = [p for p in gears_model.pert_list if p in train_pert_genes and p in gears_model.node_map_pert]")
NEW = ("    train_pert_genes = {c.replace('ctrl+', '') for c in train_conditions if c != 'ctrl'}\n"
       "    test_pert_genes = {c.replace('ctrl+', '') for c in id_test_conditions if c != 'ctrl'}\n"
       "    perturbation_names = [p for p in gears_model.pert_list if p in test_pert_genes and p in gears_model.node_map_pert]")


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?)")
        return
    shutil.copyfile(path, path + ".testperts.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
