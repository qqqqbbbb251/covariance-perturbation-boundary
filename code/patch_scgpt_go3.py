"""Broaden the predict-loop exception catch so perturbations whose target is missing
from the GEARS gene list are skipped (GEARS raises IndexError, not ValueError).
    python patch_scgpt_go3.py run_files/run_scGPT.py
"""
import shutil
import sys

OLD = ('                except ValueError:\n'
       '                    continue')
NEW = ('                except (ValueError, IndexError, KeyError):\n'
       '                    continue')


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if OLD not in src:
        print("pattern not found (already patched?)")
        return
    shutil.copyfile(path, path + ".gopatch3.bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print("patched", path)


if __name__ == "__main__":
    main()
