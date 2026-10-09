"""check_numbers.py -- scan every decimal number in the manuscript and check whether it
can be located in results/*.csv (within its displayed precision).

Purpose: catch typos / stale values introduced when editing the text. Any reported number
that is a rounded value of some result-table cell should match; unmatched decimals are
listed for manual review (some are legitimately not from a CSV, e.g. MDE margins,
external values, or numbers taken from figures).

Usage:  python check_numbers.py [paper.md]
Output: prints matched/unmatched counts and the unmatched list with line numbers.
"""

import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PAPER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper_v13.md")
RES = os.path.join(ROOT, "results")


def load_pool():
    vals = []
    for p in glob.glob(os.path.join(RES, "*.csv")):
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    for cell in row:
                        c = cell.strip().replace("\u2212", "-").replace("%", "")
                        try:
                            vals.append(abs(float(c)))
                        except ValueError:
                            pass
        except OSError:
            pass
    return np.array(vals)


def main():
    pool = load_pool()
    text = open(PAPER, encoding="utf-8").read()
    text = text.split("\n## References")[0]          # skip the reference list
    # drop fenced code blocks, if any
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = text.splitlines()

    pat = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")
    matched = 0
    unmatched = []
    for i, line in enumerate(lines, 1):
        for m in pat.finditer(line):
            tok = m.group(1)
            d = len(tok.split(".")[1])
            t = float(tok)
            tol = 0.5 * 10 ** (-d) + 1e-9
            if np.any(np.abs(pool - t) <= tol):
                matched += 1
            else:
                unmatched.append((i, tok, line.strip()[:100]))

    print("decimals in manuscript (excl. references): %d" % (matched + len(unmatched)))
    print("matched to a results/*.csv value:          %d" % matched)
    print("UNMATCHED (review these):                  %d" % len(unmatched))
    print("-" * 72)
    for ln, tok, ctx in unmatched:
        safe = ctx.encode("ascii", "replace").decode("ascii")
        print("L%-4d %-9s | %s" % (ln, tok, safe))


if __name__ == "__main__":
    main()
