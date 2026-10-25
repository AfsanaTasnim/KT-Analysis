"""Compare regenerated files in results/ with the reference files in expected/.

Numeric cells match within 1e-3; other cells must be equal. Files that have not
been regenerated are reported as NOT RUN. Exit status is 1 if any file differs.

Usage:
    python compare_expected.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOL = 1e-3


def same(a, b):
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), atol=TOL, equal_nan=True):
                return False
        elif not (x.fillna("").astype(str) == y.fillna("").astype(str)).all():
            return False
    return True


def main():
    differ = 0
    for exp in sorted(glob.glob(os.path.join(ROOT, "expected", "*.csv"))):
        name = os.path.basename(exp)
        got = os.path.join(ROOT, "results", name)
        if not os.path.isfile(got):
            print(f"NOT RUN  {name}")
            continue
        ok = same(pd.read_csv(exp), pd.read_csv(got))
        differ += 0 if ok else 1
        print(f"{'OK' if ok else 'DIFFER':8s} {name}")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
