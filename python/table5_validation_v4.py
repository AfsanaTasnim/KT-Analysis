"""Table V: KaizenTracer against the independent reference labels, on the v4 run.

Scores validation250 (n = 250) and holdout80 (n = 80) separately and pooled
(n = 330), using the v4 predictions, and writes the numbers the paper quotes in
Section V-A: per-pattern TP/FP/FN, precision/recall/F1, macro and micro
averages, the share of functions on which all seven labels agree, and 95%
bootstrap intervals for micro and macro F1.

Macro averages cover the patterns that have reference positives (precision and
recall are undefined for an empty reference class). The bootstrap resamples
functions, with the same seed and iteration count as validation_uncertainty.py.

Usage:
    python table5_validation_v4.py
"""

import os

import numpy as np
import pandas as pd

from validation_uncertainty import macro_micro

SHORT = ["training", "inference", "sampling", "loading", "memory", "context",
         "quantization"]
FULL = ["Training Pipeline", "Interactive Inference", "Advanced Sampling",
        "Model Loading and Validation", "Memory Management",
        "Context Management", "Quantization Workflow"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ITERS = 10000


def load(folder, ref_name):
    ref = pd.read_csv(os.path.join(ROOT, folder, ref_name)).sort_values("id")
    kt = pd.read_csv(os.path.join(ROOT, folder, "kaizentracer_v4.csv")).sort_values("id")
    if not ref["id"].equals(kt["id"]):
        raise SystemExit(f"id mismatch in {folder}")
    return ref[SHORT].astype(int).values, kt[SHORT].astype(int).values


def table(R, P):
    rows, TP, FP, FN = [], 0, 0, 0
    for j, name in enumerate(FULL):
        y, p = R[:, j], P[:, j]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and prec + rec > 0
              else (0.0 if tp + fn else float("nan")))
        rows.append({"Pattern": name, "Ref +": int(y.sum()), "KT +": int(p.sum()),
                     "TP": tp, "FP": fp, "FN": fn, "Precision": prec,
                     "Recall": rec, "F1": f1})
    t = pd.DataFrame(rows)
    s = t[t["Ref +"] > 0]
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    macro = {"Pattern": "Macro Average", "Precision": s["Precision"].mean(),
             "Recall": s["Recall"].mean(), "F1": s["F1"].mean()}
    micro = {"Pattern": "Micro Average", "TP": TP, "FP": FP, "FN": FN,
             "Precision": mp, "Recall": mr,
             "F1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0}
    return pd.concat([t, pd.DataFrame([macro, micro])], ignore_index=True)


def bootstrap(R, P, seed=42):
    rng = np.random.default_rng(seed)
    n = len(R)
    bm, bu = [], []
    for _ in range(ITERS):
        idx = rng.integers(0, n, n)
        a, b = macro_micro(R[idx], P[idx])
        bm.append(a)
        bu.append(b)
    return (np.percentile(bm, [2.5, 97.5]), np.percentile(bu, [2.5, 97.5]))


def main():
    R1, P1 = load("validation250", "llm_labels.csv")
    R2, P2 = load("holdout80", "reference_labels.csv")
    sets = {"validation250": (R1, P1), "holdout80": (R2, P2),
            "pooled330": (np.vstack([R1, R2]), np.vstack([P1, P2]))}
    unc = []
    for name, (R, P) in sets.items():
        t = table(R, P)
        out = t.copy()
        for c in ("Precision", "Recall", "F1"):
            out[c] = out[c].round(3)
        out.to_csv(os.path.join(ROOT, "results", f"table5_{name}_v4.csv"), index=False)
        macro, micro = macro_micro(R, P)
        exact = int((R == P).all(axis=1).sum())
        mci, uci = bootstrap(R, P)
        print(f"\n=== {name} (n={len(R)}) ===")
        print(out.fillna("-").to_string(index=False))
        print(f"all seven labels agree: {exact}/{len(R)} ({100 * exact / len(R):.0f}%)")
        print(f"macro F1 {macro:.3f} [{mci[0]:.2f}, {mci[1]:.2f}]   "
              f"micro F1 {micro:.3f} [{uci[0]:.2f}, {uci[1]:.2f}]")
        unc.append({"set": name, "n": len(R), "macro_f1": round(macro, 3),
                    "macro_ci_lo": round(mci[0], 3), "macro_ci_hi": round(mci[1], 3),
                    "micro_f1": round(micro, 3), "micro_ci_lo": round(uci[0], 3),
                    "micro_ci_hi": round(uci[1], 3), "exact_agreement": exact})
    pd.DataFrame(unc).to_csv(os.path.join(ROOT, "results", "table5_uncertainty_v4.csv"),
                              index=False)
    print("\nwrote results/table5_{validation250,holdout80,pooled330}_v4.csv "
          "and table5_uncertainty_v4.csv")


if __name__ == "__main__":
    main()
