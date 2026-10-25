"""Uncertainty and chance-baseline analysis for the validation sample.

Two questions the point estimates alone cannot answer:

  1. How precise is the estimate? A sample of 100 functions gives wide
     intervals, particularly for the rare categories. Bootstrap resampling over
     functions gives a 95% interval for macro and micro F1.

  2. Is the result above chance? A detector that fires at the same *rate* as
     KaizenTracer but at random functions would still score above zero, because
     some hits occur by coincidence. Permuting the predictions across functions
     preserves each pattern's predicted count while destroying any association
     with the source, producing the null distribution to compare against.

Reporting the observed value against this null is what establishes whether the
agreement is real, independently of how it compares to the weak-label figures.

Usage:
    python validation_uncertainty.py --dir ../validation100
"""

import argparse
import os

import numpy as np
import pandas as pd

SHORT = ["training", "inference", "sampling", "loading", "memory", "context",
         "quantization"]


def f1_from(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    if tp == 0:
        return 0.0, tp, fp, fn
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec), tp, fp, fn


def macro_micro(ref, pred):
    f1s, TP, FP, FN = [], 0, 0, 0
    for j in range(ref.shape[1]):
        f1, tp, fp, fn = f1_from(ref[:, j], pred[:, j])
        # A category with no reference positives contributes nothing to macro:
        # there is nothing to detect, so scoring it would be meaningless.
        if ref[:, j].sum() > 0:
            f1s.append(f1)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    macro = float(np.mean(f1s)) if f1s else 0.0
    if TP == 0:
        return macro, 0.0
    p, r = TP / (TP + FP), TP / (TP + FN)
    return macro, 2 * p * r / (p + r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="../validation100")
    parser.add_argument("--iters", type=int, default=10000)
    args = parser.parse_args()

    ref = pd.read_csv(os.path.join(args.dir, "llm_labels.csv")).sort_values("id")
    kt = pd.read_csv(os.path.join(args.dir, "kaizentracer.csv")).sort_values("id")
    R = ref[SHORT].astype(int).values
    P = kt[SHORT].astype(int).values
    n = len(R)

    obs_macro, obs_micro = macro_micro(R, P)
    print(f"[unc] n = {n} functions")
    print(f"[unc] observed  macro F1 {obs_macro:.3f}   micro F1 {obs_micro:.3f}")
    print()

    rng = np.random.default_rng(42)

    # --- bootstrap over functions -----------------------------------------
    bm, bu = [], []
    for _ in range(args.iters):
        idx = rng.integers(0, n, n)
        a, b = macro_micro(R[idx], P[idx])
        bm.append(a)
        bu.append(b)
    bm, bu = np.array(bm), np.array(bu)
    print("[unc] bootstrap 95% CI (resampling functions):")
    print(f"        macro F1  {np.percentile(bm, 2.5):.3f} - {np.percentile(bm, 97.5):.3f}")
    print(f"        micro F1  {np.percentile(bu, 2.5):.3f} - {np.percentile(bu, 97.5):.3f}")
    print()

    # --- permutation null --------------------------------------------------
    # Shuffle each pattern's predictions across functions independently. The
    # number of predicted positives per pattern is preserved, so the null
    # detector is exactly as "eager" as KaizenTracer -- only the association
    # with the actual function is destroyed.
    nm, nu = [], []
    for _ in range(args.iters):
        Q = P.copy()
        for j in range(Q.shape[1]):
            rng.shuffle(Q[:, j])
        a, b = macro_micro(R, Q)
        nm.append(a)
        nu.append(b)
    nm, nu = np.array(nm), np.array(nu)
    print("[unc] chance baseline (predictions permuted across functions):")
    print(f"        macro F1  mean {nm.mean():.3f}  95th pct {np.percentile(nm, 95):.3f}")
    print(f"        micro F1  mean {nu.mean():.3f}  95th pct {np.percentile(nu, 95):.3f}")
    print()
    p_macro = float((nm >= obs_macro).mean())
    p_micro = float((nu >= obs_micro).mean())
    print(f"[unc] permutation p-value  macro {p_macro:.4f}   micro {p_micro:.4f}")
    print()
    lift_macro = obs_macro - nm.mean()
    lift_micro = obs_micro - nu.mean()
    print(f"[unc] lift over chance     macro {lift_macro:+.3f}   micro {lift_micro:+.3f}")

    pd.DataFrame([{
        "n": n,
        "macro_f1": round(obs_macro, 3),
        "macro_ci_lo": round(float(np.percentile(bm, 2.5)), 3),
        "macro_ci_hi": round(float(np.percentile(bm, 97.5)), 3),
        "macro_chance_mean": round(float(nm.mean()), 3),
        "macro_p": p_macro,
        "micro_f1": round(obs_micro, 3),
        "micro_ci_lo": round(float(np.percentile(bu, 2.5)), 3),
        "micro_ci_hi": round(float(np.percentile(bu, 97.5)), 3),
        "micro_chance_mean": round(float(nu.mean()), 3),
        "micro_p": p_micro,
    }]).to_csv(os.path.join(args.dir, "validation_uncertainty.csv"), index=False)
    print(f"\n[unc] wrote {args.dir}/validation_uncertainty.csv")


if __name__ == "__main__":
    main()
