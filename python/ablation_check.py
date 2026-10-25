"""Diagnose and correct the circularity in Table IV.

The weak labels are computed by `weak_labeling.py` from the very features the
classifier is then given, so several patterns are a deterministic function of a
single input column (Advanced Sampling is exactly `ast_sampling_calls > 0`).
A Random Forest recovers such a rule on its first split, which is why those
patterns score a perfect 1.000 -- the number measures rule recoverability, not
pattern detection.

This script reports, per pattern:

  * where the forest's importance actually sits (the defining feature dominates)
  * F1 when the defining feature(s) are *withheld*, so the model must recover
    the pattern from the remaining fused AST-CFG structure

The ablated column is the informative one: it answers "is this pattern visible
in a function's structure, or only in the keyword count that defined it?" -- a
direct test of the AST-CFG fusion claim in RQ1, and a defensible Table IV that
does not require human annotation.

Usage:
    python ablation_check.py --in_csv ../results/functions_v4.csv --out ../results/table4_ablation.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train_classifier import FUSED_COLS
from weak_labeling import PATTERNS, weak_label_dataframe

# The feature(s) each rule is literally computed from. Withholding these forces
# the classifier to find the pattern in the rest of the fused vector.
DEFINING = {
    "Training Pipeline":            ["ast_training_calls"],
    "Interactive Inference":        ["ast_inference_calls", "ast_sampling_calls",
                                      "ast_training_calls"],
    "Advanced Sampling":            ["ast_sampling_calls"],
    "Model Loading and Validation": ["ast_model_loading_calls",
                                      "cfg_validation_paths"],
    "Memory Management":            ["ast_memory_ops"],
    "Context Management":           ["ast_vars", "ast_context_calls",
                                      "ast_sampling_calls",
                                      "ast_training_calls"],
    "Quantization Workflow":        ["ast_quantization_calls"],
}


def score(X, y, cols, n_estimators, kf, seed):
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan"), None
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                 class_weight="balanced_subsample", n_jobs=-1)
    pred = cross_val_predict(clf, X, y, cv=kf)
    clf.fit(X, y)
    importance = pd.Series(clf.feature_importances_, index=cols)
    return (precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0),
            importance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--out", default="../results/table4_ablation.csv")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    df = weak_label_dataframe(pd.read_csv(args.in_csv))
    # Stratified: the rarer patterns have <1% positives (Training Pipeline
    # has 44 of 6050), and unstratified folds vary from 7 to 11 positives,
    # which makes per-fold scores swing on a handful of functions.
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    rows = []
    for pattern in PATTERNS:
        y = df[f"label_{pattern}"].values
        full = df[FUSED_COLS].fillna(0).values
        p_f, r_f, f_f, imp = score(full, y, FUSED_COLS, args.n_estimators,
                                    kf, 42)

        kept = [c for c in FUSED_COLS if c not in DEFINING[pattern]]
        p_a, r_a, f_a, _ = score(df[kept].fillna(0).values, y, kept,
                                  args.n_estimators, kf, 42)

        top = "-"
        if imp is not None:
            t = imp.sort_values(ascending=False).head(1)
            top = f"{t.index[0]} ({t.iloc[0]:.2f})"

        rows.append({
            "pattern": pattern,
            "positives": int(y.sum()),
            "f1_full": round(f_f, 4) if f_f == f_f else float("nan"),
            "precision_ablated": round(p_a, 4) if p_a == p_a else float("nan"),
            "recall_ablated": round(r_a, 4) if r_a == r_a else float("nan"),
            "f1_ablated": round(f_a, 4) if f_a == f_a else float("nan"),
            "top_feature_full": top,
            "withheld": ", ".join(DEFINING[pattern]),
        })
        print(f"[ablation] {pattern}: F1 {f_f:.3f} -> {f_a:.3f} "
              f"(withheld {', '.join(DEFINING[pattern])})", flush=True)

    out = pd.DataFrame(rows)
    macro = out[["precision_ablated", "recall_ablated", "f1_ablated"]].mean()
    out = pd.concat([out, pd.DataFrame([{
        "pattern": "Macro Average",
        "f1_full": round(out.f1_full.mean(), 4),
        **{k: round(v, 4) for k, v in macro.items()},
    }])], ignore_index=True)
    out.to_csv(args.out, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\n[ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
