"""Table III: per-pattern ablation over AST-only, CFG-only and fused features.

Answers RQ1 -- does control-flow evidence add anything to syntactic evidence? --
by training the same classifier three times per pattern:

  AST Only   the ten AST columns
  CFG Only   the ten CFG columns
  AST+CFG    both, i.e. the fused vector X_f of Eq. 1

The evaluation protocol matches the one behind Fig. 3, so the AST+CFG column
reproduces those F1 values rather than contradicting them:

  * functions without a CFG are dropped (their CFG columns are zeros standing
    in for missing measurements, which would flatter the CFG-only column)
  * one row per distinct feature vector, so identical functions cannot appear
    in both the training and the test fold
  * the feature(s) each weak rule is literally computed from are withheld,
    otherwise a pattern like Advanced Sampling is just `ast_sampling_calls > 0`
    and every configuration containing that column scores 1.00
  * tau_k per Eq. 2 chosen inside each fold from out-of-bag probabilities

Withholding applies within each configuration, so "AST Only" means the AST
columns minus that pattern's defining ones.

Usage:
    python table3_feature_ablation.py --in_csv ../results/functions_v4.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from ablation_check import DEFINING
from table4_threshold import best_tau
from train_classifier import (AST_COLS, CFG_COLS, FUSED_COLS,
                              prepare_for_evaluation)
from weak_labeling import PATTERNS, weak_label_dataframe

warnings.filterwarnings("ignore")

CONFIGS = [("AST Only", AST_COLS), ("CFG Only", CFG_COLS),
           ("AST+CFG", FUSED_COLS)]


def evaluate(X, y, kf, n_estimators):
    """Out-of-fold predictions with tau_k fitted per fold on OOB scores."""
    pred = np.zeros(len(y), dtype=int)
    for tr, te in kf.split(X, y):
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     min_samples_leaf=5, random_state=42,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True,
                                     n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oob = clf.oob_decision_function_[:, 1]
        ok = ~np.isnan(oob)
        tau = best_tau(y[tr][ok], oob[ok])
        pred[te] = (clf.predict_proba(X[te])[:, 1] >= tau).astype(int)
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--out", default="../results/table3_ablation_v4.csv")
    parser.add_argument("--n_estimators", type=int, default=800)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    raw = pd.read_csv(args.in_csv)
    clean, stats = prepare_for_evaluation(raw)
    for k, v in stats.items():
        print(f"[table3] {k}: {v}")
    df = weak_label_dataframe(clean)
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    print()

    results = {name: {} for name, _ in CONFIGS}
    pooled = {name: {"true": [], "pred": []} for name, _ in CONFIGS}

    for pattern in PATTERNS:
        y = df[f"label_{pattern}"].values
        for name, cols in CONFIGS:
            usable = [c for c in cols if c not in DEFINING[pattern]]
            if not usable:
                results[name][pattern] = (float("nan"),) * 3
                continue
            pred = evaluate(df[usable].fillna(0).values, y, kf,
                            args.n_estimators)
            results[name][pattern] = (
                precision_score(y, pred, zero_division=0),
                recall_score(y, pred, zero_division=0),
                f1_score(y, pred, zero_division=0))
            pooled[name]["true"].append(y)
            pooled[name]["pred"].append(pred)
            print(f"[table3] {name:9s} {pattern:30s} "
                  f"F1={results[name][pattern][2]:.3f}", flush=True)
        print()

    rows = []
    for pattern in PATTERNS:
        row = {"Pattern": pattern}
        for name, _ in CONFIGS:
            p, r, f1 = results[name][pattern]
            row[f"{name} P"] = round(p, 3)
            row[f"{name} R"] = round(r, 3)
            row[f"{name} F1"] = round(f1, 3)
        rows.append(row)

    macro, micro = {"Pattern": "Macro Average"}, {"Pattern": "Micro Average"}
    for name, _ in CONFIGS:
        vals = np.array([results[name][p] for p in PATTERNS], dtype=float)
        macro[f"{name} P"] = round(np.nanmean(vals[:, 0]), 3)
        macro[f"{name} R"] = round(np.nanmean(vals[:, 1]), 3)
        macro[f"{name} F1"] = round(np.nanmean(vals[:, 2]), 3)
        yt = np.concatenate(pooled[name]["true"])
        yp = np.concatenate(pooled[name]["pred"])
        micro[f"{name} P"] = round(precision_score(yt, yp, zero_division=0), 3)
        micro[f"{name} R"] = round(recall_score(yt, yp, zero_division=0), 3)
        micro[f"{name} F1"] = round(f1_score(yt, yp, zero_division=0), 3)
    rows += [macro, micro]

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print("=== Table III: AST-only vs CFG-only vs fused AST+CFG ===")
    print(table.to_string(index=False))
    print(f"\n[table3] wrote {args.out}")


if __name__ == "__main__":
    main()
