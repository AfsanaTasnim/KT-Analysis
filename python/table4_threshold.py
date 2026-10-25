"""Table IV with the per-pattern decision threshold tau_k of Eq. 3.

`RandomForestClassifier.predict()` applies a fixed majority vote, i.e.
tau_k = 0.5 for every pattern. Eq. 3 of the paper does not: it defines a
*per-pattern* threshold

    yhat_{f,k} = I( (1/T) sum_t h_{t,k}(X_f) >= tau_k ).

The distinction matters for the rare patterns. Training Pipeline has 44
positives in 6050 functions (0.73%); at tau = 0.5 the forest almost never
votes positive, scoring 0.000 precision/recall/F1 -- an artefact of the
threshold, not a measurement of the pattern.

tau_k is selected inside each cross-validation fold, on the training half
only, as the value maximising F1 there; it is then applied unchanged to the
held-out half. The test fold never influences the threshold, so the reported
scores stay honest.

Usage:
    python table4_threshold.py --in_csv ../results/functions_v4.csv --out ../results/table4_final.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from ablation_check import DEFINING
from train_classifier import FUSED_COLS, prepare_for_evaluation
from weak_labeling import PATTERNS, weak_label_dataframe

# Candidate thresholds. The grid reaches well below 0.5 because a forest
# trained on a <1% class rarely puts a majority of trees on the positive side
# even for true positives.
TAU_GRID = np.round(np.arange(0.02, 0.81, 0.02), 2)


def best_tau(y_true, proba):
    """tau maximising F1; on ties the largest (most conservative) wins."""
    best, best_f1 = 0.5, -1.0
    for tau in TAU_GRID:
        f1 = f1_score(y_true, (proba >= tau).astype(int), zero_division=0)
        if f1 >= best_f1:
            best, best_f1 = tau, f1
    return best


def evaluate(X, y, kf, n_estimators, seed=42):
    """Out-of-fold predictions using a tau_k fitted per fold on training data.

    tau is selected from *out-of-bag* probabilities, not from
    ``predict_proba`` on the training rows. A forest classifies the data it was
    grown on almost perfectly, so in-sample scores pile up near 0 and 1; a
    threshold tuned against them is calibrated to a distribution that never
    occurs at test time. Each OOB score comes from the trees that did not see
    that row, which is the same regime the held-out fold is in.
    """
    pred = np.zeros(len(y), dtype=int)
    taus = []
    for train_idx, test_idx in kf.split(X, y):
        # min_samples_leaf=5 was the best of four models compared in
        # improve_table4.py (macro F1 0.446 vs 0.391-0.433); leaves of one
        # sample overfit the rarer patterns, which have only tens of positives.
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     min_samples_leaf=5,
                                     random_state=seed,
                                     class_weight="balanced_subsample",
                                     oob_score=True, bootstrap=True,
                                     n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])

        oob = clf.oob_decision_function_[:, 1]
        usable = ~np.isnan(oob)  # a row is unusable if every tree saw it
        tau = best_tau(y[train_idx][usable], oob[usable])
        taus.append(tau)

        pred[test_idx] = (clf.predict_proba(X[test_idx])[:, 1] >= tau).astype(int)
    return (precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0),
            float(np.mean(taus)),
            pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--out", default="../results/table4_final.csv")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    raw = pd.read_csv(args.in_csv)
    clean, stats = prepare_for_evaluation(raw)
    for key, value in stats.items():
        print(f"[table4] {key}: {value}")
    print()

    df = weak_label_dataframe(clean)
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    rows = []
    all_true, all_pred = [], []
    for pattern in PATTERNS:
        y = df[f"label_{pattern}"].values
        kept = [c for c in FUSED_COLS if c not in DEFINING[pattern]]
        X = df[kept].fillna(0).values
        p, r, f1, tau, pred = evaluate(X, y, kf, args.n_estimators)
        all_true.append(y)
        all_pred.append(pred)
        rows.append({
            "Pattern Category": pattern,
            "Positives": int(y.sum()),
            "Precision": round(p, 3),
            "Recall": round(r, 3),
            "F1-Score": round(f1, 3),
            "tau_k": round(tau, 2),
        })
        print(f"[table4] {pattern}: P={p:.3f} R={r:.3f} F1={f1:.3f} "
              f"(mean tau_k={tau:.2f})", flush=True)

    out = pd.DataFrame(rows)
    macro = out[["Precision", "Recall", "F1-Score"]].mean().round(3)
    out = pd.concat([out, pd.DataFrame([{
        "Pattern Category": "Macro Average", **macro.to_dict()}])],
        ignore_index=True)

    # Micro average: pool TP/FP/FN over every (function, pattern) decision, so
    # the frequent patterns dominate -- unlike the macro average, which weights
    # a 44-positive pattern the same as a 2141-positive one.
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    out = pd.concat([out, pd.DataFrame([{
        "Pattern Category": "Micro Average",
        "Precision": round(precision_score(yt, yp, zero_division=0), 3),
        "Recall": round(recall_score(yt, yp, zero_division=0), 3),
        "F1-Score": round(f1_score(yt, yp, zero_division=0), 3)}])],
        ignore_index=True)
    out.to_csv(args.out, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\n[table4] wrote {args.out}")


if __name__ == "__main__":
    main()
