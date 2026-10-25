"""Table IV: Leave-One-Project-Out generalization.

Trains on nine repositories and tests on the tenth, repeating for all ten, so
every score measures transfer to a codebase the classifier has never seen. This
is a stricter question than k-fold over pooled functions: repositories differ in
style, size and domain focus, and a model that merely memorises llama.cpp's
idioms will not carry to OpenVINO GenAI.

The protocol matches Table III and Fig. 3 so numbers are comparable:

  * functions without a CFG dropped
  * one row per distinct feature vector, computed within each split so a
    vector shared by the training and held-out projects cannot leak
  * the feature(s) each weak rule is defined on withheld
  * tau_k per Eq. 2 fitted on out-of-bag scores of the training projects only

A pattern absent from a held-out project contributes nothing to that project's
macro average -- it is skipped rather than scored zero, since there is nothing
to detect.

Usage:
    python table4_lopo.py --in_csv ../results/functions_v4.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from ablation_check import DEFINING
from table4_threshold import best_tau
from train_classifier import FUSED_COLS
from weak_labeling import PATTERNS, weak_label_dataframe

warnings.filterwarnings("ignore")


def fit_predict(train_df, test_df, cols, y_col, n_estimators):
    ytr = train_df[y_col].values
    yte = test_df[y_col].values
    if len(set(ytr)) < 2 or yte.sum() == 0:
        return None, yte
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                                 random_state=42,
                                 class_weight="balanced_subsample",
                                 oob_score=True, bootstrap=True, n_jobs=-1)
    clf.fit(train_df[cols].fillna(0).values, ytr)
    oob = clf.oob_decision_function_[:, 1]
    ok = ~np.isnan(oob)
    tau = best_tau(ytr[ok], oob[ok])
    proba = clf.predict_proba(test_df[cols].fillna(0).values)[:, 1]
    return (proba >= tau).astype(int), yte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--out", default="../results/table5_lopo_v4.csv")
    parser.add_argument("--n_estimators", type=int, default=800)
    args = parser.parse_args()

    raw = pd.read_csv(args.in_csv)
    df = weak_label_dataframe(raw[raw["cfg_nodes"] > 0].reset_index(drop=True))
    projects = sorted(df["project"].unique())
    print(f"[lopo] {len(df)} functions across {len(projects)} projects\n")

    rows = []
    for held in projects:
        train = df[df["project"] != held]
        test = df[df["project"] == held]
        # Deduplicate within each side, and drop from training any vector that
        # also occurs in the held-out project: an identical function present in
        # both would otherwise be memorised rather than generalised to.
        train = train.drop_duplicates(subset=FUSED_COLS)
        test = test.drop_duplicates(subset=FUSED_COLS)
        overlap = train.merge(test[FUSED_COLS].drop_duplicates(),
                              on=FUSED_COLS, how="inner").shape[0]
        keys = set(map(tuple, test[FUSED_COLS].values))
        mask = [tuple(v) not in keys for v in train[FUSED_COLS].values]
        train = train[mask]

        f1s, yts, yps = [], [], []
        for pattern in PATTERNS:
            cols = [c for c in FUSED_COLS if c not in DEFINING[pattern]]
            pred, yte = fit_predict(train, test, cols, f"label_{pattern}",
                                    args.n_estimators)
            if pred is None:
                continue
            f1s.append(f1_score(yte, pred, zero_division=0))
            yts.append(yte)
            yps.append(pred)

        if not f1s:
            print(f"[lopo] {held}: no evaluable pattern, skipping")
            rows.append({"Held-out Repository": held, "Patterns": 0,
                         "Macro F1": None, "Micro F1": None})
            continue

        yt, yp = np.concatenate(yts), np.concatenate(yps)
        macro = float(np.mean(f1s))
        micro = f1_score(yt, yp, zero_division=0)
        rows.append({
            "Held-out Repository": held,
            "Test Functions": len(test),
            "Patterns": len(f1s),
            "Macro F1": round(macro, 3),
            "Micro F1": round(micro, 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Recall": round(recall_score(yt, yp, zero_division=0), 3),
        })
        print(f"[lopo] {held:20s} macro F1 {macro:.3f}  micro F1 {micro:.3f}  "
              f"({len(f1s)} patterns, {len(test)} functions, "
              f"{overlap} shared vectors removed)", flush=True)

    table = pd.DataFrame(rows)
    avg = {"Held-out Repository": "Average",
           "Macro F1": round(table["Macro F1"].mean(), 3),
           "Micro F1": round(table["Micro F1"].mean(), 3)}
    table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    table.to_csv(args.out, index=False)
    print()
    print("=== Table IV: Leave-One-Project-Out generalization ===")
    print(table.to_string(index=False))
    print(f"\n[lopo] wrote {args.out}")


if __name__ == "__main__":
    main()
