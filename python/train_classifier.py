"""
train_classifier.py

Implements the ML-based Pattern Detection stage (Fig. 1, box 9) and
Algorithm 2 (line 10, MultiLabelClassifier). Trains a multi-label Random
Forest over the fused AST-CFG feature vectors X_f, using the weak labels
produced by weak_labeling.py as the training signal, then reports
per-pattern Precision/Recall/F1 (Table IV, Fig. 2) via cross-validated
held-out predictions.

Prediction rule (Eq. 3):
    yhat_{f,k} = 1[ (1/T) * sum_t h_{t,k}(X_f) >= tau_k ]
This is exactly what RandomForestClassifier.predict() computes per label
with tau_k = 0.5 (the scikit-learn default majority vote); the fused
feature vector X_f = [A_f || C_f] (Eq. 1) is the AST_COLS + CFG_COLS below.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from weak_labeling import PATTERNS, GENERAL_LABEL, label_columns, weak_label_dataframe

AST_COLS = [
    "ast_loops", "ast_branches", "ast_calls", "ast_vars", "ast_memory_ops",
    "ast_model_loading_calls", "ast_sampling_calls", "ast_training_calls",
    "ast_quantization_calls", "ast_context_calls", "ast_inference_calls",
    "ast_depth",
    # Domain-agnostic syntactic evidence: describes what the code does rather
    # than what its callees are named, so a pattern remains detectable when the
    # keyword counts are withheld.
    "ast_distinct_callees", "ast_params", "ast_returns", "ast_throws",
    "ast_try_catch", "ast_pointer_derefs", "ast_array_subscripts",
    "ast_bitwise_ops", "ast_float_ops", "ast_int_casts", "ast_comparisons",
    "ast_string_literals", "ast_sizeof_ops", "ast_assignments",
]
CFG_COLS = [
    "cfg_nodes", "cfg_edges", "cfg_branch_nodes", "cfg_loop_nodes",
    "cfg_exit_nodes", "cfg_back_edges", "cfg_cyclomatic_complexity",
    "cfg_density", "cfg_loop_depth", "cfg_validation_paths",
]
FUSED_COLS = AST_COLS + CFG_COLS  # X_f = [A_f || C_f]  (Eq. 1)


def prepare_for_evaluation(df: pd.DataFrame, require_cfg=True, dedupe=True):
    """Remove two sources of optimistic bias before cross-validation.

    * ``require_cfg`` drops functions for which clang could not build a CFG
      (``cfg_nodes == 0``). Their ten CFG columns are zeros standing in for
      *missing* measurements, not observations, so training on them teaches
      the model an artefact of parse failure.
    * ``dedupe`` keeps one row per distinct fused feature vector. Small
      functions are structurally identical in bulk -- one vector occurs 406
      times -- and because the weak label is a deterministic function of the
      vector, duplicates put the exact test row in the training fold. That is
      leakage, and it inflates every score.

    Returns the filtered frame and a dict describing what was removed, so the
    reduction can be reported rather than hidden.
    """
    stats = {"input_rows": len(df)}
    out = df
    if require_cfg:
        out = out[out["cfg_nodes"] > 0]
        stats["dropped_no_cfg"] = stats["input_rows"] - len(out)
    if dedupe:
        before = len(out)
        out = out.drop_duplicates(subset=FUSED_COLS, keep="first")
        stats["dropped_duplicate_vectors"] = before - len(out)
    stats["evaluated_rows"] = len(out)
    return out.reset_index(drop=True), stats


def train_and_evaluate(df: pd.DataFrame, n_estimators=300, n_splits=5,
                        random_state=42):
    """Trains a multi-label Random Forest (one-vs-rest ensemble across the
    fused feature vector) and returns per-pattern P/R/F1 computed from
    cross-validated held-out predictions, plus the fitted final model."""
    labeled = weak_label_dataframe(df)
    X = labeled[FUSED_COLS].fillna(0).values
    label_cols = label_columns()
    Y = labeled[label_cols].values

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    # Cross-validated held-out predictions, computed once per pattern and
    # reused for both the per-pattern scores and the micro average. (Running
    # the CV a second time for micro pooling would double the cost of the
    # whole stage and, because KFold is seeded, produce identical folds.)
    cv_preds = []
    results = []
    for i, pattern_col in enumerate(label_cols):
        y = Y[:, i]
        if len(set(y)) < 2:
            # Degenerate case: a pattern with zero positive (or zero
            # negative) weak labels in this dataset -- skip CV, report NA.
            # The trivial column is still fed to the micro pool, where a
            # constant prediction equal to the truth adds no error.
            cv_preds.append(y)
            results.append((pattern_col, float("nan"), float("nan"),
                             float("nan")))
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced_subsample",
        )
        y_pred = cross_val_predict(clf, X, y, cv=kf)
        cv_preds.append(y_pred)
        p = precision_score(y, y_pred, zero_division=0)
        r = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        results.append((pattern_col, p, r, f1))

    metrics_df = pd.DataFrame(
        results, columns=["pattern", "precision", "recall", "f1"]
    )
    metrics_df["pattern"] = metrics_df["pattern"].str.replace("label_", "",
                                                                regex=False)

    # Table IV reports the seven taxonomy patterns. "General Implementation
    # Structure" is the residual assigned when no rule matched, and it is the
    # majority class (~62% of functions), so scoring it as an eighth label and
    # folding it into the averages measures how well the model predicts
    # "none of the above" and pulls both averages up.
    metrics_df = metrics_df[metrics_df["pattern"] != GENERAL_LABEL]
    scored = [i for i, c in enumerate(label_cols)
              if c != f"label_{GENERAL_LABEL}"]

    valid = metrics_df.dropna()
    macro = valid[["precision", "recall", "f1"]].mean()
    metrics_df = pd.concat(
        [metrics_df,
         pd.DataFrame([{"pattern": "Macro Average", **macro.to_dict()}])],
        ignore_index=True,
    )

    # Micro average: pool TP/FP/FN across the seven scored patterns.
    mp, mr, mf1, _ = precision_recall_fscore_support(
        Y[:, scored].ravel(),
        np.column_stack([cv_preds[i] for i in scored]).ravel(),
        average="binary", zero_division=0,
    )
    metrics_df = pd.concat(
        [metrics_df,
         pd.DataFrame([{"pattern": "Micro Average", "precision": mp,
                         "recall": mr, "f1": mf1}])],
        ignore_index=True,
    )

    # Fit the final model (one RF per pattern) on the full dataset for
    # downstream use (project_comparison.py / detected pattern export).
    final_models = {}
    for i, pattern_col in enumerate(label_cols):
        y = Y[:, i]
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced_subsample",
        )
        if len(set(y)) >= 2:
            clf.fit(X, y)
        final_models[pattern_col] = clf

    predicted = labeled.copy()
    for i, pattern_col in enumerate(label_cols):
        clf = final_models[pattern_col]
        try:
            predicted[f"pred_{pattern_col}"] = clf.predict(X)
        except Exception:
            predicted[f"pred_{pattern_col}"] = 0

    return metrics_df, predicted, final_models


def plot_f1_scores(metrics_df: pd.DataFrame, out_path="f1_scores.png"):
    """Reproduces Fig. 2 (F1 Score Comparison for KaizenTracer)."""
    plot_df = metrics_df[~metrics_df["pattern"].isin([GENERAL_LABEL])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_df["pattern"], plot_df["f1"], marker="o")
    for x, y in zip(plot_df["pattern"], plot_df["f1"]):
        if pd.notna(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center")
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Pattern Category")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1 Score Comparison for KaizenTracer")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[train_classifier] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--metrics_csv", default="pattern_metrics.csv")
    parser.add_argument("--predictions_csv", default="functions_predicted.csv")
    parser.add_argument("--fig", default="f1_scores.png")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    metrics_df, predicted, _ = train_and_evaluate(
        df, n_estimators=args.n_estimators, n_splits=args.n_splits
    )
    metrics_df.to_csv(args.metrics_csv, index=False)
    predicted.to_csv(args.predictions_csv, index=False)
    plot_f1_scores(metrics_df, out_path=args.fig)
    print(metrics_df.to_string(index=False))
