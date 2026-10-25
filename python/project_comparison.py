"""
project_comparison.py

Implements Fig. 1 box 10 (Project Comparison) and Section V.C: aggregates
function-level predicted pattern labels into project-level pattern
*profiles* (Table II), then computes pairwise Jaccard similarity between
projects (Eq. 9) and renders the heatmap in Fig. 3.

A project's pattern profile is the set of patterns for which at least one
function in that project was predicted positive (this matches the paper's
definition: "each project is represented as a set of detected pattern
categories").
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from weak_labeling import PATTERNS, GENERAL_LABEL


def build_project_profiles(predicted: pd.DataFrame,
                            min_support_frac: float = 0.0) -> dict:
    """Returns {project_name: set(pattern_name)} using pred_label_* cols.

    ``min_support_frac`` is the fraction of a project's functions that must
    carry a pattern before the project counts as exhibiting it.

    The paper's definition is "at least one function" (``0.0``). Be aware that
    this saturates: with hundreds or thousands of functions per project, every
    project matches every pattern at least once, so all profiles become the
    full pattern set and every pairwise Jaccard similarity is exactly 1.00.
    A small positive threshold (e.g. ``0.05``) asks instead whether a pattern
    is *characteristic* of the project, which is what makes Fig. 3 informative.
    """
    profiles = {}
    all_patterns = PATTERNS + [GENERAL_LABEL]
    for project, group in predicted.groupby("project"):
        threshold = min_support_frac * len(group)
        detected = set()
        for pattern in all_patterns:
            col = f"pred_label_{pattern}"
            if col in group.columns and group[col].sum() > threshold:
                detected.add(pattern)
        profiles[project] = detected
    return profiles


def project_pattern_counts(predicted: pd.DataFrame) -> pd.DataFrame:
    """Reproduces Table II: per-project function counts for each pattern,
    plus the dominant (most frequent) pattern for that project."""
    # Table II reports the seven taxonomy patterns only. "General
    # Implementation Structure" is the residual assigned when no rule matched
    # (Algorithm 2, lines 7-9), so it is not a column of the table; it is
    # counted here solely to report what share of each project the taxonomy
    # leaves uncovered.
    rows = []
    for project, group in predicted.groupby("project"):
        row = {"project": project}
        counts = {}
        for pattern in PATTERNS:
            col = f"pred_label_{pattern}"
            n = int(group[col].sum()) if col in group.columns else 0
            row[pattern] = n
            counts[pattern] = n
        general_col = f"pred_label_{GENERAL_LABEL}"
        general = int(group[general_col].sum()) \
            if general_col in group.columns else 0
        # The dominant pattern is chosen among the seven taxonomy patterns
        # only. "General Implementation Structure" is the residual assigned
        # when no rule matched (Algorithm 2, lines 7-9), not a pattern the
        # project exhibits -- and since most functions in any codebase are
        # small helpers, it is the largest bucket in every project, which made
        # this column read "General" for all ten and carry no information.
        # A project matching no pattern has no dominant one; max() over an
        # all-zero mapping silently returns whichever key comes first, which
        # reads as a real finding (gpt2.cpp appeared "Training Pipeline
        # dominant" with a share of 0.0%).
        if max(counts.values()) == 0:
            row["Dominant Pattern"] = "None detected"
            row["Dominant Share"] = 0.0
        else:
            row["Dominant Pattern"] = max(counts, key=counts.get)
            row["Dominant Share"] = round(
                100.0 * max(counts.values()) / max(1, len(group)), 1)
        row["Uncovered"] = round(100.0 * general / max(1, len(group)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def jaccard_matrix(profiles: dict) -> pd.DataFrame:
    projects = list(profiles.keys())
    mat = pd.DataFrame(index=projects, columns=projects, dtype=float)
    for p1, p2 in itertools.product(projects, projects):
        mat.loc[p1, p2] = jaccard_similarity(profiles[p1], profiles[p2])
    return mat


def plot_jaccard_heatmap(mat: pd.DataFrame, out_path="jaccard_heatmap.png"):
    """Reproduces Fig. 3 (Jaccard Similarity Between Projects)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
                vmin=0, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Jaccard similarity"})
    ax.set_title("Jaccard Similarity Between Projects")
    ax.set_xlabel("Project")
    ax.set_ylabel("Project")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[project_comparison] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_csv", default="../results/functions_predicted_v4.csv")
    parser.add_argument("--table_out", default="table2_patterns_by_project.csv")
    parser.add_argument("--jaccard_out", default="jaccard_matrix.csv")
    parser.add_argument("--fig", default="jaccard_heatmap.png")
    args = parser.parse_args()

    predicted = pd.read_csv(args.predictions_csv)
    table2 = project_pattern_counts(predicted)
    table2.to_csv(args.table_out, index=False)
    print(table2.to_string(index=False))

    profiles = build_project_profiles(predicted)
    mat = jaccard_matrix(profiles)
    mat.to_csv(args.jaccard_out)
    plot_jaccard_heatmap(mat, out_path=args.fig)
