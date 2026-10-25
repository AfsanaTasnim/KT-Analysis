"""Fig. 4: presence-based Jaccard similarity over the seven taxonomy patterns.

Each project is the set of the seven implementation patterns that occur at
least once in it (paper Section V-E). The residual "General Implementation
Structure" bucket is not a pattern and is excluded; project_comparison.py
includes it by default, which gives an eight-category profile and different
values (e.g. llama.cpp vs gpt2.cpp = 0.25 instead of 0.14).

Rows and columns follow the paper's project order. The script also compares
the result with the matrix printed in the paper.

Usage:
    python jaccard_7cat_v4.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import project_comparison as pc
from weak_labeling import PATTERNS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ORDER = ["llama.cpp", "gemma.cpp", "onnxruntime-genai", "openvino.genai",
         "lmdeploy", "gpt2.cpp", "minchatgpt.cpp", "InferLLM", "sglang",
         "DeepSpeed-FastGen"]
LABEL = {"onnxruntime-genai": "ONNX Runtime GenAI", "openvino.genai": "OpenVINO GenAI",
         "lmdeploy": "LMDeploy", "sglang": "SGLang"}

# The matrix as printed in the paper's Fig. 4, in the order above.
PAPER = np.array([
    [1.00, 1.00, 0.86, 0.86, 1.00, 0.14, 0.71, 0.86, 0.86, 1.00],
    [1.00, 1.00, 0.86, 0.86, 1.00, 0.14, 0.71, 0.86, 0.86, 1.00],
    [0.86, 0.86, 1.00, 1.00, 0.86, 0.17, 0.83, 1.00, 1.00, 0.86],
    [0.86, 0.86, 1.00, 1.00, 0.86, 0.17, 0.83, 1.00, 1.00, 0.86],
    [1.00, 1.00, 0.86, 0.86, 1.00, 0.14, 0.71, 0.86, 0.86, 1.00],
    [0.14, 0.14, 0.17, 0.17, 0.14, 1.00, 0.20, 0.17, 0.17, 0.14],
    [0.71, 0.71, 0.83, 0.83, 0.71, 0.20, 1.00, 0.83, 0.83, 0.71],
    [0.86, 0.86, 1.00, 1.00, 0.86, 0.17, 0.83, 1.00, 1.00, 0.86],
    [0.86, 0.86, 1.00, 1.00, 0.86, 0.17, 0.83, 1.00, 1.00, 0.86],
    [1.00, 1.00, 0.86, 0.86, 1.00, 0.14, 0.71, 0.86, 0.86, 1.00],
])


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "functions_predicted_v4.csv"))
    prof = pc.build_project_profiles(df)
    prof7 = {k: {p for p in v if p in PATTERNS} for k, v in prof.items()}
    mat = pc.jaccard_matrix(prof7).loc[ORDER, ORDER].astype(float)
    mat.round(2).to_csv(os.path.join(ROOT, "results", "jaccard_7cat_v4.csv"))

    print("pattern-set sizes:", {k: len(prof7[k]) for k in ORDER})
    print(mat.round(2).rename(index=LABEL, columns=LABEL).to_string())
    diff = np.abs(mat.round(2).values - PAPER)
    print(f"\nmax |ours - paper Fig. 4| = {diff.max():.2f}; "
          f"cells differing = {int((diff > 0.005).sum())} of {diff.size}")

    fig, ax = plt.subplots(figsize=(9, 7))
    shown = mat.rename(index=LABEL, columns=LABEL)
    sns.heatmap(shown, annot=True, fmt=".2f", cmap="cividis", vmin=0, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Jaccard Similarity"})
    ax.set_xlabel("Project")
    ax.set_ylabel("Project")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "jaccard_7cat_v4.png"), dpi=200)
    print("wrote results/jaccard_7cat_v4.csv and jaccard_7cat_v4.png")


if __name__ == "__main__":
    main()
