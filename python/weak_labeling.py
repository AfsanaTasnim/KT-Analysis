"""
weak_labeling.py

Implements Algorithm 2 (weak-labeling portion, lines 1-9) from the paper:
for each function's fused AST-CFG feature vector X_f, apply domain-specific
rules Rule_p over the seven pattern categories:

    Training Pipeline, Interactive Inference, Advanced Sampling,
    Model Loading and Validation, Memory Management, Context Management,
    Quantization Workflow

A function that matches no rule is weakly labeled 'General Implementation
Structure' (Algorithm 2, lines 7-9).

These weak labels are the *training signal* used by train_classifier.py to
fit the supervised multi-label Random Forest (Algorithm 2, line 10).
"""

import pandas as pd

PATTERNS = [
    "Training Pipeline",
    "Interactive Inference",
    "Advanced Sampling",
    "Model Loading and Validation",
    "Memory Management",
    "Context Management",
    "Quantization Workflow",
]

GENERAL_LABEL = "General Implementation Structure"


def _rule_training_pipeline(row: pd.Series) -> bool:
    # Backward propagation, optimizer-step, or loss-computation evidence.
    #
    # The original rule additionally required a loop, justified as
    # "batch-generation structure". Nothing about a parameter update entails
    # iteration: a gradient-buffer update or a backward kernel dispatch is
    # often a short, loop-free function, and the conjunct excluded them.
    return row["ast_training_calls"] > 0


def _rule_interactive_inference(row: pd.Series) -> bool:
    # Prompt handling + token-generation-style call pattern: at least one
    # branch (turn-taking / EOS check) and at least one function call,
    # without dominant training signal.
    # The original rule asked only for a call, a branch, and either a sampling
    # call or a loop -- satisfied by most non-trivial C++, which made this the
    # loosest of the seven rules and, once the others were tightened, the
    # dominant pattern in eight of ten projects. Require generation evidence:
    # a token/decode/prompt-style operation, or explicit sampling.
    # The ast_branches > 0 conjunct is dropped. It was justified as a
    # "turn-taking / EOS check", but the category is a *generation step*, and
    # the generation steps that matter are routinely straight-line: a
    # transformer block's forward(), an embed_token(), a decoder Run() are all
    # branch-free, so the conjunct excluded precisely the functions the pattern
    # is meant to name while admitting nothing in return.
    return (
        row["ast_calls"] > 0
        and row["ast_training_calls"] == 0
        and (row.get("ast_inference_calls", 0) > 0
             or row["ast_sampling_calls"] > 0)
    )


def _rule_advanced_sampling(row: pd.Series) -> bool:
    # top-k / top-p / temperature / repeat-penalty logic.
    return row["ast_sampling_calls"] > 0


def _rule_model_loading_validation(row: pd.Series) -> bool:
    # Model-artefact loading calls, optionally corroborated by failure-handling
    # paths.
    #
    # The original rule was a disjunction: model-loading calls OR (a validation
    # path AND any call). The second clause supplied roughly 82% of this
    # pattern's labels while testing nothing about model loading -- it fires on
    # any function invoking a callee named check/error/valid, which is most
    # non-trivial C++ code. Requiring the loading evidence makes the rule test
    # what the category name claims; validation now corroborates rather than
    # substitutes.
    return row["ast_model_loading_calls"] > 0


def _rule_memory_management(row: pd.Series) -> bool:
    return row["ast_memory_ops"] > 0


def _rule_context_management(row: pd.Series) -> bool:
    # Context-window / KV-cache bookkeeping: state that is both substantial and
    # revisited, rather than initialised once.
    #
    # The original thresholds (>=3 locals and any loop) match roughly a fifth
    # of all functions, since three locals and a loop describe most ordinary
    # C++ code. Requiring more local state, more than one loop structure, and
    # some pointer or array indexing into that state narrows the rule to
    # functions that actually traverse a maintained buffer.
    # The structural disjunct (>=8 locals, >=2 loops, array or pointer access)
    # is dropped. It predates this category having a vocabulary, when the rule
    # had to be defined structurally or not at all. It describes any dense
    # numeric kernel -- corpus-wide it admits 657 functions on top of the 1024
    # the vocabulary finds, and in the reference sample 8 of its 10 exclusive
    # firings are wrong (ggml_gemv_* matrix kernels, graph-copy helpers). With
    # a vocabulary in place it only adds noise, and removing it makes this rule
    # consistent with the other six, which are all purely name-based.
    named = row.get("ast_context_calls", 0) > 0
    return (
        named
        and row["ast_training_calls"] == 0
        and row["ast_sampling_calls"] == 0
    )


def _rule_quantization_workflow(row: pd.Series) -> bool:
    return row["ast_quantization_calls"] > 0


RULES = {
    "Training Pipeline": _rule_training_pipeline,
    "Interactive Inference": _rule_interactive_inference,
    "Advanced Sampling": _rule_advanced_sampling,
    "Model Loading and Validation": _rule_model_loading_validation,
    "Memory Management": _rule_memory_management,
    "Context Management": _rule_context_management,
    "Quantization Workflow": _rule_quantization_workflow,
}


def weak_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one binary column per pattern (weak label y^w_{f,k}, Eq. 2)."""
    df = df.copy()
    for pattern, rule in RULES.items():
        col = f"label_{pattern}"
        df[col] = df.apply(rule, axis=1).astype(int)

    label_cols = [f"label_{p}" for p in PATTERNS]
    no_match = df[label_cols].sum(axis=1) == 0
    df[f"label_{GENERAL_LABEL}"] = no_match.astype(int)
    return df


def label_columns():
    return [f"label_{p}" for p in PATTERNS] + [f"label_{GENERAL_LABEL}"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", default="../results/functions_v4.csv")
    parser.add_argument("--out_csv", default="functions_weak_labeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    labeled = weak_label_dataframe(df)
    labeled.to_csv(args.out_csv, index=False)
    print(f"[weak_labeling] wrote {len(labeled)} rows -> {args.out_csv}")
    print(labeled[label_columns()].sum().sort_values(ascending=False))
