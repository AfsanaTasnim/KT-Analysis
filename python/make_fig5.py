"""Fig. 5: Spearman correlation between AST/CFG-derived features and the
number of detected pattern categories per function, on the corrected corpus."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from weak_labeling import PATTERNS, weak_label_dataframe

d = weak_label_dataframe(pd.read_csv("../results/functions_v4.csv"))
d = d.drop_duplicates(subset=["project","file","function","start_line"])
d = d[d["cfg_nodes"] > 0]
d["n_pat"] = d[[f"label_{p}" for p in PATTERNS]].sum(axis=1)

FEATURES = {
    "Function-call\ncount": "ast_calls", "CFG loop\ndepth": "cfg_loop_depth",
    "Loop\ncount": "ast_loops", "CFG node\ncount": "cfg_nodes",
    "CFG edge\ncount": "cfg_edges", "AST\ndepth": "ast_depth",
    "Cyclomatic\ncomplexity": "cfg_cyclomatic_complexity",
    "CFG branch-\nnode count": "cfg_branch_nodes",
    "Branch\ncount": "ast_branches", "Validation-\npath count": "cfg_validation_paths",
}
rows = [(n, spearmanr(d[c], d["n_pat"]).statistic) for n, c in FEATURES.items()]
rows.sort(key=lambda r: -r[1])
names = [r[0] for r in rows]; vals = [r[1] for r in rows]

colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#64B5CD",
          "#8172B3","#FFA6B1","#8C6D5C","#C4BCB2","#7D6B8A"]
fig, ax = plt.subplots(figsize=(12.5, 7.2))
bars = ax.bar(names, vals, color=colors[:len(names)], width=0.72)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.2f}",
            ha="center", va="bottom", fontsize=12)
ax.set_ylim(0, 0.40)
ax.set_ylabel(r"Spearman Correlation ($\rho_s$)", fontsize=12)
ax.set_xlabel("AST/CFG-Derived Feature", fontsize=12)
ax.set_title("Spearman Correlation Between AST/CFG-Derived Features "
             "and Detected Pattern Evidence", fontsize=13)
ax.yaxis.grid(True, color="#E8E8E8"); ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
plt.tight_layout(); plt.savefig("../results/fig5_spearman_v4.png", dpi=200)
print("wrote ../results/fig5_spearman_v4.png")
for n, v in rows: print(f"  {n.replace(chr(10),' '):26s} {v:.3f}")
