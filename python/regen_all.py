"""Regenerate every label-dependent table and figure from one feature CSV."""
import argparse, os
import pandas as pd
from weak_labeling import PATTERNS, GENERAL_LABEL, weak_label_dataframe
import project_comparison as pc

ap=argparse.ArgumentParser()
ap.add_argument("--functions",default="../results/functions_v4.csv")
ap.add_argument("--tag",default="v4")
a=ap.parse_args()
T=a.tag

df=weak_label_dataframe(pd.read_csv(a.functions))
df=df.drop_duplicates(subset=["project","file","function","start_line"],keep="first")
for p in PATTERNS+[GENERAL_LABEL]:
    df[f"pred_label_{p}"]=df[f"label_{p}"]
df.to_csv(f"../results/functions_predicted_{T}.csv",index=False)
print(f"[regen] {len(df)} distinct functions")

# ---- Table II -----------------------------------------------------------
t2=pc.project_pattern_counts(df)
t2.to_csv(f"../results/table2_patterns_by_project_{T}.csv",index=False)
short={"Training Pipeline":"Train","Interactive Inference":"Infer","Advanced Sampling":"Sampl",
       "Model Loading and Validation":"Load","Memory Management":"Mem",
       "Context Management":"Ctx","Quantization Workflow":"Quant"}
print("\n=== TABLE II ===")
print(t2.rename(columns=short).to_string(index=False))

# ---- Fig 2: normalized per-project distribution -------------------------
rows=[]
for proj,g in df.groupby("project"):
    tot=sum(int(g[f"label_{p}"].sum()) for p in PATTERNS)
    r={"project":proj}
    for p in PATTERNS:
        r[short[p]]=round(100.0*int(g[f"label_{p}"].sum())/tot,1) if tot else 0.0
    rows.append(r)
fig2=pd.DataFrame(rows); fig2.to_csv(f"../results/fig2_normalized_{T}.csv",index=False)
print("\n=== FIG 2: share of detected patterns within each project (%) ===")
print(fig2.to_string(index=False))

# ---- Jaccard (Fig 4) ----------------------------------------------------
prof=pc.build_project_profiles(df, min_support_frac=0.0)
mat=pc.jaccard_matrix(prof); mat.to_csv(f"../results/jaccard_{T}.csv")
print("\n=== FIG 4: Jaccard (presence-based, paper definition) ===")
print(mat.round(2).to_string())
prof5=pc.build_project_profiles(df, min_support_frac=0.05)
pc.jaccard_matrix(prof5).to_csv(f"../results/jaccard_thresholded_{T}.csv")
print("\npattern-set sizes (presence):",{k:len(v) for k,v in prof.items()})
