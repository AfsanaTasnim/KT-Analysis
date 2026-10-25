# KaizenTracer
## Replication Pacakge

Static AST + CFG feature extraction for C++ local LLM repositories, and the
analysis that produces the paper's tables and figures.

## Layout

```
src/ASTCFGExtractor.cpp   Clang LibTooling extractor
CMakeLists.txt, CMakePresets.json, vcpkg.json   extractor build
scripts/                  collect_sources.py, run_all_projects.py, configure_repos.sh
python/                   labeling, ablation, LOPO, validation, Jaccard, Spearman
results/functions_v4.csv  feature table, one row per function (input)
results/*_v4.*            derived tables and figures
expected/                 reference outputs for compare_expected.py
validation250/, holdout80/  reference labels and v4 predictions (Table V)
repos/                    analyzed repositories (not committed)
```

## 1. Setup

```
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r python/requirements.txt
cd python
```

Tested with Python 3.12.

## 2. Reproduce the results

Run from `python/`. Input: `results/functions_v4.csv` (11,257 rows).

```
python regen_all.py
python table3_feature_ablation.py
python table4_lopo.py
python table5_validation_v4.py
python jaccard_7cat_v4.py
python make_fig5.py
python compare_expected.py
```

| Paper item | Script | Output in `results/` |
|---|---|---|
| Table III | `regen_all.py` | `table2_patterns_by_project_v4.csv` |
| Table IV, Fig. 2 | `table3_feature_ablation.py` | `table3_ablation_v4.csv` |
| Table V | `table5_validation_v4.py` | `table5_validation250_v4.csv`, `table5_holdout80_v4.csv`, `table5_pooled330_v4.csv`, `table5_uncertainty_v4.csv` |
| Table VI | `table4_lopo.py` | `table5_lopo_v4.csv` |
| Fig. 3 | `jaccard_7cat_v4.py` | `jaccard_7cat_v4.csv`, `jaccard_7cat_v4.png` |
| Fig. 4 | `make_fig5.py` | `fig5_spearman_v4.png` (values also printed) |

`compare_expected.py` checks every regenerated CSV against `expected/`.
`table1_parse_stats_v4.csv` (Table II) reports NOT RUN unless step 3 is done.

Runtime: `table3_feature_ablation.py` about 7 min, `table4_lopo.py` about 2 min, `regen_all.py` about 30 s, others under 5 s.

## 3. Re-extract the features (optional)

Requires CMake, Ninja, a C++17 compiler, and LLVM/Clang 22.1.8 with LibTooling.
Run from the repository root.

Build the extractor:

```
cmake --preset windows-llvm          # Linux/macOS: unix-llvm
cmake --build --preset windows-llvm
```

Fetch the repositories into the directories below (do not modify them):

| Project | URL | Version | Directory |
|---|---|---|---|
| llama.cpp | https://github.com/ggerganov/llama.cpp | commit `01818e4956858fe225a6875aae88b4cbee0b319e` | `repos/llama.cpp` |
| gemma.cpp | https://github.com/google/gemma.cpp | commit `3ed403e28707c0e3eb5f48b0e487b63a446d8e2e` | `repos/gemma.cpp` |
| ONNX Runtime GenAI | https://github.com/microsoft/onnxruntime-genai | main, snapshot 2026-08-18 (`VERSION_INFO` 0.16.0-dev) | `repos/onnxruntime-genai/onnxruntime-genai-main` |
| OpenVINO GenAI | https://github.com/openvinotoolkit/openvino.genai | master, snapshot 2026-08-18 (2026.4.0.0) | `repos/openvino-genai/openvino.genai-master` |
| LMDeploy | https://github.com/InternLM/lmdeploy | main, snapshot 2026-08-18 (0.15.0) | `repos/lmdeploy/lmdeploy-main` |
| gpt2.cpp | https://github.com/gf712/gpt2-cpp | main, snapshot 2026-08-18 | `repos/gpt2.cpp/gpt2-cpp-main` |
| minchatgpt.cpp | https://github.com/iangitonga/minchatgpt.cpp | main, snapshot 2026-08-18 | `repos/minchatgpt.cpp/minchatgpt.cpp-main` |
| InferLLM | https://github.com/MegEngine/InferLLM | main, snapshot 2026-08-18 | `repos/InferLLM-main/InferLLM-main` |
| SGLang | https://github.com/sgl-project/sglang | main, snapshot 2026-08-18 | `repos/sglang` |
| DeepSpeed-FastGen | https://github.com/microsoft/DeepSpeed | master, snapshot 2026-08-18 (0.19.6) | `repos/deepspeed-fastgen/DeepSpeed-master` |

Configure and extract:

```
bash scripts/configure_repos.sh repos
python scripts/run_all_projects.py --bin build/Release/kaizentracer.exe --repos repos --out results/functions_rerun.csv
```

Parse coverage depends on the installed headers and SDKs. `results/functions_v4.csv` is the reference table; `expected/table1_parse_stats_v4.csv` is the reference for Table II.
