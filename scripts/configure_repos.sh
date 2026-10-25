#!/usr/bin/env bash
# Best-effort CMake configure of each analyzed repository, purely to obtain a
# compile_commands.json (and any headers CMake generates, e.g. config.hpp).
#
# Nothing in a target repository is ever modified: incompatibilities are worked
# around with command-line flags only. Repositories requiring a vendor SDK
# (CUDA, ONNX Runtime, OpenVINO) are expected to fail here; run_all_projects.py
# then falls back to a fixed command line and reports the lower parse rate.
#
# Usage: bash configure_repos.sh [repos_dir]

set -u
REPOS="${1:-repos}"
NINJA="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
export PATH="/c/LLVM/bin:$PATH"
TIMEOUT="${CONFIGURE_TIMEOUT:-420}"

# project path (relative to $REPOS) -> extra flags
declare -A EXTRA=(
  ["llama.cpp"]="-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF"
  ["gemma.cpp"]="-DGEMMA_ENABLE_TESTS=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF"
  ["gpt2.cpp/gpt2-cpp-main"]=""
  ["InferLLM-main/InferLLM-main"]=""
  ["lmdeploy/lmdeploy-main"]=""
  ["onnxruntime-genai/onnxruntime-genai-main"]="-DENABLE_TESTS=OFF -DENABLE_PYTHON=OFF"
  ["openvino-genai/openvino.genai-master"]=""
)

for rel in "${!EXTRA[@]}"; do
  root="$REPOS/$rel"
  [ -f "$root/CMakeLists.txt" ] || { echo "[configure] $rel: no CMakeLists, skipping"; continue; }
  if [ -f "$root/build/compile_commands.json" ]; then
    echo "[configure] $rel: compile_commands.json already present, skipping"
    continue
  fi
  echo "[configure] $rel ..."
  timeout "$TIMEOUT" cmake -S "$root" -B "$root/build" -G Ninja \
      -DCMAKE_MAKE_PROGRAM="$NINJA" \
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      ${EXTRA[$rel]} > "$root/build_configure.log" 2>&1
  if [ -f "$root/build/compile_commands.json" ]; then
    n=$(grep -c '"file"' "$root/build/compile_commands.json")
    echo "[configure] $rel: OK ($n entries)"
  else
    echo "[configure] $rel: FAILED (see $root/build_configure.log) - will use glob fallback"
  fi
done
