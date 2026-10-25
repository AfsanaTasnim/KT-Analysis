"""Run the kaizentracer extractor over every repository in the study.

Produces the accumulated `functions.csv` plus `table1_parse_stats.csv`, the
per-repository coverage numbers reported in Table I (source files, parsed
files, functions, CFGs, parse rate).

Repositories that can be configured with CMake get a real `compile_commands.json`
and are parsed with the project's own include paths. Repositories that need a
vendor SDK (CUDA, ONNX Runtime, OpenVINO) to configure fall back to a fixed
command line with heuristic include directories; those are marked `glob` in the
`mode` column and are expected to show a lower parse rate. Reporting that rate
honestly is preferable to silently dropping the repository.

Usage:
    python run_all_projects.py --bin build/Release/kaizentracer.exe \
                               --repos repos --out functions.csv
"""

import argparse
import os
import re
import subprocess
import sys

import collect_sources

# project name (as it appears in the paper) -> path under --repos
PROJECTS = [
    ("llama.cpp",           "llama.cpp"),
    ("gemma.cpp",           "gemma.cpp"),
    ("onnxruntime-genai",   "onnxruntime-genai/onnxruntime-genai-main"),
    ("openvino.genai",      "openvino-genai/openvino.genai-master"),
    ("lmdeploy",            "lmdeploy/lmdeploy-main"),
    ("gpt2.cpp",            "gpt2.cpp/gpt2-cpp-main"),
    ("minchatgpt.cpp",      "minchatgpt.cpp/minchatgpt.cpp-main"),
    ("InferLLM",            "InferLLM-main/InferLLM-main"),
    ("sglang",              "sglang"),
    ("DeepSpeed-FastGen",   "deepspeed-fastgen/DeepSpeed-master"),
]

# Directories worth putting on the include path when there is no compile
# database: the conventional public/private header roots of a C++ project.
INCLUDE_CANDIDATES = [
    "", "include", "src", "csrc", "lib", "core", "cpp", "source",
    "src/include", "include/src",
]

ERROR_RE = re.compile(r"^Error while processing (.+)\.$", re.MULTILINE)


HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".hxx", ".inc", ".ipp")


def include_flags(repo_root: str, sources) -> list:
    """-I flags for a repo we cannot configure.

    Most parse failures in fallback mode are the project's *own* headers living
    in a subdirectory we did not guess, not genuinely external SDK headers. So
    rather than guess, put every directory that actually contains a header on
    the include path, plus each directory's parent (headers are commonly
    included as "subdir/foo.h"). External SDK headers still fail, which is what
    --tolerate-errors is for.
    """
    flags, seen = [], set()

    def add(path):
        norm = os.path.abspath(path).replace(os.sep, "/")
        if norm not in seen and os.path.isdir(norm):
            seen.add(norm)
            flags.append("-I" + norm)

    for cand in INCLUDE_CANDIDATES:
        add(os.path.join(repo_root, cand) if cand else repo_root)

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames
                       if not collect_sources.is_pruned_dir(d)]
        if any(f.lower().endswith(HEADER_SUFFIXES) for f in filenames):
            add(dirpath)
            add(os.path.dirname(dirpath))

    for src in sources:
        add(os.path.dirname(src))
    return flags


def run_project(binary, name, repo_root, out_csv, list_dir):
    build_dir = os.path.join(repo_root, "build")
    sources, mode = collect_sources.collect(repo_root, build_dir)
    if not sources:
        print(f"[run_all] {name}: no C++ sources found, skipping")
        return {"project": name, "mode": mode, "source_files": 0,
                "parsed_files": 0, "functions": 0, "parse_rate": 0.0}

    list_file = os.path.join(list_dir, f"{name}.txt")
    with open(list_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sources))

    before = count_rows(out_csv)
    # --tolerate-errors applies in both modes: even with a real compile
    # database, repositories written for POSIX (mmap/munmap, <unistd.h>) or for
    # CUDA cascade errors when parsed on this host, and without it those files
    # contribute nothing at all instead of their parseable subset.
    args = [f"--project={name}", f"--out={os.path.abspath(out_csv)}",
            "--tolerate-errors"]
    if mode == "compile_db":
        args += ["-p", os.path.abspath(build_dir)]
        args += sources
    else:
        args += sources
        args += ["--", "-std=c++17", "-w", "-ferror-limit=0"]
        args += include_flags(repo_root, sources)

    # Windows caps a command line at 32767 characters and a large repository
    # needs far more than that in source paths plus -I flags, so hand the
    # argument list over in a response file instead.
    rsp = os.path.join(list_dir, f"{name}.rsp")
    with open(rsp, "w", encoding="utf-8") as fh:
        fh.write("\n".join('"%s"' % a.replace("\\", "/") for a in args))
    cmd = [binary, "@" + rsp]

    print(f"[run_all] {name}: {len(sources)} files ({mode}) ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace")
    with_errors = len(set(ERROR_RE.findall(proc.stderr or "")))

    # "Parsed" means the file actually yielded function-level features. A file
    # clang accepted but from which nothing was extracted is not evidence of
    # coverage, so counting it would inflate the parse rate.
    contributing, functions, cfgs = project_yield(out_csv, name, before)
    rate = 100.0 * contributing / len(sources) if sources else 0.0
    print(f"[run_all] {name}: {contributing}/{len(sources)} files yielded "
          f"features ({rate:.1f}%), {functions} functions, "
          f"{with_errors} with errors", flush=True)
    return {"project": name, "mode": mode, "source_files": len(sources),
            "parsed_files": contributing, "functions": functions,
            "cfgs": cfgs, "files_with_errors": with_errors,
            "parse_rate": round(rate, 1)}


def project_yield(out_csv, name, skip_rows):
    """Distinct files, function rows and built CFGs contributed by this run."""
    import csv
    files, functions, cfgs = set(), 0, 0
    with open(out_csv, encoding="utf-8", errors="replace", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i < skip_rows or row.get("project") != name:
                continue
            files.add(row.get("file", ""))
            functions += 1
            try:
                if int(row.get("cfg_nodes") or 0) > 0:
                    cfgs += 1
            except ValueError:
                pass
    return len(files), functions, cfgs


def count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="build/Release/kaizentracer.exe")
    parser.add_argument("--repos", default="repos")
    parser.add_argument("--out", default="functions.csv")
    parser.add_argument("--stats", default="table1_parse_stats.csv")
    parser.add_argument("--only", default="",
                        help="comma-separated subset of project names")
    parser.add_argument("--append", action="store_true",
                        help="keep an existing out CSV and append to it, for "
                             "resuming a partial run")
    args = parser.parse_args()

    if os.path.isfile(args.out) and not args.append:
        os.remove(args.out)
    list_dir = os.path.join(os.path.dirname(args.out) or ".", "_filelists")
    os.makedirs(list_dir, exist_ok=True)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = []
    for name, rel in PROJECTS:
        if wanted and name not in wanted:
            continue
        repo_root = os.path.join(args.repos, rel)
        if not os.path.isdir(repo_root):
            print(f"[run_all] {name}: not found at {repo_root}, skipping")
            continue
        rows.append(run_project(args.bin, name, repo_root, args.out, list_dir))

    import pandas as pd
    stats = pd.DataFrame(rows)
    stats.to_csv(args.stats, index=False)
    print("\n=== Table I: repository coverage ===")
    print(stats.to_string(index=False))
    print(f"\n[run_all] {count_rows(args.out)} function rows -> {args.out}")


if __name__ == "__main__":
    main()
