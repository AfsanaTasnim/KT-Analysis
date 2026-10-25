"""Select a repository's own C++ translation units.

Two modes:

* **compile database** - read `compile_commands.json` and take the entries that
  belong to the repository. This is the accurate mode: it yields exactly the
  translation units the project actually compiles, with real include paths.
* **glob** - walk the tree for C++ sources. Used for repositories that cannot
  be configured without a vendor SDK (CUDA, ONNX Runtime, OpenVINO), where a
  compile database is unavailable.

Both modes drop third-party code (fetched into build/_deps, or checked in under
vendor/ and third_party/), generated sources under the build tree, and tests.
Leaving them in would attribute another project's structural patterns to this
one.

Usage:
    python collect_sources.py <repo_root> <build_dir|-> <out_list.txt>
"""

import json
import os
import sys

# Path fragments that mark code the project did not write itself.
THIRD_PARTY = (
    "/_deps/", "/vendor/", "/third_party/", "/thirdparty/", "/3rdparty/",
    "/external/", "/extern/", "/deps/", "/submodules/", "/site-packages/",
)
TESTS = ("_test", "test_", "/tests/", "/test/", "/testing/", "/gtest/")

# Well-known single-file / amalgamated third-party libraries that projects drop
# straight into their own source directories, where no path-based rule catches
# them. simdjson.cpp alone is ~12k lines and would contribute several hundred
# functions to whichever project vendored it, swamping that project's real
# structural profile.
THIRD_PARTY_BASENAMES = {
    "simdjson", "cxxopts", "ctre", "ctre-unicode", "json", "json11",
    "jsoncpp", "nlohmann_json", "sqlite3", "pugixml", "tinyxml", "tinyxml2",
    "miniz", "cjson", "httplib", "picojson", "base64", "md5", "sha1", "sha256",
    "utf8", "stb_image", "stb_image_write", "dr_wav", "miniaudio", "tinyexpr",
    "whereami", "getopt", "fmt", "spdlog", "catch", "doctest", "lodepng",
}

# .cu / .cuh are CUDA, not C++, and need the CUDA toolkit to parse; excluded so
# a repository is not penalised in the parse rate for files clang was never
# going to accept as C++.
CXX_SUFFIXES = (".cpp", ".cc", ".cxx", ".c++")


def is_project_source(path_lower: str, root_lower: str,
                      build_lower: str = "") -> bool:
    """True if this path is one of the project's own, non-test sources."""
    if not path_lower.startswith(root_lower):
        return False
    if build_lower and path_lower.startswith(build_lower + "/"):
        return False
    if any(frag in path_lower for frag in THIRD_PARTY):
        return False
    if any(frag in path_lower for frag in TESTS):
        return False
    stem = os.path.splitext(os.path.basename(path_lower))[0]
    if stem in THIRD_PARTY_BASENAMES:
        return False
    return True


def _norm(path: str) -> str:
    return os.path.abspath(path).replace(os.sep, "/")


def from_compile_db(repo_root: str, build_dir: str):
    """Translation units listed in build_dir/compile_commands.json."""
    with open(os.path.join(build_dir, "compile_commands.json"),
              encoding="utf-8") as fh:
        entries = json.load(fh)

    root = _norm(repo_root).lower()
    build = _norm(build_dir).lower().rstrip("/")
    files = set()
    for entry in entries:
        path = _norm(entry["file"])
        if not path.lower().endswith(CXX_SUFFIXES):
            continue
        if is_project_source(path.lower(), root, build):
            files.add(path)
    return sorted(files)


def from_glob(repo_root: str):
    """C++ sources found by walking the tree, for repos with no compile db."""
    root = _norm(repo_root)
    root_lower = root.lower()
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune third-party and build directories in place so the walk stays
        # cheap on large repositories.
        dirnames[:] = [
            d for d in dirnames
            if not is_pruned_dir(d)
        ]
        for name in filenames:
            if not name.lower().endswith(CXX_SUFFIXES):
                continue
            path = _norm(os.path.join(dirpath, name))
            if is_project_source(path.lower(), root_lower):
                files.add(path)
    return sorted(files)


PRUNED_DIRS = {
    "build", "_deps", "vendor", "third_party", "thirdparty", "3rdparty",
    "external", "extern", "deps", "submodules", "site-packages", ".git",
    "tests", "test", "testing", "gtest", "node_modules", "__pycache__",
}


def is_pruned_dir(name: str) -> bool:
    return name.lower() in PRUNED_DIRS


def collect(repo_root: str, build_dir: str = ""):
    """Compile-database selection when available, glob otherwise."""
    if build_dir and os.path.isfile(
            os.path.join(build_dir, "compile_commands.json")):
        return from_compile_db(repo_root, build_dir), "compile_db"
    return from_glob(repo_root), "glob"


def main():
    repo_root, build_dir, out_list = sys.argv[1:4]
    if build_dir == "-":
        build_dir = ""
    files, mode = collect(repo_root, build_dir)
    with open(out_list, "w", encoding="utf-8") as fh:
        fh.write("\n".join(files))
    print(f"[collect_sources] {len(files)} source files ({mode}) -> {out_list}")


if __name__ == "__main__":
    main()
