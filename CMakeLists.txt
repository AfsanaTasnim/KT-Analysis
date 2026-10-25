cmake_minimum_required(VERSION 3.13)

# Needed so -DCMAKE_MSVC_RUNTIME_LIBRARY=... is honoured: the CRT flavour must
# match how the LLVM/Clang libraries were built (official LLVM Windows releases
# are /MT, vcpkg's x64-windows LLVM is /MD), or every clang*.lib fails LNK2038.
if (POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(KaizenTracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where to look for LLVM/Clang when neither -DLLVM_DIR=... nor a toolchain file
# (vcpkg) points at one. These cover the official LLVM Windows installer and the
# usual Linux llvm-dev/libclang-dev layouts. HINTS are consulted *after* an
# explicit LLVM_DIR and after CMAKE_PREFIX_PATH, so passing either still wins.
set(KAIZENTRACER_LLVM_HINTS
  "$ENV{LLVM_ROOT}"
  "C:/LLVM"
  "C:/Program Files/LLVM"
  "/usr/lib/llvm"
  CACHE STRING "Extra prefixes to search for an LLVM/Clang installation")

find_package(LLVM REQUIRED CONFIG
  HINTS ${KAIZENTRACER_LLVM_HINTS}
  PATH_SUFFIXES lib/cmake/llvm)
find_package(Clang REQUIRED CONFIG
  HINTS ${KAIZENTRACER_LLVM_HINTS} "${LLVM_DIR}/../clang"
  PATH_SUFFIXES lib/cmake/clang)

# CLANG_PACKAGE_VERSION is not exported by every ClangConfig.cmake; fall back to
# the LLVM version, which Clang is always released in lockstep with.
if (NOT CLANG_PACKAGE_VERSION)
  set(CLANG_PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION} (${Clang_DIR})")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(kaizentracer src/ASTCFGExtractor.cpp)

# Clang resolves its builtin headers (stddef.h, the *intrin.h family, ...)
# relative to the running executable, which only works for tools installed
# beside clang itself. kaizentracer lives in its own build tree, so bake in
# the resource directory of the LLVM it was linked against; --resource-dir
# overrides it at runtime.
target_compile_definitions(kaizentracer PRIVATE
  KAIZENTRACER_DEFAULT_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")

target_link_libraries(kaizentracer
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  clangAST
  clangAnalysis
  clangFrontend
  clangSerialization
)

if (MSVC)
  # Clang/LLVM headers are large enough to overflow MSVC's default object
  # section count, and RTTI/exceptions must match how LLVM itself was built.
  target_compile_options(kaizentracer PRIVATE /bigobj /EHsc /GR)
  # The clang*.lib files drag in a lot of LLVM; MSVC's default 4096-symbol
  # warning noise from the LLVM headers is not actionable here.
  target_compile_options(kaizentracer PRIVATE /wd4267 /wd4244 /wd4146 /wd4805 /wd4624)
  target_compile_definitions(kaizentracer PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
  )
endif()
