//===----------------------------------------------------------------------===//
// KaizenTracer :: ASTCFGExtractor
//
// Implements the AST Generation, CFG Generation, and Feature Fusion stages
// of the KaizenTracer pipeline (see Fig. 1 of the paper).
//
// For every function definition found in the input C++ source files, this
// tool:
//   1. Walks the function body's AST to compute syntactic ("what the code
//      contains") features: loops, branches, calls, memory operations,
//      model-loading calls, sampling calls, training calls, quantization
//      calls, variable declarations, and AST depth.
//   2. Builds the function's Control Flow Graph with Clang's clang::CFG and
//      computes execution ("how the code runs") features: nodes, edges,
//      branch nodes, loop nodes, exit nodes, back edges, cyclomatic
//      complexity, graph density, loop depth, and validation/error-handling
//      paths.
//   3. Fuses both feature vectors (X_f = [A_f || C_f], Eq. 1 in the paper)
//      and writes one CSV row per function.
//
// The resulting CSV (functions.csv) is the input to the Python side of the
// pipeline (python/weak_labeling.py, train_classifier.py,
// project_comparison.py, spearman_analysis.py), which implements the
// rule-based weak labeling + multi-label Random Forest pattern detector
// (Algorithm 2) and the project-level comparison (Jaccard / Spearman).
//
// Build:
//   cmake -B build -DLLVM_DIR=<path-to-llvm-cmake> -DClang_DIR=<path-to-clang-cmake>
//   cmake --build build
//
// Usage:
//   ./kaizentracer --project=llama.cpp -p <compile_commands_dir> file1.cpp file2.cpp ...
//
// Output:
//   Appends rows to functions.csv (created with header if missing) in the
//   current working directory.
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/thread.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory KaizenCategory("KaizenTracer options");
static llvm::cl::opt<std::string> ProjectName(
    "project", llvm::cl::desc("Name of the repository being analyzed "
                               "(written into the 'project' CSV column)"),
    llvm::cl::cat(KaizenCategory), llvm::cl::init("unknown_project"));
static llvm::cl::opt<std::string> OutputCsv(
    "out", llvm::cl::desc("Path to the output CSV file (appended, "
                           "created with a header if it does not exist)"),
    llvm::cl::cat(KaizenCategory), llvm::cl::init("functions.csv"));

static llvm::cl::opt<bool> TolerateErrors(
    "tolerate-errors",
    llvm::cl::desc("Keep analyzing a translation unit after unrecoverable "
                   "errors such as a missing #include, extracting features "
                   "from whatever clang could still parse. Needed for repos "
                   "that cannot be configured without a vendor SDK; features "
                   "from such files are necessarily partial."),
    llvm::cl::cat(KaizenCategory), llvm::cl::init(false));

#ifndef KAIZENTRACER_DEFAULT_RESOURCE_DIR
#define KAIZENTRACER_DEFAULT_RESOURCE_DIR ""
#endif
static llvm::cl::opt<std::string> ResourceDir(
    "resource-dir",
    llvm::cl::desc("Clang resource directory holding the builtin headers "
                   "(stddef.h, *intrin.h, ...). Defaults to the resource "
                   "directory of the LLVM this tool was built against; pass "
                   "an empty string to let Clang locate it itself."),
    llvm::cl::cat(KaizenCategory),
    llvm::cl::init(KAIZENTRACER_DEFAULT_RESOURCE_DIR));

//===----------------------------------------------------------------------===//
// Domain keyword tables (Section III.A / III.B of the paper)
//
// These implement the "domain-specific rules over AST and CFG features"
// mentioned in the paper (Algorithm 2, Rule_p). Keyword matching is done on
// the lowercase callee name (function/method identifier) of each CallExpr.
//===----------------------------------------------------------------------===//

namespace keywords {

// Matching is whole-word over identifier tokens (see matchesAny), so short
// stems are safe -- "free" no longer fires on "freeze", "new" on "renew".
// The cost is that morphological variants must be listed explicitly, since
// "quant" is no longer a substring hit for "quantize".

// Explicit memory management. NOT "resize"/"reserve"/"release"/"destroy":
// those are ordinary std:: container and RAII operations that appear in most
// C++ regardless of domain, and they were the single largest false-positive
// source in validation (20 of 65). Retained terms denote allocation and raw
// buffer manipulation, which is what the category is about.
static const std::vector<std::string> MemoryOps = {
    "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
    "alloc", "allocate", "allocator", "allocation",
    "dealloc", "deallocate", "mmap", "munmap", "pin", "unpin",
    "host2device", "device2host", "copy to device", "copy from device"};

// Model-artefact acquisition. Beyond the literal "load" family this covers the
// operations by which C++ inference engines actually bring a model into
// memory: parsing the configuration and metadata that accompany the weights,
// constructing weight tensors from a loader, reading typed keys out of a GGUF
// or safetensors header, and decrypting protected artefacts. Restricting the
// vocabulary to "load*" misses most real loading code, since the entry point is
// usually named for the artefact being read rather than for the act of
// loading.
static const std::vector<std::string> ModelLoading = {
    "load", "loads", "loaded", "loader", "loading", "reload",
    "mmap", "gguf", "safetensors", "checkpoint", "ckpt",
    "deserialize", "serialize", "restore", "weights", "weight",
    "from file", "read weights", "open model", "state dict",
    "parse config", "parseconfig", "read config", "get key", "getkey",
    "create tensor", "createtensor", "read tensor", "read data",
    "decrypt", "from pretrained", "init from file", "vocab", "tokenizer",
    // LoRA/adapter weights are model artefacts read from disk exactly as the
    // base weights are; the vocabulary named every other artefact format
    // (gguf, safetensors, checkpoint) but omitted this one.
    "lora", "adapter", "adapters",
    "hparams", "arch tensors", "model config", "read raw"};

static const std::vector<std::string> Sampling = {
    "sample", "samples", "sampled", "sampling", "sampler",
    "top k", "topk", "top p", "topp", "temperature", "softmax",
    "penalty", "repeat penalty", "greedy", "beam", "logit", "logits",
    "multinomial", "mirostat", "argmax", "nucleus"};

static const std::vector<std::string> Training = {
    // NOT "back": tokenisation splits push_back / pop_back / emplace_back into
    // {push, back}, so the bare token matches most C++ that touches a vector.
    // Adding it raised this pattern from 25 to 1031 functions corpus-wide,
    // almost all spurious. The ggml "<op>_back" backward-kernel convention is
    // covered by the compound forms below instead.
    "backward", "norm back", "silu back", "rope back", "soft max back",
    "backprop", "optimizer", "optim", "zero grad", "zerograd",
    "grad", "grads", "gradient", "gradients", "loss", "losses", "train",
    "training", "sgd", "adam", "adamw", "crossentropy", "cross entropy",
    "optimizer step", "update weights"};

static const std::vector<std::string> Quantization = {
    "quantize", "quantized", "quantizer", "quantization", "quant",
    "dequantize", "dequantized", "dequant", "int8", "int4", "uint8",
    "fp16", "bf16", "fp8", "qtype", "scale zero point"};

// Token generation. Like Context, this category had no vocabulary: its rule
// asked only for a call, a branch and a loop, which describes most non-trivial
// C++ and made it the loosest of the seven. These are the operations that
// actually constitute a generation step.
static const std::vector<std::string> Inference = {
    "generate", "generation", "decode", "decoding", "token", "tokens",
    "next token", "prompt", "infer", "inference", "predict", "prediction",
    "completion", "complete", "chat", "forward", "step", "stream",
    "streaming", "eos", "stop token", "encode", "embd", "embed"};

// Context / session state. Every other category matches domain terms; this one
// was defined purely by structural thresholds, so functions named for exactly
// this concern (trim_kv_cache, CappedSeqLen, chat history accessors) carried no
// evidence for it while unrelated functions with several locals and a loop did.
static const std::vector<std::string> Context = {
    "kv cache", "kvcache", "cache", "context", "ctx", "session",
    "history", "chat history", "conversation", "seq len", "seqlen",
    "sequence length", "past", "prefix", "window", "state", "slot",
    "position ids", "positions", "rewind", "trim", "evict"};

static const std::vector<std::string> Validation = {
    "check", "checks", "valid", "validate", "invalid", "isvalid",
    "assert", "verify", "error", "throw", "fail", "failed", "failure",
    "abort", "ensure", "exception", "sanity"};

inline std::string toLower(llvm::StringRef S) {
  std::string Out = S.str();
  std::transform(Out.begin(), Out.end(), Out.begin(),
                  [](unsigned char c) { return std::tolower(c); });
  return Out;
}

// Split an identifier into lowercase words: snake_case, camelCase, PascalCase
// and digit boundaries all separate tokens. "loadModelFromFile" and
// "load_model_from_file" both yield {load, model, from, file}.
inline std::vector<std::string> tokenize(const std::string &Name) {
  std::vector<std::string> Tokens;
  std::string Cur;
  for (size_t i = 0; i < Name.size(); ++i) {
    const unsigned char C = static_cast<unsigned char>(Name[i]);
    // Any non-alphanumeric character separates words. This must include the
    // space: the vocabularies spell multi-word terms with spaces ("kv cache",
    // "seq len", "top k"), and while tokenize() is applied to both the
    // identifier and the keyword, a space kept inside a token made every such
    // keyword a single token containing a space -- which no identifier token
    // can ever equal. All space-delimited terms were dead entries.
    if (!std::isalnum(C)) {
      if (!Cur.empty()) { Tokens.push_back(Cur); Cur.clear(); }
      continue;
    }
    // A lower-to-upper transition starts a new word, as does letter-to-digit.
    if (i > 0 && !Cur.empty()) {
      const unsigned char P = static_cast<unsigned char>(Name[i - 1]);
      const bool CamelBoundary = std::islower(P) && std::isupper(C);
      const bool DigitBoundary = std::isalpha(P) && std::isdigit(C);
      if (CamelBoundary || DigitBoundary) {
        Tokens.push_back(Cur);
        Cur.clear();
      }
    }
    Cur += static_cast<char>(std::tolower(C));
  }
  if (!Cur.empty())
    Tokens.push_back(Cur);
  return Tokens;
}

// Whole-word matching against the identifier's tokens.
//
// The previous implementation tested for the keyword as a *substring* of the
// whole name, which fired constantly on unrelated code: "load" matched
// payload/download/workload/upload, "quant" matched quantity, "free" matched
// freeze, "new" matched renew, "step" matched timestep. Every such hit is a
// label a structural classifier cannot possibly predict, so the noise placed a
// hard ceiling on precision. Matching tokens instead requires the keyword to
// be a word of the identifier: "load_model" and "loadModel" match "load",
// "payload" does not.
//
// A multi-word keyword ("top_k") must appear as consecutive tokens.
inline bool matchesAny(const std::string &Name,
                        const std::vector<std::string> &Table) {
  const std::vector<std::string> Tokens = tokenize(Name);
  if (Tokens.empty())
    return false;
  for (const auto &K : Table) {
    const std::vector<std::string> KeyTokens = tokenize(K);
    if (KeyTokens.empty() || KeyTokens.size() > Tokens.size())
      continue;
    for (size_t i = 0; i + KeyTokens.size() <= Tokens.size(); ++i) {
      bool AllMatch = true;
      for (size_t j = 0; j < KeyTokens.size(); ++j) {
        if (Tokens[i + j] != KeyTokens[j]) { AllMatch = false; break; }
      }
      if (AllMatch)
        return true;
    }
  }
  return false;
}

} // namespace keywords

//===----------------------------------------------------------------------===//
// Feature vector: A_f (AST) concatenated with C_f (CFG)  ->  X_f = [A_f || C_f]
//===----------------------------------------------------------------------===//

struct FunctionFeatures {
  std::string project;
  std::string file;
  std::string function;
  unsigned startLine = 0;

  // ---- A_f : AST-based syntactic features ----
  unsigned ast_loops = 0;
  unsigned ast_branches = 0;
  unsigned ast_calls = 0;
  unsigned ast_vars = 0;
  unsigned ast_memory_ops = 0;
  unsigned ast_model_loading_calls = 0;
  unsigned ast_sampling_calls = 0;
  unsigned ast_training_calls = 0;
  unsigned ast_quantization_calls = 0;
  unsigned ast_context_calls = 0;
  unsigned ast_inference_calls = 0;
  unsigned ast_depth = 0;

  // ---- A_f (extended) : domain-agnostic syntactic evidence ----
  // The original nine AST features are dominated by keyword counts over callee
  // names, so a pattern is only detectable when a function happens to call
  // something conveniently named. These describe *what the code does* rather
  // than what it calls: bit manipulation and integer narrowing characterise
  // quantization, floating-point comparison over indexed buffers
  // characterises sampling, throw/catch and string literals characterise
  // loading and validation, and pointer arithmetic with sizeof characterises
  // memory management -- none of which requires a recognisable callee name.
  unsigned ast_distinct_callees = 0;
  unsigned ast_params = 0;
  unsigned ast_returns = 0;
  unsigned ast_throws = 0;
  unsigned ast_try_catch = 0;
  unsigned ast_pointer_derefs = 0;
  unsigned ast_array_subscripts = 0;
  unsigned ast_bitwise_ops = 0;
  unsigned ast_float_ops = 0;
  unsigned ast_int_casts = 0;
  unsigned ast_comparisons = 0;
  unsigned ast_string_literals = 0;
  unsigned ast_sizeof_ops = 0;
  unsigned ast_assignments = 0;

  // ---- C_f : CFG-based execution features ----
  unsigned cfg_nodes = 0;
  unsigned cfg_edges = 0;
  unsigned cfg_branch_nodes = 0;
  unsigned cfg_loop_nodes = 0;
  unsigned cfg_exit_nodes = 0;
  unsigned cfg_back_edges = 0;
  unsigned cfg_cyclomatic_complexity = 0;
  double cfg_density = 0.0;
  unsigned cfg_loop_depth = 0;
  unsigned cfg_validation_paths = 0;

  static void writeHeader(std::ostream &Out) {
    Out << "project,file,function,start_line,"
        << "ast_loops,ast_branches,ast_calls,ast_vars,ast_memory_ops,"
        << "ast_model_loading_calls,ast_sampling_calls,ast_training_calls,"
        << "ast_quantization_calls,ast_context_calls,ast_inference_calls,ast_depth,"
        << "ast_distinct_callees,ast_params,ast_returns,ast_throws,"
        << "ast_try_catch,ast_pointer_derefs,ast_array_subscripts,"
        << "ast_bitwise_ops,ast_float_ops,ast_int_casts,ast_comparisons,"
        << "ast_string_literals,ast_sizeof_ops,ast_assignments,"
        << "cfg_nodes,cfg_edges,cfg_branch_nodes,cfg_loop_nodes,"
        << "cfg_exit_nodes,cfg_back_edges,cfg_cyclomatic_complexity,"
        << "cfg_density,cfg_loop_depth,cfg_validation_paths\n";
  }

  void writeRow(std::ostream &Out) const {
    Out << csvEscape(project) << ',' << csvEscape(file) << ','
        << csvEscape(function) << ',' << startLine << ','
        << ast_loops << ',' << ast_branches << ',' << ast_calls << ','
        << ast_vars << ',' << ast_memory_ops << ','
        << ast_model_loading_calls << ',' << ast_sampling_calls << ','
        << ast_training_calls << ',' << ast_quantization_calls << ',' << ast_context_calls << ','
        << ast_inference_calls << ','
        << ast_depth << ','
        << ast_distinct_callees << ',' << ast_params << ',' << ast_returns
        << ',' << ast_throws << ',' << ast_try_catch << ','
        << ast_pointer_derefs << ',' << ast_array_subscripts << ','
        << ast_bitwise_ops << ',' << ast_float_ops << ',' << ast_int_casts
        << ',' << ast_comparisons << ',' << ast_string_literals << ','
        << ast_sizeof_ops << ',' << ast_assignments << ','
        << cfg_nodes << ',' << cfg_edges << ',' << cfg_branch_nodes << ','
        << cfg_loop_nodes << ',' << cfg_exit_nodes << ',' << cfg_back_edges
        << ',' << cfg_cyclomatic_complexity << ',' << cfg_density << ','
        << cfg_loop_depth << ',' << cfg_validation_paths << '\n';
  }

private:
  static std::string csvEscape(const std::string &S) {
    if (S.find(',') == std::string::npos &&
        S.find('"') == std::string::npos)
      return S;
    std::string Out = "\"";
    for (char c : S) {
      if (c == '"')
        Out += "\"\"";
      else
        Out += c;
    }
    Out += "\"";
    return Out;
  }
};

//===----------------------------------------------------------------------===//
// AST feature extraction: RecursiveASTVisitor over a single function body.
//===----------------------------------------------------------------------===//

class ASTFeatureVisitor : public RecursiveASTVisitor<ASTFeatureVisitor> {
public:
  explicit ASTFeatureVisitor(FunctionFeatures &F) : F(F) {}

  bool VisitForStmt(ForStmt *) { ++F.ast_loops; return true; }
  bool VisitWhileStmt(WhileStmt *) { ++F.ast_loops; return true; }
  bool VisitDoStmt(DoStmt *) { ++F.ast_loops; return true; }
  bool VisitCXXForRangeStmt(CXXForRangeStmt *) { ++F.ast_loops; return true; }

  bool VisitIfStmt(IfStmt *) { ++F.ast_branches; return true; }
  bool VisitSwitchStmt(SwitchStmt *) { ++F.ast_branches; return true; }
  bool VisitConditionalOperator(ConditionalOperator *) {
    ++F.ast_branches;
    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    if (D->isLocalVarDecl())
      ++F.ast_vars;
    return true;
  }

  bool VisitCXXNewExpr(CXXNewExpr *) { ++F.ast_memory_ops; return true; }
  bool VisitCXXDeleteExpr(CXXDeleteExpr *) { ++F.ast_memory_ops; return true; }

  bool VisitReturnStmt(ReturnStmt *) { ++F.ast_returns; return true; }
  bool VisitCXXThrowExpr(CXXThrowExpr *) { ++F.ast_throws; return true; }
  bool VisitCXXTryStmt(CXXTryStmt *) { ++F.ast_try_catch; return true; }
  bool VisitStringLiteral(StringLiteral *) {
    ++F.ast_string_literals;
    return true;
  }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *) {
    ++F.ast_array_subscripts;
    return true;
  }
  bool VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
    if (E->getKind() == UETT_SizeOf)
      ++F.ast_sizeof_ops;
    return true;
  }

  bool VisitUnaryOperator(UnaryOperator *UO) {
    if (UO->getOpcode() == UO_Deref)
      ++F.ast_pointer_derefs;
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    if (ME->isArrow())
      ++F.ast_pointer_derefs;
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *BO) {
    const BinaryOperatorKind Op = BO->getOpcode();
    if (BO->isComparisonOp())
      ++F.ast_comparisons;
    if (BO->isAssignmentOp())
      ++F.ast_assignments;
    if (Op == BO_And || Op == BO_Or || Op == BO_Xor || Op == BO_Shl ||
        Op == BO_Shr || Op == BO_AndAssign || Op == BO_OrAssign ||
        Op == BO_XorAssign || Op == BO_ShlAssign || Op == BO_ShrAssign)
      ++F.ast_bitwise_ops;
    // Floating-point arithmetic and comparison: the numeric core of sampling
    // and normalisation code, independent of what the callees are named.
    const bool Arithmetic = (Op == BO_Add || Op == BO_Sub || Op == BO_Mul ||
                             Op == BO_Div || Op == BO_Rem);
    if ((Arithmetic || BO->isComparisonOp()) &&
        !BO->getType().isNull() && BO->getLHS() &&
        !BO->getLHS()->getType().isNull() &&
        BO->getLHS()->getType()->isFloatingType())
      ++F.ast_float_ops;
    return true;
  }

  bool VisitCastExpr(CastExpr *CE) {
    // Narrowing a float to an integer, or between integer widths, is the
    // signature of quantization/dequantization arithmetic.
    QualType To = CE->getType();
    const Expr *Sub = CE->getSubExpr();
    if (To.isNull() || !Sub || Sub->getType().isNull())
      return true;
    if (To->isIntegerType() && !To->isBooleanType() &&
        (Sub->getType()->isFloatingType() || Sub->getType()->isIntegerType()) &&
        CE->getCastKind() != CK_LValueToRValue)
      ++F.ast_int_casts;
    return true;
  }

  bool VisitCallExpr(CallExpr *CE) {
    ++F.ast_calls;
    std::string Callee = calleeName(CE);
    if (Callee.empty())
      return true;
    DistinctCallees.insert(Callee);
    // Match on the identifier as written. Lower-casing here would erase the
    // camelCase boundaries tokenize() splits on, collapsing "ReuseBuffer" into
    // the single token "reusebuffer" -- see the note on matchesAny.
    const std::string &Lower = Callee;
    if (keywords::matchesAny(Lower, keywords::MemoryOps))
      ++F.ast_memory_ops;
    if (keywords::matchesAny(Lower, keywords::ModelLoading))
      ++F.ast_model_loading_calls;
    if (keywords::matchesAny(Lower, keywords::Sampling))
      ++F.ast_sampling_calls;
    if (keywords::matchesAny(Lower, keywords::Training))
      ++F.ast_training_calls;
    if (keywords::matchesAny(Lower, keywords::Quantization))
      ++F.ast_quantization_calls;
    if (keywords::matchesAny(Lower, keywords::Context))
      ++F.ast_context_calls;
    if (keywords::matchesAny(Lower, keywords::Inference))
      ++F.ast_inference_calls;
    return true;
  }

  // Track maximum AST nesting depth within the function body.
  bool TraverseStmt(Stmt *S) {
    if (!S)
      return RecursiveASTVisitor::TraverseStmt(S);
    ++CurDepth;
    F.ast_depth = std::max(F.ast_depth, CurDepth);
    bool Result = RecursiveASTVisitor::TraverseStmt(S);
    --CurDepth;
    return Result;
  }

  // Call once after traversal: counts only known after the whole body is seen.
  void finalize() {
    F.ast_distinct_callees = static_cast<unsigned>(DistinctCallees.size());
  }

private:
  static std::string calleeName(CallExpr *CE) {
    if (const FunctionDecl *FD = CE->getDirectCallee())
      return FD->getNameInfo().getAsString();
    return {};
  }

  FunctionFeatures &F;
  unsigned CurDepth = 0;
  // Fan-out: a dispatcher calling 20 different functions and a loop calling
  // one 20 times both score ast_calls == 20, but are structurally unalike.
  std::set<std::string> DistinctCallees;
};

//===----------------------------------------------------------------------===//
// CFG feature extraction: build clang::CFG for the function and compute
// execution-oriented features described in Section III.B.
//===----------------------------------------------------------------------===//

// Maximum nesting depth of loop statements inside a body: 0 with no loops,
// 1 for a flat loop, 2 for a loop within a loop, and so on.
static unsigned maxLoopNesting(const Stmt *S) {
  if (!S)
    return 0;
  unsigned Deepest = 0;
  for (const Stmt *Child : S->children())
    Deepest = std::max(Deepest, maxLoopNesting(Child));
  if (isa<ForStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S) ||
      isa<CXXForRangeStmt>(S))
    return Deepest + 1;
  return Deepest;
}

static void extractCFGFeatures(FunctionDecl *FD, ASTContext &Ctx,
                                FunctionFeatures &F) {
  Stmt *Body = FD->getBody();
  if (!Body)
    return;

  CFG::BuildOptions Options;
  Options.AddInitializers = true;
  Options.AddImplicitDtors = true;
  std::unique_ptr<CFG> Cfg = CFG::buildCFG(FD, Body, &Ctx, Options);
  if (!Cfg)
    return;

  unsigned NumNodes = Cfg->size();
  unsigned NumEdges = 0;
  unsigned BranchNodes = 0;
  unsigned ExitNodes = 0;
  unsigned ValidationPaths = 0;

  // Position of every block, so the depth-first classification below is O(1)
  // per edge rather than a linear scan (which made large CFGs quadratic).
  llvm::DenseMap<const CFGBlock *, unsigned> Index;
  Index.reserve(NumNodes);
  {
    unsigned Next = 0;
    for (const CFGBlock *B : *Cfg)
      Index[B] = Next++;
  }

  for (const CFGBlock *B : *Cfg) {
    unsigned Succs = 0;
    for (CFGBlock::const_succ_iterator SI = B->succ_begin();
         SI != B->succ_end(); ++SI) {
      if (!*SI)
        continue;
      ++Succs;
      ++NumEdges;
    }
    if (Succs >= 2)
      ++BranchNodes;
    if (Succs == 0)
      ++ExitNodes;

    // Validation / failure-handling path: a block whose terminator or
    // statements reference a validation-keyword call.
    bool HasValidationCall = false;
    for (const CFGElement &Elem : *B) {
      if (auto CS = Elem.getAs<CFGStmt>()) {
        if (const auto *CE = dyn_cast<CallExpr>(CS->getStmt())) {
          if (const FunctionDecl *Callee = CE->getDirectCallee()) {
            const std::string Lower = Callee->getNameInfo().getAsString();
            if (keywords::matchesAny(Lower, keywords::Validation))
              HasValidationCall = true;
          }
        }
      }
    }
    if (HasValidationCall)
      ++ValidationPaths;
  }

  // Back edges, by the textbook definition: an edge u -> v where v is still
  // on the depth-first stack, i.e. v is an ancestor of u in the DFS tree.
  // Comparing raw block positions instead does not work -- clang numbers CFG
  // blocks in reverse (the entry block has the *highest* id), so an
  // index-order test classifies almost every forward edge as a back edge.
  // Loop nodes are then exactly the blocks such an edge targets: the loop
  // headers.
  enum DfsState : unsigned char { Unvisited, OnStack, Finished };
  std::vector<DfsState> Status(NumNodes, Unvisited);
  llvm::SmallPtrSet<const CFGBlock *, 16> LoopHeaders;
  unsigned BackEdges = 0;

  std::vector<std::pair<const CFGBlock *, CFGBlock::const_succ_iterator>> Stack;
  const CFGBlock *Entry = &Cfg->getEntry();
  if (Index.count(Entry)) {
    Status[Index[Entry]] = OnStack;
    Stack.emplace_back(Entry, Entry->succ_begin());
    while (!Stack.empty()) {
      const CFGBlock *Cur = Stack.back().first;
      if (Stack.back().second == Cur->succ_end()) {
        Status[Index[Cur]] = Finished;
        Stack.pop_back();
        continue;
      }
      const CFGBlock *Succ = *(Stack.back().second++);
      if (!Succ)
        continue;
      auto It = Index.find(Succ);
      if (It == Index.end())
        continue;
      if (Status[It->second] == OnStack) {
        ++BackEdges;
        LoopHeaders.insert(Succ);
      } else if (Status[It->second] == Unvisited) {
        Status[It->second] = OnStack;
        // Safe to grow the stack here: no reference into it is held.
        Stack.emplace_back(Succ, Succ->succ_begin());
      }
    }
  }
  unsigned LoopNodes = static_cast<unsigned>(LoopHeaders.size());

  // Loop *nesting* depth. The previous code incremented a counter once per
  // loop header and never decremented it, so this column was always just a
  // copy of cfg_loop_nodes. clang's CFG flattens loops into blocks, so the
  // nesting relation is recovered from the AST instead.
  unsigned MaxLoopDepth = maxLoopNesting(Body);

  F.cfg_nodes = NumNodes;
  F.cfg_edges = NumEdges;
  F.cfg_branch_nodes = BranchNodes;
  F.cfg_loop_nodes = LoopNodes;
  F.cfg_exit_nodes = ExitNodes;
  F.cfg_back_edges = BackEdges;
  // Cyclomatic complexity: E - N + 2 (single connected component). A
  // straight-line function has E = N - 1 and must score 1; the previous
  // fallback returned 2 in exactly that case, which is why no function in a
  // corpus could ever score below 2.
  long long Cyclomatic =
      static_cast<long long>(NumEdges) - static_cast<long long>(NumNodes) + 2;
  F.cfg_cyclomatic_complexity =
      static_cast<unsigned>(std::max<long long>(1, Cyclomatic));
  F.cfg_density = (NumNodes > 1)
                       ? static_cast<double>(NumEdges) /
                             (static_cast<double>(NumNodes) * (NumNodes - 1))
                       : 0.0;
  F.cfg_loop_depth = MaxLoopDepth;
  F.cfg_validation_paths = ValidationPaths;
}

//===----------------------------------------------------------------------===//
// Output sink (thread-safety not required: ClangTool runs actions
// sequentially per translation unit by default).
//===----------------------------------------------------------------------===//

class CsvSink {
public:
  explicit CsvSink(const std::string &Path) : Path(Path) {
    std::ifstream Test(Path);
    NeedsHeader = !Test.good();
  }

  void write(const FunctionFeatures &F) {
    std::ofstream Out(Path, std::ios::app);
    if (NeedsHeader) {
      FunctionFeatures::writeHeader(Out);
      NeedsHeader = false;
    }
    F.writeRow(Out);
  }

private:
  std::string Path;
  bool NeedsHeader;
};

static CsvSink *GlobalSink = nullptr;
static std::mutex SinkMutex;

//===----------------------------------------------------------------------===//
// AST consumer: for every top-level function definition in the TU, extract
// AST features, then CFG features, then fuse and emit a CSV row.
//===----------------------------------------------------------------------===//

class KaizenASTConsumer : public ASTConsumer {
public:
  explicit KaizenASTConsumer(ASTContext &Ctx, StringRef FileName)
      : Ctx(Ctx), FileName(FileName.str()) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    // Walking the declaration tree by hand missed every construct not
    // explicitly enumerated -- in particular FunctionTemplateDecl and the
    // methods of ClassTemplateDecl, i.e. all templated code. That is a large
    // and non-random slice of a modern C++ codebase, so collect definitions
    // with a full recursive visitor instead.
    FunctionCollector Collector;
    Collector.TraverseDecl(Context.getTranslationUnitDecl());
    for (FunctionDecl *FD : Collector.Functions)
      processFunction(FD);
  }

private:
  // Collects every function *written* in the translation unit. Template
  // instantiations are deliberately not visited: the paper counts functions as
  // authored, and one row per instantiation would multiply-count a single
  // definition by however many types it happens to be used with.
  class FunctionCollector : public RecursiveASTVisitor<FunctionCollector> {
  public:
    bool shouldVisitTemplateInstantiations() const { return false; }

    bool VisitFunctionDecl(FunctionDecl *FD) {
      if (!FD->hasBody() || !FD->isThisDeclarationADefinition())
        return true;
      // A lambda's operator() is already accounted for by the enclosing
      // function's AST/CFG traversal; emitting it again would double-count
      // its body.
      if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
        if (MD->getParent() && MD->getParent()->isLambda())
          return true;
      Functions.push_back(FD);
      return true;
    }

    std::vector<FunctionDecl *> Functions;
  };

  void processFunction(FunctionDecl *FD) {
    // Skip functions that don't originate from the main file being
    // analyzed (e.g. transitively-included header definitions), matching
    // the paper's "core source files" preprocessing step.
    SourceManager &SM = Ctx.getSourceManager();
    if (!SM.isInMainFile(SM.getExpansionLoc(FD->getLocation())))
      return;

    FunctionFeatures F;
    F.project = ProjectName;
    F.file = FileName;
    F.function = FD->getQualifiedNameAsString();
    F.startLine = SM.getExpansionLineNumber(FD->getLocation());

    F.ast_params = FD->getNumParams();

    // A function's own identifier is domain evidence in its own right, and
    // until now it was ignored: only callee names were matched, so a function
    // named load_arch_tensors scored zero model-loading evidence unless it
    // happened to call something named "load". Developers name functions for
    // what they do, which is the first thing a human reader uses to classify
    // them. Count the declared name once against each vocabulary, alongside
    // the callees counted during traversal.
    {
      // The function's own identifier only. Walking out to the enclosing
      // class was tried and reverted: it applies a class's domain to every
      // member including trivial accessors, and measured 9 recovered true
      // positives against 28 new false positives on the reference set
      // (llama_mmap::size, llama_vocab::type_name, State::SetRunOption, and
      // above all llm_graph_context::build_attn -- llama.cpp's graph-builder
      // base class contains "context", so every graph builder deriving from
      // it read as Context Management).
      const std::string SelfName = FD->getNameInfo().getAsString();
      if (keywords::matchesAny(SelfName, keywords::MemoryOps))
        ++F.ast_memory_ops;
      if (keywords::matchesAny(SelfName, keywords::ModelLoading))
        ++F.ast_model_loading_calls;
      if (keywords::matchesAny(SelfName, keywords::Sampling))
        ++F.ast_sampling_calls;
      if (keywords::matchesAny(SelfName, keywords::Training))
        ++F.ast_training_calls;
      if (keywords::matchesAny(SelfName, keywords::Quantization))
        ++F.ast_quantization_calls;
      if (keywords::matchesAny(SelfName, keywords::Context))
        ++F.ast_context_calls;
      if (keywords::matchesAny(SelfName, keywords::Inference))
        ++F.ast_inference_calls;
    }

    ASTFeatureVisitor Visitor(F);
    Visitor.TraverseStmt(FD->getBody());
    Visitor.finalize();

    extractCFGFeatures(FD, Ctx, F);

    std::lock_guard<std::mutex> Lock(SinkMutex);
    GlobalSink->write(F);
  }

  ASTContext &Ctx;
  std::string FileName;
};

class KaizenFrontendAction : public ASTFrontendAction {
public:
  bool BeginSourceFileAction(CompilerInstance &CI) override {
    if (TolerateErrors) {
      // A missing #include is a *fatal* diagnostic, which aborts the whole
      // translation unit and yields no rows at all. Repositories that need a
      // vendor SDK (CUDA, ONNX Runtime, OpenVINO) to configure hit this on
      // nearly every file, so whole projects would silently contribute
      // nothing. Downgrading fatals to ordinary errors lets clang keep going
      // and build a partial AST covering everything it *could* resolve, which
      // is the difference between a low parse rate and no data.
      CI.getDiagnostics().setFatalsAsError(true);
      CI.getDiagnostics().setSuppressSystemWarnings(true);
    }
    return ASTFrontendAction::BeginSourceFileAction(CI);
  }

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef File) override {
    return std::make_unique<KaizenASTConsumer>(CI.getASTContext(), File);
  }
};

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

// Stack reserved for the analysis thread. 64 MB clears the deepest ASTs seen
// in the repositories from Table I with room to spare; it is reserved address
// space, not committed memory, so the cost of over-provisioning is negligible.
static constexpr unsigned kStackSizeBytes = 64u * 1024u * 1024u;

int main(int argc, const char **argv) {
  // Expand @response-files before option parsing. Analyzing a repository means
  // passing hundreds of source paths plus an include path per header
  // directory, which overruns the 32767-character command line Windows allows
  // CreateProcess -- the caller can put the whole argument list in a file
  // instead and pass @file.
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  llvm::SmallVector<const char *, 256> Args(argv, argv + argc);
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeWindowsCommandLine,
                                 Args);
  argc = static_cast<int>(Args.size());
  argv = Args.data();

  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, KaizenCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();

  // ClangTool chdir()s into each compile command's "directory" before parsing,
  // so a relative --out path would resolve against the *target repository's*
  // build tree rather than where the tool was invoked, scattering rows into
  // the analyzed projects. Pin it to an absolute path up front.
  llvm::SmallString<256> AbsoluteOut(OutputCsv.getValue());
  if (std::error_code EC = llvm::sys::fs::make_absolute(AbsoluteOut)) {
    llvm::errs() << "[KaizenTracer] cannot resolve --out path '" << OutputCsv
                 << "': " << EC.message() << '\n';
    return 1;
  }
  const std::string OutPath(AbsoluteOut.str());

  CsvSink Sink(OutPath);
  GlobalSink = &Sink;

  ClangTool Tool(OptionsParser.getCompilations(),
                  OptionsParser.getSourcePathList());

  // Without this, any translation unit that reaches for a compiler-provided
  // header fails with "'avxintrin.h' file not found" and contributes no rows,
  // silently shrinking the dataset rather than reporting an error.
  if (!ResourceDir.empty())
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        ("-resource-dir=" + ResourceDir).c_str(),
        ArgumentInsertPosition::BEGIN));

  // Machine-generated sources (long initializer chains, deeply nested
  // expressions) recurse far enough through RecursiveASTVisitor to overflow
  // the default 1 MB stack MSVC gives the main thread, taking the whole run
  // down with STATUS_STACK_OVERFLOW partway through a batch. Clang's own
  // driver sidesteps this the same way: do the work on a thread with a stack
  // large enough to absorb the deepest ASTs found in real repositories.
  int Result = 0;
  llvm::thread Runner(std::optional<unsigned>(kStackSizeBytes), [&] {
    Result = Tool.run(newFrontendActionFactory<KaizenFrontendAction>().get());
  });
  Runner.join();

  std::cout << "[KaizenTracer] Finished extracting AST/CFG features for "
            << "project '" << ProjectName << "'. Output: " << OutPath
            << std::endl;
  return Result;
}
