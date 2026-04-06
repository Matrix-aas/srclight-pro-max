# Lab Book: Community Detection & Execution Flow Tracing

## 2026-04-06 — Session 1: Design & Initial Implementation

### Background

Studying GitNexus (graph-powered code intelligence, PolyForm Noncommercial) revealed two features
Srclight lacks: automatic community/module detection via Leiden clustering, and execution flow
tracing via BFS from entry points. Both are well-published algorithms with open implementations.

### Baseline Data

Workspace edge counts across repos (srclight's `symbol_edges` table, `edge_type='calls'`):

| Repo | Symbols | Call Edges |
|------|---------|------------|
| bitcoin | 29,287 | 196,920 |
| nomad-builder | 39,424 | 77,442 |
| intuition-2019 | 20,648 | 63,126 |
| motacoin | 14,479 | 47,778 |
| kumquat | 3,978 | 18,665 |
| qi | 6,323 | 16,249 |
| web | 22,525 | 13,622 |
| srclight | 1,008 | 4,044 |
| zhcorpus | 3,758 | 3,037 |
| ice | 4,043 | 2,802 |

### Hypothesis

Louvain community detection on the call graph will produce meaningful functional clusters
that correspond to human-recognizable modules (e.g., "database", "indexing", "git", "MCP server").

### Experiment 1: Louvain on Srclight's Own Call Graph

**Method:** Load 4,044 call edges from srclight's index.db into networkx. Run Louvain
(resolution=1.0, seed=42). Examine resulting communities against known module structure.

**Expected:** ~5-10 communities mapping roughly to: db, indexer, server, git, embeddings,
workspace, cli, build, vector_math/cache.

**Results:**
- **19 communities detected in 0.160s** (more granular than expected)
- Recognizable modules confirmed:
  - #5 "Provider Embed Batch" = embeddings module (Ollama, Voyage, Cohere, OpenAI providers)
  - #7 "Learning Conversation Similarity" = learnings system
  - #8 "Cache Sidecar Vector" = vector cache
  - #9 "Hooks Hook Uninstall" = git hooks
  - #12 "Top Cosine Decode" = vector math
  - #10 "Platform Cmake Variants" = build system parsing
  - #11 "Import Include Javascript" = import extraction
- Larger clusters (#0, #1, #2, #3) are somewhat mixed — the "Projects Api List" cluster (#0, 160 symbols) combines workspace, server, and test code
- Cohesion scores low (0.065-0.195) for large clusters, high (0.4-1.0) for small focused ones

**Observations:**
- More communities than expected (19 vs 5-10). Louvain finds fine-grained structure.
- Test code gets mixed into production clusters because tests call production functions.
- Auto-labeling via TF-IDF works surprisingly well — keywords are meaningful.
- Small, focused modules (vector_math, hooks) have high cohesion and clean labels.
- Large modules with many cross-cutting concerns get lower cohesion.

### Experiment 2: Execution Flow Tracing on Srclight

**Method:** Score entry points by out-degree/in-degree ratio + name heuristics.
BFS from top entry points, max depth 10, max branching 4. Deduplicate subset flows.

**Expected:** Flows like: CLI entry -> indexer -> db store, Server tool -> db query -> return.

**Results:**
- **75 flows traced in 12.632s** (too slow — needs optimization)
- Entry point: `_extract_symbols` dominates (highest out-degree)
- Flows are 11 steps deep, crossing 5-7 communities
- Flow quality is mixed — BFS follows all call edges including constructors and data classes, leading to noisy paths (e.g., SymbolRecord constructors appear in many paths)

**Observations:**
- Flow tracing is **too slow** for production use on even a medium repo. The BFS generates a combinatorial explosion of paths because branching factor is high.
- Constructors and data class references pollute flows — should filter to "behavioral" edges only.
- Need: (a) reduce branching, (b) filter edge types, (c) memoize common subpaths.
- The concept is sound but needs tuning before it's production-quality.

### Experiment 3: Scale Test on Bitcoin (196K edges)

**Method:** Run Louvain on bitcoin's call graph. Measure wall time and community count.

**Expected:** Sub-second for Louvain. ~20-50 communities.

**Results:**
- **58 communities detected in 3.634s** (slower than expected, but still reasonable)
- Community labels are domain-accurate:
  - #4 "Sig2 Signing Mu" = cryptographic signing (MuSig2)
  - #5 "Undo Validation Locator" = block validation
  - #7 "Service Local In6" = network/service layer (IPv6)
  - #11 "Cha20 Decrypt Encrypt" = ChaCha20 encryption
  - #12 "Selection Database Records" = wallet/UTXO selection
  - #13 "Ge Ecmult Gej" = elliptic curve math (secp256k1)
  - #14 "Flags Psbt Signer" = PSBT transaction signing
- Largest community: 3,068 symbols (10% of total) — likely the core/general bucket
- Cohesion scores very low for large clusters (0.004-0.016)

**Observations:**
- 3.6s for 196K edges is acceptable for an index-time operation (runs once after indexing).
- Louvain correctly identifies domain-specific modules even in a large C++ codebase.
- The auto-labeling picks up domain terms (musig2, chacha20, secp256k1) from function names.
- Low cohesion in large clusters suggests resolution parameter could be tuned higher.

### Summary & Next Steps

1. **Community detection: VALIDATED.** Louvain produces meaningful, recognizable clusters.
   Labels are surprisingly accurate. Performance is acceptable at index time.

2. **Flow tracing: NEEDS WORK.** BFS is too slow and too noisy. Key improvements needed:
   - Filter out constructor/data class edges before tracing
   - Reduce max_branching or prioritize high-confidence edges
   - Consider DFS with path memoization instead of BFS
   - Perhaps limit to cross-community flows only (most interesting)

3. **Resolution tuning:** Consider adaptive resolution based on graph size:
   - Small graphs (<1K nodes): resolution=1.5 for finer granularity
   - Large graphs (>10K nodes): resolution=1.0 (current default)

4. **Next experiment:** Run on nomad-builder (C++ heavy, 77K edges) to test
   language-specific quality.
