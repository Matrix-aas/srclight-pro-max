# srclight v2.1 Agent Ergonomics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `srclight` cheaper to use for AI agents, better at file-level navigation without grep, and richer on TS/Vue/type-heavy fullstack repos without regressing indexing performance.

**Architecture:** Implement this in two waves. Wave A changes server- and DB-facing tool contracts first: compact outputs, new file-navigation tools, and better fallback/orientation behavior. Wave B deepens the index: cleaner TS docs, richer Vue metadata, file-level semantic summaries, type-reference edges, and improved embedding text, all behind incremental schema/indexer changes that preserve compatibility where practical.

**Tech Stack:** Python 3.13, SQLite, MCP over stdio, existing `srclight` DB/query layer, tree-sitter based indexing, pytest, ruff.

---

## File Map

### Runtime and tool surface

- `src/srclight/server.py`
  - MCP tool entrypoints and JSON response shaping
  - Add `list_files` and `get_file_summary`
  - Change default summary/verbose behavior for communities and execution flows
  - Add `get_community()` fallback behavior and search output cleanup

- `src/srclight/workspace.py`
  - Workspace-scoped versions of search/navigation/orientation
  - Mirror new file tools and project-aware fallback behavior
  - Keep workspace behavior aligned with single-repo behavior

### Storage and query layer

- `src/srclight/db.py`
  - Add file-summary/file-metadata storage and migration
  - Add compact flow/community queries and file-navigation queries
  - Add helpers for file-candidate fallback and type-usage queries
  - Keep query cost bounded for summary-first tools

### Indexing and metadata extraction

- `src/srclight/indexer.py`
  - Improve `_extract_doc_comment()`
  - Enrich Vue SFC metadata and file-level summaries
  - Add `uses_type` edges
  - Update `INDEXER_BUILD_ID` when extraction/storage semantics change

- `src/srclight/embeddings.py`
  - Improve `prepare_embedding_text()` to serialize richer Vue/file/type/fullstack context without bloating payloads

### Tests

- `tests/test_community.py`
  - Flow/community summary and truncation behavior

- `tests/test_features.py`
  - Search/ranking/fallback behavior and server-facing output shape

- `tests/test_new_tools.py`
  - New MCP tool coverage for `list_files` and `get_file_summary`

- `tests/test_workspace.py`
  - Project-scoped parity for new tools and fallback/orientation behavior

- `tests/test_indexer.py`
  - TS doc extraction, Vue metadata, type-reference indexing

- `tests/test_embeddings.py`
  - Embedding text regression tests for richer metadata

- `tests/test_db.py`
  - File metadata migration and storage helpers

### Docs

- `README.md`
- `README.ru.md`
- `docs/usage-guide.md`
  - Document new tool surface and summary/verbose behavior once code is stable

## Implementation Notes Before Starting

- Keep Wave A and Wave B separately shippable.
- Do not add extra MCP tools beyond `list_files` and `get_file_summary`.
- Keep response-shape compatibility explicit:
  - summary-first becomes default
  - `verbose=true` preserves detail
- Define concrete thresholds while implementing:
  - “materially cheaper” means summary responses are structurally smaller than current verbose payloads and omit low-value repeated fields
  - “no meaningful regression” means targeted indexing tests stay green and no newly added regression test shows obvious scan-time blowups

### Task 1: Compact Communities and Execution Flows

**Files:**
- Modify: `src/srclight/db.py:1979-2085`
- Modify: `src/srclight/server.py:3393-3524`
- Test: `tests/test_community.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for summary-first flow and community outputs**

Add targeted tests that assert:

- `get_communities()` returns compact entries by default
- `get_communities(verbose=True)` still returns detailed member payload
- `get_communities(member_limit=3, path_prefix="server/", layer="server")` respects the filters
- `get_execution_flows()` returns compact entries by default
- `get_execution_flows(verbose=True)` includes steps
- `get_execution_flows(path_prefix="server/", layer="server")` respects the filters
- truncated flows expose explicit truncation fields

```python
communities = json.loads(server.get_communities(member_limit=2, path_prefix="server/", layer="server"))
assert communities["communities"][0]["member_count"] >= len(communities["communities"][0]["members"])
assert len(communities["communities"][0]["members"]) <= 2

payload = json.loads(server.get_execution_flows())
assert payload["flows"][0]["truncated"] is True
assert "steps" not in payload["flows"][0]

verbose = json.loads(server.get_execution_flows(verbose=True, max_depth=4))
assert verbose["flows"][0]["steps"]
assert verbose["flows"][0]["max_depth_applied"] == 4

filtered = json.loads(server.get_execution_flows(path_prefix="server/", layer="server"))
assert filtered["flows"]
```

- [ ] **Step 2: Run targeted tests to verify the current behavior fails**

Run:

```bash
pytest tests/test_community.py tests/test_features.py -k "execution_flows or communities" -v
```

Expected: FAIL because current tools only return verbose payloads and do not expose truncation metadata or summary-first shape.

- [ ] **Step 3: Add compact query/shape helpers in the DB layer**

Implement DB helpers that separate retrieval from shaping:

- `Database.get_communities(...)`
- `Database.get_execution_flows(...)`
- optionally `Database.get_flow_steps(...)` plus lightweight summary wrappers

Use lightweight structures for summary mode and avoid repeated `metadata: null`-style fields.

```python
def get_execution_flows(self, limit: int = 50, *, max_depth: int | None = None,
                        path_prefix: str | None = None, layer: str | None = None) -> list[dict]:
    ...

def get_communities(self, *, limit: int = 25, member_limit: int = 5,
                    path_prefix: str | None = None, layer: str | None = None) -> list[dict]:
    ...

def summarize_flow(flow: dict, steps: list[dict], *, max_depth: int | None) -> dict:
    return {
        "label": flow["label"],
        "entry": flow["entry_name"],
        "terminal": flow["terminal_name"],
        "step_count": flow["step_count"],
        "truncated": truncated,
        "max_depth_applied": max_depth,
        "key_steps": compact_steps,
    }
```

- [ ] **Step 4: Update MCP tool signatures and JSON shaping in `server.py`**

Add summary/verbose controls and filtering parameters without breaking simple existing usage.

```python
def get_communities(
    project: str | None = None,
    verbose: bool = False,
    limit: int = 25,
    member_limit: int = 5,
    path_prefix: str | None = None,
    layer: str | None = None,
) -> str:
    ...

def get_execution_flows(
    project: str | None = None,
    verbose: bool = False,
    limit: int = 25,
    max_depth: int | None = None,
    path_prefix: str | None = None,
    layer: str | None = None,
) -> str:
    ...
```

- [ ] **Step 5: Re-run the targeted tests and then a broader sanity slice**

Run:

```bash
pytest tests/test_community.py tests/test_features.py -k "execution_flows or communities" -v
pytest tests/test_workspace.py -k "codebase_map or project" -v
```

Expected: PASS for the new summary-first behavior and no regressions in nearby project/tool logic.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/db.py src/srclight/server.py tests/test_community.py tests/test_features.py tests/test_workspace.py
git commit -m "feat: compact flow and community outputs"
```

### Task 2: Search Output Cleanup and `get_community()` Fallback

**Files:**
- Modify: `src/srclight/server.py:1280-1437`
- Modify: `src/srclight/server.py:3432-3489`
- Modify: `src/srclight/db.py:1160-1400`
- Modify: `src/srclight/workspace.py:311-520`
- Test: `tests/test_features.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for cleaner search output and file-aware community fallback**

Add tests that assert:

- hybrid/search output does not expose noisy `rrf_score` by default
- search results expose `rank_source` and/or `match_reasons`
- `get_community("LayoutEngine")` can recover through file-level or nearest-file fallback
- workspace-mode fallback stays project-scoped

```python
payload = json.loads(server.hybrid_search("auth routes"))
assert "rrf_score" not in payload["results"][0]
assert payload["results"][0]["rank_source"] in {"keyword", "semantic", "hybrid"}

fallback = json.loads(server.get_community("LayoutEngine"))
assert fallback["next_step"]["tool"] in {"get_file_summary", "symbols_in_file"}
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/test_features.py tests/test_workspace.py -k "fallback or hybrid_search or route_query or get_community" -v
```

Expected: FAIL because current output still surfaces raw score internals and community lookup stops at symbol misses.

- [ ] **Step 3: Add DB/file-candidate helpers for fallback lookups**

Add a small helper that can find likely file matches by exact filename, compacted token phrase, or nearest symbol-bearing file.

```python
def suggest_file_candidates(self, query: str, limit: int = 5) -> list[dict]:
    ...
```

Keep this cheap and deterministic. Prefer indexed file rows and symbol-bearing files over filesystem scans.

- [ ] **Step 4: Update search and community tool shaping**

In `server.py` and `workspace.py`:

- hide raw `rrf_score` by default
- add `rank_source` / `match_reasons`
- change `get_community()` miss handling to:
  1. exact symbol
  2. nearest symbol
  3. file candidate
  4. explicit suggested next tool

- [ ] **Step 5: Re-run targeted tests**

Run:

```bash
pytest tests/test_features.py tests/test_workspace.py -k "fallback or hybrid_search or get_community" -v
```

Expected: PASS with project-aware fallback behavior and cleaner search payloads.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/db.py src/srclight/server.py src/srclight/workspace.py tests/test_features.py tests/test_workspace.py
git commit -m "feat: improve search hints and community fallback"
```

### Task 3: Add `list_files` and `get_file_summary`

**Files:**
- Modify: `src/srclight/db.py:246-580`
- Modify: `src/srclight/db.py:1160-1901`
- Modify: `src/srclight/server.py:1437-1546`
- Modify: `src/srclight/workspace.py:311-560`
- Test: `tests/test_db.py`
- Test: `tests/test_new_tools.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for file listing, file summaries, and file metadata persistence**

Cover:

- schema migration for file summary metadata
- `list_files(path_prefix="shared/src/domain/level")`
- `list_files(path_prefix="shared/src/domain", recursive=True, limit=50)`
- `get_file_summary("client/src/components/Foo.vue")`
- workspace project-scoped versions

```python
payload = json.loads(server.list_files(path_prefix="shared/src/domain", recursive=False))
assert payload["files"][0]["path"].startswith("shared/src/domain/")

deep_payload = json.loads(server.list_files(path_prefix="shared/src/domain", recursive=True, limit=2))
assert len(deep_payload["files"]) == 2
assert deep_payload["recursive"] is True

summary = json.loads(server.get_file_summary("client/src/components/ProfileCard.vue"))
assert summary["file"] == "client/src/components/ProfileCard.vue"
assert "top_level_symbols" in summary
```

- [ ] **Step 2: Run the new-tool and DB tests to confirm they fail**

Run:

```bash
pytest tests/test_db.py tests/test_new_tools.py tests/test_workspace.py -k "list_files or file_summary or migration" -v
```

Expected: FAIL because the schema and tool surface do not exist yet.

- [ ] **Step 3: Extend the file schema and DB helpers**

Bump the schema version and extend `files` storage with lightweight summary metadata.

Suggested minimal shape:

- `summary TEXT`
- `metadata TEXT`

Update `FileRecord`, `initialize()`, and `upsert_file()`.

```python
class FileRecord:
    ...
    summary: str | None = None
    metadata: dict | None = None
```

Add DB helpers:

- `list_files(...)`
- `get_file_summary(...)`
- `update_file_summary(...)`

- [ ] **Step 4: Add MCP tools and workspace plumbing**

Add:

- `server.list_files(...)`
- `server.get_file_summary(...)`
- matching workspace methods that keep results project-scoped

Use the explicit contract:

```python
def list_files(
    path_prefix: str | None = None,
    project: str | None = None,
    recursive: bool = True,
    limit: int = 100,
) -> str:
    ...
```

Use DB-backed data only. Do not fall back to raw filesystem traversal inside these tools.

- [ ] **Step 5: Re-run the targeted tests**

Run:

```bash
pytest tests/test_db.py tests/test_new_tools.py tests/test_workspace.py -k "list_files or file_summary or migration" -v
```

Expected: PASS with working tool outputs and workspace support.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/db.py src/srclight/server.py src/srclight/workspace.py tests/test_db.py tests/test_new_tools.py tests/test_workspace.py
git commit -m "feat: add file listing and summary tools"
```

### Task 4: Make `codebase_map()` More File-Aware

**Files:**
- Modify: `src/srclight/server.py:1170-1269`
- Modify: `src/srclight/workspace.py:370-560`
- Modify: `src/srclight/db.py:721-820`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing orientation tests that depend on file summaries and indexed file hints**

Add tests that assert:

- `codebase_map()` uses indexed file summaries when present
- route/controller/config files appear in `start_here` and topology hints
- unconventional but indexed backend surfaces remain discoverable

```python
payload = wdb.codebase_map(project="meanjong")
assert any(item["path"].endswith("src/main.ts") for item in payload["start_here"])
assert payload["topology"]["routes"]["files"]
```

- [ ] **Step 2: Run the orientation tests**

Run:

```bash
pytest tests/test_workspace.py -k "codebase_map" -v
```

Expected: FAIL for the new file-summary-aware expectations.

- [ ] **Step 3: Add file-level orientation helpers**

Use file metadata first, then fall back to representative file heuristics.

```python
def orientation_files(self, limit: int = 100) -> list[dict[str, Any]]:
    ...
```

Thread these hints through `server.codebase_map()` and `WorkspaceDB.codebase_map()`.

- [ ] **Step 4: Re-run the orientation tests**

Run:

```bash
pytest tests/test_workspace.py -k "codebase_map" -v
```

Expected: PASS with stronger file-aware orientation.

- [ ] **Step 5: Commit**

```bash
git add src/srclight/db.py src/srclight/server.py src/srclight/workspace.py tests/test_workspace.py
git commit -m "feat: enrich codebase map with file-level orientation"
```

### Task 5: Clean Up TS/JS Doc Extraction

**Files:**
- Modify: `src/srclight/indexer.py:205-320`
- Test: `tests/test_indexer.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests for meaningful docs and separator-noise rejection**

Add tests for:

- nearest JSDoc wins
- meaningful block comments beat decorative separators
- `// -----` is ignored
- TODO-only comments are ignored

```python
symbol = db.get_symbol_by_name("handleMessage")
assert symbol.doc_comment == "Handle an inbound websocket payload."
```

- [ ] **Step 2: Run the indexer/embed tests to confirm failure**

Run:

```bash
pytest tests/test_indexer.py tests/test_embeddings.py -k "doc_comment or prepare_embedding_text" -v
```

Expected: FAIL because current extraction still accepts decorative noise.

- [ ] **Step 3: Tighten `_extract_doc_comment()`**

Implement a small scoring/filtering pass:

- reject decorative separators
- prefer JSDoc
- fall back to meaningful comments only

```python
def _is_noise_comment(text: str) -> bool:
    ...

def _extract_doc_comment(source_bytes: bytes, node: Node) -> str | None:
    ...
```

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
pytest tests/test_indexer.py tests/test_embeddings.py -k "doc_comment or prepare_embedding_text" -v
```

Expected: PASS with cleaner doc extraction and preserved embedding coverage.

- [ ] **Step 5: Commit**

```bash
git add src/srclight/indexer.py tests/test_indexer.py tests/test_embeddings.py
git commit -m "fix: prefer meaningful ts and js doc comments"
```

### Task 6: Enrich Vue SFC Metadata and File-Level Summaries

**Files:**
- Modify: `src/srclight/indexer.py:3238-3894`
- Modify: `src/srclight/indexer.py:3997-4060`
- Modify: `src/srclight/embeddings.py:132-309`
- Test: `tests/test_indexer.py`
- Test: `tests/test_embeddings.py`
- Test: `tests/test_new_tools.py`

- [ ] **Step 1: Write failing tests for richer Vue metadata and file summaries**

Cover:

- props/emits/slots in component metadata
- routes_used/css_modules/scoped_styles in component metadata
- stores/composables/GraphQL hints in component metadata
- child component tags, named slots, event handlers, and `v-model` in component metadata
- `defineModel`, `defineExpose`, and `useTemplateRef` handling in `<script setup>`
- `get_file_summary()` surfaces component identity instead of raw style noise
- embedding text includes structured Vue hints

```python
component = db.get_symbol_by_name("ProfileCard")
assert component.metadata["props"] == ["user", "editable"]
assert "default" in component.metadata["slots"]
assert component.metadata["css_modules"] == ["card", "avatar"]
assert component.metadata["routes_used"] == ["/checkout"]
assert component.metadata["template_components"] == ["AppButton", "UserAvatar"]
assert "onSave" in component.metadata["event_handlers"]

summary = db.get_file_summary("app/components/ProfileCard.vue")
assert summary["framework"] == "vue"
assert summary["summary"].startswith("Vue component")
```

- [ ] **Step 2: Run the failing Vue-focused tests**

Run:

```bash
pytest tests/test_indexer.py tests/test_embeddings.py tests/test_new_tools.py -k "vue or component or file_summary" -v
```

Expected: FAIL because current Vue extraction mostly pushes hints into `doc_comment` and does not expose structured file summaries.

- [ ] **Step 3: Add structured Vue metadata extraction and file summary generation**

Implement:

- richer script/template/style signal aggregation
- file-level summary builder for Vue pages/components
- `db.update_file_summary(...)` calls during indexing
- script-setup macro handling for `defineProps`, `defineEmits`, `defineModel`, `defineExpose`, `useTemplateRef`
- template signal handling for child components, named slots, event handlers, and `v-model`
- style signal handling for CSS modules and scoped styles

```python
file_summary = {
    "framework": "vue",
    "role": "component",
    "summary": "Vue component using auth store and GetCatalog query.",
    "metadata": {...},
}
```

Update `INDEXER_BUILD_ID` once the new extraction/storage contract is in place.

- [ ] **Step 4: Teach `prepare_embedding_text()` to serialize Vue structure compactly**

Prefer compact structured context over dumping giant component prose.

```python
if resource == "component":
    metadata_parts.append("props: " + ", ".join(props))
    metadata_parts.append("emits: " + ", ".join(emits))
```

- [ ] **Step 5: Re-run the targeted Vue tests**

Run:

```bash
pytest tests/test_indexer.py tests/test_embeddings.py tests/test_new_tools.py -k "vue or component or file_summary" -v
```

Expected: PASS with useful Vue metadata, file summaries, and embedding text.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/indexer.py src/srclight/embeddings.py tests/test_indexer.py tests/test_embeddings.py tests/test_new_tools.py
git commit -m "feat: enrich vue metadata and file summaries"
```

### Task 7: Add Type-Reference Edges and Surface Them Through Existing Tools

**Files:**
- Modify: `src/srclight/indexer.py:3030-4551`
- Modify: `src/srclight/db.py:1500-1760`
- Modify: `src/srclight/server.py:1546-2304`
- Modify: `src/srclight/workspace.py:560-760`
- Test: `tests/test_indexer.py`
- Test: `tests/test_features.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for `uses_type` edges and type-aware lookups**

Cover:

- interface and type alias usage produces `uses_type` edges
- existing lookups can surface type-driven relationships without pretending they are calls
- workspace parity stays intact

```python
edges = db.get_dependents_for_symbol(symbol.id, edge_types=("uses_type",))
assert any(edge["name"] == "AuthService" for edge in edges)
```

- [ ] **Step 2: Run the type-focused tests to confirm failure**

Run:

```bash
pytest tests/test_indexer.py tests/test_features.py tests/test_workspace.py -k "interface or type_alias or uses_type or dependents" -v
```

Expected: FAIL because the current graph only tracks call/inheritance-style edges.

- [ ] **Step 3: Add `uses_type` edge extraction in the indexer**

Implement a narrow first pass for TypeScript:

- interface references
- type alias references
- keep call edges separate from type-usage edges

```python
EDGE_TARGET_KINDS = {..., "interface", "type_alias"}
...
self.db.insert_edge(EdgeRecord(source_id=source_id, target_id=target_id, edge_type="uses_type"))
```

- [ ] **Step 4: Extend existing query/tool surfaces without adding a new MCP tool**

Add optional type-awareness to existing graph tools where useful, for example via explicit edge-type selection or separate sections in the response.

```python
{
    "callers": [...],
    "type_users": [...],
}
```

Keep defaults intuitive and do not silently merge type usage into ordinary call counts.

- [ ] **Step 5: Re-run the targeted tests**

Run:

```bash
pytest tests/test_indexer.py tests/test_features.py tests/test_workspace.py -k "interface or type_alias or uses_type or dependents" -v
```

Expected: PASS with visible type-driven architecture edges.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/indexer.py src/srclight/db.py src/srclight/server.py src/srclight/workspace.py tests/test_indexer.py tests/test_features.py tests/test_workspace.py
git commit -m "feat: add type usage graph edges"
```

### Task 8: Improve Embedding Text for Fullstack Agent Queries

**Files:**
- Modify: `src/srclight/embeddings.py:132-309`
- Modify: `tests/test_embeddings.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for improved structured embedding text**

Cover:

- route path + method
- controller prefix
- file-summary-driven framework identity
- queue/event names
- Vue props/emits/slots/stores/composables
- entity/schema/table hints

```python
prepared = prepare_embedding_text(symbol)
assert "POST /api/auth/refresh" in prepared
assert "props: user, editable" in prepared
```

- [ ] **Step 2: Run the embedding tests to confirm failure**

Run:

```bash
pytest tests/test_embeddings.py tests/test_features.py -k "prepare_embedding_text or hybrid_search" -v
```

Expected: FAIL for the new richer structured assertions.

- [ ] **Step 3: Extend `prepare_embedding_text()` without leaking noisy or sensitive values**

Prefer compact serialized context, not raw JSON blobs.

```python
if file_summary:
    metadata_parts.append(file_summary)
```

Preserve current connection URL redaction behavior and do not reintroduce secrets into embedding text.

- [ ] **Step 4: Re-run the embedding-focused tests**

Run:

```bash
pytest tests/test_embeddings.py tests/test_features.py -k "prepare_embedding_text or hybrid_search" -v
```

Expected: PASS with stronger retrieval context and no secret leakage.

- [ ] **Step 5: Commit**

```bash
git add src/srclight/embeddings.py tests/test_embeddings.py tests/test_features.py
git commit -m "feat: enrich embedding text for agent retrieval"
```

### Task 9: Document the New Tool Surface and Run Final Verification

**Files:**
- Modify: `docs/usage-guide.md`
- Modify: `README.md`
- Modify: `README.ru.md`

- [ ] **Step 1: Update usage docs for the new behavior**

Document:

- `list_files`
- `get_file_summary`
- summary vs verbose flow/community usage
- file-aware/community fallback behavior

```markdown
Use `get_file_summary()` when `symbols_in_file()` is too shallow but a full read would waste tokens.
```

- [ ] **Step 2: Run the focused full verification sweep**

Run:

```bash
ruff check src/srclight tests
pytest tests/test_db.py tests/test_new_tools.py tests/test_community.py tests/test_features.py tests/test_workspace.py tests/test_indexer.py tests/test_embeddings.py -v
```

Expected: PASS across the touched surfaces.

- [ ] **Step 3: Run real-repo acceptance checks**

From indexed local repos, manually validate a small fixed question set using `srclight` only:

```bash
srclight codebase-map
srclight list-files --path-prefix shared/src/domain
srclight get-file-summary shared/src/domain/level/solver.ts
```

Use:

- `/Users/matrix/WebstormProjects/meanjong`
- `/Users/matrix/PhpstormProjects/casik-x/nuxt-client2`
- `/Users/matrix/PhpstormProjects/tgifts-front`
- `/Users/matrix/PhpstormProjects/tgifts-admin`
- `/Users/matrix/PhpstormProjects/rootmycar/mercedes-portal`
- `/Users/matrix/PhpstormProjects/Orakul-Project`

Check that:

- file navigation no longer requires grep in common cases
- `get_execution_flows()` and `get_communities()` are cheaper by default
- Vue/component/type-heavy queries land on better artifacts on the first try

- [ ] **Step 4: Commit**

```bash
git add docs/usage-guide.md README.md README.ru.md
git commit -m "docs: explain agent ergonomics tools and compact outputs"
```

## Final Handoff Checklist

- [ ] All task commits are in place
- [ ] Summary/verbose compatibility is documented
- [ ] Schema migration for file summaries is covered by tests
- [ ] `INDEXER_BUILD_ID` changed if extraction/storage behavior changed
- [ ] No new default response is bloated or ambiguous
- [ ] No scan-time regression was introduced by richer metadata extraction
