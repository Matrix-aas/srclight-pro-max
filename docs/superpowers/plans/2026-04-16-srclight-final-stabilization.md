# srclight Final Stabilization and Task Context v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the last reported framework-aware dead-code and graph-precision bugs, add compact output shaping, and introduce `context_for_task(...)` as a deterministic AI-first context tool.

**Architecture:** Implement this as four focused slices: framework dead-code awareness, tighter same-name callee resolution, reusable compact response shaping, and a new `task_context.py` orchestration module. Extract only the seams introduced by this work so the batch improves maintainability without a risky broad refactor.

**Tech Stack:** Python 3.13, SQLite, MCP over stdio, pytest, ruff, existing srclight DB/query/indexer stack.

---

## File Map

- `src/srclight/indexer.py`
  - add framework-owned metadata/edges for Nuxt auto-import composables and Nest `useFactory`
  - keep extractor changes tightly scoped
- `src/srclight/db.py`
  - support any new dead/live filtering helpers and callee filtering needs
- `src/srclight/server.py`
  - wire compact outputs and expose `context_for_task(...)`
- `src/srclight/task_context.py`
  - new module for task-context orchestration and ranking
- `tests/test_new_tools.py`
  - add `context_for_task(...)` coverage
- `tests/test_features.py`
  - graph precision and compact output regressions
- `tests/test_new_tools.py`
  - dead-code/runtime-usage regressions where server-facing
- `tests/test_indexer.py`
  - Nuxt auto-import and Nest `useFactory` extraction behavior
- `tests/test_community.py`
  - only if risk/output changes spill there
- `README.md`, `README.ru.md`, `docs/usage-guide.md`
  - document `context_for_task(...)` and compact-output options once stable

## Task 1: Framework-Aware Dead Code Fixes

**Files:**
- Modify: `src/srclight/indexer.py`
- Modify: `src/srclight/db.py`
- Test: `tests/test_indexer.py`
- Test: `tests/test_new_tools.py`

- [ ] **Step 1: Write failing tests for Nuxt auto-import composables and Nest `useFactory`**

Add tests that prove:
- a Vue/Nuxt SFC using `useBreakpoint()`, `usePagination()`, or another custom composable without explicit import does not leave the target composable dead
- a Nest module with a `useFactory` provider does not leave the factory symbol dead

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
pytest tests/test_indexer.py tests/test_new_tools.py -k "useFactory or auto_import or dead_code" -v
```

- [ ] **Step 3: Implement conservative Nuxt auto-import resolution**

Add a minimal framework-aware path that:
- recognizes likely Nuxt repos
- resolves bare `useXxx()` calls against indexed composable locations and `.nuxt/imports.d.ts` when available
- creates incoming liveness edges or equivalent ownership/runtime edges

- [ ] **Step 4: Implement Nest `useFactory` liveness**

Ensure module/provider extraction records enough metadata for `useFactory` symbols to receive framework-owned incoming edges.

- [ ] **Step 5: Re-run the targeted tests**

Run the same pytest slice and confirm the dead-code regressions are fixed.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/indexer.py src/srclight/db.py tests/test_indexer.py tests/test_new_tools.py
git commit -m "fix: recognize framework-owned live code"
```

## Task 2: Tighten Same-Name Callee Resolution

**Files:**
- Modify: `src/srclight/indexer.py`
- Modify: `src/srclight/server.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for misleading same-name service edges**

Add tests covering receiver-aware resolution where:
- `this.codingModel.findOne()` must not resolve to `BlockService.findOne`
- same-name methods across services remain ambiguous unless there is strong ownership/import evidence

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
pytest tests/test_features.py -k "callee and same_name" -v
```

- [ ] **Step 3: Tighten receiver-context filtering**

Implement the smallest safe fix:
- suppress low-confidence same-name method edges when the receiver looks framework/model-owned
- preserve good edges with strong owner/import evidence

- [ ] **Step 4: Re-run the targeted tests**

Run the same pytest slice and verify misleading callee edges are gone.

- [ ] **Step 5: Commit**

```bash
git add src/srclight/indexer.py src/srclight/server.py tests/test_features.py
git commit -m "fix: tighten same-name callee resolution"
```

## Task 3: Compact Output Shaping

**Files:**
- Modify: `src/srclight/server.py`
- Create or Modify: helper module if it reduces `server.py` size cleanly
- Test: `tests/test_features.py`
- Test: `tests/test_new_tools.py`

- [ ] **Step 1: Write failing tests for compact heavy outputs**

Add tests for:
- `get_symbol(...)` preserving compact behavior for high-frequency exact matches during any shared output-shaping refactor
- `detect_changes(compact=True)` returning summary entries without full heavy payloads
- any new reusable compact shaper behavior needed for multi-match symbol or change outputs

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
pytest tests/test_features.py tests/test_new_tools.py -k "compact and detect_changes" -v
```

- [ ] **Step 3: Implement reusable compact response shaping**

Keep it deterministic and compact:
- summary identity
- file
- line
- risk
- key reasons / next steps

- [ ] **Step 4: Re-run the targeted tests**

Confirm compact outputs are smaller and still useful.

- [ ] **Step 5: Commit**

```bash
git add src/srclight/server.py tests/test_features.py tests/test_new_tools.py
git commit -m "feat: add compact output shaping for heavy tools"
```

## Task 4: Add `context_for_task(...)` v1

**Files:**
- Create: `src/srclight/task_context.py`
- Modify: `src/srclight/server.py`
- Test: `tests/test_new_tools.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for `context_for_task(...)`**

Add tests that assert the tool:
- works in single-repo mode
- requires `project` in workspace mode
- returns compact structured results with:
  - seeds
  - primary symbols/files
  - related API/tests/types
  - next steps
- respects `budget` levels

Use an explicit contract shape for `v1` so tests are stable:

```python
{
    "task": str,
    "budget": "small" | "medium" | "large",
    "seeds": list[dict],
    "primary_symbols": list[dict],
    "primary_files": list[dict],
    "related_api": list[dict],
    "related_tests": list[dict],
    "data_types": list[dict],
    "call_chain": list[dict],
    "next_steps": list[str],
    "why_these_results": list[str],
}
```

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
pytest tests/test_new_tools.py tests/test_workspace.py -k "context_for_task" -v
```

- [ ] **Step 3: Implement `task_context.py`**

Build a deterministic orchestration layer that:
- seeds from `hybrid_search`/`search_symbols`
- resolves compact file and symbol summaries
- attaches nearby API/tests/types
- ranks and trims under `small|medium|large` budgets

- [ ] **Step 4: Wire the MCP tool in `server.py`**

Expose:

```python
def context_for_task(task: str, project: str | None = None, budget: str = "medium") -> str:
    ...
```

- [ ] **Step 5: Re-run the targeted tests**

Run the same pytest slice and verify both single-repo and workspace behavior.

- [ ] **Step 6: Commit**

```bash
git add src/srclight/task_context.py src/srclight/server.py tests/test_new_tools.py tests/test_workspace.py
git commit -m "feat: add task context tool"
```

## Task 5: Docs and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `README.ru.md`
- Modify: `docs/usage-guide.md`

- [ ] **Step 1: Update docs for compact outputs and `context_for_task(...)`**

- [ ] **Step 2: Run full verification**

Run:

```bash
ruff check src/srclight tests
pytest -q
```

- [ ] **Step 3: Live smoke on real repos**

Run focused manual checks on:
- `mercedes-portal`
- one Nuxt/Vue repo already used during prior validation

- [ ] **Step 4: Final commit**

```bash
git add README.md README.ru.md docs/usage-guide.md
git commit -m "docs: cover final stabilization and task context"
```
