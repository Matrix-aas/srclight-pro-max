# srclight Final Stabilization and Task Context v1

Status: Draft approved in conversation, implementation in progress
Date: 2026-04-16
Repo: /Users/matrix/WebstormProjects/srclight

## Summary

This spec defines the final stabilization batch for `srclight` before considering the current line production-ready for wider day-to-day use.

The user-reported remaining gaps come from real usage on `mercedes-portal` and fall into four clusters:

1. `find_dead_code()` still produces a few framework-specific false positives.
2. `get_callees()` can still show misleading same-name method targets.
3. several tools still return more payload than an AI agent needs.
4. there is still no single task-oriented context tool that packages the minimum useful project context for a concrete coding job.

This batch closes all four clusters without reopening broad architectural churn.

## Goals

1. Eliminate the remaining dead-code false positives caused by Nuxt auto-imported composables and Nest `useFactory` providers.
2. Tighten graph resolution so `get_callees()` stops surfacing misleading cross-service same-name method edges.
3. Add a consistent compact-output path for heavy tools so agents spend fewer tokens on boilerplate JSON.
4. Introduce `context_for_task(...)` as a deterministic, compact, explainable orchestration tool for AI agents.
5. Opportunistically split out new logic into focused modules instead of making `server.py` and `indexer.py` even larger.

## Non-Goals

1. Do not build a generalized whole-program static analyzer.
2. Do not attempt a full Nuxt runtime model of every auto-import/plugin/module behavior.
3. Do not replace existing tools with `context_for_task(...)`; it is additive.
4. Do not perform a large-scale refactor of all oversized files in this wave.

## Design Principles

1. False confidence is worse than missing data.
2. Framework-owned runtime usages should count as “live” when the framework actually invokes them.
3. Large MCP payloads should be opt-in, not the default, when a compact answer is sufficient.
4. A task-context tool should orchestrate existing indexed knowledge, not hide a second planner inside the server.
5. File splitting should follow seams introduced by the new work, not abstract cleanliness for its own sake.

## Scope

This initiative has four implementation tracks.

### Track 1: Dead-Code Awareness for Implicit Framework Usage

#### 1.1 Nuxt auto-import composables

The remaining false positives such as `useDescriptionEditor`, `useBreakpoint`, and `usePagination` come from Nuxt/Vue runtime resolution, not explicit imports.

`srclight` should recognize these as live when:

- they are called from Vue/Nuxt SFC script blocks without local declaration or import
- the containing repo has Nuxt signals (`nuxt.config.*`, Nuxt dependencies, or server/runtime hints)
- the target function can be matched to likely auto-import locations such as:
  - `composables/`
  - `utils/`
  - `stores/`
  - `.nuxt/imports.d.ts` when present

This does not require perfect emulation of Nuxt. It requires conservative enough resolution to stop reporting obviously live composables as dead.

#### 1.2 Nest `useFactory`

Nest factory providers should be treated like DI-owned runtime entrypoints.

When a module contains:

```ts
providers: [
  {
    provide: TOKEN,
    useFactory: (...) => ...
  }
]
```

the indexed factory symbol should receive a framework-owned incoming edge so it is not reported as dead code.

The same mechanism should work whether the factory is inline or references a named function where practical.

### Track 2: Graph Precision for Same-Name Methods

Current misleading edges happen when method names such as `findOne` collide across services and the graph prefers a same-name symbol even when the receiver clearly points to a framework/model object.

The fix should:

- use receiver context more aggressively
- avoid emitting a confident callee edge when the receiver looks like:
  - model/document/query builder/framework object
  - a property without strong service/class ownership evidence
- prefer dropping ambiguous low-confidence same-name edges rather than returning misleading results

This is especially important for `get_callees()` because a wrong callee is actively harmful.

### Track 3: Compact Output Surface

Several tools still produce large payloads by default or in common multi-match scenarios.

This wave should introduce a consistent compact shaping layer, applied where it materially reduces waste.

Initial scope:

- `get_symbol(...)`
  - already compact for high-frequency exact matches; keep and harden this path
- `detect_changes(...)`
  - add `compact=true` option returning only symbol identity, file, risk, and key impact summary
- other heavy tools may use the same response-shaping helpers where cheap and safe

Compact mode must stay deterministic and preserve enough information for the next tool call.

### Track 4: New Tool — `context_for_task(...)`

Add a new MCP tool backed by a dedicated module, for example `src/srclight/task_context.py`.

The tool contract for `v1`:

- inputs:
  - `task: str`
  - `project: str | None = None`
  - `budget: "small" | "medium" | "large" = "medium"`
- works in:
  - single-repo mode
  - workspace mode with mandatory `project`

The tool should:

1. find seed symbols/files from the task text
2. gather the smallest useful related context:
   - primary symbols
   - owning files
   - nearby API endpoints
   - related tests
   - important types/schemas
   - callers/callees/dependents where relevant
3. rank and trim results to fit the requested budget
4. return structured next steps and the reasons each result was included

It must not:

- run an internal LLM
- dump full source for every hit
- behave opaquely

The output should be compact, explainable, and optimized for an AI agent about to make the next query or code change.

## Selective File Extraction

Because `indexer.py` and `server.py` are already large, this wave should extract only the seams it introduces:

- a dedicated `task_context.py`
- compact-output helpers where practical
- framework dead-code helpers if they can be isolated cleanly

Do not attempt a broad “split every giant file” refactor in this wave.

## Validation

Success criteria:

1. `mercedes-portal` no longer reports the remaining false positives for Nuxt auto-imported composables and Nest `useFactory` providers.
2. misleading `get_callees()` same-name service edges no longer appear.
3. compact output modes materially reduce payload size while preserving usability.
4. `context_for_task(...)` can answer realistic tasks such as:
   - “add validation to CodingService.updateDescription”
   - “trace comment posting flow”
   - “find where auth session is refreshed”
5. no regression in current test suite or live smoke checks on indexed real repos.
