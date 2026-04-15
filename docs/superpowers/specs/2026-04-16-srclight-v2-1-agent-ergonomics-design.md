# srclight v2.1 Agent Ergonomics

Status: Draft approved in conversation, awaiting spec review
Date: 2026-04-16
Repo: /Users/matrix/WebstormProjects/srclight

## Summary

This spec defines the next product wave for `srclight` after the `v2` fullstack extraction work.

The goal is to make `srclight` materially better for AI agents working inside indexed repositories:

- cheaper to use in tokens
- better at file- and layer-level navigation without falling back to `grep`/`find`
- richer and more trustworthy on TypeScript, Vue SFCs, and type-heavy fullstack codebases

The user-provided trigger for this work was an honest external evaluation of `srclight` on `meanjong`. The evaluation confirmed that core orientation, communities, hybrid search, symbol navigation, and graph traversal are already useful, but also exposed three remaining weaknesses:

1. some outputs are too payload-heavy for agent workflows
2. file-level navigation still has gaps that push agents toward filesystem tools
3. the index still loses useful structure in TS/Vue/type-driven architectures

This wave addresses all three weaknesses as one coherent initiative.

## Goals

1. Make `get_execution_flows()` and `get_communities()` cheap enough to use as first-step orientation tools.
2. Let agents navigate indexed projects without immediately reaching for filesystem search.
3. Improve TypeScript and Vue extraction so the index reflects feature structure, not just raw symbol lists.
4. Make semantic and hybrid search more agent-shaped by improving indexed metadata and embedding text.
5. Preserve current `stdio`/MCP usability and avoid introducing another scan-time performance regression.

## Non-Goals

1. Do not add a large family of new specialized MCP tools in this wave.
2. Do not replace the existing call graph with a generalized whole-program static analyzer.
3. Do not build a full CSS or template compiler for Vue SFCs.
4. Do not break existing clients with abrupt incompatible response changes where a compatibility path is practical.

## User Outcomes

After this wave, an agent should be able to:

- get an overview of a repo with fewer wasted tokens
- list files within a layer or directory without leaving `srclight`
- ask for a concise file summary instead of reading the whole file
- recover from symbol-like queries that are actually file-level concepts
- see useful Vue component structure instead of noisy style/template fragments
- reason about TypeScript architectures that rely heavily on interfaces and type aliases

## Design Principles

1. Summary-first beats raw-data-first for agent workflows.
2. File-level navigation is a first-class need, not an edge case.
3. Richer indexing is only worthwhile if it remains cheap and stable.
4. New tools should be added only where they clearly unlock workflows that existing tools cannot cover cleanly.
5. Real-repo ergonomics matter more than elegant but hypothetical abstractions.

## Scope Overview

This initiative is organized into three tracks and two implementation waves.

Tracks:

1. Token-efficient outputs
2. Navigation without grep
3. Better index quality

Implementation waves:

- Wave A: output contracts, navigation tools, fallback behavior, ranking cleanup
- Wave B: TS/Vue/type-reference index improvements and richer embedding context

## Track 1: Token-Efficient Outputs

### 1.1 `get_execution_flows()`

`get_execution_flows()` should default to a compact summary mode instead of returning the full verbose flow payload.

Required changes:

- add a summary-first default output shape
- add `verbose=true` for full details
- add `max_depth` control
- add `limit`
- add `path_prefix` filtering
- add `layer` filtering where the layer can map to common repo surfaces such as `client`, `server`, `shared`, `docs`, or other repo-specific top-level roots
- explicitly report truncation with fields such as:
  - `truncated`
  - `max_depth_applied`
  - `remaining_steps`

The compact output should focus on:

- entrypoint
- short label
- step count
- whether the result is truncated
- a concise list of key steps or representative nodes
- major communities touched

The response must not emit repeated `metadata: null`-style noise.

### 1.2 `get_communities()`

`get_communities()` should also default to compact output.

Required changes:

- add summary-first default output
- add `verbose=true`
- add `limit`
- add `member_limit`
- add `path_prefix`
- add `layer`

The summary response should highlight:

- community label
- member count
- cohesion score if meaningful
- representative members
- dominant paths or layers

### 1.3 Ranking output cleanup

`hybrid_search()` and related search outputs should avoid exposing low-value scoring noise by default.

Required changes:

- stop surfacing raw `rrf_score` by default if it is not helping interpretation
- add cleaner ranking explanations such as:
  - `rank_source`: `keyword`, `semantic`, or `hybrid`
  - `match_reasons`
  - possibly a simple `score_band`

The goal is to help agents decide what to open next, not to expose internal ranking internals.

## Track 2: Navigation Without Grep

### 2.1 New tool: `list_files`

Add a small file-navigation MCP tool:

- `list_files(path_prefix?, project?, recursive?, limit?)`

Behavior:

- returns a lightweight file list
- supports listing a layer, folder, or sub-tree
- does not return heavy file payloads
- should be cheap enough to call early in exploration

This tool closes a current gap between symbol search and full filesystem inspection.

### 2.2 New tool: `get_file_summary`

Add a file-summary MCP tool:

- `get_file_summary(path, project?)`

Behavior:

- returns concise file identity and top-level structure
- acts as a middle ground between `symbols_in_file()` and reading the entire file

Expected contents:

- top-level exports
- important symbols
- short doc summary if available
- framework hints
- high-value metadata such as route/controller/component identity

This should be designed for agent consumption, not as a full AST dump.

### 2.3 `get_community()` fallback behavior

`get_community(name)` should not fail hard on file-shaped or namespace-shaped queries when the underlying concept clearly exists in the repo.

Fallback order:

1. exact symbol community match
2. nearest symbol match
3. file-level match
4. best candidate files with a suggested next tool

The response should guide the next step instead of ending at "not found".

### 2.4 `codebase_map()` file-aware orientation

`codebase_map()` should provide stronger file-level orientation hints when useful, especially for fullstack repos.

This does not replace symbol-level orientation. It complements it with better answers to:

- where the route files are
- where the server surfaces are
- where the main domain layers live
- where the likely entry files are

## Track 3: Better Index Quality

### 3.1 TS/JS doc extraction

Improve comment extraction so `srclight` prefers meaningful documentation and ignores decorative noise.

Required behavior:

- prefer nearest real JSDoc block
- fall back to meaningful block comments
- only use line comments when they look like actual documentation
- ignore separator comments and decorative comment leaders
- do not treat TODO/FIXME-only lines as documentation

This should improve:

- `get_symbol()`
- `get_file_summary()`
- embeddings

### 3.2 Vue SFC structure

Upgrade Vue component extraction so the index exposes component structure rather than mostly template/style fragments.

For `.vue` component metadata, extract and surface where reasonably detectable:

- `props`
- `emits`
- `slots`
- `stores_used`
- `composables_used`
- `graphql_ops_used`
- `routes_used`
- `css_modules`
- `scoped_styles`

For `<script setup>`, handle patterns such as:

- `defineProps`
- `defineEmits`
- `defineModel`
- `defineExpose`
- `useCssModule`
- `useTemplateRef`
- Nuxt composables
- store hooks
- common GraphQL composables

For `<template>`, surface only high-value structure:

- child component tags
- named slots
- event handlers
- `v-model` bindings

For `<style lang="postcss" module>` and related variants, index class/module selectors and attach them to the file/component without trying to build a heavyweight CSS compiler.

### 3.3 Type-reference graph

Add a distinct graph layer for type usage instead of trying to overload call edges.

Minimum requirement for this wave:

- capture `uses_type` relationships for interfaces and type aliases

Possible future extensions are explicitly out of scope for now:

- `implements_type`
- `returns_type`
- `accepts_type`

The first implementation only needs to make type-driven architectures visible and searchable.

### 3.4 File-level semantic summaries

Some repo concepts are file identities first and symbols second.

Examples:

- a Nitro route file
- a Nest controller file
- a Vue page component file
- a config file that defines a subsystem boundary

The index should produce lightweight file-level semantic summaries that can be used by:

- `get_file_summary()`
- `codebase_map()`
- embeddings

### 3.5 Better embedding text

Embedding text should become more structured and more aligned with agent intent.

For relevant symbols and files, include high-value context such as:

- route method and path
- controller prefix
- props, emits, slots
- queue or event names
- stores and composables used
- entity, schema, table, or collection names

The goal is to improve semantic retrieval quality without bloating index-time work or response payloads.

## Compatibility Strategy

1. Evolve existing tools compatibly where possible.
2. Use summary-first defaults, but preserve an opt-in verbose path.
3. Add only the two new tools that clearly unlock missing workflows:
   - `list_files`
   - `get_file_summary`

No broader MCP tool explosion is part of this wave.

## Performance Constraints

This wave must not introduce another indexing slowdown like the earlier import-resolution regression.

Hard constraints:

- no new per-symbol filesystem walks
- no repeated expensive import resolution without caching
- no default responses that are unnecessarily large
- `list_files` and `get_file_summary` must stay cheap

Any richer Vue or type extraction must be implemented with scan-time cost in mind.

## Verification Strategy

### Automated tests

Add or update tests for:

- compact vs verbose flow output
- compact vs verbose community output
- truncation signaling
- path and layer filters
- `get_community()` fallback behavior
- `list_files`
- `get_file_summary`
- TS/JS doc extraction noise filtering
- Vue metadata extraction
- type-reference edges
- ranking output cleanup

### Real-repo acceptance checks

Run acceptance checks against indexed repos that already exist in the local environment:

- `meanjong`
- `nuxt-client2`
- `tgifts-front`
- `tgifts-admin`
- `mercedes-portal`
- `Orakul-Project`

Acceptance criterion:

An agent should be able to orient itself, list relevant files, inspect file summaries, find feature entrypoints, and follow feature flow with fewer steps and fewer tokens than before.

## Success Criteria

This wave is successful if all of the following are true:

1. `get_execution_flows()` and `get_communities()` are materially cheaper to use by default.
2. `list_files` and `get_file_summary` remove the most obvious file-navigation gap.
3. `get_community()` no longer fails unhelpfully on file-shaped queries.
4. Vue component summaries become structurally useful.
5. Type-heavy TS architectures become more visible through indexed relationships.
6. Hybrid and semantic search become easier to interpret and more likely to surface the right artifact on the first try.
7. Indexing performance does not meaningfully regress on real repos.

## Proposed Implementation Order

Wave A:

1. compact output contracts for flows and communities
2. search output cleanup
3. `list_files`
4. `get_file_summary`
5. `get_community()` fallback
6. `codebase_map()` file-aware orientation improvements

Wave B:

1. TS/JS doc extraction cleanup
2. Vue metadata enrichment
3. type-reference edges
4. file-level semantic summaries
5. embedding text improvements

## Risks

1. Over-designing compact payloads and accidentally hiding useful detail.
2. Making Vue extraction deeper in a way that slows indexing or produces brittle heuristics.
3. Making compatibility promises that preserve too much legacy response noise.
4. Letting new file-level features drift into ad hoc filesystem duplication instead of using index-aware semantics.

## Open Questions

There are no unresolved product-level questions blocking planning.

Implementation-level questions such as exact field names, output shapes, and internal storage strategies should be resolved during the planning phase as long as they preserve the goals and constraints in this spec.
