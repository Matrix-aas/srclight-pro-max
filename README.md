<!-- mcp-name: io.github.srclight/srclight -->

# Srclight

AI-first code intelligence for fullstack repos.

This repository is a heavily upgraded fork of `srclight`, tuned for the kind of codebases AI agents actually suffer in: Vue, Nuxt, Nitro, NestJS, Drizzle, MikroORM, Mongoose, GraphQL, BullMQ, RabbitMQ, Redis, and the usual TypeScript/JavaScript backend mess that sends normal grep workflows straight to hell.

Srclight builds a local code index with symbols, call graphs, semantic search, impact analysis, framework-aware extraction, and MCP tools that let Claude, Codex, Cursor, and other agents understand a repo without burning half the session budget on blind file hunting.

## Why this fork exists

The old public `srclight` release line is fine if you want the original project.

This fork exists because fullstack repos need better AI-oriented orientation:

- stronger Vue / Nuxt / Nitro understanding
- better NestJS, routes, services, modules, resolvers, queues, cron jobs
- better data-layer extraction for Drizzle, Mongoose, MikroORM
- cleaner stdio-first MCP behavior
- better index / embedding UX
- better project topology and "where the hell does this feature live?" answers

Short version: less grep, less token waste, less psychic damage.

## What you get

- local-first code indexing with SQLite FTS5 and tree-sitter
- MCP server for Claude Code, Cursor, and other MCP clients
- keyword, semantic, and hybrid search
- symbol graph: callers, callees, dependents, impact, communities, execution flows
- fullstack-aware extraction for:
  - Vue SFCs
  - Nuxt / Nitro server routes and middleware
  - NestJS modules, controllers, services, resolvers, guards, pipes, filters, interceptors
  - Drizzle, Mongoose, MikroORM entities / schemas / tables / repositories
  - BullMQ, RabbitMQ, Redis, message patterns, async handlers
- workspace mode for multi-repo search
- local Ollama embeddings by default with `ollama:qwen3-embedding:4b`

## Install

### The easy way

```bash
curl -fsSL https://raw.githubusercontent.com/Matrix-aas/srclight/main/scripts/install.sh | bash
```

That installer:

- installs with `pipx`
- upgrades cleanly
- refuses to quietly sit on top of the old broken `0.15.x` pipx install
- tells you exactly what to delete if you've still got that fossil lying around

### Manual install

Recommended:

```bash
pipx install --force 'git+https://github.com/Matrix-aas/srclight.git@main'
```

From source:

```bash
git clone https://github.com/Matrix-aas/srclight.git
cd srclight
python3 -m pip install -e '.[dev]'
```

If you want docs / PDF / OCR extras:

```bash
python3 -m pip install -e '.[docs,pdf]'
python3 -m pip install -e '.[docs,pdf,ocr]'
python3 -m pip install -e '.[all]'
```

## Upgrade from the old pipx install

If you previously did this:

```bash
pipx install srclight
```

and got the old `0.15.x` line, uninstall that thing first:

```bash
pipx uninstall srclight
pipx install --force 'git+https://github.com/Matrix-aas/srclight.git@main'
```

Do not install the old PyPI build and expect this repo. That is the wrong timeline.

## Quick start

```bash
# index a repo
cd /path/to/project
srclight index

# index with embeddings (default: ollama:qwen3-embedding:4b)
srclight index --embed

# basic CLI search
srclight search "auth"
srclight symbols app/stores/auth.store.ts

# start MCP server for local agents
srclight serve --transport stdio
```

If you use a workspace:

```bash
srclight workspace init fullstack
srclight workspace add /path/to/repo-a -w fullstack
srclight workspace add /path/to/repo-b -w fullstack
srclight workspace index -w fullstack --embed
srclight serve --workspace fullstack --transport stdio
```

## Ollama embeddings

Recommended local model:

```bash
ollama pull qwen3-embedding:4b
srclight index --embed
```

`--embed` uses `ollama:qwen3-embedding:4b` by default in this fork.

If you want a multilingual alternative:

```bash
ollama pull nomic-embed-text-v2-moe
srclight index --embed ollama:nomic-embed-text-v2-moe
```

## MCP setup

### Claude Code

```bash
# single repo
claude mcp add srclight -- srclight serve --transport stdio

# workspace
claude mcp add srclight -- srclight serve --workspace fullstack --transport stdio
```

### Cursor

`stdio` config:

```json
{
  "mcpServers": {
    "srclight": {
      "command": "srclight",
      "args": ["serve", "--workspace", "fullstack", "--transport", "stdio"]
    }
  }
}
```

`SSE` is still available if you want a long-lived server, but this fork treats stdio as the default local-agent path.

## Why agents like it

Without srclight, AI agents waste time on:

- repeated `rg` passes just to find entrypoints
- reading random files to infer ownership
- missing callers and hidden async edges
- guessing how routes, modules, stores, queues, and DB pieces fit together

With srclight, they can ask for:

- `codebase_map()` for topology
- `search_symbols()` or `hybrid_search()` for targeted lookup
- `get_callers()` / `get_callees()` / `get_dependents()` for flow
- `detect_changes()` for blast radius
- `recent_changes()` / `git_hotspots()` for churn and risk

That saves tokens, saves time, and makes the toolchain feel less dumb.

## Current focus

This fork is intentionally opinionated toward modern fullstack work:

- TypeScript / JavaScript
- Vue / Nuxt / Nitro
- NestJS
- GraphQL + REST
- Drizzle / MikroORM / Mongoose
- BullMQ / RabbitMQ / Redis

It still supports the broader upstream language set, but the main upgrade energy goes into helping AI agents understand real web app repos end-to-end.

## Docs

- [Usage guide](docs/usage-guide.md)
- [Cursor MCP example](docs/cursor-mcp-example.json)
- [Releasing notes](docs/releasing.md)

## License

MIT.
