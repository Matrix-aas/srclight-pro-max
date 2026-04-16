"""Srclight MCP Server.

Exposes code indexing tools to AI agents via the Model Context Protocol.
Supports both single-repo mode and workspace mode (multi-repo via ATTACH+UNION).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shlex
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .db import Database, tokenized_query_hint
from .embeddings import DEFAULT_OLLAMA_EMBED_MODEL
from .indexer import IndexConfig, Indexer

logger = logging.getLogger("srclight.server")

_INSTRUCTIONS_TEMPLATE = """Welcome to Srclight — deep code indexing for AI agents.

{dynamic_section}## Default Routing
1. Start every new session with `codebase_map()` to get orientation.
2. In workspace mode, call `list_projects()` if you do not already know which repo matters.
3. Use `hybrid_search(query)` first for most code discovery.
4. Use `get_symbol(name)` or `symbols_in_file(path)` to drill into exact code once you have a hit.

## Search Decision Tree
- `hybrid_search(query)` — default choice. Best for feature flow, natural language, mixed keyword/concept searches, and new contexts.
- `search_symbols(query)` — prefer for exact names, code fragments, or when you want fast keyword-only results.
- `semantic_search(query)` — prefer only when terminology is unclear and embeddings are available.
- `get_symbol(name)` — prefer when you already know the symbol and want the full source + metadata.
- `get_signature(name)` — prefer when you only need the API shape, not the full body.

## Structure And Impact
- `symbols_in_file(path, project)` — table of contents for one file.
- `get_callers(symbol_name, project)` / `get_callees(symbol_name, project)` — local dependency tracing.
- `get_dependents(symbol_name, project)` / `get_impact(symbol_name, project)` — blast radius before changing code.
- `detect_changes(ref, project)` — map git changes to changed symbols and impact after edits.

## Repo And Server Health
- `index_status()` — check whether srclight is indexed and how fresh the index is.
- `embedding_status(project)` / `embedding_health(project)` — confirm embedding coverage and provider health.
- `reindex(path)` — refresh the index after meaningful file changes.
- `setup_guide()` — structured setup + recovery instructions for agents and users.
- `show_status(message)` / `server_stats()` — lightweight liveness and status checks.

## Workspace Rules
- Omit `project` to search across all projects.
- Pass `project` to narrow results to one repo.
- In workspace mode, graph + git + impact tools require `project`.
- If a tool says `project` is required, call `list_projects()` and retry with one of the returned names.

## AI Query Recipes
- Exact symbol: `search_symbols("AuthStore")`
- Feature flow: `hybrid_search("refresh token auth session apollo logout")`
- Vue/Nuxt frontend: `hybrid_search("locale path i18n css module template ref")`
- GraphQL frontend: `hybrid_search("query mutation subscription auth store gql")`
- Dependency question: `get_dependents("configureApolloClients", project="repo")`

## Recovery Hints
- If symbol lookup fails, try `hybrid_search(...)` first, then `search_symbols(...)`.
- If all tools fail with stale-parameter errors after a restart, tell the user to restart the editor/CLI so the MCP client reconnects.
- Prefer Srclight over grep/find/cat when indexed data is available: search first, then open symbols or file TOCs only as needed.
"""


def _build_dynamic_instructions() -> str:
    """Build the dynamic section of the server instructions from current workspace state."""
    lines = []
    try:
        if _is_workspace_mode():
            wdb = _get_workspace_db()
            projects = wdb.list_projects()
            project_count = len(projects)
            total_files = sum(p.get("files", 0) for p in projects)
            total_symbols = sum(p.get("symbols", 0) for p in projects)
            total_edges = sum(p.get("edges", 0) for p in projects)
            project_names = [p.get("project", p.get("name", "?")) for p in projects[:10]]

            lines.append(f"## Your Workspace: {_workspace_name}")
            lines.append(f"You have access to **{project_count} indexed project{'s' if project_count != 1 else ''}** "
                         f"containing **{total_files:,} files**, **{total_symbols:,} symbols**, "
                         f"and **{total_edges:,} relationships**.")
            if project_names:
                names_str = ", ".join(project_names)
                if project_count > 10:
                    names_str += f", ... and {project_count - 10} more"
                lines.append(f"Projects: {names_str}")
            lines.append("")
            lines.append("You can search across all projects at once, trace function calls, "
                         "find who changed code and why, and discover relationships between symbols.")
            lines.append("")
        elif _db_path is not None or _repo_root is not None:
            repo_root, db_path = _resolve_single_repo_context()
            if repo_root is not None and db_path is not None:
                if db_path.exists() and db_path.stat().st_size > 0:
                    db = _get_db()
                    stats = db.stats()
                    lines.append("## Your Codebase")
                    lines.append(f"You have access to **{stats['files']:,} files**, "
                                 f"**{stats['symbols']:,} symbols**, "
                                 f"and **{stats['edges']:,} relationships**.")
                    if stats.get("languages"):
                        lang_list = ", ".join(stats["languages"].keys())
                        lines.append(f"Languages: {lang_list}")
                    lines.append("")
                    lines.append("You can search code, trace function calls, "
                                 "find who changed code and why, and discover relationships between symbols.")
                    lines.append("")
                else:
                    representative_files = _find_representative_files(repo_root)
                    framework_hints = _detect_framework_hints(repo_root, representative_files)
                    start_here = _build_start_here(representative_files, framework_hints)
                    lines.append("## Your Codebase")
                    lines.append(f"Repo root: {repo_root}")
                    lines.append("Index status: not indexed yet.")
                    if start_here:
                        start_paths = ", ".join(item["path"] for item in start_here[:4])
                        lines.append(f"Start here: {start_paths}")
                    lines.append("")
                    lines.append("Call `codebase_map()` for a filesystem-only brief or run `srclight index --embed`.")
                    lines.append("")
    except Exception:
        # If we can't get stats (e.g. DB not yet initialized), fall back to generic text
        lines.append("You have access to a code index with searchable symbols, call graphs, and git history.")
        lines.append("")

    return "\n".join(lines)


def _refresh_instructions() -> None:
    """Update the MCP server instructions with current workspace state."""
    try:
        dynamic = _build_dynamic_instructions()
        mcp._mcp_server.instructions = _INSTRUCTIONS_TEMPLATE.format(dynamic_section=dynamic)
    except Exception:
        pass  # Keep existing instructions on error


mcp = FastMCP(
    "srclight",
    instructions=_INSTRUCTIONS_TEMPLATE.format(
        dynamic_section="You have access to a code index with searchable symbols, call graphs, and git history.\n\n"
    ),
)

# Global state — initialized on first tool call or via configure()
_db: Database | None = None
_db_path: Path | None = None
_repo_root: Path | None = None

# Workspace mode state
_workspace_name: str | None = None
_workspace_db = None  # WorkspaceDB instance (lazy import to avoid circular)
_workspace_config_mtime: float = 0.0  # mtime of workspace config at last load

# Vector cache (GPU-resident embedding matrix)
_vector_cache = None  # VectorCache instance (single-repo mode)

# Learnings DB (workspace-level, lazy)
_learnings_db = None  # LearningsDB instance


def _is_workspace_mode() -> bool:
    return _workspace_name is not None


def _read_index_signal(root: Path | None) -> dict | None:
    """Read the last-indexed signal file for a project root."""
    if root is None:
        return None
    signal_file = root / ".srclight" / "last-indexed"
    try:
        if signal_file.exists():
            return json.loads(signal_file.read_text())
    except Exception:
        pass
    return None


def _discover_repo_root_and_db_path(start: Path | None = None) -> tuple[Path, Path]:
    """Discover the repo root and expected DB path without opening SQLite."""
    check = (start or Path.cwd()).resolve()
    while check != check.parent:
        if (check / ".srclight" / "index.db").exists():
            return check, check / ".srclight" / "index.db"
        if (check / ".codelight" / "index.db").exists():
            return check, check / ".codelight" / "index.db"
        if (check / ".srclight.db").exists():
            return check, check / ".srclight.db"
        if (check / ".git").exists():
            return check, check / ".srclight" / "index.db"
        check = check.parent
    root = (start or Path.cwd()).resolve()
    return root, root / ".srclight" / "index.db"


def _resolve_single_repo_context() -> tuple[Path | None, Path | None]:
    """Resolve repo root and DB path for single-repo mode."""
    global _db_path, _repo_root

    if _repo_root is not None:
        root = _repo_root.resolve()
        db_path = _db_path or root / ".srclight" / "index.db"
        _db_path = db_path
        return root, db_path

    if _db_path is not None:
        db_path = _db_path.resolve()
        root = db_path.parent.parent if db_path.name == "index.db" else db_path.parent
        _repo_root = root
        return root, db_path

    root, db_path = _discover_repo_root_and_db_path()
    _repo_root = root
    _db_path = db_path
    return root, db_path


def _load_package_manifest(root: Path) -> dict:
    """Best-effort package.json loader for repo fingerprinting."""
    manifest_path = root / "package.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def _package_dependency_names(manifest: dict) -> set[str]:
    """Collect dependency names across common package.json sections."""
    names: set[str] = set()
    for field in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        values = manifest.get(field)
        if isinstance(values, dict):
            names.update(values.keys())
    return names


def _collect_repo_files(
    root: Path,
    relative_dirs: tuple[str, ...],
    *,
    limit: int,
    allowed_suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Collect a few representative files under the given directories."""
    results: list[str] = []
    seen: set[str] = set()
    suffixes = tuple(s.lower() for s in allowed_suffixes) if allowed_suffixes else None

    for relative_dir in relative_dirs:
        directory = root / relative_dir
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative in seen:
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue
            seen.add(relative)
            results.append(relative)
            if len(results) >= limit:
                return results
    return results


def _collect_existing_paths(root: Path, candidates: tuple[str, ...], *, limit: int) -> list[str]:
    """Return the existing candidate paths relative to root."""
    results: list[str] = []
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            results.append(candidate)
        if len(results) >= limit:
            break
    return results


def _merge_representative_paths(*groups: list[str], limit: int) -> list[str]:
    """Merge path groups while preserving order and removing duplicates."""
    results: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            if path in seen:
                continue
            seen.add(path)
            results.append(path)
            if len(results) >= limit:
                return results
    return results


_DATA_SYSTEMS = ("prisma", "drizzle", "mongoose", "mikroorm")
_ASYNC_SYSTEMS = ("bullmq", "rabbitmq", "redis")
_ASYNC_SYSTEM_ALIASES = {"rmq": "rabbitmq"}
_ASYNC_RESOURCES = {"processor", "consumer", "worker", "queue", "scheduler"}


def _normalize_orientation_token(value: object) -> str:
    """Normalize framework-ish metadata values into stable tokens."""
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return "nest" if token == "nestjs" else token


def _canonical_async_orientation_token(value: object) -> str:
    """Normalize async metadata tokens and collapse transport aliases."""
    token = _normalize_orientation_token(value)
    return _ASYNC_SYSTEM_ALIASES.get(token, token)


def _infer_app_type(
    representative_files: dict[str, list[str]],
    signals: set[str],
    *,
    manifest_present: bool,
) -> str:
    """Infer the primary app type from representative files and merged signals."""
    has_frontend = "nuxt" in signals or "vue" in signals
    has_backend = (
        "nest" in signals
        or bool(representative_files.get("data"))
        or bool(representative_files.get("async"))
        or any(
            path.startswith(("src/main", "src/controllers", "src/modules", "src/http"))
            for path in representative_files.get("backend", [])
        )
    )

    if has_frontend and has_backend:
        return "fullstack"
    if "nuxt" in signals:
        return "nuxt"
    if "nest" in signals:
        return "nest"
    if "vue" in signals:
        return "vue"
    if manifest_present:
        return "node"
    return "codebase"


def _indexed_orientation_hints(symbol_rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize indexed symbol metadata into orientation hints."""
    representative_files = {
        "backend": [],
        "data": [],
        "async": [],
        "config": [],
    }
    route_systems: list[str] = []
    route_files: list[str] = []
    data_systems: list[str] = []
    async_systems: list[str] = []
    runtime_files: list[str] = []
    signals: list[str] = []
    backend_priorities: dict[str, int] = {}
    async_priorities: dict[str, int] = {}
    runtime_priorities: dict[str, int] = {}

    def _append(bucket: list[str], value: str) -> None:
        if value and value not in bucket:
            bucket.append(value)

    def _ordered_systems(values: list[str], preferred: tuple[str, ...]) -> list[str]:
        order = {name: index for index, name in enumerate(preferred)}
        return sorted(values, key=lambda name: (order.get(name, len(order)), name))

    def _priority_update(priorities: dict[str, int], path: str, priority: int) -> None:
        if not path:
            return
        priorities[path] = min(priority, priorities.get(path, priority))

    def _ordered_paths(values: list[str], priorities: dict[str, int]) -> list[str]:
        return sorted(values, key=lambda path: (priorities.get(path, 10), path))

    for row in symbol_rows:
        path = str(row.get("file_path") or "")
        kind = str(row.get("kind") or "").lower()
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        framework = _canonical_async_orientation_token(metadata.get("framework"))
        has_framework_metadata = bool(framework)
        transport = _canonical_async_orientation_token(metadata.get("transport"))
        resource = str(metadata.get("resource") or "").lower()
        has_route_metadata = any(metadata.get(key) for key in ("route_prefix", "route_path", "http_method"))
        system_tokens = [
            token
            for token in (framework, transport)
            if token
        ]

        is_route_surface = (
            kind in {"controller", "route"}
            or resource in {"controller", "route"}
            or has_route_metadata
        )
        is_nest_controller = (
            kind == "controller"
            or resource == "controller"
            or path.startswith("src/controllers/")
            or path.endswith(".controller.ts")
        ) and (
            framework == "nest"
            or (
                not has_framework_metadata
                and (
                    path.startswith("src/controllers/")
                    or path.endswith(".controller.ts")
                )
            )
        )
        is_runtime = (
            resource in {"module", "config", "bootstrap"}
            or kind in {"module", "config"}
            or "/bootstrap/" in path
        )
        is_data = framework in _DATA_SYSTEMS
        is_async = (
            kind in {"queue_processor", "microservice_handler", "scheduled_job"}
            or resource in _ASYNC_RESOURCES
            or any(token in _ASYNC_SYSTEMS for token in system_tokens)
        )

        if path in {"src/main.ts", "src/main.js", "main.ts", "main.js", "server.ts"} or resource == "bootstrap":
            _append(representative_files["backend"], path)
            _priority_update(backend_priorities, path, 0)

        if is_route_surface:
            _append(representative_files["backend"], path)
            _append(route_files, path)
            _priority_update(backend_priorities, path, 1)
            if is_nest_controller and "nest_controllers" not in route_systems:
                route_systems.append("nest_controllers")
            if framework == "nest":
                _append(signals, framework)

        if is_runtime:
            _append(representative_files["config"], path)
            _append(runtime_files, path)
            _priority_update(runtime_priorities, path, 0 if resource == "module" else 1)
            if framework == "nest":
                _append(signals, framework)

        if is_data and path:
            _append(representative_files["data"], path)
            _append(data_systems, framework)
            _append(signals, framework)

        if is_async and path:
            _append(representative_files["async"], path)
            _priority_update(async_priorities, path, 1 if resource == "config" else 0)
            for token in system_tokens:
                if token in _ASYNC_SYSTEMS:
                    _append(async_systems, token)
                    _append(signals, token)
            if framework == "nest":
                _append(signals, framework)
            if resource == "config":
                _append(representative_files["config"], path)
                _append(runtime_files, path)

    return {
        "signals": signals,
        "representative_files": {
            "backend": _ordered_paths(representative_files["backend"], backend_priorities),
            "data": representative_files["data"],
            "async": _ordered_paths(representative_files["async"], async_priorities),
            "config": _ordered_paths(representative_files["config"], runtime_priorities),
        },
        "route_systems": route_systems,
        "route_files": route_files,
        "data_systems": _ordered_systems(data_systems, _DATA_SYSTEMS),
        "async_systems": _ordered_systems(async_systems, _ASYNC_SYSTEMS),
        "runtime_files": _ordered_paths(runtime_files, runtime_priorities),
    }


def _merge_indexed_representative_files(
    representative_files: dict[str, list[str]],
    indexed_hints: dict[str, object] | None,
) -> dict[str, list[str]]:
    """Merge metadata-derived file hints into representative files."""
    if not indexed_hints:
        return {key: list(value) for key, value in representative_files.items()}

    merged = {key: list(value) for key, value in representative_files.items()}
    indexed_files = indexed_hints.get("representative_files")
    if not isinstance(indexed_files, dict):
        return merged

    merged["backend"] = _merge_representative_paths(
        merged.get("backend", []),
        list(indexed_files.get("backend") or []),
        limit=3,
    )
    merged["data"] = _merge_representative_paths(
        merged.get("data", []),
        list(indexed_files.get("data") or []),
        limit=3,
    )
    merged["async"] = _merge_representative_paths(
        merged.get("async", []),
        list(indexed_files.get("async") or []),
        limit=3,
    )
    merged["config"] = _merge_representative_paths(
        merged.get("config", []),
        list(indexed_files.get("config") or []),
        limit=4,
    )
    return merged


def _find_representative_files(root: Path) -> dict[str, list[str]]:
    """Collect high-signal files that help agents orient quickly."""
    representative_files: dict[str, list[str]] = {}

    config_files = _collect_existing_paths(root, (
        "nuxt.config.ts",
        "nuxt.config.js",
        "nuxt.config.mjs",
        "vite.config.ts",
        "vite.config.js",
        "nest-cli.json",
        "tsconfig.json",
        "package.json",
    ), limit=4)
    if config_files:
        representative_files["config"] = config_files

    entrypoints = _collect_existing_paths(root, (
        "app.vue",
        "src/main.ts",
        "src/main.js",
        "main.ts",
        "main.js",
        "server.ts",
        "src/main.tsx",
    ), limit=3)
    if entrypoints:
        representative_files["entrypoints"] = entrypoints

    docs = _collect_existing_paths(root, (
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/README.md",
    ), limit=4)
    if docs:
        representative_files["docs"] = docs

    backend = _merge_representative_paths(
        _collect_repo_files(root, ("server/api", "server/routes", "api"), limit=1, allowed_suffixes=(".ts", ".js", ".mjs")),
        _collect_existing_paths(root, ("src/main.ts", "src/main.js", "server.ts", "main.ts", "main.js"), limit=1),
        _collect_repo_files(root, ("src/controllers", "src/modules", "src/server"), limit=1, allowed_suffixes=(".ts", ".js")),
        limit=3,
    )
    if backend:
        representative_files["backend"] = backend

    data = _merge_representative_paths(
        _collect_existing_paths(root, ("prisma/schema.prisma",), limit=1),
        _collect_repo_files(
            root,
            ("prisma", "src/db", "db", "src/database", "database", "src/entities", "src/models", "src/repositories"),
            limit=3,
            allowed_suffixes=(".prisma", ".sql", ".ts", ".js"),
        ),
        limit=3,
    )
    if data:
        representative_files["data"] = data

    async_files = _collect_repo_files(
        root,
        ("src/queues", "queues", "src/workers", "workers", "src/jobs", "jobs", "src/events", "events", "src/consumers", "consumers"),
        limit=3,
        allowed_suffixes=(".ts", ".js", ".mjs"),
    )
    if async_files:
        representative_files["async"] = async_files

    category_specs = (
        ("routes", ("app/pages", "pages", "src/pages", "app/router", "src/router"), (".vue", ".ts", ".js"), 3),
        ("components", ("app/components", "components", "src/components"), (".vue", ".ts", ".js"), 3),
        ("composables", ("app/composables", "composables", "src/composables", "hooks", "src/hooks"), (".ts", ".js", ".vue"), 3),
        ("stores", ("app/stores", "stores", "src/stores"), (".ts", ".js"), 3),
        ("plugins", ("app/plugins", "plugins", "src/plugins"), (".ts", ".js"), 3),
        ("server", ("server", "server/api", "api", "src/server"), (".ts", ".js", ".mjs"), 3),
        ("graphql", ("app/graphql", "graphql", "src/graphql"), (".gql", ".graphql", ".ts", ".js"), 3),
        ("styles", ("app/assets", "assets", "styles", "src/styles"), (".css", ".pcss", ".postcss", ".scss", ".sass", ".less"), 3),
    )
    for category, relative_dirs, suffixes, limit in category_specs:
        files = _collect_repo_files(root, relative_dirs, limit=limit, allowed_suffixes=suffixes)
        if files:
            representative_files[category] = files

    return representative_files


def _detect_framework_hints(
    root: Path,
    representative_files: dict[str, list[str]],
    *,
    extra_signals: set[str] | None = None,
) -> dict[str, object]:
    """Infer the dominant framework/runtime signals from repo layout + package.json."""
    manifest = _load_package_manifest(root)
    dependencies = _package_dependency_names(manifest)
    signals: set[str] = set(extra_signals or [])

    if "nuxt" in dependencies or any(path.startswith("nuxt.config.") for path in representative_files.get("config", [])):
        signals.add("nuxt")
    if (
        "vue" in dependencies
        or "nuxt" in signals
        or bool(representative_files.get("components"))
        or bool(representative_files.get("routes"))
        or "app.vue" in representative_files.get("entrypoints", [])
    ):
        signals.add("vue")
    if (
        "graphql" in dependencies
        or "@apollo/client" in dependencies
        or "@nuxtjs/apollo" in dependencies
        or bool(representative_files.get("graphql"))
    ):
        signals.add("graphql")
    if (
        "@pinia/nuxt" in dependencies
        or "pinia" in dependencies
        or bool(representative_files.get("stores"))
    ):
        signals.add("pinia")
    if (
        "postcss" in dependencies
        or any(path.endswith(".postcss") for path in representative_files.get("styles", []))
    ):
        signals.add("postcss")
    if (
        "@nestjs/core" in dependencies
        or "nest-cli.json" in representative_files.get("config", [])
    ):
        signals.add("nest")
    if representative_files.get("server") and "nuxt" in signals:
        signals.add("nitro")
    if (
        "@prisma/client" in dependencies
        or "prisma" in dependencies
        or any(path.endswith(".prisma") for path in representative_files.get("data", []))
    ):
        signals.add("prisma")
    if "bullmq" in dependencies or any("queue" in path or "processor" in path for path in representative_files.get("async", [])):
        signals.add("bullmq")

    return {
        "app_type": _infer_app_type(representative_files, signals, manifest_present=bool(manifest)),
        "signals": sorted(signals),
    }


def _build_topology(
    root: Path,
    representative_files: dict[str, list[str]],
    framework_hints: dict[str, object],
    indexed_hints: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Summarize the main architectural surfaces visible from repo layout."""
    manifest = _load_package_manifest(root)
    dependencies = _package_dependency_names(manifest)
    signals = set(framework_hints.get("signals") or [])
    indexed_hints = indexed_hints or {}
    topology: dict[str, dict[str, object]] = {}

    backend_files = representative_files.get("backend") or representative_files.get("server") or []
    if backend_files:
        topology["backend"] = {
            "files": backend_files,
            "summary": "Primary backend entrypoints, HTTP surfaces, and server modules.",
        }

    route_systems = list(indexed_hints.get("route_systems") or [])
    has_controller_surfaces = any(path.startswith("src/controllers") for path in backend_files)
    if has_controller_surfaces and "nest_controllers" not in route_systems:
        route_systems.append("nest_controllers")
    if any(path.startswith(("server/api", "server/routes")) for path in backend_files + (representative_files.get("server") or [])) and "nitro_file_routes" not in route_systems:
        route_systems.append("nitro_file_routes")
    if route_systems:
        route_files = _merge_representative_paths(
            [path for path in representative_files.get("server") or [] if path.startswith(("server/api", "server/routes"))],
            list(indexed_hints.get("route_files") or []),
            [path for path in backend_files if path.startswith("src/controllers")],
            [path for path in backend_files if path.startswith(("server/api", "server/routes"))],
            limit=3,
        )
        topology["routes"] = {
            "systems": route_systems,
            "files": route_files,
            "summary": _route_summary(route_systems),
        }

    data_files = representative_files.get("data") or []
    if data_files:
        data_systems = list(indexed_hints.get("data_systems") or [])
        if not data_systems:
            data_systems = [system for system in _DATA_SYSTEMS if system in signals]
        if not data_systems:
            data_systems = ["prisma"] if any(path.endswith(".prisma") for path in data_files) else ["generic"]
        topology["data"] = {
            "systems": data_systems,
            "files": data_files,
        }

    async_files = representative_files.get("async") or []
    if async_files:
        async_systems = list(indexed_hints.get("async_systems") or [])
        if not async_systems:
            async_systems = [system for system in _ASYNC_SYSTEMS if system in signals]
        if not async_systems:
            async_systems = ["bullmq"] if "bullmq" in dependencies else ["generic"]
        topology["async"] = {
            "systems": async_systems,
            "files": async_files,
        }

    runtime_files = _merge_representative_paths(
        [path for path in representative_files.get("config", []) if path in {"nuxt.config.ts", "nuxt.config.js", "nuxt.config.mjs", "package.json", "nest-cli.json"}],
        list(indexed_hints.get("runtime_files") or []),
        _collect_repo_files(root, ("src/config", "config"), limit=2, allowed_suffixes=(".ts", ".js", ".json", ".yaml", ".yml")),
        limit=3,
    )
    if runtime_files:
        topology["runtime"] = {
            "files": runtime_files,
            "summary": "Runtime configuration, environment wiring, and framework bootstrap settings.",
        }

    return topology


def _route_summary(route_systems: list[str]) -> str:
    """Summarize route systems without implying surfaces we did not detect."""
    labels = {
        "nest_controllers": "Nest controllers",
        "nitro_file_routes": "Nitro file routes",
    }
    parts = [labels[system] for system in route_systems if system in labels]
    if not parts:
        return "HTTP route and transport surfaces."
    if len(parts) == 1:
        return f"HTTP route and transport surfaces from {parts[0]}."
    return f"HTTP route and transport surfaces from {parts[0]} and {parts[1]}."


def _start_here_reason(category: str, app_type: str) -> str:
    """Explain why a representative file is a good orientation entrypoint."""
    reasons = {
        "config": {
            "nuxt": "Nuxt runtime and module configuration.",
            "fullstack": "Framework and runtime configuration across frontend and backend.",
            "default": "Framework and build configuration.",
        },
        "entrypoints": {
            "nuxt": "Global Nuxt app shell.",
            "default": "Application entrypoint.",
        },
        "routes": {
            "default": "Top-level route or page surface.",
        },
        "components": {
            "default": "Reusable UI building blocks.",
        },
        "composables": {
            "default": "Shared frontend behavior and stateful logic.",
        },
        "stores": {
            "default": "Centralized client state.",
        },
        "plugins": {
            "default": "Framework plugin/bootstrap integration.",
        },
        "server": {
            "nuxt": "Nitro server endpoints and server-only logic.",
            "fullstack": "Server-only endpoints and transport surfaces.",
            "default": "HTTP or backend entrypoints.",
        },
        "backend": {
            "fullstack": "Backend bootstrap, controllers, and server entrypoints.",
            "nest": "Backend bootstrap, controllers, and server entrypoints.",
            "default": "Backend entrypoints and transport surfaces.",
        },
        "data": {
            "default": "Persistence layer and database schema entrypoints.",
        },
        "async": {
            "default": "Queue, worker, and async processing entrypoints.",
        },
        "graphql": {
            "default": "GraphQL operations or schema-adjacent documents.",
        },
        "styles": {
            "default": "Shared styling, tokens, or PostCSS modules.",
        },
    }
    category_reasons = reasons.get(category, {})
    return category_reasons.get(app_type) or category_reasons.get("default") or "Representative file."


def _build_start_here(
    representative_files: dict[str, list[str]],
    framework_hints: dict[str, object],
) -> list[dict[str, str]]:
    """Build a small ordered entrypoint list for repo orientation."""
    app_type = str(framework_hints.get("app_type") or "codebase")
    if app_type == "fullstack":
        priority = ("config", "backend", "routes", "data", "async", "entrypoints", "server", "stores", "composables", "plugins", "graphql", "styles", "components")
        category_limits = {"backend": 2}
    elif app_type == "nuxt":
        priority = ("config", "entrypoints", "routes", "stores", "composables", "plugins", "server", "graphql", "styles", "components")
        category_limits = {}
    elif app_type == "nest":
        priority = ("config", "backend", "data", "async", "entrypoints", "server", "graphql", "styles", "components")
        category_limits = {"config": 2, "backend": 2}
    else:
        priority = ("config", "entrypoints", "routes", "components", "composables", "stores", "plugins", "server", "graphql", "styles")
        category_limits = {}

    start_here: list[dict[str, str]] = []
    seen: set[str] = set()
    for category in priority:
        files = representative_files.get(category) or []
        if not files:
            continue
        for path in files[:category_limits.get(category, 1)]:
            if path in seen:
                continue
            seen.add(path)
            start_here.append({
                "path": path,
                "reason": _start_here_reason(category, app_type),
            })
            if len(start_here) >= 6:
                break
        if len(start_here) >= 6:
            break

    if not start_here:
        for path in representative_files.get("config", []):
            start_here.append({
                "path": path,
                "reason": "Top-level project manifest or framework configuration.",
            })
            break

    if not start_here:
        for path in representative_files.get("docs", []):
            start_here.append({
                "path": path,
                "reason": "Top-level project documentation.",
            })
            break

    if not start_here:
        start_here.append({
            "path": ".",
            "reason": "Repo root. Start with top-level files and directories.",
        })
    return start_here


def _build_repo_brief(
    framework_hints: dict[str, object],
    start_here: list[dict[str, str]],
    *,
    indexed: bool,
) -> str:
    """Build a compact AI-oriented repo summary."""
    app_type = str(framework_hints.get("app_type") or "codebase")
    signals = list(framework_hints.get("signals") or [])
    label = "unindexed" if not indexed else "indexed"
    app_label = app_type.upper() if app_type == "nuxt" else app_type
    signal_text = f" with {', '.join(signals[:4])} signals" if signals else ""
    start_paths = ", ".join(item["path"] for item in start_here[:4])
    start_text = f" Start with {start_paths}." if start_paths else ""
    return f"{label.capitalize()} {app_label} repo{signal_text}.{start_text}"


def _bootstrap_codebase_map_result(
    repo_root: Path,
    db_path: Path,
    framework_hints: dict[str, object],
    representative_files: dict[str, list[str]],
    start_here: list[dict[str, str]],
    topology: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build a filesystem-only codebase map for repos without a usable index."""
    quoted_root = shlex.quote(str(repo_root))
    return {
        "mode": "single",
        "repo_root": str(repo_root),
        "bootstrap_mode": "filesystem_only",
        "index": {
            "status": "not_indexed",
            "db_path": str(db_path),
            "present": db_path.exists(),
        },
        "framework_hints": framework_hints,
        "representative_files": representative_files,
        "topology": topology,
        "start_here": start_here,
        "brief": _build_repo_brief(framework_hints, start_here, indexed=False),
        "hint": (
            "Repo is not indexed yet. "
            f"Run `srclight index --embed` from {quoted_root} "
            f"or `srclight index {quoted_root} --embed`."
        ),
        "next_actions": [
            f"Run: srclight index {quoted_root} --embed",
            "Call codebase_map() again after indexing.",
            "Use setup_guide() if the user needs setup help.",
        ],
    }


def _get_workspace_db():
    """Get or create the WorkspaceDB connection.

    Hot-reloads if the workspace config file has been modified (e.g. after
    `srclight workspace add`). This means you never need to restart the
    MCP server to pick up new repos.
    """
    global _workspace_db, _workspace_config_mtime

    from .workspace import WorkspaceConfig, WorkspaceDB

    config_path = WorkspaceConfig(name=_workspace_name).config_path

    # Check if config has changed since last load
    try:
        current_mtime = config_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    if _workspace_db is not None and current_mtime == _workspace_config_mtime:
        return _workspace_db

    # Config changed (or first load) — (re)create workspace connection
    if _workspace_db is not None:
        logger.info("Workspace config changed, reloading...")
        try:
            _workspace_db.close()
        except Exception:
            pass
        _workspace_db = None

    config = WorkspaceConfig.load(_workspace_name)
    _workspace_db = WorkspaceDB(config)
    _workspace_db.open()
    _workspace_config_mtime = current_mtime
    return _workspace_db


def _get_learnings_db():
    """Get or create the workspace-level LearningsDB."""
    global _learnings_db

    if _learnings_db is not None:
        return _learnings_db

    from .learnings import LearningsDB
    from .workspace import WorkspaceConfig

    if _workspace_name is None:
        raise RuntimeError("Learnings require workspace mode")

    config = WorkspaceConfig.load(_workspace_name)
    _learnings_db = LearningsDB(config.learnings_db_path)
    _learnings_db.open()
    _learnings_db.initialize()
    return _learnings_db


def _get_db() -> Database:
    """Get or create the database connection (single-repo mode)."""
    global _db, _db_path, _repo_root

    if _db is not None:
        return _db

    # Default: look for .srclight/index.db, walk up to find repo root.
    # Legacy: migrate .codelight/ → .srclight/ if found on disk.
    root, db_path = _resolve_single_repo_context()
    if root is not None:
        legacy = root / ".codelight"
        new_dir = root / ".srclight"
        if db_path == legacy / "index.db" and legacy.exists() and not new_dir.exists():
            try:
                legacy.rename(new_dir)
                logger.info("Migrated %s -> %s", legacy, new_dir)
                db_path = new_dir / "index.db"
            except OSError:
                pass

    _repo_root = root
    _db_path = db_path

    _db = Database(_db_path)
    _db.open()

    # Initialize schema if this is a new database
    if not _db_path.exists() or _db_path.stat().st_size == 0:
        _db.initialize()

    return _db


def _get_vector_cache():
    """Get or create the VectorCache (single-repo mode)."""
    global _vector_cache
    if _vector_cache is not None:
        return _vector_cache

    from .vector_cache import VectorCache

    _get_db()
    srclight_dir = _db_path.parent if _db_path else None
    if srclight_dir is None:
        return None

    cache = VectorCache(srclight_dir)
    if cache.sidecar_exists():
        try:
            cache.load_sidecar()
        except Exception as e:
            logger.warning("Failed to load vector cache sidecar: %s", e)
            return None
    _vector_cache = cache
    return _vector_cache


def _symbol_to_dict(sym) -> dict:
    """Convert a SymbolRecord to a clean dict for MCP response."""
    return {
        "id": sym.id,
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "kind": sym.kind,
        "signature": sym.signature,
        "file": sym.file_path,
        "start_line": sym.start_line,
        "end_line": sym.end_line,
        "line_count": sym.line_count,
        "doc_comment": sym.doc_comment,
        "visibility": sym.visibility,
    }


def _project_required_error(tool_name: str) -> str:
    """Return a JSON error with the list of valid project names."""
    wdb = _get_workspace_db()
    project_names = sorted(e.name for e in wdb._all_indexable)
    return json.dumps({
        "error": f"In workspace mode, 'project' parameter is required for {tool_name}.",
        "available_projects": project_names,
        "hint": (
            f"Call list_projects() and retry with {tool_name}(..., project=\"{project_names[0]}\")"
            if project_names else
            "Call list_projects() to discover valid project names."
        ),
    })


def _symbol_name_suggestions(name: str, project: str | None = None) -> list[str]:
    """Return nearby symbol names for miss hints."""
    try:
        if _is_workspace_mode():
            wdb = _get_workspace_db()
            suggestions = wdb.suggest_symbol_names(name, project=project)
            return [item["name"] for item in suggestions]

        db = _get_db()
        return db.suggest_symbol_names(name)
    except Exception:
        return []


def _tool_call(tool_name: str, value: str, *, project: str | None = None) -> str:
    """Format a recovery suggestion tool call."""
    project_part = f', project="{project}"' if project else ""
    return f'{tool_name}("{value}"{project_part})'


def _rank_source(result: dict[str, object]) -> str:
    """Collapse internal search source metadata into agent-friendly buckets."""
    sources = {str(source) for source in result.get("sources", []) if source}
    if "fts" in sources and "embedding" in sources:
        return "hybrid"
    if "embedding" in sources:
        return "semantic"

    if result.get("similarity") is not None and not result.get("source"):
        return "semantic"
    return "keyword"


def _match_reasons(result: dict[str, object]) -> list[str]:
    """Translate internal ranking signals into terse retrieval reasons."""
    reasons: list[str] = []
    source = result.get("source")
    if source in {"name", "name_like"}:
        reasons.append("symbol name match")
    elif source == "tokenized_like":
        reasons.append("tokenized identifier match")
    elif source == "metadata_like":
        reasons.append("metadata match")
    elif source == "content":
        reasons.append("symbol content match")
    elif source == "docs":
        reasons.append("documentation match")

    sources = {str(item) for item in result.get("sources", []) if item}
    if "fts" in sources and not reasons:
        reasons.append("keyword match")
    if "embedding" in sources or _rank_source(result) == "semantic":
        reasons.append("semantic similarity")

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        deduped.append(reason)
        seen.add(reason)
    return deduped


def _shape_search_result(result: dict[str, object]) -> dict[str, object]:
    """Hide low-signal internals and expose concise ranking hints."""
    shaped = {
        key: value
        for key, value in result.items()
        if key != "rrf_score"
    }
    shaped["rank_source"] = _rank_source(result)
    reasons = _match_reasons(result)
    if reasons:
        shaped["match_reasons"] = reasons
    return shaped


def _format_community_payload(
    db: Database,
    symbol_name: str,
    sym,
    *,
    project: str | None = None,
    resolved_via: str = "exact_symbol",
) -> dict[str, object]:
    """Build the normal get_community payload for an exact or inferred symbol hit."""
    comm_id = db.get_community_for_symbol(sym.id)
    if comm_id is None:
        payload: dict[str, object] = {
            "symbol": symbol_name,
            "community": None,
            "info": "Symbol not assigned to any community",
        }
        if resolved_via != "exact_symbol":
            payload["resolved_via"] = resolved_via
            payload["matched_symbol"] = sym.name
        if project is not None:
            payload["project"] = project
        return payload

    members = db.get_community_members(comm_id)
    communities = db.get_community_records(limit=None)
    comm_info = next((c for c in communities if c["id"] == comm_id), None)

    payload = {
        "symbol": symbol_name,
        "community_id": comm_id,
        "label": comm_info["label"] if comm_info else "unknown",
        "keywords": comm_info.get("keywords", []) if comm_info else [],
        "cohesion": comm_info["cohesion"] if comm_info else None,
        "member_count": len(members),
        "members": members,
    }
    if resolved_via != "exact_symbol":
        payload["resolved_via"] = resolved_via
        payload["matched_symbol"] = sym.name
    if project is not None:
        payload["project"] = project
    return payload


def _community_fallback_payload(
    db: Database,
    symbol_name: str,
    *,
    project: str | None = None,
) -> dict[str, object]:
    """Build a miss payload that escalates from nearest symbol to file candidates."""
    nearest_matches = db.suggest_symbol_name_matches(symbol_name, limit=1)
    if nearest_matches:
        nearest_name = nearest_matches[0]["name"]
        nearest_sym = db.get_symbol_by_name(nearest_name)
        if nearest_sym is not None:
            nearest_symbol = {
                "name": nearest_sym.name,
                "kind": nearest_sym.kind,
                "file": nearest_sym.file_path,
            }
            comm_id = db.get_community_for_symbol(nearest_sym.id)
            if comm_id is not None:
                payload = _format_community_payload(
                    db,
                    symbol_name,
                    nearest_sym,
                    project=project,
                    resolved_via="nearest_symbol",
                )
                payload["next_step"] = {
                    "tool": "get_symbol",
                    "call": _tool_call("get_symbol", nearest_sym.name, project=project),
                }
                return payload
            payload = {
                "symbol": symbol_name,
                "community": None,
                "fallback_stage": "nearest_symbol",
                "nearest_symbol": nearest_symbol,
                "next_step": {
                    "tool": "get_symbol",
                    "call": _tool_call("get_symbol", nearest_sym.name, project=project),
                },
            }
            if project is not None:
                payload["project"] = project
            return payload

    file_candidates = db.suggest_file_candidates(symbol_name, limit=5)
    payload: dict[str, object] = {
        "symbol": symbol_name,
        "community": None,
    }
    if project is not None:
        payload["project"] = project

    if file_candidates:
        if project is not None:
            file_candidates = [{**candidate, "project": project} for candidate in file_candidates]
        payload["fallback_stage"] = "file_candidate"
        payload["file_candidates"] = file_candidates
        payload["next_step"] = {
            "tool": "symbols_in_file",
            "call": _tool_call("symbols_in_file", str(file_candidates[0]["path"]), project=project),
        }
        return payload

    payload["fallback_stage"] = "suggested_tool"
    payload["next_step"] = {
        "tool": "hybrid_search",
        "call": _tool_call("hybrid_search", symbol_name, project=project),
    }
    did_you_mean = _symbol_name_suggestions(symbol_name, project=project)
    if did_you_mean:
        payload["did_you_mean"] = did_you_mean
    return payload


def _symbol_not_found_error(name: str, project: str | None = None) -> str:
    """Return a JSON error with recovery hints when a symbol lookup fails."""
    ctx = f" in {project}" if project else ""
    payload: dict[str, object] = {
        "error": f"Symbol '{name}' not found{ctx}",
        "suggestions": [
            f"Try {_tool_call('hybrid_search', name, project=project)} for concept + keyword search",
            f"Try {_tool_call('search_symbols', name, project=project)} for exact/fuzzy keyword search",
        ],
    }

    tokenized = tokenized_query_hint(name)
    if tokenized:
        payload["suggestions"].append(
            f"Try {_tool_call('search_symbols', tokenized, project=project)} for tokenized identifier search"
        )

    did_you_mean = _symbol_name_suggestions(name, project=project)
    if did_you_mean:
        payload["did_you_mean"] = did_you_mean

    return json.dumps(payload)


def _project_not_found_error(project: str) -> str:
    """Return a JSON error with fuzzy 'did you mean' suggestions for project names."""
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        project_names = sorted(e.name for e in wdb._all_indexable)
    else:
        project_names = []
    result: dict[str, object] = {"error": f"Project '{project}' not found"}
    if project_names:
        close = difflib.get_close_matches(project, project_names, n=3, cutoff=0.4)
        if close:
            result["did_you_mean"] = close
        result["available_projects"] = project_names
    return json.dumps(result)


def _workspace_project_not_found_error(project: str | None) -> str | None:
    """Return the project-not-found payload for unknown workspace projects."""
    if not _is_workspace_mode() or project is None:
        return None

    wdb = _get_workspace_db()
    workspace = getattr(wdb, "workspace", None)
    if workspace is not None and hasattr(workspace, "get_entries"):
        entries = workspace.get_entries()
    else:
        entries = getattr(wdb, "_all_indexable", [])

    if not any(getattr(entry, "name", None) == project for entry in entries):
        return _project_not_found_error(project)
    return None


# --- Tier 1: Instant tools ---


@mcp.tool()
def codebase_map(project: str | None = None) -> str:
    """Get a complete overview of the indexed codebase.

    Returns project stats, language breakdown, symbol counts by kind,
    directory structure with symbol counts, and hotspot files.
    Call this FIRST in any new session to orient yourself.

    In workspace mode, returns aggregated stats across all projects.

    Args:
        project: Optional project filter (workspace mode only)
    """
    _record_query()
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        try:
            result = wdb.codebase_map(project=project)
        except LookupError:
            if project is None:
                raise
            return _project_not_found_error(project)
        return json.dumps(result, indent=2)

    repo_root, db_path = _resolve_single_repo_context()
    if repo_root is None or db_path is None:
        return json.dumps({"error": "Unable to determine repo root."}, indent=2)

    representative_files = _find_representative_files(repo_root)
    framework_hints = _detect_framework_hints(repo_root, representative_files)
    start_here = _build_start_here(representative_files, framework_hints)
    topology = _build_topology(repo_root, representative_files, framework_hints)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0

    if db_size == 0 and _db is None:
        return json.dumps(
            _bootstrap_codebase_map_result(
                repo_root,
                db_path,
                framework_hints,
                representative_files,
                start_here,
                topology,
            ),
            indent=2,
        )

    try:
        db = _get_db()
        stats = db.stats()
        state = db.get_index_state(str(repo_root))
    except (AssertionError, OSError, sqlite3.Error):
        return json.dumps(
            _bootstrap_codebase_map_result(
                repo_root,
                db_path,
                framework_hints,
                representative_files,
                start_here,
                topology,
            ),
            indent=2,
        )

    indexed_hints = None
    try:
        indexed_hints = _indexed_orientation_hints(db.orientation_symbols())
    except (AttributeError, OSError, sqlite3.Error, json.JSONDecodeError):
        indexed_hints = None

    representative_files = _merge_indexed_representative_files(representative_files, indexed_hints)
    extra_signals = set(indexed_hints.get("signals") or []) if indexed_hints else None
    framework_hints = _detect_framework_hints(repo_root, representative_files, extra_signals=extra_signals)
    start_here = _build_start_here(representative_files, framework_hints)
    topology = _build_topology(repo_root, representative_files, framework_hints, indexed_hints=indexed_hints)

    result = {
        "mode": "single",
        "repo_root": str(_repo_root),
        "index": {
            "status": "ready" if stats["files"] > 0 else "empty",
            "db_path": str(_db_path),
            "files": stats["files"],
            "symbols": stats["symbols"],
            "edges": stats["edges"],
            "db_size_mb": stats["db_size_mb"],
        },
        "brief": _build_repo_brief(framework_hints, start_here, indexed=True),
        "framework_hints": framework_hints,
        "representative_files": representative_files,
        "topology": topology,
        "start_here": start_here,
        "languages": stats["languages"],
        "symbol_kinds": stats["symbol_kinds"],
        "directories": db.directory_summary(max_depth=2),
        "hotspot_files": db.hotspot_files(limit=10),
    }

    if state:
        result["index"]["last_commit"] = state.get("last_commit")
        result["index"]["indexed_at"] = state.get("indexed_at")

    signal = _read_index_signal(_repo_root)
    if signal:
        result["index"]["last_indexed_at"] = signal.get("timestamp")

    return json.dumps(result, indent=2)


@mcp.tool()
def search_symbols(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 20,
) -> str:
    """Search for code symbols (functions, classes, methods, structs, etc.).

    Uses tiered search: symbol names → source code content → documentation.
    In workspace mode, searches across all projects simultaneously.

    Args:
        query: Search query — can be a symbol name, code fragment, or natural language
        kind: Optional filter: 'function', 'class', 'method', 'struct', 'enum', etc.
        project: Optional project filter (workspace mode only, e.g. 'intuition')
        limit: Max results to return (default 20)
    """
    _record_query()
    if _is_workspace_mode():
        project_error = _workspace_project_not_found_error(project)
        if project_error is not None:
            return project_error
        wdb = _get_workspace_db()
        results = wdb.search_symbols(query, kind=kind, project=project, limit=limit)
    else:
        db = _get_db()
        results = db.search_symbols(query, kind=kind, limit=limit)

    if not results:
        payload: dict[str, object] = {
            "query": query,
            "result_count": 0,
            "results": [],
            "hint": f"No keyword matches. Try {_tool_call('hybrid_search', query, project=project)} for semantic matching.",
        }
        tokenized = tokenized_query_hint(query)
        if tokenized:
            payload["suggestions"] = [
                f"Try {_tool_call('search_symbols', tokenized, project=project)} for tokenized identifier search",
                f"Try {_tool_call('hybrid_search', tokenized, project=project)} for concept + keyword search",
            ]
            payload["hint"] = (
                f"No keyword matches. Try {_tool_call('search_symbols', tokenized, project=project)} "
                f"or {_tool_call('hybrid_search', tokenized, project=project)}."
            )
        else:
            suggestions = _symbol_name_suggestions(query, project=project)
            if suggestions:
                payload["did_you_mean"] = suggestions
        return json.dumps(payload, indent=2)

    return json.dumps([_shape_search_result(result) for result in results], indent=2)


@mcp.tool()
def get_symbol(name: str, project: str | None = None) -> str:
    """Get full details of a symbol by name.

    Returns the complete source code, signature, documentation,
    file location, and metadata. If multiple symbols share the name,
    all are returned. Falls back to substring matching if no exact match.

    In workspace mode, searches across all projects.

    Args:
        name: Symbol name (e.g., 'Dictionary', 'lookup', 'main')
        project: Optional project filter (workspace mode only)
    """
    _record_query()
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        results = wdb.get_symbol(name, project=project)
        if not results:
            return _symbol_not_found_error(name, project)
        if len(results) == 1:
            return json.dumps(results[0], indent=2)
        return json.dumps({"match_count": len(results), "symbols": results}, indent=2)

    db = _get_db()
    symbols = db.get_symbols_by_name(name)
    if not symbols:
        return _symbol_not_found_error(name)

    if len(symbols) == 1:
        sym = symbols[0]
        result = _symbol_to_dict(sym)
        result["content"] = sym.content
        result["parameters"] = sym.parameters
        result["return_type"] = sym.return_type
        result["metadata"] = sym.metadata
        return json.dumps(result, indent=2)

    results = []
    for sym in symbols:
        d = _symbol_to_dict(sym)
        d["content"] = sym.content
        d["parameters"] = sym.parameters
        d["return_type"] = sym.return_type
        results.append(d)

    return json.dumps({
        "match_count": len(results),
        "symbols": results,
    }, indent=2)


@mcp.tool()
def get_signature(name: str) -> str:
    """Get just the signature of a symbol (lightweight, for planning).

    Returns only the function/method signature without the full body.
    Use this when you need to understand an API without reading all the code.
    Returns all matches if multiple symbols share the name.

    Args:
        name: Symbol name
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        results = wdb.get_symbol(name)
        if not results:
            return _symbol_not_found_error(name)
        sigs = [
            {
                "project": r["project"],
                "name": r["name"],
                "signature": r.get("signature"),
                "kind": r["kind"],
                "file": r["file"],
                "line": r["start_line"],
                "doc": r.get("doc_comment"),
            }
            for r in results
        ]
        if len(sigs) == 1:
            return json.dumps(sigs[0], indent=2)
        return json.dumps({"match_count": len(sigs), "signatures": sigs}, indent=2)

    db = _get_db()
    symbols = db.get_symbols_by_name(name, limit=10)
    if not symbols:
        return _symbol_not_found_error(name)

    results = []
    for sym in symbols:
        results.append({
            "name": sym.name,
            "signature": sym.signature,
            "kind": sym.kind,
            "file": sym.file_path,
            "line": sym.start_line,
            "doc": sym.doc_comment,
        })

    if len(results) == 1:
        return json.dumps(results[0], indent=2)
    return json.dumps({"match_count": len(results), "signatures": results}, indent=2)


@mcp.tool()
def symbols_in_file(path: str, project: str | None = None) -> str:
    """List all symbols defined in a specific file.

    Returns a table of contents: every function, class, method, struct, etc.
    in the file, ordered by line number. Use this instead of reading a file
    to understand its structure.

    Args:
        path: Relative file path (e.g., 'src/libdict/dictionary.cpp')
        project: Project name (required in workspace mode if ambiguous)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("symbols_in_file")
        wdb = _get_workspace_db()
        all_results = []
        for batch in wdb._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    rows = wdb.conn.execute(
                        f"""SELECT s.name, s.kind, s.signature, s.start_line, s.end_line, s.doc_comment
                           FROM [{schema}].symbols s
                           JOIN [{schema}].files f ON s.file_id = f.id
                           WHERE f.path = ?
                           ORDER BY s.start_line""",
                        (path,),
                    ).fetchall()
                    all_results.extend({
                        "name": r["name"],
                        "kind": r["kind"],
                        "signature": r["signature"],
                        "line": r["start_line"],
                        "end_line": r["end_line"],
                        "doc": r["doc_comment"][:100] if r["doc_comment"] else None,
                    } for r in rows)
                except Exception:
                    pass
        return json.dumps({
            "project": project,
            "file": path,
            "symbol_count": len(all_results),
            "symbols": all_results,
        }, indent=2)

    db = _get_db()
    symbols = db.symbols_in_file(path)
    if not symbols:
        return json.dumps({"error": f"No symbols found in '{path}'"})

    result = []
    for sym in symbols:
        result.append({
            "name": sym.name,
            "kind": sym.kind,
            "signature": sym.signature,
            "line": sym.start_line,
            "end_line": sym.end_line,
            "doc": sym.doc_comment[:100] if sym.doc_comment else None,
        })

    return json.dumps({
        "file": path,
        "symbol_count": len(result),
        "symbols": result,
    }, indent=2)


# --- Tier 2: Graph tools ---


def _dedup_edges(edges: list[dict]) -> list[dict]:
    """Deduplicate edges by symbol name, keeping the highest-confidence entry."""
    by_name: dict[str, dict] = {}
    for c in edges:
        s = c["symbol"]
        name = s.name
        confidence = c["confidence"]
        entry = {
            "name": name,
            "kind": s.kind,
            "file": s.file_path,
            "line": s.start_line,
            "edge_type": c["edge_type"],
            "confidence": confidence,
        }
        if name not in by_name:
            by_name[name] = entry
            by_name[name]["_locations"] = [(s.file_path, s.start_line)]
        else:
            by_name[name]["_locations"].append((s.file_path, s.start_line))
            if confidence > by_name[name]["confidence"]:
                by_name[name].update(entry)
                by_name[name]["_locations"] = by_name[name]["_locations"]

    result = []
    for entry in by_name.values():
        locations = entry.pop("_locations")
        if len(locations) > 1:
            entry["locations"] = [{"file": file_path, "line": line} for file_path, line in locations]
        result.append(entry)

    result.sort(key=lambda r: (
        0 if r["edge_type"] == "inherits" else 1,
        -r["confidence"],
        r["name"],
    ))
    return result


@mcp.tool()
def get_callers(symbol_name: str, project: str | None = None) -> str:
    """Find all symbols that call or reference a given symbol.

    Answers: "Who calls this function?" / "What depends on this?"
    Note: In workspace mode, requires 'project' to specify which repo's graph to search.

    Args:
        symbol_name: Name of the symbol to find callers for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("graph queries (get_callers/get_callees)")
        # Use a temporary single-project Database for graph queries
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        callers = db.get_callers(sym.id)
        result = _dedup_edges(callers)
        db.close()
        return json.dumps({
            "project": project,
            "symbol": symbol_name,
            "caller_count": len(result),
            "callers": result,
        }, indent=2)

    db = _get_db()
    sym = db.get_symbol_by_name(symbol_name)
    if sym is None:
        return _symbol_not_found_error(symbol_name)

    callers = db.get_callers(sym.id)
    result = _dedup_edges(callers)

    return json.dumps({
        "symbol": symbol_name,
        "caller_count": len(result),
        "callers": result,
    }, indent=2)


@mcp.tool()
def get_callees(symbol_name: str, project: str | None = None) -> str:
    """Find all symbols that a given symbol calls or references.

    Answers: "What does this function call?" / "What are this symbol's dependencies?"
    Note: In workspace mode, requires 'project' to specify which repo's graph to search.

    Args:
        symbol_name: Name of the symbol to find callees for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("graph queries (get_callers/get_callees)")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        callees = db.get_callees(sym.id)
        result = _dedup_edges(callees)
        db.close()
        return json.dumps({
            "project": project,
            "symbol": symbol_name,
            "callee_count": len(result),
            "callees": result,
        }, indent=2)

    db = _get_db()
    sym = db.get_symbol_by_name(symbol_name)
    if sym is None:
        return _symbol_not_found_error(symbol_name)

    callees = db.get_callees(sym.id)
    result = _dedup_edges(callees)

    return json.dumps({
        "symbol": symbol_name,
        "callee_count": len(result),
        "callees": result,
    }, indent=2)


@mcp.tool()
def get_type_hierarchy(name: str, project: str | None = None) -> str:
    """Get the inheritance hierarchy for a class or struct.

    Shows both base classes (parents) and subclasses (children).
    Note: In workspace mode, requires 'project' to specify which repo.

    Args:
        name: Class or struct name (e.g., 'ICaptureService', 'TtsProvider')
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("get_type_hierarchy")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(name, project)
        base_classes = db.get_base_classes(sym.id)
        subclasses = db.get_subclasses(sym.id)
        db.close()
        return json.dumps({
            "project": project,
            "symbol": {"name": sym.name, "kind": sym.kind, "file": sym.file_path, "line": sym.start_line},
            "base_classes": [{"name": c["symbol"].name, "kind": c["symbol"].kind, "file": c["symbol"].file_path} for c in base_classes],
            "subclasses": [{"name": c["symbol"].name, "kind": c["symbol"].kind, "file": c["symbol"].file_path} for c in subclasses],
        }, indent=2)

    db = _get_db()
    sym = db.get_symbol_by_name(name)
    if sym is None:
        return _symbol_not_found_error(name)

    base_classes = db.get_base_classes(sym.id)
    subclasses = db.get_subclasses(sym.id)

    result = {
        "symbol": {
            "name": sym.name,
            "kind": sym.kind,
            "file": sym.file_path,
            "line": sym.start_line,
        },
        "base_classes": [
            {
                "name": c["symbol"].name,
                "kind": c["symbol"].kind,
                "file": c["symbol"].file_path,
                "line": c["symbol"].start_line,
            }
            for c in base_classes
        ],
        "subclasses": [
            {
                "name": c["symbol"].name,
                "kind": c["symbol"].kind,
                "file": c["symbol"].file_path,
                "line": c["symbol"].start_line,
            }
            for c in subclasses
        ],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_tests_for(symbol_name: str, project: str | None = None) -> str:
    """Find test functions that cover a given symbol.

    Uses heuristic matching: test file paths + test function names containing
    the symbol name. Also returns any explicit 'tests' edges from the graph.

    Args:
        symbol_name: Name of the symbol to find tests for
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("this tool")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        tests = db.get_tests_for(symbol_name)
        db.close()
    else:
        db = _get_db()
        tests = db.get_tests_for(symbol_name)

    results = []
    for t in tests:
        s = t["symbol"]
        results.append({
            "name": s.name,
            "kind": s.kind,
            "file": s.file_path,
            "line": s.start_line,
            "confidence": t["confidence"],
        })

    return json.dumps({
        "symbol": symbol_name,
        "test_count": len(results),
        "tests": results,
    }, indent=2)


@mcp.tool()
def get_dependents(symbol_name: str, transitive: bool = False, project: str | None = None) -> str:
    """Find all symbols that depend on (call/reference) a given symbol.

    Answers: "What would break if I change this?" / "What's the blast radius?"

    With transitive=True, walks the caller graph recursively to show the full
    impact chain (up to 5 levels deep).

    Args:
        symbol_name: Name of the symbol to find dependents for
        transitive: If True, follow the dependency chain recursively
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("this tool")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        deps = db.get_dependents(sym.id, transitive=transitive)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            return _symbol_not_found_error(symbol_name)
        deps = db.get_dependents(sym.id, transitive=transitive)

    result = _dedup_edges(deps)
    return json.dumps({
        "symbol": symbol_name,
        "transitive": transitive,
        "dependent_count": len(result),
        "dependents": result,
    }, indent=2)


@mcp.tool()
def get_implementors(interface_name: str, project: str | None = None) -> str:
    """Find all classes that implement or inherit from an interface/base class.

    Answers: "What classes implement this interface?" / "What are the concrete types?"

    Args:
        interface_name: Name of the interface or base class
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("get_implementors")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(interface_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(interface_name, project)
        impls = db.get_implementors(sym.id)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(interface_name)
        if sym is None:
            return _symbol_not_found_error(interface_name)
        impls = db.get_implementors(sym.id)

    results = [
        {
            "name": c["symbol"].name,
            "kind": c["symbol"].kind,
            "file": c["symbol"].file_path,
            "line": c["symbol"].start_line,
        }
        for c in impls
    ]

    return json.dumps({
        "interface": interface_name,
        "implementor_count": len(results),
        "implementors": results,
    }, indent=2)


@mcp.tool()
def index_status() -> str:
    """Check the current state of the code index.

    In workspace mode, shows per-project stats.
    """
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        projects = wdb.list_projects()
        return json.dumps({
            "mode": "workspace",
            "workspace": _workspace_name,
            "projects": projects,
        }, indent=2)

    repo_root, db_path = _resolve_single_repo_context()
    if repo_root is None or db_path is None or not db_path.exists() or db_path.stat().st_size == 0:
        return json.dumps({
            "mode": "single",
            "repo_root": str(repo_root) if repo_root is not None else None,
            "db_path": str(db_path) if db_path is not None else None,
            "bootstrap_mode": "filesystem_only",
            "index": {
                "status": "not_indexed",
                "db_path": str(db_path) if db_path is not None else None,
                "present": bool(db_path and db_path.exists()),
            },
            "hint": (
                "Repo is not indexed yet. Run `srclight index --embed` "
                f"from {shlex.quote(str(repo_root))}."
            ),
            "next_actions": [
                f"Run: srclight index {shlex.quote(str(repo_root))} --embed",
                "Call index_status() again after indexing.",
                "Use codebase_map() for a filesystem-only brief.",
            ],
        }, indent=2)

    try:
        db = _get_db()
    except (AssertionError, OSError, sqlite3.Error):
        return json.dumps({
            "mode": "single",
            "repo_root": str(repo_root),
            "db_path": str(db_path),
            "bootstrap_mode": "filesystem_only",
            "index": {
                "status": "not_indexed",
                "db_path": str(db_path),
                "present": bool(db_path.exists()),
            },
            "hint": (
                "Repo is not indexed yet. Run `srclight index --embed` "
                f"from {shlex.quote(str(repo_root))}."
            ),
            "next_actions": [
                f"Run: srclight index {shlex.quote(str(repo_root))} --embed",
                "Call index_status() again after indexing.",
                "Use codebase_map() for a filesystem-only brief.",
            ],
        }, indent=2)

    stats = db.stats()
    state = db.get_index_state(str(_repo_root)) if _repo_root else None

    result = {
        "mode": "single",
        "repo_root": str(_repo_root),
        "db_path": str(_db_path),
        "files": stats["files"],
        "symbols": stats["symbols"],
        "edges": stats["edges"],
        "db_size_mb": stats["db_size_mb"],
        "languages": stats["languages"],
    }

    if state:
        result["last_commit"] = state.get("last_commit")
        result["indexed_at"] = state.get("indexed_at")

    signal = _read_index_signal(_repo_root)
    if signal:
        result["last_indexed_at"] = signal.get("timestamp")

    return json.dumps(result, indent=2)


@mcp.tool()
def list_projects() -> str:
    """List all projects in the workspace with stats.

    Only available in workspace mode. Shows files, symbols, languages,
    and DB size for each project.
    """
    if not _is_workspace_mode():
        return json.dumps({"error": "Not in workspace mode. Start with --workspace NAME"})

    wdb = _get_workspace_db()
    projects = wdb.list_projects()
    return json.dumps({
        "workspace": _workspace_name,
        "project_count": len(projects),
        "projects": projects,
    }, indent=2)


@mcp.tool()
async def reindex(path: str | None = None) -> str:
    """Trigger re-indexing of the codebase or a specific path.

    Incrementally updates the index — only re-parses files whose content
    has changed since the last index.

    Args:
        path: Optional specific directory to re-index (default: entire repo)
    """
    global _vector_cache
    root = Path(path) if path else _repo_root
    if root is None:
        return json.dumps({"error": "No repo root configured"})

    root = root.resolve()
    db = _get_db()
    config = IndexConfig(root=root)
    indexer = Indexer(db, config)
    stats = indexer.index(root)

    # Invalidate vector cache so next query reloads from fresh sidecar
    _vector_cache = None

    result = {
        "files_indexed": stats.files_indexed,
        "files_unchanged": stats.files_unchanged,
        "files_removed": stats.files_removed,
        "symbols_extracted": stats.symbols_extracted,
        "errors": stats.errors,
        "elapsed_seconds": round(stats.elapsed_seconds, 2),
    }

    # Send notification to connected clients (MCP logging)
    try:
        ctx = mcp.get_context()
        await ctx.info(
            f"Reindex complete: {stats.files_indexed} files, "
            f"{stats.symbols_extracted} symbols indexed in {stats.elapsed_seconds:.1f}s"
        )
    except Exception:
        pass  # Best effort — client may not support notifications

    return json.dumps(result, indent=2)


# --- Tier 4: Git Change Intelligence ---


def _resolve_repo_root(project: str | None = None) -> Path | None:
    """Resolve repo root for git operations."""
    if _is_workspace_mode() and project:
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        return Path(path) if path else None
    return _repo_root


@mcp.tool()
def blame_symbol(symbol_name: str, project: str | None = None) -> str:
    """Get git blame info for a symbol — who changed it, when, and why.

    Returns the last modifier, total unique commits/authors, age in days,
    and the list of commits that touched this symbol's line range.

    Args:
        symbol_name: Name of the symbol to blame
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Find the symbol in the index
    if _is_workspace_mode():
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)

    if sym is None:
        return _symbol_not_found_error(symbol_name)

    result = git_mod.blame_symbol(
        repo_root, sym.file_path, sym.start_line, sym.end_line
    )
    result["symbol"] = symbol_name
    result["file"] = sym.file_path
    result["lines"] = f"{sym.start_line}-{sym.end_line}"

    return json.dumps(result, indent=2)


@mcp.tool()
def recent_changes(
    n: int = 20, author: str | None = None,
    path_filter: str | None = None, project: str | None = None,
) -> str:
    """Get recent git commits with files changed.

    Answers: "What changed recently?" / "What has this author been working on?"

    Args:
        n: Number of commits to return (default 20)
        author: Filter by author name (substring match)
        path_filter: Filter by file path prefix (e.g., 'src/libdict/')
        project: Project name (workspace mode) or uses current repo
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        # Show recent changes across all projects
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        all_changes = []
        for entry in config.get_entries():
            root = Path(entry.path)
            if root.exists():
                commits = git_mod.recent_changes(
                    root, n=n, author=author, path_filter=path_filter
                )
                for c in commits:
                    c["project"] = entry.name
                all_changes.extend(commits)
        # Sort by date descending across all projects
        all_changes.sort(key=lambda c: c.get("date", ""), reverse=True)
        return json.dumps(all_changes[:n], indent=2)

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    commits = git_mod.recent_changes(
        repo_root, n=n, author=author, path_filter=path_filter
    )
    return json.dumps(commits, indent=2)


@mcp.tool()
def git_hotspots(
    n: int = 20, since: str | None = None, project: str | None = None,
) -> str:
    """Find most frequently changed files (churn hotspots / bug magnets).

    Files that change often are more likely to have bugs and be fragile.
    Use this to identify risky areas before making changes.

    Args:
        n: Number of files to return (default 20)
        since: Time period (e.g., '30.days', '3.months', '1.year')
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("git_hotspots")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    spots = git_mod.hotspots(repo_root, n=n, since=since)
    return json.dumps({
        "project": project or str(repo_root),
        "period": since or "all time",
        "hotspot_count": len(spots),
        "hotspots": spots,
    }, indent=2)


@mcp.tool()
def whats_changed(project: str | None = None) -> str:
    """Show uncommitted changes (work in progress).

    Returns staged, unstaged, and untracked files. Use this instead of
    running git status + git diff manually.

    Args:
        project: Project name (workspace mode) or uses current repo
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        # Show changes across all projects
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        all_results = {}
        for entry in config.get_entries():
            root = Path(entry.path)
            if root.exists():
                changes = git_mod.whats_changed(root)
                if changes["total_changes"] > 0:
                    all_results[entry.name] = changes
        return json.dumps({
            "projects_with_changes": len(all_results),
            "projects": all_results,
        }, indent=2)

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    result = git_mod.whats_changed(repo_root)
    return json.dumps(result, indent=2)


@mcp.tool()
def changes_to(symbol_name: str, n: int = 20, project: str | None = None) -> str:
    """Get the change history for a specific symbol's file.

    Shows commits that modified the file containing the symbol.
    Useful for understanding "why is it this way?" and recent activity.

    Args:
        symbol_name: Symbol name to track
        n: Number of commits to return
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Find the symbol to get its file
    if _is_workspace_mode():
        db_path = repo_root / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)

    if sym is None:
        return _symbol_not_found_error(symbol_name)

    commits = git_mod.changes_to_file(repo_root, sym.file_path, n=n)
    return json.dumps({
        "symbol": symbol_name,
        "file": sym.file_path,
        "commit_count": len(commits),
        "commits": commits,
    }, indent=2)


# --- Tier 5: Build & Configuration Intelligence ---


@mcp.tool()
def get_build_targets(project: str | None = None) -> str:
    """Get all build targets (libraries, executables) from the build system.

    Parses CMakeLists.txt, .csproj, package.json, Cargo.toml to extract
    targets with their sources, dependencies, and platform conditions.

    Args:
        project: Project name (required in workspace mode)
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    info = build_mod.get_build_info(repo_root)
    return json.dumps(info, indent=2)


@mcp.tool()
def get_platform_variants(symbol_name: str, project: str | None = None) -> str:
    """Find platform-specific variants of a symbol.

    Scans C/C++/C# source files for #ifdef platform guards near the symbol.
    Essential for cross-platform projects — shows which platforms have
    specialized implementations.

    Args:
        symbol_name: Symbol name to search for
        project: Project name (required in workspace mode)
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    variants = build_mod.get_platform_variants(repo_root, symbol_name)
    return json.dumps({
        "symbol": symbol_name,
        "variant_count": len(variants),
        "variants": variants,
    }, indent=2)


@mcp.tool()
def platform_conditionals(project: str | None = None, platform: str | None = None) -> str:
    """List all platform-conditional code blocks in the project.

    Scans for #ifdef, #if defined(), and similar preprocessor guards.
    Useful for understanding which code is platform-specific.

    Args:
        project: Project name (required in workspace mode)
        platform: Optional filter (e.g., 'windows', 'linux', 'apple', 'android')
    """
    from . import build as build_mod

    if _is_workspace_mode() and not project:
        return _project_required_error("this tool")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    conditionals = build_mod.scan_platform_conditionals(repo_root)

    if platform:
        conditionals = [c for c in conditionals if platform in c["platforms"]]

    # Group by platform for summary
    platform_counts: dict[str, int] = {}
    for c in conditionals:
        for p in c["platforms"]:
            platform_counts[p] = platform_counts.get(p, 0) + 1

    return json.dumps({
        "total": len(conditionals),
        "platform_summary": platform_counts,
        "conditionals": conditionals[:100],  # Cap at 100 for readability
    }, indent=2)


# --- Tier 6: Semantic Search (Embeddings) ---


@mcp.tool()
def semantic_search(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 10,
) -> str:
    """Find semantically similar code using embeddings.

    Unlike search_symbols (keyword-based), this finds conceptually similar
    code even when the exact terms don't match. Good for natural language
    queries like "find code that handles dictionary lookup" or
    "where is the authentication logic".

    Requires embeddings to be generated first (srclight index --embed).

    Args:
        query: Natural language description of what you're looking for
        kind: Optional filter by symbol kind (function, class, method, etc.)
        project: Project name (workspace mode) or uses current repo
        limit: Max results (default 10)
    """
    _record_query()
    from .embeddings import get_provider, vector_to_bytes

    project_error = _workspace_project_not_found_error(project)
    if project_error is not None:
        return project_error

    # Determine which model was used for embeddings
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        emb_stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        emb_stats = db.embedding_stats()

    if not emb_stats.get("model"):
        return json.dumps({
            "error": "No embeddings found. Run 'srclight index --embed' first.",
            "hint": f"Try: srclight index --embed (defaults to {DEFAULT_OLLAMA_EMBED_MODEL})",
        })

    model_name = emb_stats["model"]
    dims = emb_stats["dimensions"]

    try:
        provider = get_provider(model_name)
        query_vec = provider.embed_one(query)
        query_bytes = vector_to_bytes(query_vec)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to embed query: {e}",
            "model": model_name,
        })

    if _is_workspace_mode():
        results = wdb.vector_search(query_bytes, dims, project=project, kind=kind, limit=limit)
    else:
        cache = _get_vector_cache()
        results = db.vector_search(query_bytes, dims, kind=kind, limit=limit, cache=cache)

    return json.dumps({
        "query": query,
        "model": model_name,
        "result_count": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
def hybrid_search(
    query: str, kind: str | None = None, project: str | None = None, limit: int = 20,
) -> str:
    """Search using both keyword matching AND semantic similarity.

    Combines FTS5 text search with embedding-based semantic search using
    Reciprocal Rank Fusion (RRF). This is the most powerful search mode —
    it finds results that match either by exact keywords or by meaning.

    Falls back to keyword-only search if embeddings aren't available.

    Args:
        query: Search query (works with both keywords and natural language)
        kind: Optional filter by symbol kind
        project: Project name (workspace mode) or uses current repo
        limit: Max results (default 20)
    """
    _record_query()
    from .embeddings import get_provider, rrf_merge, vector_to_bytes

    project_error = _workspace_project_not_found_error(project)
    if project_error is not None:
        return project_error

    # Get FTS results
    if _is_workspace_mode():
        wdb = _get_workspace_db()
        fts_results = wdb.search_symbols(query, kind=kind, project=project, limit=limit * 2)
    else:
        db = _get_db()
        fts_results = db.search_symbols(query, kind=kind, limit=limit * 2)

    # Try to get embedding results
    embedding_results = []
    model_used = None
    embedding_error: str | None = None

    if _is_workspace_mode():
        emb_stats = wdb.embedding_stats(project=project)
    else:
        emb_stats = db.embedding_stats()

    if emb_stats.get("model"):
        model_name = emb_stats["model"]
        dims = emb_stats["dimensions"]
        try:
            provider = get_provider(model_name)
            query_vec = provider.embed_one(query)
            query_bytes = vector_to_bytes(query_vec)

            if _is_workspace_mode():
                embedding_results = wdb.vector_search(
                    query_bytes, dims, project=project, kind=kind, limit=limit * 2
                )
            else:
                cache = _get_vector_cache()
                embedding_results = db.vector_search(
                    query_bytes, dims, kind=kind, limit=limit * 2, cache=cache
                )
            model_used = model_name
        except Exception as e:
            # Fail fast and report clearly when the embedding provider
            # (e.g. Ollama) is unreachable or misconfigured. We still
            # return FTS-only results, but include the error so clients
            # can surface it instead of silently degrading.
            embedding_error = str(e)
            logger.warning("Embedding search failed, using FTS only: %s", e)

    if embedding_results:
        merged = rrf_merge(fts_results, embedding_results)
        final = merged[:limit]
        payload: dict[str, object] = {
            "query": query,
            "mode": "hybrid (FTS5 + embeddings)",
            "model": model_used,
            "result_count": len(final),
            "results": [_shape_search_result(result) for result in final],
        }
        if not final:
            payload["hint"] = "No results. Try broadening your query or check that the index is up to date with reindex()."
        return json.dumps(payload, indent=2)
    else:
        payload = {
            "query": query,
            "mode": "keyword only (no embeddings available)",
            "result_count": min(len(fts_results), limit),
            "results": [_shape_search_result(result) for result in fts_results[:limit]],
        }
        if not fts_results:
            payload["hint"] = "No results. Try broadening your query or check that the index is up to date with reindex()."
        if embedding_error is not None:
            payload["embedding_error"] = embedding_error
        return json.dumps(payload, indent=2)


@mcp.tool()
def embedding_status(project: str | None = None) -> str:
    """Check embedding coverage and model info.

    Shows how many symbols have embeddings, which model was used,
    and the coverage percentage.

    Args:
        project: Project name (workspace mode) or uses current repo
    """
    project_error = _workspace_project_not_found_error(project)
    if project_error is not None:
        return project_error

    if _is_workspace_mode():
        wdb = _get_workspace_db()
        stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        stats = db.embedding_stats()

    if not stats.get("model"):
        stats["hint"] = (
            "Run 'srclight index --embed' to generate embeddings "
            f"(defaults to {DEFAULT_OLLAMA_EMBED_MODEL})"
        )

    return json.dumps(stats, indent=2)


@mcp.tool()
def embedding_health(project: str | None = None) -> str:
    """Check if the configured embedding provider is reachable.

    Uses embedding_stats() to find the active model, then performs a
    lightweight provider-specific health check (e.g. Ollama /api/tags).
    Returns a JSON blob with status, model, and any error message so
    clients can surface problems instead of silently degrading.
    """
    project_error = _workspace_project_not_found_error(project)
    if project_error is not None:
        return project_error

    if _is_workspace_mode():
        wdb = _get_workspace_db()
        stats = wdb.embedding_stats(project=project)
    else:
        db = _get_db()
        stats = db.embedding_stats()

    if not stats.get("model"):
        return json.dumps({
            "status": "no_embeddings",
            "detail": (
                "No embeddings found in the index. "
                f"Run 'srclight index --embed' first (defaults to {DEFAULT_OLLAMA_EMBED_MODEL})."
            ),
            "stats": stats,
        }, indent=2)

    model_name = stats["model"]
    from .embeddings import get_provider

    result: dict[str, object] = {
        "status": "unknown",
        "model": model_name,
        "dimensions": stats.get("dimensions"),
    }

    try:
        provider = get_provider(model_name)

        # OllamaProvider exposes is_available(), which hits /api/tags with a short timeout.
        is_available = getattr(provider, "is_available", None)
        if callable(is_available):
            ok = bool(is_available())
            result["provider"] = provider.name
            result["reachable"] = ok
            if ok:
                result["status"] = "ok"
            else:
                result["status"] = "error"
                result["error"] = "Embedding provider reported is_available() == False"
        else:
            # Fallback: we don't know how to health-check this provider without
            # running a full embed call. Leave status as unknown but include name.
            result["provider"] = getattr(provider, "name", model_name)
            result["status"] = "unknown"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


# --- Tier 7: Import Resolution ---


# Import extraction patterns by language (regex-based, not tree-sitter)
IMPORT_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)",
    ],
    "javascript": [
        r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "typescript": [
        r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "c": [r'#include\s*[<"]([^>"]+)[>"]'],
    "cpp": [r'#include\s*[<"]([^>"]+)[>"]'],
    "go": [r'"([^"]+)"'],
    "java": [r"^import\s+([\w.]+);"],
    "kotlin": [r"^import\s+([\w.]+)"],
    "dart": [r"import\s+['\"]([^'\"]+)['\"]"],
    "swift": [r"^import\s+(\w+)"],
    "csharp": [r"^using\s+([\w.]+);"],
    "php": [
        r"^use\s+([\w\\]+)",
        r"(?:require|include)(?:_once)?\s*['\"]([^'\"]+)['\"]",
    ],
}


def _extract_imports(content: str, language: str) -> list[dict]:
    """Extract import statements from file content using regex patterns."""
    patterns = IMPORT_PATTERNS.get(language, [])
    if not patterns:
        return []

    imports = []
    seen_statements = set()

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") and language != "c" and language != "cpp":
            if not stripped.startswith("#include"):
                continue

        for pat in patterns:
            for m in re.finditer(pat, stripped):
                statement = stripped
                if statement in seen_statements:
                    continue
                seen_statements.add(statement)

                groups = [g for g in m.groups() if g is not None]
                if not groups:
                    continue

                if language == "python":
                    from_module = m.group(1)
                    import_names = m.group(2)
                    if from_module:
                        names = [n.strip() for n in import_names.split(",") if n.strip()]
                        imports.append({
                            "statement": statement,
                            "module": from_module,
                            "names": names,
                        })
                    else:
                        for name in import_names.split(","):
                            name = name.strip().split(" as ")[0].strip()
                            if name:
                                imports.append({
                                    "statement": statement,
                                    "module": name,
                                    "names": [],
                                })
                else:
                    module = groups[0]
                    imports.append({
                        "statement": statement,
                        "module": module,
                        "names": [],
                    })

    return imports


@mcp.tool()
def find_imports(path: str, project: str | None = None) -> str:
    """Find and resolve import statements in a source file.

    Extracts all import/include/require statements and attempts to resolve
    them to indexed symbols. Answers: "What does this file depend on?"

    Supports: Python (import/from...import), JavaScript/TypeScript (import/require),
    C/C++ (#include), Go (import), Java/Kotlin (import), Dart (import),
    Swift (import), C# (using), PHP (use/require/include).

    Args:
        path: Relative file path (e.g., 'src/srclight/server.py')
        project: Project name (required in workspace mode if ambiguous)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_imports")
        wdb = _get_workspace_db()

        # Find the file in the workspace
        file_info = None
        for batch in wdb._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    row = wdb.conn.execute(
                        f"SELECT * FROM [{schema}].files WHERE path = ?",
                        (path,),
                    ).fetchone()
                    if row:
                        file_info = {"language": row["language"], "schema": schema}
                        break
                except Exception:
                    pass
            if file_info:
                break

        if not file_info:
            return json.dumps({"error": f"File '{path}' not found in project '{project}'"})

        language = file_info["language"]
        if not language or language not in IMPORT_PATTERNS:
            return json.dumps({
                "file": path,
                "language": language,
                "import_count": 0,
                "resolved_count": 0,
                "imports": [],
                "note": f"Import extraction not supported for language: {language}",
            }, indent=2)

        # Read file content from disk
        config_entries = [e for e in wdb._all_indexable if e.name == project]
        if not config_entries:
            return _project_not_found_error(project)
        project_root = Path(config_entries[0].root)
        file_path = project_root / path
        try:
            content = file_path.read_text(errors="replace")
        except OSError as e:
            return json.dumps({"error": f"Cannot read file: {e}"})

        raw_imports = _extract_imports(content, language)

        imports = []
        resolved_count = 0
        for imp in raw_imports:
            entry: dict = {
                "statement": imp["statement"],
                "module": imp["module"],
            }
            if imp["names"]:
                entry["names"] = imp["names"]

            resolved_to = None
            names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
            for name in names_to_try:
                for batch in wdb._iter_batches(project_filter=project):
                    for schema, pname in batch:
                        try:
                            row = wdb.conn.execute(
                                f"""SELECT s.name, s.kind, s.start_line, f.path as file_path
                                    FROM [{schema}].symbols s
                                    JOIN [{schema}].files f ON s.file_id = f.id
                                    WHERE s.name = ?
                                    LIMIT 1""",
                                (name,),
                            ).fetchone()
                            if row:
                                resolved_to = {
                                    "name": row["name"],
                                    "file": row["file_path"],
                                    "line": row["start_line"],
                                    "kind": row["kind"],
                                }
                                break
                        except Exception:
                            pass
                    if resolved_to:
                        break
                if resolved_to:
                    break

            if resolved_to:
                entry["resolved_to"] = resolved_to
                entry["status"] = "resolved"
                resolved_count += 1
            else:
                entry["resolved_to"] = None
                entry["status"] = "external"

            imports.append(entry)

        return json.dumps({
            "file": path,
            "language": language,
            "import_count": len(imports),
            "resolved_count": resolved_count,
            "imports": imports,
        }, indent=2)

    # Single-repo mode
    db = _get_db()
    file_rec = db.get_file(path)
    if not file_rec:
        return json.dumps({"error": f"File '{path}' not found in index"})

    language = file_rec.language
    if not language or language not in IMPORT_PATTERNS:
        return json.dumps({
            "file": path,
            "language": language,
            "import_count": 0,
            "resolved_count": 0,
            "imports": [],
            "note": f"Import extraction not supported for language: {language}",
        }, indent=2)

    if _repo_root:
        file_path = _repo_root / path
    else:
        file_path = Path(path)
    try:
        content = file_path.read_text(errors="replace")
    except OSError as e:
        return json.dumps({"error": f"Cannot read file: {e}"})

    raw_imports = _extract_imports(content, language)

    imports = []
    resolved_count = 0
    for imp in raw_imports:
        entry: dict = {
            "statement": imp["statement"],
            "module": imp["module"],
        }
        if imp["names"]:
            entry["names"] = imp["names"]

        resolved_to = None
        names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
        for name in names_to_try:
            result = db.resolve_import(name, hint_path=path)
            if result:
                resolved_to = result
                break

        if resolved_to:
            entry["resolved_to"] = resolved_to
            entry["status"] = "resolved"
            resolved_count += 1
        else:
            entry["resolved_to"] = None
            entry["status"] = "external"

        imports.append(entry)

    return json.dumps({
        "file": path,
        "language": language,
        "import_count": len(imports),
        "resolved_count": resolved_count,
        "imports": imports,
    }, indent=2)


# --- Tier 8: Code Analysis ---


@mcp.tool()
def find_dead_code(project: str | None = None, kind: str | None = None) -> str:
    """Find symbols that have no callers or references — potential dead code.

    Returns symbols that are defined but never referenced by any other symbol
    in the indexed codebase. Useful for cleanup and understanding which code
    is actually used.

    Excludes: main/entry points, __init__/__main__, test functions, and
    symbols in vendored/third-party code.

    Args:
        project: Project name (required in workspace mode)
        kind: Filter by symbol kind (e.g., 'function', 'class', 'method')
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_dead_code")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        dead = db.get_dead_symbols(kind=kind)
        db.close()
    else:
        db = _get_db()
        dead = db.get_dead_symbols(kind=kind)

    # Group by file for readability
    by_file: dict[str, list[dict]] = {}
    for sym in dead:
        entry = {
            "name": sym.name,
            "kind": sym.kind,
            "line": sym.start_line,
            "signature": sym.signature,
        }
        file_path = sym.file_path or "unknown"
        by_file.setdefault(file_path, []).append(entry)

    result: dict[str, object] = {
        "total_unreferenced": len(dead),
        "file_count": len(by_file),
        "by_file": by_file,
    }
    if project:
        result["project"] = project
    if kind:
        result["kind_filter"] = kind
    if not dead:
        result["hint"] = "No unreferenced symbols found. This may mean the codebase is well-connected, or edges haven't been indexed yet."

    return json.dumps(result, indent=2)


@mcp.tool()
def find_pattern(
    pattern: str,
    project: str | None = None,
    language: str | None = None,
    kind: str | None = None,
    limit: int = 50,
) -> str:
    """Search for structural code patterns in symbol bodies.

    Goes beyond text grep by searching within parsed symbol boundaries.
    Patterns are matched against symbol source code with context about
    the containing function/class/method.

    Unlike grep, results include:
    - The symbol name and kind containing the match
    - File path and line numbers of the symbol
    - The match context within the symbol

    Pattern supports regex. Common patterns:
    - "Color\\\\(0x" — find raw color literals
    - "requests\\\\.get\\\\(" — find HTTP calls
    - "TODO|FIXME|HACK" — find code annotations
    - "except.*Exception" — find broad exception handlers
    - "sleep\\\\(" — find sleep calls
    - "eval\\\\(|exec\\\\(" — find dynamic code execution

    Args:
        pattern: Regex pattern to search for in symbol source code
        project: Optional project filter (workspace mode: filters to one project)
        language: Filter by language (e.g., 'python', 'javascript')
        kind: Filter by symbol kind (e.g., 'function', 'method')
        limit: Maximum results (default 50)
    """
    import re as _re

    # Validate regex
    try:
        _re.compile(pattern)
    except _re.error as e:
        return json.dumps({"error": f"Invalid regex pattern: {e}"}, indent=2)

    if _is_workspace_mode():
        if not project:
            return _project_required_error("find_pattern")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        matches = db.search_pattern(pattern, language=language, kind=kind, limit=limit)
        db.close()
    else:
        db = _get_db()
        matches = db.search_pattern(pattern, language=language, kind=kind, limit=limit)

    # Group by file for readability
    by_file: dict[str, list[dict]] = {}
    for m in matches:
        file_path = m.pop("file", "unknown")
        by_file.setdefault(file_path, []).append(m)

    result: dict[str, object] = {
        "pattern": pattern,
        "match_count": len(matches),
        "file_count": len(by_file),
        "by_file": by_file,
    }
    if project:
        result["project"] = project
    if language:
        result["language_filter"] = language
    if kind:
        result["kind_filter"] = kind
    if not matches:
        result["hint"] = "No matches found. Try a broader pattern or check that symbols have been indexed."

    return json.dumps(result, indent=2)


# Set when run_server() or first tool runs — for server_stats
_server_start_time: float | None = None

# Query activity tracking (for Flutter app / web dashboard)
_last_query_time: float | None = None
_last_query_client: str | None = None
_query_count: int = 0

# UI event queue — polled by Flutter app via /api/ui_events.
_ui_events: list[dict] = []


def _record_query(client: str | None = None) -> None:
    """Record that a tool was called (timestamp + optional client identifier)."""
    global _last_query_time, _last_query_client, _query_count
    _last_query_time = time.time()
    _query_count += 1
    if client:
        _last_query_client = client


@mcp.tool()
async def server_stats() -> str:
    """Return when this server process started and how long it has been running."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    now = time.time()
    uptime = now - _server_start_time
    started_at = datetime.fromtimestamp(_server_start_time, tz=timezone.utc)
    return json.dumps({
        "started_at": started_at.isoformat(),
        "started_at_epoch": _server_start_time,
        "uptime_seconds": round(uptime, 2),
        "uptime_human": f"{int(uptime)}s",
    }, indent=2)


@mcp.tool()
async def restart_server() -> str:
    """Request the server to exit so a process manager can restart it (SSE only).

    Exits with code 0 so a wrapper can start a fresh process (e.g. loads updated
    code). Client must reconnect after restart. Restart is allowed by default;
    set SRCLIGHT_ALLOW_RESTART=0 to disable.
    """
    allow = os.environ.get("SRCLIGHT_ALLOW_RESTART", "1").strip().lower()
    if allow in ("0", "false", "no"):
        return json.dumps({
            "ok": False,
            "message": "Restart is disabled (SRCLIGHT_ALLOW_RESTART=0). Remove it or set to 1 to allow.",
            "hint": "Example: srclight serve --workspace NAME --transport sse --port 8742",
        }, indent=2)

    def _exit():
        os._exit(0)

    asyncio.get_running_loop().call_later(0, _exit)
    return json.dumps({
        "ok": True,
        "message": "Server will exit now. Reconnect after your process manager restarts it.",
    }, indent=2)


@mcp.tool()
async def show_status(message: str = "") -> str:
    """Show the srclight dashboard window and return current status.

    Pops up the desktop app window (if running) and returns indexing stats.
    Use this when a user asks about their indexing status, project health,
    or wants to see what srclight is doing.

    Args:
        message: Optional message to display in the dashboard.
    """
    global _ui_events
    _ui_events.append({
        "type": "show_status",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    # Return current stats so the AI has the data in context too.
    stats: dict = {
        "query_count": _query_count,
    }
    if _last_query_time is not None:
        stats["last_query_ago_seconds"] = round(time.time() - _last_query_time, 1)
    if _last_query_client is not None:
        stats["last_query_client"] = _last_query_client
    try:
        map_result = await codebase_map()
        stats["codebase"] = json.loads(map_result)
    except Exception:
        pass
    return json.dumps(stats, indent=2)


@mcp.tool()
async def setup_guide() -> str:
    """Structured setup instructions for AI agents and users.
    Returns: how to add a workspace, connect Cursor, where config lives, how to index with embeddings, hook install."""
    from .workspace import WORKSPACES_DIR

    return json.dumps({
        "title": "Srclight setup guide for agents",
        "quick_start": [
            "Call codebase_map() first in every new session.",
            "Prefer stdio for local agent clients: srclight serve --transport stdio.",
            "Use hybrid_search(query) as the default code-discovery tool.",
            "Bare --embed defaults to "
            f"{DEFAULT_OLLAMA_EMBED_MODEL}.",
            "After server upgrades, restart both the server and the MCP client/editor session.",
        ],
        "decision_tree": [
            {
                "when": "Need repo orientation",
                "use": "codebase_map()",
            },
            {
                "when": "Need to find code by concept or mixed keywords",
                "use": "hybrid_search(query)",
            },
            {
                "when": "Need exact symbol source",
                "use": "get_symbol(name)",
            },
            {
                "when": "Need setup or indexing help for the user",
                "use": "setup_guide()",
            },
        ],
        "config_location": {
            "workspaces_dir": str(WORKSPACES_DIR),
            "description": "Workspace configs are JSON files: ~/.srclight/workspaces/{name}.json",
        },
        "client_snippets": {
            "stdio_command": "srclight serve --transport stdio",
            "cursor_mcp_url": "http://127.0.0.1:8742/mcp",
            "cursor_transport_notes": "Prefer stdio for local agents. Use Streamable HTTP /mcp or SSE /sse only for HTTP clients such as Cursor.",
        },
        "steps": [
            {
                "step": 1,
                "title": "Create or use a workspace",
                "commands": [
                    "srclight workspace init WORKSPACE_NAME",
                    "srclight workspace add /path/to/repo -w WORKSPACE_NAME",
                ],
            },
            {
                "step": 2,
                "title": "Index the workspace (optionally with embeddings)",
                "commands": [
                    "srclight workspace index -w WORKSPACE_NAME",
                    "srclight workspace index -w WORKSPACE_NAME --embed",
                ],
                "notes": (
                    "Bare --embed defaults to "
                    f"{DEFAULT_OLLAMA_EMBED_MODEL}. Ollama runs on localhost:11434. "
                    "The MCP reindex() tool refreshes the live server in place. "
                    "After external CLI indexing, restart the server/client if you need fresh sidecars immediately."
                ),
            },
            {
                "step": 3,
                "title": "Install git hooks (optional, for auto-reindex)",
                "commands": ["srclight hook install --workspace WORKSPACE_NAME"],
            },
            {
                "step": 4,
                "title": "Start stdio for local agents or SSE for HTTP clients",
                "commands": [
                    "srclight serve --workspace WORKSPACE_NAME --transport stdio",
                    "# Cursor / HTTP: srclight serve --workspace WORKSPACE_NAME --transport sse",
                    "# Or with web dashboard: srclight serve --workspace WORKSPACE_NAME --transport sse --web",
                ],
                "notes": "Prefer stdio for Codex/Claude/local MCP clients. SSE/Streamable HTTP binds to 127.0.0.1:8742 for Cursor or the optional web dashboard.",
            },
        ],
        "troubleshooting": [
            {
                "problem": "No embeddings found",
                "action": f"Run srclight index --embed or srclight workspace index --embed (defaults to {DEFAULT_OLLAMA_EMBED_MODEL}).",
            },
            {
                "problem": "MCP tools fail after server restart",
                "action": "Restart the editor/CLI so the MCP client reconnects and rediscovers tools.",
            },
            {
                "problem": "Workspace tool asks for project",
                "action": "Call list_projects() and retry with one of the returned project names.",
            },
        ],
        "for_agents": "Call codebase_map() at session start. Use list_projects() to see repos. Use setup_guide() to get these steps for the user.",
        "after_upgrade": "After upgrading srclight (pip install -U srclight), restart the server and then restart your editor/CLI to pick up new tools. Existing MCP sessions only discover tools at connect time.",
        "next_action": "If the repo is not indexed yet, run the index command from step 2. If the server is already running, call codebase_map() next.",
    }, indent=2)


# --- Tier 7: Learnings (workspace-level conversation intelligence) ---


@mcp.tool()
def record_learning(
    kind: str,
    content: str,
    reasoning: str | None = None,
    project: str | None = None,
    scope: str = "workspace",
    confidence: float = 1.0,
    ttl_days: int | None = None,
    symbols: list[str] | None = None,
    source_type: str | None = None,
    source_ref: str | None = None,
) -> str:
    """Record a learning — a decision, correction, discovery, pattern, blocker, or convention.

    Learnings persist across sessions and are searchable via relevant_learnings().
    Use this to capture important decisions, corrections from the user, discovered
    patterns, blockers, or coding conventions.

    Args:
        kind: One of 'decision', 'correction', 'discovery', 'pattern', 'blocker', 'convention'
        content: The learning itself (what was decided/discovered/corrected)
        reasoning: Why this learning matters or the context behind it
        project: Project this applies to (omit for cross-project learnings)
        scope: 'workspace', 'project', 'file', or 'symbol'
        confidence: 0.0-1.0 confidence level (default 1.0)
        ttl_days: Auto-expire after N days (omit for permanent)
        symbols: Symbol names this learning relates to
        source_type: 'conversation', 'labbook', 'decisions_md', 'claude_md', 'agent_log'
        source_ref: Reference identifier (session ID, file path, etc.)
    """
    from .learnings import LearningRecord

    ldb = _get_learnings_db()
    rec = LearningRecord(
        kind=kind,
        content=content,
        reasoning=reasoning,
        scope=scope,
        project=project,
        confidence=confidence,
        ttl_days=ttl_days,
    )

    sources = None
    if source_type:
        sources = [{"type": source_type, "ref": source_ref or ""}]

    learning_id = ldb.record_learning(rec, symbols=symbols, sources=sources)
    return json.dumps({"learning_id": learning_id, "status": "recorded"})


@mcp.tool()
def conversation_summary(
    session_id: str,
    task_summary: str,
    project: str | None = None,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """Record a conversation session summary.

    Call at the end of a session to log what was accomplished.

    Args:
        session_id: Unique session identifier
        task_summary: Brief description of what was done
        project: Primary project worked on (if any)
        model: Model used (e.g. 'claude-opus-4-6')
        tokens_in: Input tokens consumed
        tokens_out: Output tokens generated
        cost_usd: Estimated cost in USD
    """
    from .learnings import ConversationRecord

    ldb = _get_learnings_db()
    rec = ConversationRecord(
        session_id=session_id,
        project=project,
        task_summary=task_summary,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
    )
    conv_id = ldb.record_conversation(rec)
    return json.dumps({"conversation_id": conv_id, "status": "recorded"})


@mcp.tool()
def relevant_learnings(
    query: str,
    project: str | None = None,
    kind: str | None = None,
    limit: int = 10,
) -> str:
    """Find relevant learnings using hybrid search (keyword + semantic).

    Searches past decisions, corrections, discoveries, patterns, blockers,
    and conventions. Returns the most relevant learnings for your current context.

    Call this at the START of a session or when making a decision that might
    have been addressed before.

    Args:
        query: What you're looking for (natural language or keywords)
        project: Filter to a specific project (omit for all)
        kind: Filter by kind: 'decision', 'correction', 'discovery', 'pattern', 'blocker', 'convention'
        limit: Max results (default 10)
    """
    ldb = _get_learnings_db()

    # FTS search
    fts_results = ldb.search_fts(query, kind=kind, project=project, limit=limit * 2)

    # Try embedding search if available
    embedding_results = []
    # (Embedding search is a bonus — FTS alone is sufficient)

    if embedding_results:
        results = ldb.hybrid_search(fts_results, embedding_results, limit=limit)
    else:
        results = fts_results[:limit]

    if not results:
        return json.dumps({"results": [], "message": "No relevant learnings found."})

    # Format results
    formatted = []
    for r in results:
        entry = {
            "kind": r["kind"],
            "content": r["content"],
            "project": r["project"],
            "created_at": r["created_at"],
        }
        if r.get("reasoning"):
            entry["reasoning"] = r["reasoning"]
        if r.get("rrf_score"):
            entry["rrf_score"] = r["rrf_score"]
        formatted.append(entry)

    return json.dumps({"results": formatted, "count": len(formatted)}, indent=2)


@mcp.tool()
def learning_stats(
    project: str | None = None,
    days: int | None = None,
) -> str:
    """Get learning statistics — counts by kind over time.

    Shows how many decisions, corrections, discoveries, etc. have been captured.

    Args:
        project: Filter to a specific project (omit for all)
        days: Look back N days (omit for all time)
    """
    ldb = _get_learnings_db()
    s = ldb.stats(project=project, days=days)
    return json.dumps(s, indent=2)


@mcp.tool()
def get_communities(
    project: str | None = None,
    verbose: bool = False,
    limit: int = 25,
    member_limit: int = 5,
    path_prefix: str | None = None,
    layer: str | None = None,
) -> str:
    """Get detected functional communities (module clusters) in the call graph.

    Communities are auto-detected using the Louvain algorithm on call-graph edges.
    Each community has a TF-IDF auto-label, member count, and cohesion score.

    Args:
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("community analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        communities = db.get_communities(
            verbose=verbose,
            limit=limit,
            member_limit=member_limit,
            path_prefix=path_prefix,
            layer=layer,
        )
        db.close()
    else:
        db = _get_db()
        communities = db.get_communities(
            verbose=verbose,
            limit=limit,
            member_limit=member_limit,
            path_prefix=path_prefix,
            layer=layer,
        )

    if not communities:
        return json.dumps({"info": "No communities detected. Run reindex to generate.", "communities": []})

    payload = {
        "community_count": len(communities),
        "communities": communities,
    }
    if project is not None:
        payload["project"] = project
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_community(symbol_name: str, project: str | None = None) -> str:
    """Get the community that a specific symbol belongs to, with all co-members.

    Answers: "What functional module does this symbol belong to?"

    Args:
        symbol_name: Name of the symbol to look up
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("community lookup")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        try:
            sym = db.get_symbol_by_name(symbol_name)
            if sym is None:
                return json.dumps(_community_fallback_payload(db, symbol_name, project=project), indent=2)
            return json.dumps(
                _format_community_payload(db, symbol_name, sym, project=project),
                indent=2,
            )
        finally:
            db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            return json.dumps(_community_fallback_payload(db, symbol_name), indent=2)
        return json.dumps(_format_community_payload(db, symbol_name, sym), indent=2)


@mcp.tool()
def get_execution_flows(
    project: str | None = None,
    verbose: bool = False,
    limit: int = 25,
    max_depth: int | None = None,
    path_prefix: str | None = None,
    layer: str | None = None,
) -> str:
    """Get traced execution flows through the call graph.

    Flows are paths from entry points (like main, run, handle) through the call graph,
    showing how execution moves across functional communities.

    Args:
        project: Project name (required in workspace mode)
    """
    if _is_workspace_mode():
        if not project:
            return _project_required_error("execution flow analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        flows = db.get_execution_flows(
            limit=limit,
            verbose=verbose,
            max_depth=max_depth,
            path_prefix=path_prefix,
            layer=layer,
        )
        db.close()
    else:
        db = _get_db()
        flows = db.get_execution_flows(
            limit=limit,
            verbose=verbose,
            max_depth=max_depth,
            path_prefix=path_prefix,
            layer=layer,
        )

    if not flows:
        return json.dumps({"info": "No execution flows traced. Run reindex to generate.", "flows": []})

    payload = {
        "flow_count": len(flows),
        "flows": flows,
    }
    if project is not None:
        payload["project"] = project
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_impact(symbol_name: str, project: str | None = None) -> str:
    """Compute blast radius and risk for modifying a symbol.

    Answers: "How risky is it to change this?" Analyzes direct/transitive dependents,
    affected communities, and affected execution flows to assign a risk level
    (LOW / MEDIUM / HIGH / CRITICAL).

    Args:
        symbol_name: Name of the symbol to analyze
        project: Project name (required in workspace mode)
    """
    from .community import compute_impact

    if _is_workspace_mode():
        if not project:
            return _project_required_error("impact analysis")
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        if not path:
            return _project_not_found_error(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            db.close()
            return _symbol_not_found_error(symbol_name, project)
        # Build sym_to_community map from stored data
        communities = db.get_community_records(limit=None)
        sym_to_comm: dict[int, int] = {}
        for c in communities:
            for m in db.get_community_members(c["id"]):
                sym_to_comm[m["id"]] = c["id"]
        flows = db.get_execution_flow_records(limit=None)
        # Reconstruct flow step dicts for compute_impact
        flow_dicts = _reconstruct_flows(db, flows)
        result = compute_impact(db, sym.id, sym_to_comm, flow_dicts)
        db.close()
    else:
        db = _get_db()
        sym = db.get_symbol_by_name(symbol_name)
        if sym is None:
            return _symbol_not_found_error(symbol_name)
        communities = db.get_community_records(limit=None)
        sym_to_comm = {}
        for c in communities:
            for m in db.get_community_members(c["id"]):
                sym_to_comm[m["id"]] = c["id"]
        flows = db.get_execution_flow_records(limit=None)
        flow_dicts = _reconstruct_flows(db, flows)
        result = compute_impact(db, sym.id, sym_to_comm, flow_dicts)

    return json.dumps({
        "symbol": symbol_name,
        "project": project,
        **result,
    }, indent=2)


def _reconstruct_flows(db: Database, stored_flows: list[dict]) -> list[dict]:
    """Reconstruct flow dicts with steps from stored flow data."""
    result = []
    for flow in stored_flows:
        steps = db.get_flow_steps(flow["id"])
        result.append({
            "entry_symbol_id": flow["entry_symbol_id"],
            "terminal_symbol_id": flow["terminal_symbol_id"],
            "label": flow["label"],
            "step_count": flow["step_count"],
            "communities_crossed": flow["communities_crossed"],
            "steps": [{"symbol_id": s["symbol_id"], "community_id": s["community_id"], "order": s["step_order"]} for s in steps],
        })
    return result


@mcp.tool()
def detect_changes(
    ref: str | None = None,
    project: str | None = None,
) -> str:
    """Detect which symbols were changed and compute their aggregate blast radius.

    Maps git diff hunks to indexed symbols, then runs impact analysis on each
    to show what breaks. Call this after editing files or before committing to
    understand the full impact of your changes.

    Args:
        ref: Git ref to diff against (default: uncommitted changes vs HEAD).
             Use "HEAD~1" for last commit's impact, or a branch name.
        project: Project name (required in workspace mode)
    """
    from . import git as git_mod
    from .community import compute_impact

    if _is_workspace_mode() and not project:
        return _project_required_error("detect_changes")

    repo_root = _resolve_repo_root(project)
    if not repo_root:
        return _project_not_found_error(project)

    # Get per-project DB
    if _is_workspace_mode():
        from .workspace import WorkspaceConfig
        config = WorkspaceConfig.load(_workspace_name)
        path = config.projects.get(project)
        db_path = Path(path) / ".srclight" / "index.db"
        if not db_path.exists():
            return json.dumps({"error": f"Project '{project}' not indexed"})
        db = Database(db_path)
        db.open()
    else:
        db = _get_db()

    # Parse diff into changed file/line ranges
    changed_files = git_mod.detect_changes(repo_root, ref=ref)
    if not changed_files:
        if _is_workspace_mode():
            db.close()
        return json.dumps({"info": "No changes detected", "changed_symbols": []})

    # Map hunks to symbols
    changed_symbols: list[dict] = []
    seen_sym_ids: set[int] = set()

    for file_change in changed_files:
        file_path = file_change["file"]
        symbols = db.symbols_in_file(file_path)
        if not symbols:
            continue

        for sym in symbols:
            if sym.id in seen_sym_ids:
                continue
            # Check if any hunk overlaps this symbol's line range
            for hunk in file_change["hunks"]:
                hunk_start = hunk["new_start"]
                hunk_end = hunk_start + max(hunk["new_count"] - 1, 0)
                if hunk_start <= sym.end_line and hunk_end >= sym.start_line:
                    seen_sym_ids.add(sym.id)
                    changed_symbols.append({
                        "id": sym.id,
                        "name": sym.name,
                        "qualified_name": sym.qualified_name,
                        "kind": sym.kind,
                        "file": file_path,
                        "lines": f"{sym.start_line}-{sym.end_line}",
                    })
                    break

    if not changed_symbols:
        if _is_workspace_mode():
            db.close()
        return json.dumps({
            "info": "Changes detected but no indexed symbols affected",
            "changed_files": [f["file"] for f in changed_files],
            "changed_symbols": [],
        })

    # Load community and flow data for impact analysis
    communities = db.get_community_records(limit=None)
    sym_to_comm: dict[int, int] = {}
    for c in communities:
        for m in db.get_community_members(c["id"]):
            sym_to_comm[m["id"]] = c["id"]

    flows = db.get_execution_flow_records(limit=None)
    flow_dicts = _reconstruct_flows(db, flows)

    # Run impact analysis on each changed symbol
    all_affected_comms: set[int] = set()
    all_affected_flows: set[str] = set()
    total_direct = 0
    total_transitive = 0
    max_risk = "LOW"
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    symbol_impacts: list[dict] = []

    for sym_info in changed_symbols:
        impact = compute_impact(db, sym_info["id"], sym_to_comm, flow_dicts)
        sym_info["risk"] = impact["risk"]
        sym_info["direct_dependents"] = impact["direct_dependents"]
        sym_info["affected_flows"] = impact["affected_flows"]
        symbol_impacts.append(sym_info)

        total_direct += impact["direct_dependents"]
        total_transitive += impact["transitive_dependents"]
        all_affected_comms.update(impact["affected_communities"])
        all_affected_flows.update(impact["affected_flows"])
        if risk_order.get(impact["risk"], 0) > risk_order.get(max_risk, 0):
            max_risk = impact["risk"]

    if _is_workspace_mode():
        db.close()

    # Sort by risk descending
    symbol_impacts.sort(key=lambda s: risk_order.get(s["risk"], 0), reverse=True)

    return json.dumps({
        "project": project,
        "ref": ref or "HEAD (uncommitted)",
        "overall_risk": max_risk,
        "changed_symbol_count": len(symbol_impacts),
        "total_direct_dependents": total_direct,
        "total_transitive_dependents": total_transitive,
        "communities_affected": len(all_affected_comms),
        "flows_affected": len(all_affected_flows),
        "changed_symbols": symbol_impacts,
    }, indent=2)


def make_sse_and_streamable_http_app(mount_path: str | None = "/"):
    """Return a Starlette app serving both SSE and Streamable HTTP on one port (Cursor compatibility)."""
    streamable_app = mcp.streamable_http_app()
    sse_app = mcp.sse_app(mount_path=mount_path)
    sse_routes = [r for r in sse_app.routes if getattr(r, "path", None) in ("/sse", "/messages")]
    streamable_app.router.routes.extend(sse_routes)
    return streamable_app


def configure(db_path: Path | None = None, repo_root: Path | None = None) -> None:
    """Configure the server for single-repo mode."""
    global _db_path, _repo_root, _db, _vector_cache
    if _db is not None:
        _db.close()
        _db = None
    _vector_cache = None
    _db_path = db_path
    _repo_root = repo_root
    _refresh_instructions()


def configure_workspace(workspace_name: str) -> None:
    """Configure the server for workspace (multi-repo) mode."""
    global _workspace_name, _workspace_db, _learnings_db
    _workspace_name = workspace_name
    if _workspace_db is not None:
        _workspace_db.close()
        _workspace_db = None
    if _learnings_db is not None:
        _learnings_db.close()
        _learnings_db = None
    _refresh_instructions()


def run_server(transport: str = "sse", port: int = 8742):
    """Start the MCP server."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    if transport == "sse":
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = port
    mcp.run(transport=transport)
