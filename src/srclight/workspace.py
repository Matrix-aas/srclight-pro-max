"""Workspace management for multi-repo indexing.

A workspace groups multiple repos under one name. Each repo has its own
.srclight/index.db. At query time, we ATTACH all per-repo databases to a
:memory: connection and UNION across them — same pattern as MultiDict in
intuition/nomad-builder.

Config lives at ~/.srclight/workspaces/{name}.json
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("srclight.workspace")

_LEGACY_DIR = Path.home() / ".codelight"
_NEW_DIR = Path.home() / ".srclight"

# Auto-migrate ~/.codelight/ → ~/.srclight/ on first access
if _LEGACY_DIR.exists() and not _NEW_DIR.exists():
    try:
        _LEGACY_DIR.rename(_NEW_DIR)
        logger.info("Migrated %s -> %s", _LEGACY_DIR, _NEW_DIR)
    except OSError as e:
        logger.warning("Could not migrate %s -> %s: %s", _LEGACY_DIR, _NEW_DIR, e)

WORKSPACES_DIR = _NEW_DIR / "workspaces"


@dataclass
class ProjectEntry:
    """A project within a workspace."""
    name: str
    path: str  # Absolute path to repo root

    @property
    def index_db(self) -> Path:
        return Path(self.path) / ".srclight" / "index.db"

    @property
    def has_index(self) -> bool:
        return self.index_db.exists()


@dataclass
class WorkspaceConfig:
    """Configuration for a workspace (group of repos)."""
    name: str
    projects: dict[str, str] = field(default_factory=dict)  # name -> path

    @property
    def config_path(self) -> Path:
        return WORKSPACES_DIR / f"{self.name}.json"

    @property
    def learnings_db_path(self) -> Path:
        """Path to the workspace-level learnings database."""
        return WORKSPACES_DIR / f"{self.name}_learnings.db"

    def save(self) -> None:
        if not self.name or not re.match(r"^[a-zA-Z0-9_-]+$", self.name):
            raise ValueError(
                f"Invalid workspace name '{self.name}': must be non-empty, "
                "alphanumeric with hyphens/underscores only"
            )
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "projects": self.projects}
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, name: str) -> WorkspaceConfig:
        path = WORKSPACES_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workspace '{name}' not found at {path}")
        data = json.loads(path.read_text())
        return cls(name=data["name"], projects=data.get("projects", {}))

    @classmethod
    def list_all(cls) -> list[str]:
        if not WORKSPACES_DIR.exists():
            return []
        return sorted(
            p.stem for p in WORKSPACES_DIR.glob("*.json")
        )

    def add_project(self, name: str, path: str) -> None:
        self.projects[name] = str(Path(path).resolve())
        self.save()

    def remove_project(self, name: str) -> None:
        self.projects.pop(name, None)
        self.save()

    def get_entries(self) -> list[ProjectEntry]:
        return [ProjectEntry(name=n, path=p) for n, p in sorted(self.projects.items())]


_RESERVED_SCHEMA_NAMES = {"main", "temp", "memory"}


def _sanitize_schema_name(name: str) -> str:
    """Convert a project name to a valid SQLite schema identifier.

    SQLite ATTACH AS names must be valid identifiers.
    Replace hyphens/dots with underscores, strip non-alphanumeric.
    Guards against SQLite reserved schema names (main, temp).
    """
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    if not s:
        s = "_unnamed"
    if s.lower() in _RESERVED_SCHEMA_NAMES:
        s = f"p_{s}"
    return s


MAX_ATTACH = 10  # SQLite default SQLITE_MAX_ATTACHED


class WorkspaceDB:
    """Cross-repo search via ATTACH + UNION.

    Opens :memory: as the primary connection, ATTACHes each project's
    .srclight/index.db, and queries across all with UNION ALL.

    SQLite limits ATTACH to 10 databases. When there are more projects,
    we query in batches: attach up to 10, run queries, detach, attach next batch.
    """

    def __init__(self, workspace: WorkspaceConfig):
        self.workspace = workspace
        self.conn: sqlite3.Connection | None = None
        self._attached: dict[str, str] = {}  # schema_name -> project_name
        self._all_indexable: list[ProjectEntry] = []  # all entries with an index
        self._caches: dict[str, Any] = {}  # project_name -> VectorCache

    def open(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Discover all indexable projects
        self._all_indexable = [
            e for e in self.workspace.get_entries() if e.has_index
        ]
        # Attach first batch
        self._attach_batch(self._all_indexable[:MAX_ATTACH])

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        self._attached.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _detach_all(self) -> None:
        """Detach all currently attached databases."""
        assert self.conn is not None
        for schema in list(self._attached.keys()):
            try:
                self.conn.execute(f"DETACH DATABASE [{schema}]")
            except sqlite3.OperationalError:
                pass
        self._attached.clear()

    def _attach_batch(self, entries: list[ProjectEntry]) -> None:
        """ATTACH a batch of project databases."""
        assert self.conn is not None
        for entry in entries:
            schema = _sanitize_schema_name(entry.name)
            try:
                self.conn.execute(
                    f"ATTACH DATABASE ? AS [{schema}]",
                    (str(entry.index_db),),
                )
                self._attached[schema] = entry.name
                logger.debug("Attached %s as [%s]", entry.index_db, schema)
            except sqlite3.OperationalError as e:
                logger.warning("Failed to attach %s: %s", entry.name, e)

    def _iter_batches(self, project_filter: str | None = None):
        """Yield batches of (schema, project_name) tuples.

        If all indexable projects fit in one batch (<= MAX_ATTACH), yields once
        with the already-attached schemas. Otherwise, detaches and re-attaches
        in batches of MAX_ATTACH.
        """
        entries = self._all_indexable
        if project_filter:
            entries = [e for e in entries if e.name == project_filter]

        if len(entries) <= MAX_ATTACH:
            # Ensure these specific entries are attached
            needed = {_sanitize_schema_name(e.name) for e in entries}
            if not needed.issubset(set(self._attached.keys())):
                self._detach_all()
                self._attach_batch(entries)
            yield list(
                (s, p) for s, p in self._attached.items()
                if project_filter is None or p == project_filter
            )
        else:
            # Need to batch
            for i in range(0, len(entries), MAX_ATTACH):
                batch = entries[i:i + MAX_ATTACH]
                self._detach_all()
                self._attach_batch(batch)
                yield list(self._attached.items())

    @property
    def project_count(self) -> int:
        return len(self._all_indexable)

    @property
    def attached_projects(self) -> dict[str, str]:
        """schema_name -> project_name mapping for currently attached projects."""
        return dict(self._attached)

    def _iter_entries(self, project_filter: str | None = None) -> list[ProjectEntry]:
        entries = self._all_indexable
        if project_filter is not None:
            entries = [entry for entry in entries if entry.name == project_filter]
        return entries

    @contextmanager
    def _open_project_db(self, entry: ProjectEntry):
        from .db import Database

        db = Database(entry.index_db)
        db.open()
        try:
            yield db
        finally:
            db.close()

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in the workspace with their stats."""
        assert self.conn is not None
        results = []
        seen_projects: set[str] = set()

        for batch in self._iter_batches():
            for schema, project_name in sorted(batch, key=lambda x: x[1]):
                if project_name in seen_projects:
                    continue
                seen_projects.add(project_name)
                try:
                    files = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].files"
                    ).fetchone()["n"]
                    symbols = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbols"
                    ).fetchone()["n"]
                    edges = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbol_edges"
                    ).fetchone()["n"]

                    lang_rows = self.conn.execute(
                        f"SELECT language, COUNT(*) as n FROM [{schema}].files "
                        f"GROUP BY language ORDER BY n DESC"
                    ).fetchall()
                    languages = {r["language"] or "unknown": r["n"] for r in lang_rows}

                    entry = next(
                        e for e in self.workspace.get_entries() if e.name == project_name
                    )
                    db_size = entry.index_db.stat().st_size if entry.index_db.exists() else 0

                    results.append({
                        "project": project_name,
                        "path": entry.path,
                        "files": files,
                        "symbols": symbols,
                        "edges": edges,
                        "languages": languages,
                        "db_size_mb": round(db_size / (1024 * 1024), 2),
                    })
                except sqlite3.OperationalError as e:
                    logger.warning("Error reading stats for %s: %s", project_name, e)
                    results.append({
                        "project": project_name,
                        "error": str(e),
                    })

        # Also list unindexed projects
        for entry in self.workspace.get_entries():
            if entry.name not in seen_projects:
                results.append({
                    "project": entry.name,
                    "path": entry.path,
                    "files": 0,
                    "symbols": 0,
                    "indexed": False,
                })

        return results

    def search_symbols(
        self, query: str, kind: str | None = None,
        project: str | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search symbols across projects using the single-repo ranker per DB."""
        assert self.conn is not None

        results: list[dict[str, Any]] = []
        for entry in self._iter_entries(project_filter=project):
            try:
                with self._open_project_db(entry) as db:
                    project_results = db.search_symbols(query, kind=kind, limit=limit * 3)
                    for row in project_results:
                        results.append({"project": entry.name, **row})
            except (OSError, sqlite3.Error) as e:
                logger.warning("Skipping search for %s due to project DB error: %s", entry.name, e)
                continue

        results.sort(
            key=lambda r: (
                r.get("vendored", False),
                r.get("rank", 0),
                r.get("project", ""),
                r.get("name", ""),
            )
        )
        return results[:limit]

    def suggest_symbol_names(
        self, query: str, project: str | None = None, limit: int = 5
    ) -> list[dict[str, str]]:
        """Suggest nearby symbol names across workspace projects."""
        suggestions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for entry in self._iter_entries(project_filter=project):
            try:
                with self._open_project_db(entry) as db:
                    for match in db.suggest_symbol_name_matches(query, limit=limit * 2):
                        name = match["name"]
                        key = (entry.name, name)
                        if key in seen:
                            continue
                        suggestions.append({
                            "project": entry.name,
                            "name": name,
                            "score": match["score"],
                        })
                        seen.add(key)
            except (OSError, sqlite3.Error) as e:
                logger.warning("Skipping suggestions for %s due to project DB error: %s", entry.name, e)
                continue

        suggestions.sort(key=lambda item: (item["score"], item["project"], item["name"]))
        return [
            {"project": item["project"], "name": item["name"]}
            for item in suggestions[:limit]
        ]

    def suggest_file_candidates(
        self, query: str, project: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Suggest likely files across workspace projects using per-project DB indexes."""
        candidates: list[dict[str, Any]] = []

        for entry in self._iter_entries(project_filter=project):
            try:
                with self._open_project_db(entry) as db:
                    project_candidates = db.suggest_file_candidates(query, limit=limit * 2)
                    for order, candidate in enumerate(project_candidates):
                        candidates.append({
                            "project": entry.name,
                            **candidate,
                            "_project_order": order,
                        })
            except (OSError, sqlite3.Error) as e:
                logger.warning("Skipping file suggestions for %s due to project DB error: %s", entry.name, e)
                continue

        candidates.sort(
            key=lambda item: (
                item["_project_order"],
                item["project"],
                item["path"],
            )
        )
        return [
            {k: v for k, v in item.items() if k != "_project_order"}
            for item in candidates[:limit]
        ]

    def list_files(
        self,
        path_prefix: str | None = None,
        *,
        project: str | None = None,
        recursive: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List indexed files across workspace projects."""
        results: list[dict[str, Any]] = []

        for entry in self._iter_entries(project_filter=project):
            remaining = limit - len(results)
            if remaining <= 0:
                break
            try:
                with self._open_project_db(entry) as db:
                    project_files = db.list_files(
                        path_prefix=path_prefix,
                        recursive=recursive,
                        limit=remaining,
                    )
                    for item in project_files:
                        results.append({"project": entry.name, **item})
                        if len(results) >= limit:
                            return results
            except (OSError, sqlite3.Error) as e:
                logger.warning("Skipping list_files for %s due to project DB error: %s", entry.name, e)
                continue

        return results

    def get_file_summary(
        self,
        path: str,
        *,
        project: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Get one file summary, optionally scoped to a single project."""
        summaries: list[dict[str, Any]] = []

        for entry in self._iter_entries(project_filter=project):
            try:
                with self._open_project_db(entry) as db:
                    summary = db.get_file_summary(path)
                    if summary is not None:
                        summaries.append({"project": entry.name, **summary})
            except (OSError, sqlite3.Error) as e:
                logger.warning("Skipping get_file_summary for %s due to project DB error: %s", entry.name, e)
                continue

        if project is not None:
            return summaries[0] if summaries else None
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]
        return summaries

    def codebase_map(self, project: str | None = None) -> dict[str, Any]:
        """Get aggregated stats across all projects (or a single one)."""
        assert self.conn is not None

        if project is not None and not any(entry.name == project for entry in self.workspace.get_entries()):
            raise LookupError(project)

        total_files = 0
        total_symbols = 0
        total_edges = 0
        all_languages: dict[str, int] = {}
        all_kinds: dict[str, int] = {}
        project_summaries: list[dict] = []

        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    files = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].files"
                    ).fetchone()["n"]
                    symbols = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbols"
                    ).fetchone()["n"]
                    edges = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbol_edges"
                    ).fetchone()["n"]

                    total_files += files
                    total_symbols += symbols
                    total_edges += edges

                    for row in self.conn.execute(
                        f"SELECT language, COUNT(*) as n FROM [{schema}].files GROUP BY language"
                    ):
                        lang = row["language"] or "unknown"
                        all_languages[lang] = all_languages.get(lang, 0) + row["n"]

                    for row in self.conn.execute(
                        f"SELECT kind, COUNT(*) as n FROM [{schema}].symbols GROUP BY kind"
                    ):
                        all_kinds[row["kind"]] = all_kinds.get(row["kind"], 0) + row["n"]

                    project_summaries.append({
                        "project": project_name,
                        "files": files,
                        "symbols": symbols,
                        "edges": edges,
                    })
                except sqlite3.OperationalError as e:
                    logger.warning("Error reading %s: %s", project_name, e)

        result = {
            "workspace": self.workspace.name,
            "projects_attached": self.project_count,
            "totals": {
                "files": total_files,
                "symbols": total_symbols,
                "edges": total_edges,
            },
            "languages": dict(sorted(all_languages.items(), key=lambda x: -x[1])),
            "symbol_kinds": dict(sorted(all_kinds.items(), key=lambda x: -x[1])),
            "projects": project_summaries,
        }

        if project is not None:
            from .server import (
                _build_repo_brief,
                _build_start_here,
                _build_topology,
                _detect_framework_hints,
                _find_representative_files,
                _indexed_file_orientation_hints,
                _indexed_orientation_hints,
                _merge_indexed_representative_files,
                _merge_orientation_hints,
            )

            entry = next(
                (candidate for candidate in self.workspace.get_entries() if candidate.name == project),
                None,
            )
            if entry is not None:
                repo_root = Path(entry.path)
                representative_files = _find_representative_files(repo_root)
                indexed_hints = None
                try:
                    with self._open_project_db(entry) as db:
                        indexed_hints = _merge_orientation_hints(
                            _indexed_file_orientation_hints(db.orientation_files()),
                            _indexed_orientation_hints(db.orientation_symbols()),
                        )
                except (OSError, sqlite3.Error, json.JSONDecodeError):
                    indexed_hints = None

                representative_files = _merge_indexed_representative_files(representative_files, indexed_hints)
                extra_signals = set(indexed_hints.get("signals") or []) if indexed_hints else None
                framework_hints = _detect_framework_hints(
                    repo_root,
                    representative_files,
                    extra_signals=extra_signals,
                )
                start_here = _build_start_here(representative_files, framework_hints)
                topology = _build_topology(
                    repo_root,
                    representative_files,
                    framework_hints,
                    indexed_hints=indexed_hints,
                    mode="workspace",
                )
                result.update({
                    "repo_root": str(repo_root),
                    "framework_hints": framework_hints,
                    "representative_files": representative_files,
                    "topology": topology,
                    "start_here": start_here,
                    "brief": _build_repo_brief(framework_hints, start_here, indexed=True),
                })

        return result

    def get_symbol(self, name: str, project: str | None = None) -> list[dict[str, Any]]:
        """Get full symbol details by name across projects."""
        assert self.conn is not None
        results = []

        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    rows = self.conn.execute(
                        f"""SELECT s.*, f.path as file_path
                           FROM [{schema}].symbols s
                           JOIN [{schema}].files f ON s.file_id = f.id
                           WHERE s.name = ?
                           ORDER BY f.path, s.start_line""",
                        (name,),
                    ).fetchall()
                    if not rows:
                        rows = self.conn.execute(
                            f"""SELECT s.*, f.path as file_path
                               FROM [{schema}].symbols s
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE s.name LIKE ? COLLATE NOCASE
                               ORDER BY f.path, s.start_line
                               LIMIT 20""",
                            (f"%{name}%",),
                        ).fetchall()

                    for row in rows:
                        results.append({
                            "project": project_name,
                            "id": row["id"],
                            "name": row["name"],
                            "qualified_name": row["qualified_name"],
                            "kind": row["kind"],
                            "signature": row["signature"],
                            "file": row["file_path"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "content": row["content"],
                            "doc_comment": row["doc_comment"],
                            "line_count": row["line_count"],
                        })
                except sqlite3.OperationalError:
                    pass

        return results

    def _get_project_cache(self, project_name: str):
        """Get or create a VectorCache for a project.

        Returns a loaded VectorCache, or None if no sidecar exists.
        Re-checks for sidecars that may have appeared since last call
        (e.g. after running `srclight index --embed`).
        """
        cached = self._caches.get(project_name)
        if cached is not None and cached.is_loaded():
            return cached

        from .vector_cache import VectorCache

        entry = next(
            (e for e in self._all_indexable if e.name == project_name), None
        )
        if entry is None:
            return None

        srclight_dir = Path(entry.path) / ".srclight"
        cache = VectorCache(srclight_dir)
        if cache.sidecar_exists():
            try:
                cache.load_sidecar()
                self._caches[project_name] = cache
                return cache
            except Exception as e:
                logger.warning("Failed to load sidecar for %s: %s", project_name, e)

        # No sidecar or load failed — return None but don't permanently cache it.
        # Next call will re-check sidecar existence (fast filesystem stat).
        return None

    def vector_search(
        self, query_embedding: bytes, dimensions: int,
        project: str | None = None, kind: str | None = None, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search symbols by cosine similarity across workspace projects.

        Fast path: uses per-project VectorCache when sidecars exist (~3ms/project).
        Slow path: fetches all embeddings from SQLite via ATTACH+UNION.
        """
        assert self.conn is not None

        # Try fast path: per-project cache search + merge
        all_candidates: list[tuple[str, int, float, int]] = []  # (proj, row_idx, sim, sym_id)
        cache_miss_projects: list[str] = []

        entries = self._all_indexable
        if project:
            entries = [e for e in entries if e.name == project]

        for entry in entries:
            cache = self._get_project_cache(entry.name)
            if cache is not None and cache.is_loaded():
                # Trust loaded cache — validity is checked on load and after reindex.
                # Skipping per-query SQLite connect saves ~60ms across 10 projects.
                candidates = cache.search(query_embedding, dimensions, limit * 2, kind)
                for row_idx, sim, sym_id in candidates:
                    all_candidates.append((entry.name, row_idx, sim, sym_id))
            else:
                # No loaded sidecar. Fall back if the project actually has embeddings;
                # otherwise keep the fast path for unindexed projects.
                if self._project_has_embeddings(entry):
                    cache_miss_projects.append(entry.name)

        # If we got cache hits and no misses, use fast enrichment
        if all_candidates and not cache_miss_projects:
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            all_candidates = all_candidates[:limit]
            return self._enrich_workspace_results(all_candidates)

        # Fall back to slow path for any projects without valid caches
        return self._vector_search_slow(
            query_embedding, dimensions, project=project, kind=kind, limit=limit
        )

    def _enrich_workspace_results(
        self, candidates: list[tuple[str, int, float, int]],
    ) -> list[dict[str, Any]]:
        """Fetch full metadata for cache-based search results.

        Groups lookups by project to minimize connection overhead.
        """
        # Group by project
        by_project: dict[str, list[tuple[int, float, int]]] = {}
        for proj_name, row_idx, sim, sym_id in candidates:
            by_project.setdefault(proj_name, []).append((row_idx, sim, sym_id))

        # Fetch metadata per-project (one connection per project)
        enriched: dict[tuple[str, int], dict] = {}
        for proj_name, items in by_project.items():
            entry = next(
                (e for e in self._all_indexable if e.name == proj_name), None
            )
            if entry is None:
                continue
            try:
                proj_conn = sqlite3.connect(str(entry.index_db))
                proj_conn.row_factory = sqlite3.Row
                for _row_idx, sim, sym_id in items:
                    row = proj_conn.execute(
                        """SELECT s.name, s.qualified_name, s.kind, s.signature,
                                  f.path as file_path, s.start_line, s.end_line,
                                  s.line_count, s.doc_comment
                           FROM symbols s
                           JOIN files f ON s.file_id = f.id
                           WHERE s.id = ?""",
                        (sym_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    enriched[(proj_name, sym_id)] = {
                        "project": proj_name,
                        "symbol_id": sym_id,
                        "name": row["name"],
                        "qualified_name": row["qualified_name"],
                        "kind": row["kind"],
                        "signature": row["signature"],
                        "file": row["file_path"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "line_count": row["line_count"],
                        "doc_comment": row["doc_comment"],
                        "similarity": round(sim, 4),
                    }
                proj_conn.close()
            except Exception as e:
                logger.warning("Error enriching results from %s: %s", proj_name, e)

        # Return in original order (sorted by similarity)
        return [
            enriched[(proj_name, sym_id)]
            for proj_name, _row_idx, _sim, sym_id in candidates
            if (proj_name, sym_id) in enriched
        ]

    def _project_has_embeddings(self, entry: ProjectEntry) -> bool:
        """Return True when a project has embeddings worth falling back for."""
        try:
            conn = sqlite3.connect(str(entry.index_db))
            try:
                row = conn.execute("SELECT 1 FROM symbol_embeddings LIMIT 1").fetchone()
                return row is not None
            finally:
                conn.close()
        except (OSError, sqlite3.Error):
            # If we cannot inspect the DB, prefer correctness and fall back.
            return True

    def _vector_search_slow(
        self, query_embedding: bytes, dimensions: int,
        project: str | None = None, kind: str | None = None, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Slow path: fetch all embeddings from SQLite via ATTACH+UNION."""
        assert self.conn is not None
        import struct

        from .vector_math import cosine_top_k, decode_matrix

        n_floats = len(query_embedding) // 4
        query_vec = struct.unpack(f'{n_floats}f', query_embedding)

        all_rows: list[tuple[str, Any]] = []
        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    table_check = self.conn.execute(
                        f"SELECT name FROM [{schema}].sqlite_master "
                        f"WHERE type='table' AND name='symbol_embeddings'"
                    ).fetchone()
                    if not table_check:
                        continue

                    if kind:
                        rows = self.conn.execute(
                            f"""SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                                      s.kind, s.signature, f.path as file_path,
                                      s.start_line, s.end_line, s.line_count, s.doc_comment
                               FROM [{schema}].symbol_embeddings e
                               JOIN [{schema}].symbols s ON e.symbol_id = s.id
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE e.dimensions = ? AND s.kind = ?""",
                            (dimensions, kind),
                        ).fetchall()
                    else:
                        rows = self.conn.execute(
                            f"""SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                                      s.kind, s.signature, f.path as file_path,
                                      s.start_line, s.end_line, s.line_count, s.doc_comment
                               FROM [{schema}].symbol_embeddings e
                               JOIN [{schema}].symbols s ON e.symbol_id = s.id
                               JOIN [{schema}].files f ON s.file_id = f.id
                               WHERE e.dimensions = ?""",
                            (dimensions,),
                        ).fetchall()

                    for row in rows:
                        all_rows.append((project_name, row))
                except sqlite3.OperationalError as e:
                    logger.warning("Vector search error in %s: %s", project_name, e)

        if not all_rows:
            return []

        blobs = [row["embedding"] for _, row in all_rows]
        matrix = decode_matrix(blobs, n_floats)
        top_k = cosine_top_k(query_vec, matrix, limit)

        results = []
        for idx, sim in top_k:
            proj, row = all_rows[idx]
            results.append({
                "project": proj,
                "symbol_id": row["symbol_id"],
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "kind": row["kind"],
                "signature": row["signature"],
                "file": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "line_count": row["line_count"],
                "doc_comment": row["doc_comment"],
                "similarity": round(sim, 4),
            })
        return results

    def embedding_stats(self, project: str | None = None) -> dict[str, Any]:
        """Get embedding statistics across workspace projects."""
        assert self.conn is not None
        total_symbols = 0
        total_embedded = 0
        model = None
        dimensions = None

        for batch in self._iter_batches(project_filter=project):
            for schema, project_name in batch:
                try:
                    n_sym = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbols"
                    ).fetchone()["n"]
                    total_symbols += n_sym

                    # Check if embeddings table exists
                    table_check = self.conn.execute(
                        f"SELECT name FROM [{schema}].sqlite_master "
                        f"WHERE type='table' AND name='symbol_embeddings'"
                    ).fetchone()
                    if not table_check:
                        continue

                    n_emb = self.conn.execute(
                        f"SELECT COUNT(*) as n FROM [{schema}].symbol_embeddings"
                    ).fetchone()["n"]
                    total_embedded += n_emb

                    if model is None and n_emb > 0:
                        row = self.conn.execute(
                            f"SELECT model, dimensions FROM [{schema}].symbol_embeddings LIMIT 1"
                        ).fetchone()
                        if row:
                            model = row["model"]
                            dimensions = row["dimensions"]
                except sqlite3.OperationalError:
                    pass

        return {
            "total_symbols": total_symbols,
            "embedded_symbols": total_embedded,
            "coverage_pct": round(total_embedded / total_symbols * 100, 1) if total_symbols else 0,
            "model": model,
            "dimensions": dimensions,
        }
