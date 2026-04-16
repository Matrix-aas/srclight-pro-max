"""SQLite database layer for Srclight.

Schema: files, symbols, 3x FTS5 indexes, symbol_edges, index_state.
External content FTS5 with trigger sync.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .vector_math import cosine_top_k, decode_matrix

# Paths that indicate vendored/third-party code
VENDORED_PREFIXES = ("third_party/", "third-party/", "vendor/", "ext/", "depends/")
DOC_PATH_HINTS = ("docs/",)
DOC_FILENAMES = {"readme.md", "claude.md", "agents.md", "contributing.md"}


def is_vendored_path(path: str) -> bool:
    """Check if a file path is in a vendored/third-party directory."""
    return any(path.startswith(p) or f"/{p}" in path for p in VENDORED_PREFIXES)


def is_documentation_path(path: str) -> bool:
    """Check if a file path is primarily documentation-oriented."""
    lower = path.lower()
    return (
        lower.endswith(".md")
        or any(lower.startswith(prefix) for prefix in DOC_PATH_HINTS)
        or Path(lower).name in DOC_FILENAMES
    )


def is_code_like_query(query: str) -> bool:
    """Heuristic for queries that are likely targeting code, not prose docs."""
    query_lower = query.lower()
    if re.search(r"\b(?:use|define)[A-Z]", query):
        return True

    keywords = {
        "css", "emits", "fetch", "graphql", "i18n", "locale", "module", "mutation",
        "nuxt", "props", "query", "route", "router", "store", "subscription",
        "template", "vue",
    }
    matched = sum(1 for token in keywords if token in query_lower)
    return matched >= 2


def split_identifier(name: str) -> str:
    """Split a code identifier into searchable tokens.

    Handles CamelCase, snake_case, :: qualifiers, and mixed styles.
    Examples:
        "SQLiteDictionary" -> "SQLite Dictionary sqlite dictionary"
        "get_callers" -> "get callers"
        "OCRManager" -> "OCR Manager ocr manager"
        "myapp::util::ConfigManager" -> "myapp util Config Manager config manager"
    """
    if not name:
        return ""

    # Split on :: and -> first (C++ qualifiers)
    parts = re.split(r"::|->|\.", name)

    tokens = []
    for part in parts:
        # Split on underscores
        sub_parts = part.split("_")
        for sp in sub_parts:
            if not sp:
                continue
            # Split CamelCase with proper handling of acronyms:
            # "SQLiteDict" -> ["SQLite", "Dict"]
            # "OCRManager" -> ["OCR", "Manager"]
            # "getHTTPSUrl" -> ["get", "HTTPS", "Url"]
            # Step 1: insert boundary between lowercase/digit and uppercase
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", sp)
            # Step 2: insert boundary between acronym and next word
            # "SQLite" -> keep as-is, "OCRManager" -> "OCR Manager"
            s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
            camel_parts = s.split()
            tokens.extend(p for p in camel_parts if p)

    # Return both original tokens and lowercased for case-insensitive matching
    result_parts = []
    for t in tokens:
        result_parts.append(t)
    # Add lowercased versions
    lower_parts = [t.lower() for t in tokens if t.lower() != t]
    result_parts.extend(lower_parts)

    return " ".join(result_parts)


def tokenized_query_hint(query: str) -> str | None:
    """Return a spaced identifier variant for compact code-like queries."""
    if not query:
        return None

    raw_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", query)]
    split_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", split_identifier(query))]
    if not split_tokens:
        return None

    deduped: list[str] = []
    seen: set[str] = set()
    for token in split_tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)

    variant = " ".join(deduped).strip()
    if not variant:
        return None

    raw_normalized = " ".join(raw_tokens)
    if variant == raw_normalized and not re.search(r"[^A-Za-z0-9\s]", query):
        return None
    return variant


def _search_query_variants(query: str) -> list[str]:
    """Return a small set of retrieval-friendly query variants."""
    variants: list[str] = []
    seen: set[str] = set()

    def _add(text: str | None) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    query_tokens = _search_query_tokens(query)

    _add(query)
    _add(split_identifier(query))
    _add(tokenized_query_hint(query))

    if "rmq" in query_tokens:
        _add(re.sub(r"\brmq\b", "rabbitmq", query, flags=re.IGNORECASE))
        _add("rabbitmq consumer handler")
    if "rabbitmq" in query_tokens:
        _add("rabbitmq consumer handler")
    if "message_pattern" in query_tokens:
        _add("message pattern")
        _add("messagepattern")
        _add("message pattern handler")
    if "event_handler" in query_tokens or "event" in query_tokens:
        _add("event handler")
    if "queue" in query_tokens or "worker" in query_tokens:
        _add("queue worker")
    if "cron" in query_tokens or "schedule" in query_tokens:
        _add("scheduled job")

    return variants


def _metadata_like_patterns(query: str) -> list[str]:
    """Return narrow metadata LIKE patterns for retrieval fallbacks."""
    patterns: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        normalized = (value or "").strip().lower()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        patterns.append(f"%{normalized}%")

    query_tokens = _search_query_tokens(query)
    tokenized_hint = tokenized_query_hint(query)

    _add(query)
    _add(tokenized_hint)

    if "message_pattern" in query_tokens:
        _add("message_pattern")
        _add("event_pattern")
        _add("pattern")
    if "event_handler" in query_tokens:
        _add("event_pattern")
    if "rmq" in query_tokens:
        _add("rmq")
        _add("rabbitmq")
    if "rabbitmq" in query_tokens:
        _add("rabbitmq")
    if "redis" in query_tokens:
        _add("redis")
    if query_tokens & {"bootstrap", "config", "configuration", "module", "runtime", "setup"}:
        _add("bootstrap")
        _add("config")
        _add("connection_url")

    return patterns


def _search_query_tokens(query: str) -> set[str]:
    """Normalize query text into tokens for ranking heuristics."""
    tokens: set[str] = set()
    for text in (query, split_identifier(query)):
        for token in re.findall(r"[A-Za-z0-9]+", (text or "").lower()):
            if not token:
                continue
            tokens.add(token)
            if token.endswith("ies") and len(token) > 4:
                tokens.add(token[:-3] + "y")
            elif token.endswith("s") and len(token) > 3:
                tokens.add(token[:-1])

    if {"mikro", "orm"} <= tokens:
        tokens.add("mikroorm")
    if {"type", "orm"} <= tokens:
        tokens.add("typeorm")
    if {"graph", "ql"} <= tokens or "graphql" in tokens:
        tokens.add("graphql")
    if "rmq" in tokens:
        tokens.add("rabbitmq")
    if {"message", "pattern"} <= tokens:
        tokens.add("message_pattern")
    if {"event", "handler"} <= tokens:
        tokens.add("event_handler")

    return tokens


def _compact_identifier(text: str) -> str:
    """Normalize an identifier-like string into a compact comparison key."""
    return "".join(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))


def _normalized_token_phrase(text: str) -> str:
    """Normalize arbitrary text into a comparable token phrase."""
    return " ".join(re.findall(r"[A-Za-z0-9]+", (text or "").lower())).strip()


def _escape_like_literal(value: str) -> str:
    """Escape SQL LIKE metacharacters so path prefixes are treated literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _file_embedding_context_hash(summary: str | None, metadata: dict | None) -> str | None:
    """Hash file summary context used by embeddings."""
    if summary is None and metadata is None:
        return None

    payload = {"summary": summary, "metadata": metadata}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


_UNSET = object()

SCHEMA_VERSION = 7

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- File registry (incremental indexing via hash)
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    language TEXT,
    size INTEGER,
    line_count INTEGER,
    summary TEXT,
    metadata TEXT,
    embedding_context_hash TEXT,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Symbols (functions, classes, methods, structs — AST-level chunks)
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT,
    qualified_name TEXT,
    signature TEXT,
    return_type TEXT,
    parameters TEXT,         -- JSON array
    visibility TEXT,
    is_async INTEGER DEFAULT 0,
    is_static INTEGER DEFAULT 0,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    doc_comment TEXT,
    body_hash TEXT,
    line_count INTEGER,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    metadata TEXT,           -- JSON
    UNIQUE(file_id, kind, name, start_line)
);

-- FTS5 Index 1: Symbol names (code-aware tokenization)
-- name_tokens stores split identifiers (CamelCase/snake_case -> words)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_names_fts USING fts5(
    qualified_name,
    name,
    signature,
    name_tokens,
    file_path UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='unicode61'
);

-- FTS5 Index 2: Source code content (trigram for substring matching)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_content_fts USING fts5(
    content,
    file_path UNINDEXED,
    name UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='trigram'
);

-- FTS5 Index 3: Documentation (natural language with stemming)
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_docs_fts USING fts5(
    doc_comment,
    name UNINDEXED,
    file_path UNINDEXED,
    kind UNINDEXED,
    symbol_id UNINDEXED,
    tokenize='porter unicode61'
);

-- Symbol relationships (graph layer)
CREATE TABLE IF NOT EXISTS symbol_edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    metadata TEXT,           -- JSON
    UNIQUE(source_id, target_id, edge_type)
);

-- Indexing state (resumable)
CREATE TABLE IF NOT EXISTS index_state (
    id INTEGER PRIMARY KEY,
    repo_root TEXT UNIQUE NOT NULL,
    last_commit TEXT,
    config_hash TEXT,
    files_indexed INTEGER DEFAULT 0,
    symbols_indexed INTEGER DEFAULT 0,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    indexer_version TEXT
);

-- Embeddings (optional, populated when embed model is configured)
CREATE TABLE IF NOT EXISTS symbol_embeddings (
    symbol_id INTEGER PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    body_hash TEXT,
    embedded_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Regular indexes
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind_name ON symbols(kind, name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON symbol_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON symbol_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON symbol_edges(edge_type);

-- Communities (Louvain clusters of call-graph symbols)
CREATE TABLE IF NOT EXISTS communities (
    id INTEGER PRIMARY KEY,
    label TEXT,
    symbol_count INTEGER DEFAULT 0,
    cohesion REAL,
    keywords TEXT,
    metadata TEXT
);

-- Junction: which symbols belong to which community
CREATE TABLE IF NOT EXISTS symbol_communities (
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    PRIMARY KEY (symbol_id, community_id)
);

CREATE INDEX IF NOT EXISTS idx_sym_comm_community ON symbol_communities(community_id);

-- Execution flows (BFS traces from entry points)
CREATE TABLE IF NOT EXISTS execution_flows (
    id INTEGER PRIMARY KEY,
    label TEXT,
    entry_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    terminal_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
    step_count INTEGER,
    communities_crossed INTEGER DEFAULT 0,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_flows_entry ON execution_flows(entry_symbol_id);

-- Ordered steps in a flow
CREATE TABLE IF NOT EXISTS flow_steps (
    flow_id INTEGER NOT NULL REFERENCES execution_flows(id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    community_id INTEGER,
    PRIMARY KEY (flow_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_flow_steps_symbol ON flow_steps(symbol_id);
"""


@dataclass
class FileRecord:
    id: int | None = None
    path: str = ""
    content_hash: str = ""
    mtime: float = 0.0
    language: str | None = None
    size: int = 0
    line_count: int = 0
    summary: str | None = None
    metadata: dict | None = None
    indexed_at: str | None = None


@dataclass
class SymbolRecord:
    id: int | None = None
    file_id: int = 0
    kind: str = ""
    name: str | None = None
    qualified_name: str | None = None
    signature: str | None = None
    return_type: str | None = None
    parameters: list[dict] | None = None
    visibility: str | None = None
    is_async: bool = False
    is_static: bool = False
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    doc_comment: str | None = None
    body_hash: str | None = None
    line_count: int = 0
    parent_symbol_id: int | None = None
    metadata: dict | None = None
    # Joined fields (not in symbols table directly)
    file_path: str | None = None
    file_summary: str | None = None
    file_summary_metadata: dict | None = None


@dataclass
class EdgeRecord:
    id: int | None = None
    source_id: int = 0
    target_id: int = 0
    edge_type: str = ""
    confidence: float = 1.0
    metadata: dict | None = None


class Database:
    """SQLite database for Srclight index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def initialize(self) -> None:
        """Create all tables and indexes."""
        assert self.conn is not None
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_info (key, value) VALUES (?, ?)",
            ("embedding_cache_version", "0"),
        )
        # Migrate: add indexer_version column if missing (pre-0.10.1 DBs)
        try:
            self.conn.execute("ALTER TABLE index_state ADD COLUMN indexer_version TEXT")
        except Exception:
            pass  # column already exists
        # Migrate v5 -> v6: file summary metadata
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN summary TEXT")
        except Exception:
            pass  # column already exists
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN metadata TEXT")
        except Exception:
            pass  # column already exists
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN embedding_context_hash TEXT")
        except Exception:
            pass  # column already exists
        self._refresh_all_file_embedding_context_hashes()
        # Migrate v4 -> v5: add community/flow tables
        try:
            self.conn.execute("SELECT 1 FROM communities LIMIT 1")
        except Exception:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS communities (
                    id INTEGER PRIMARY KEY, label TEXT,
                    symbol_count INTEGER DEFAULT 0, cohesion REAL,
                    keywords TEXT, metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS symbol_communities (
                    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                    community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                    PRIMARY KEY (symbol_id, community_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sym_comm_community ON symbol_communities(community_id);
                CREATE TABLE IF NOT EXISTS execution_flows (
                    id INTEGER PRIMARY KEY, label TEXT,
                    entry_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
                    terminal_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
                    step_count INTEGER, communities_crossed INTEGER DEFAULT 0, metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_flows_entry ON execution_flows(entry_symbol_id);
                CREATE TABLE IF NOT EXISTS flow_steps (
                    flow_id INTEGER NOT NULL REFERENCES execution_flows(id) ON DELETE CASCADE,
                    step_order INTEGER NOT NULL,
                    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                    community_id INTEGER,
                    PRIMARY KEY (flow_id, step_order)
                );
                CREATE INDEX IF NOT EXISTS idx_flow_steps_symbol ON flow_steps(symbol_id);
            """)
        self.conn.commit()

    # --- Files ---

    def upsert_file(self, rec: FileRecord) -> int:
        """Insert or update a file record. Returns file ID."""
        assert self.conn is not None
        metadata_json = json.dumps(rec.metadata) if rec.metadata is not None else None
        self.conn.execute(
            """INSERT INTO files (path, content_hash, mtime, language, size, line_count, summary, metadata, embedding_context_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   content_hash=excluded.content_hash,
                   mtime=excluded.mtime,
                   language=excluded.language,
                   size=excluded.size,
                   line_count=excluded.line_count,
                   summary=COALESCE(excluded.summary, files.summary),
                   metadata=COALESCE(excluded.metadata, files.metadata),
                   embedding_context_hash=COALESCE(excluded.embedding_context_hash, files.embedding_context_hash),
                   indexed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (
                rec.path,
                rec.content_hash,
                rec.mtime,
                rec.language,
                rec.size,
                rec.line_count,
                rec.summary,
                metadata_json,
                _file_embedding_context_hash(rec.summary, rec.metadata),
            ),
        )
        # lastrowid is unreliable for ON CONFLICT DO UPDATE — fetch the actual ID
        row = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (rec.path,)
        ).fetchone()
        self._refresh_file_embedding_context_hash(row["id"])
        return row["id"]

    def get_file(self, path: str) -> FileRecord | None:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return self._row_to_file(row)

    def get_file_by_id(self, file_id: int) -> FileRecord | None:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_file(row)

    def _row_to_file(self, row: sqlite3.Row) -> FileRecord:
        data = {k: row[k] for k in row.keys()}
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        valid_fields = {f.name for f in fields(FileRecord)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return FileRecord(**data)

    def file_needs_reindex(self, path: str, content_hash: str) -> bool:
        """Check if file needs re-indexing by comparing content hash."""
        existing = self.get_file(path)
        if existing is None:
            return True
        return existing.content_hash != content_hash

    def delete_file(self, file_id: int) -> None:
        """Delete a file and all its symbols (CASCADE)."""
        assert self.conn is not None
        # First remove FTS entries for symbols in this file
        symbols = self.conn.execute(
            "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        for sym in symbols:
            self._delete_symbol_fts(sym["id"])
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def all_file_paths(self) -> set[str]:
        assert self.conn is not None
        rows = self.conn.execute("SELECT path FROM files").fetchall()
        return {row["path"] for row in rows}

    def list_files(
        self,
        path_prefix: str | None = None,
        *,
        recursive: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List indexed files, optionally filtered by a path prefix."""
        assert self.conn is not None

        normalized_prefix = (path_prefix or "").strip().strip("/")
        params: list[Any] = []
        query = """
            SELECT path, language, size, line_count, summary
            FROM files
        """
        if normalized_prefix:
            escaped_prefix = _escape_like_literal(normalized_prefix)
            query += " WHERE path LIKE ? ESCAPE '\\' COLLATE NOCASE"
            params.append(f"{escaped_prefix}/%")
            if not recursive:
                query += " AND path NOT LIKE ? ESCAPE '\\' COLLATE NOCASE"
                params.append(f"{escaped_prefix}/%/%")
        elif not recursive:
            query += " WHERE path NOT LIKE '%/%'"

        query += " ORDER BY path LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        files: list[dict[str, Any]] = []
        for row in rows:
            files.append({
                "path": row["path"],
                "language": row["language"],
                "size": row["size"],
                "line_count": row["line_count"],
                "summary": row["summary"],
            })
        return files

    def update_file_summary(
        self,
        path: str,
        *,
        summary: str | None | object = _UNSET,
        metadata: dict | None | object = _UNSET,
    ) -> None:
        """Persist lightweight summary fields for an indexed file."""
        assert self.conn is not None
        updates: list[str] = []
        params: list[Any] = []

        if summary is not _UNSET:
            updates.append("summary = ?")
            params.append(summary)
        if metadata is not _UNSET:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata) if metadata is not None else None)
        if not updates:
            return

        params.append(path)
        self.conn.execute(
            f"UPDATE files SET {', '.join(updates)} WHERE path = ?",
            params,
        )
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is not None:
            self._refresh_file_embedding_context_hash(row["id"])

    def get_file_summary(self, path: str) -> dict[str, Any] | None:
        """Return lightweight summary metadata and top-level symbols for a file."""
        file_rec = self.get_file(path)
        if file_rec is None or file_rec.id is None:
            return None

        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT name, kind, signature, start_line, end_line
               FROM symbols
               WHERE file_id = ? AND parent_symbol_id IS NULL
               ORDER BY start_line, id""",
            (file_rec.id,),
        ).fetchall()

        return {
            "file": file_rec.path,
            "language": file_rec.language,
            "size": file_rec.size,
            "line_count": file_rec.line_count,
            "summary": file_rec.summary,
            "metadata": file_rec.metadata,
            "top_level_symbols": [
                {
                    "name": row["name"],
                    "kind": row["kind"],
                    "signature": row["signature"],
                    "line": row["start_line"],
                    "end_line": row["end_line"],
                }
                for row in rows
            ],
        }

    def _refresh_file_embedding_context_hash(self, file_id: int) -> None:
        """Persist the current embedding context hash for a file."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT summary, metadata FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            return

        metadata = row["metadata"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = None

        context_hash = _file_embedding_context_hash(row["summary"], metadata)
        self.conn.execute(
            "UPDATE files SET embedding_context_hash = ? WHERE id = ?",
            (context_hash, file_id),
        )

    def _refresh_all_file_embedding_context_hashes(self) -> None:
        """Backfill file embedding context hashes after migrations."""
        assert self.conn is not None
        rows = self.conn.execute(
            """
            SELECT id, summary, metadata
            FROM files
            WHERE (summary IS NOT NULL OR metadata IS NOT NULL)
              AND embedding_context_hash IS NULL
            """
        ).fetchall()
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            context_hash = _file_embedding_context_hash(row["summary"], metadata)
            self.conn.execute(
                "UPDATE files SET embedding_context_hash = ? WHERE id = ?",
                (context_hash, row["id"]),
            )

    def orientation_files(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return high-signal indexed files for repo orientation."""
        assert self.conn is not None

        row_limit = max(limit * 3, limit)
        rows = self.conn.execute(
            """SELECT id, path, language, size, line_count, summary, metadata
               FROM files
               WHERE summary IS NOT NULL
                  OR metadata IS NOT NULL
               ORDER BY
                   CASE
                       WHEN metadata IS NOT NULL AND summary IS NOT NULL THEN 0
                       WHEN metadata IS NOT NULL THEN 1
                       WHEN summary IS NOT NULL THEN 2
                       ELSE 3
                   END,
                   path
               LIMIT ?""",
            (row_limit,),
        ).fetchall()
        if not rows:
            return []

        file_ids = [int(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in file_ids)
        symbol_rows = self.conn.execute(
            f"""SELECT file_id, name, kind, signature, start_line, end_line
                FROM symbols
                WHERE parent_symbol_id IS NULL
                  AND file_id IN ({placeholders})
                ORDER BY file_id, start_line, id""",
            file_ids,
        ).fetchall()

        top_level_symbols: dict[int, list[dict[str, Any]]] = {}
        for row in symbol_rows:
            bucket = top_level_symbols.setdefault(int(row["file_id"]), [])
            if len(bucket) >= 3:
                continue
            bucket.append({
                "name": row["name"],
                "kind": row["kind"],
                "signature": row["signature"],
                "line": row["start_line"],
                "end_line": row["end_line"],
            })

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            results.append({
                "path": row["path"],
                "language": row["language"],
                "size": row["size"],
                "line_count": row["line_count"],
                "summary": row["summary"],
                "metadata": metadata or {},
                "top_level_symbols": top_level_symbols.get(int(row["id"]), []),
            })
            if len(results) >= limit:
                break

        return results

    # --- Symbols ---

    def insert_symbol(self, rec: SymbolRecord, file_path: str) -> int:
        """Insert a symbol and its FTS entries. Returns symbol ID."""
        assert self.conn is not None
        params_json = json.dumps(rec.parameters) if rec.parameters else None
        meta_json = json.dumps(rec.metadata) if rec.metadata else None

        cur = self.conn.execute(
            """INSERT OR REPLACE INTO symbols
               (file_id, kind, name, qualified_name, signature, return_type,
                parameters, visibility, is_async, is_static,
                start_line, end_line, content, doc_comment, body_hash,
                line_count, parent_symbol_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.file_id, rec.kind, rec.name, rec.qualified_name,
                rec.signature, rec.return_type, params_json, rec.visibility,
                int(rec.is_async), int(rec.is_static),
                rec.start_line, rec.end_line, rec.content, rec.doc_comment,
                rec.body_hash, rec.line_count, rec.parent_symbol_id, meta_json,
            ),
        )
        symbol_id = cur.lastrowid

        # Insert into all 3 FTS tables
        name_tokens = split_identifier(rec.name) if rec.name else ""
        self.conn.execute(
            """INSERT INTO symbol_names_fts
               (rowid, qualified_name, name, signature, name_tokens, file_path, kind, symbol_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol_id, rec.qualified_name or "", rec.name or "", rec.signature or "",
             name_tokens, file_path, rec.kind, str(symbol_id)),
        )

        self.conn.execute(
            """INSERT INTO symbol_content_fts
               (rowid, content, file_path, name, kind, symbol_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (symbol_id, rec.content, file_path, rec.name or "", rec.kind, str(symbol_id)),
        )

        if rec.doc_comment:
            self.conn.execute(
                """INSERT INTO symbol_docs_fts
                   (rowid, doc_comment, name, file_path, kind, symbol_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (symbol_id, rec.doc_comment, rec.name or "", file_path, rec.kind,
                 str(symbol_id)),
            )

        return symbol_id

    def _delete_symbol_fts(self, symbol_id: int) -> None:
        """Remove a symbol's FTS entries."""
        assert self.conn is not None
        for table in ("symbol_names_fts", "symbol_content_fts", "symbol_docs_fts"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE rowid = ?", (symbol_id,)
            )

    def delete_symbols_for_file(self, file_id: int) -> None:
        """Delete all symbols for a file (edges, FTS entries, then symbols)."""
        assert self.conn is not None
        symbols = self.conn.execute(
            "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        if not symbols:
            return
        sym_ids = [s["id"] for s in symbols]
        # Explicitly delete edges first (don't rely solely on CASCADE)
        self.delete_edges_for_symbols(sym_ids)
        for sid in sym_ids:
            self._delete_symbol_fts(sid)
        self.conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))

    def get_symbol_by_name(self, name: str) -> SymbolRecord | None:
        """Get first symbol matching exact name. Use get_symbols_by_name for all matches."""
        assert self.conn is not None
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name = ? LIMIT 1""",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def get_symbols_by_name(self, name: str, limit: int = 20) -> list[SymbolRecord]:
        """Get all symbols matching exact name, with LIKE fallback."""
        assert self.conn is not None

        # Try exact match first
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name = ?
               ORDER BY f.path, s.start_line
               LIMIT ?""",
            (name, limit),
        ).fetchall()

        if rows:
            return [self._row_to_symbol(r) for r in rows]

        # Fallback: case-insensitive LIKE match
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name LIKE ? COLLATE NOCASE
               ORDER BY
                   CASE WHEN s.name = ? THEN 0
                        WHEN s.name LIKE ? THEN 1
                        ELSE 2 END,
                   f.path, s.start_line
               LIMIT ?""",
            (f"%{name}%", name, f"{name}%", limit),
        ).fetchall()

        return [self._row_to_symbol(r) for r in rows]

    def orientation_symbols(self, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch a bounded, category-aware sample of orientation-relevant symbols."""
        assert self.conn is not None
        chunk = max(24, limit // 5)
        bucket_queries = (
            (
                """SELECT s.kind, s.name, s.signature, s.metadata, f.path AS file_path
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE f.path IN ('src/main.ts', 'src/main.js', 'main.ts', 'main.js', 'server.ts')
                      OR json_extract(s.metadata, '$.resource') = 'bootstrap'
                   ORDER BY CASE
                        WHEN json_extract(s.metadata, '$.resource') = 'bootstrap' THEN 0
                        WHEN f.path IN ('src/main.ts', 'src/main.js', 'main.ts', 'main.js', 'server.ts') THEN 1
                        ELSE 2
                   END, f.path, s.start_line
                   LIMIT ?""",
                chunk,
            ),
            (
                """SELECT s.kind, s.name, s.signature, s.metadata, f.path AS file_path
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.kind IN ('controller', 'route')
                      OR json_extract(s.metadata, '$.resource') IN ('controller', 'route')
                      OR json_extract(s.metadata, '$.route_prefix') IS NOT NULL
                      OR json_extract(s.metadata, '$.route_path') IS NOT NULL
                      OR json_extract(s.metadata, '$.http_method') IS NOT NULL
                   ORDER BY CASE
                        WHEN s.kind = 'controller' THEN 0
                        WHEN json_extract(s.metadata, '$.resource') = 'controller' THEN 1
                        ELSE 2
                   END, f.path, s.start_line
                   LIMIT ?""",
                chunk,
            ),
            (
                """SELECT s.kind, s.name, s.signature, s.metadata, f.path AS file_path
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE lower(COALESCE(json_extract(s.metadata, '$.framework'), '')) IN ('prisma', 'drizzle', 'mongoose', 'mikroorm')
                      OR lower(COALESCE(json_extract(s.metadata, '$.resource'), '')) IN ('database', 'model', 'entity', 'repository')
                   ORDER BY f.path, s.start_line
                   LIMIT ?""",
                chunk,
            ),
            (
                """SELECT s.kind, s.name, s.signature, s.metadata, f.path AS file_path
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.kind IN ('queue_processor', 'microservice_handler', 'scheduled_job')
                      OR lower(COALESCE(json_extract(s.metadata, '$.resource'), '')) IN ('processor', 'consumer', 'worker', 'queue', 'scheduler')
                      OR lower(COALESCE(json_extract(s.metadata, '$.framework'), '')) IN ('bullmq', 'rabbitmq', 'redis')
                      OR lower(COALESCE(json_extract(s.metadata, '$.transport'), '')) IN ('bullmq', 'rabbitmq', 'redis')
                   ORDER BY CASE
                        WHEN s.kind IN ('queue_processor', 'microservice_handler', 'scheduled_job') THEN 0
                        ELSE 1
                   END, f.path, s.start_line
                   LIMIT ?""",
                chunk,
            ),
            (
                """SELECT s.kind, s.name, s.signature, s.metadata, f.path AS file_path
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE lower(COALESCE(json_extract(s.metadata, '$.resource'), '')) IN ('module', 'config', 'bootstrap')
                      OR s.kind IN ('module', 'config')
                      OR f.path LIKE '%/bootstrap/%'
                   ORDER BY CASE
                        WHEN lower(COALESCE(json_extract(s.metadata, '$.resource'), '')) = 'module' THEN 0
                        WHEN s.kind = 'module' THEN 1
                        WHEN lower(COALESCE(json_extract(s.metadata, '$.resource'), '')) = 'config' THEN 2
                        ELSE 3
                   END, f.path, s.start_line
                   LIMIT ?""",
                chunk,
            ),
        )

        rows = []
        seen: set[tuple[str, str, str]] = set()
        for query, bucket_limit in bucket_queries:
            for row in self.conn.execute(query, (bucket_limit,)).fetchall():
                row_key = (row["file_path"], row["kind"], row["name"])
                if row_key in seen:
                    continue
                seen.add(row_key)
                rows.append(row)
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            results.append({
                "kind": row["kind"],
                "name": row["name"],
                "signature": row["signature"],
                "file_path": row["file_path"],
                "metadata": metadata or {},
            })
        return results

    def suggest_symbol_name_matches(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Suggest nearby symbol names with ranking metadata."""
        assert self.conn is not None
        tokens = [token for token in re.findall(r"[A-Za-z0-9]+", split_identifier(query).lower()) if token]
        if not tokens:
            return []

        query_tokens = _search_query_tokens(query)
        compact_query = _compact_identifier(query)
        compact_pattern = "%" + "%".join(dict.fromkeys(tokens[:4])) + "%"
        clauses = ["s.name LIKE ? COLLATE NOCASE"]
        params: list[Any] = [compact_pattern]

        for token in dict.fromkeys(tokens[:4]):
            clauses.append("s.name LIKE ? COLLATE NOCASE")
            params.append(f"%{token}%")

        rows = self.conn.execute(
            f"""SELECT DISTINCT s.name
               FROM symbols s
               WHERE {" OR ".join(clauses)}
               ORDER BY
                   CASE WHEN s.name = ? THEN 0
                        WHEN s.name LIKE ? COLLATE NOCASE THEN 1
                        WHEN s.name LIKE ? COLLATE NOCASE THEN 2
                        ELSE 3 END,
                   length(s.name),
                   s.name
               LIMIT ?""",
            params + [query, f"{query}%", compact_pattern, limit * 4],
        ).fetchall()

        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            name = row["name"]
            if name and name not in seen:
                name_tokens = _search_query_tokens(name)
                compact_name = _compact_identifier(name)
                missing_tokens = len(query_tokens - name_tokens)
                overlap = len(query_tokens & name_tokens)
                score = (
                    missing_tokens,
                    0 if compact_name.startswith(compact_query) else 1 if compact_query and compact_query in compact_name else 2,
                    -overlap,
                    abs(len(name_tokens) - len(query_tokens)),
                    len(name),
                    name.lower(),
                )
                matches.append({"name": name, "score": score})
                seen.add(name)
            if len(matches) >= limit:
                break

        matches.sort(key=lambda item: item["score"])
        return matches[:limit]

    def suggest_symbol_names(self, query: str, limit: int = 5) -> list[str]:
        """Suggest nearby symbol names for miss hints."""
        return [item["name"] for item in self.suggest_symbol_name_matches(query, limit=limit)]

    def suggest_file_candidates(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Suggest likely symbol-bearing files for a query.

        Uses indexed file rows only and prefers deterministic path/name matches
        over broad scans of the filesystem.
        """
        assert self.conn is not None

        raw_query = (query or "").strip()
        compact_query = _compact_identifier(raw_query)
        normalized_query = _normalized_token_phrase(raw_query)
        query_tokens = _search_query_tokens(raw_query)
        if not compact_query and not normalized_query and not query_tokens:
            return []

        exact_filename_patterns: list[str] = []
        for text in dict.fromkeys(filter(None, [raw_query, tokenized_query_hint(raw_query)])):
            filename = "".join(re.findall(r"[A-Za-z0-9]+", text))
            if filename:
                exact_filename_patterns.extend([f"%/{filename}.%", f"{filename}.%"])

        compact_expr = (
            "lower(replace(replace(replace(replace(replace(f.path, '/', ''), '_', ''), '-', ''), '.', ''), ' ', ''))"
        )
        clauses = ["f.path LIKE ? COLLATE NOCASE"]
        params: list[Any] = [f"%{raw_query}%"]
        if compact_query:
            clauses.append(f"{compact_expr} LIKE ?")
            params.append(f"%{compact_query}%")
        for token in list(dict.fromkeys(sorted(query_tokens)))[:4]:
            clauses.append("f.path LIKE ? COLLATE NOCASE")
            params.append(f"%{token}%")

        rows = self.conn.execute(
            f"""SELECT f.path, COUNT(DISTINCT s.id) AS symbol_count
                FROM files f
                JOIN symbols s ON s.file_id = f.id
                WHERE {" OR ".join(clauses)}
                GROUP BY f.id, f.path
                ORDER BY
                    CASE
                        WHEN {" OR ".join("f.path LIKE ? COLLATE NOCASE" for _ in exact_filename_patterns) if exact_filename_patterns else "0"} THEN 0
                        WHEN {compact_expr} LIKE ? THEN 1
                        ELSE 2
                    END,
                    length(f.path),
                    f.path,
                    symbol_count DESC
                LIMIT ?""",
            [
                *params,
                *exact_filename_patterns,
                f"%{compact_query}%" if compact_query else "%",
                max(limit * 8, 24),
            ],
        ).fetchall()

        candidates: list[dict[str, Any]] = []
        for row in rows:
            path = row["path"]
            basename = Path(path).name
            stem = Path(path).stem
            basename_compact = _compact_identifier(basename)
            stem_compact = _compact_identifier(stem)
            path_compact = _compact_identifier(path)
            path_phrase = _normalized_token_phrase(path)
            path_tokens = _search_query_tokens(path)
            overlap = len(query_tokens & path_tokens)

            reasons: list[str] = []
            if compact_query and stem_compact == compact_query:
                reasons.append("exact filename match")
            elif compact_query and basename_compact == compact_query:
                reasons.append("exact basename match")
            elif compact_query and compact_query in basename_compact:
                reasons.append("compact filename match")
            elif compact_query and compact_query in path_compact:
                reasons.append("compact path match")

            if normalized_query and normalized_query in path_phrase:
                reasons.append("token phrase match")
            if overlap:
                reasons.append("token overlap")

            if not reasons:
                continue

            score = (
                0 if "exact filename match" in reasons else
                1 if "exact basename match" in reasons else
                2 if "compact filename match" in reasons else
                3 if "compact path match" in reasons else
                4,
                -overlap,
                -int(row["symbol_count"] or 0),
                len(path),
                path.lower(),
            )
            candidates.append({
                "path": path,
                "symbol_count": int(row["symbol_count"] or 0),
                "match_reason": reasons[0],
                "match_reasons": reasons,
                "_score": score,
            })

        candidates.sort(key=lambda item: item["_score"])
        return [
            {k: v for k, v in candidate.items() if k != "_score"}
            for candidate in candidates[:limit]
        ]

    def _search_rank_context(self, symbol_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch metadata needed for post-FTS ranking."""
        assert self.conn is not None
        if not symbol_ids:
            return {}

        placeholders = ",".join("?" for _ in symbol_ids)
        params = symbol_ids + symbol_ids + symbol_ids
        rows = self.conn.execute(
            f"""WITH outgoing AS (
                   SELECT source_id AS symbol_id, COUNT(*) AS n
                   FROM symbol_edges
                   WHERE source_id IN ({placeholders})
                   GROUP BY source_id
               ),
               incoming AS (
                   SELECT target_id AS symbol_id, COUNT(*) AS n
                   FROM symbol_edges
                   WHERE target_id IN ({placeholders})
                   GROUP BY target_id
               )
               SELECT s.id AS symbol_id,
                      s.metadata AS metadata,
                      COALESCE(outgoing.n, 0) + COALESCE(incoming.n, 0) AS edge_count
               FROM symbols s
               LEFT JOIN outgoing ON outgoing.symbol_id = s.id
               LEFT JOIN incoming ON incoming.symbol_id = s.id
               WHERE s.id IN ({placeholders})""",
            params,
        ).fetchall()

        context: dict[int, dict[str, Any]] = {}
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = None
            context[int(row["symbol_id"])] = {
                "metadata": metadata or {},
                "edge_count": int(row["edge_count"] or 0),
            }
        return context

    def _rerank_search_results(
        self, results: list[dict[str, Any]], query: str, code_like_query: bool
    ) -> None:
        """Apply query-aware boosts after candidate collection."""
        if not results:
            return

        context = self._search_rank_context([int(result["symbol_id"]) for result in results])
        query_tokens = _search_query_tokens(query)
        framework_terms = {
            token
            for token in query_tokens
            if token in {"drizzle", "elysia", "graphql", "mikroorm", "mongoose", "nestjs", "nitro", "nuxt", "typeorm"}
        }
        route_intent = bool(query_tokens & {"api", "endpoint", "http", "path", "route", "router"})
        persistence_intent = bool(
            query_tokens
            & {"database", "db", "drizzle", "entity", "mikroorm", "mongoose", "orm", "persistence", "repository", "schema", "table", "typeorm"}
        )
        async_intent_tokens = {
            "bootstrap",
            "config",
            "configuration",
            "consumer",
            "cron",
            "event",
            "event_handler",
            "handler",
            "message_pattern",
            "queue",
            "rabbitmq",
            "redis",
            "rmq",
            "runtime",
            "schedule",
            "setup",
            "worker",
        }
        async_intent = bool(query_tokens & async_intent_tokens)
        async_transport_query = bool(query_tokens & {"rabbitmq", "redis", "rmq"})
        async_config_intent = async_transport_query and bool(
            query_tokens & {"bootstrap", "config", "configuration", "module", "runtime", "setup"}
        )
        backend_intent = (
            code_like_query or route_intent or persistence_intent or async_intent or bool(framework_terms)
        )

        route_kinds = {"route", "router", "route_handler"}
        persistence_kinds = {"database", "entity", "repository"}
        entrypoint_kinds = route_kinds | {"database", "mutation", "plugin", "query", "subscription"}
        route_paths = ("api/", "controllers/", "endpoints/", "routes/", "server/api/")
        persistence_paths = ("database/", "db/", "entities/", "models/", "persistence/", "repositories/", "schemas/")
        async_kinds = {"microservice_handler", "queue_processor", "scheduled_job"}
        async_support_kinds = {"queue", "transport", "worker"}
        async_paths = (
            "async/",
            "bootstrap",
            "bullmq",
            "config/",
            "configs/",
            "consumer",
            "consumers/",
            "cron",
            "events/",
            "jobs/",
            "messaging/",
            "queues/",
            "scheduler",
            "workers/",
        )
        async_transport_terms = {"amqplib", "rabbitmq", "redis", "rmq"}
        async_config_paths = ("bootstrap", "config/", "configs/", "main.", "module", "runtime")
        query_compact = _compact_identifier(query)
        query_phrase = _normalized_token_phrase(query)
        pattern_query = bool(re.search(r"[./:-]", query)) and len(query_tokens) >= 2

        for row in results:
            rank = float(row.get("rank", 0))
            file_path = str(row.get("file", "")).lower()
            sym_kind = str(row.get("kind", "")).lower()
            name_tokens = _search_query_tokens(row.get("name", "") or "")
            overlap = len(query_tokens & name_tokens)
            if overlap:
                rank -= min(overlap * 0.8, 3.2)

            row_context = context.get(int(row["symbol_id"]), {})
            metadata = row_context.get("metadata") or {}
            framework = re.sub(r"[^a-z0-9]+", "", str(metadata.get("framework") or "").lower())
            resource = str(metadata.get("resource") or "").lower()
            edge_count = int(row_context.get("edge_count") or 0)
            transport = str(metadata.get("transport") or "").lower()
            role = str(metadata.get("role") or "").lower()
            pattern_values = [
                str(metadata.get(key) or "")
                for key in ("pattern", "message_pattern", "event_pattern")
                if metadata.get(key)
            ]
            pattern_compact_matches = any(_compact_identifier(value) == query_compact for value in pattern_values)
            pattern_phrase_matches = any(_normalized_token_phrase(value) == query_phrase for value in pattern_values)
            has_async_metadata = bool(
                pattern_values
                or transport
                or role
                or any(
                    metadata.get(key)
                    for key in ("queue_name", "cron", "every_ms", "interval_name", "connection_url")
                )
            )

            if backend_intent and is_documentation_path(file_path):
                rank += 8.0

            if pattern_query:
                if pattern_compact_matches or pattern_phrase_matches:
                    rank -= 14.0
                elif is_documentation_path(file_path):
                    rank += 6.0

            if route_intent:
                if sym_kind in route_kinds or resource in route_kinds:
                    rank -= 12.0
                elif sym_kind in {"controller", "module"} or resource in {"controller", "module"}:
                    rank -= 4.0
                if any(marker in file_path for marker in route_paths):
                    rank -= 5.0
                if metadata.get("http_method") or metadata.get("route_path") or metadata.get("route_prefix"):
                    rank -= 2.5

            if persistence_intent:
                if sym_kind in persistence_kinds or resource in persistence_kinds:
                    rank -= 12.0
                elif sym_kind in {"schema", "table"} or resource in {"schema", "table"}:
                    rank -= 7.0
                if any(marker in file_path for marker in persistence_paths):
                    rank -= 5.0
                if any(
                    metadata.get(key)
                    for key in ("collection_name", "entity_name", "entity_names", "table_name")
                ):
                    rank -= 2.5

            if async_intent:
                if sym_kind in async_kinds or resource in async_kinds:
                    rank -= 12.0
                elif sym_kind in async_support_kinds or resource in async_support_kinds:
                    rank -= 7.0
                if any(marker in file_path for marker in async_paths):
                    rank -= 5.0
                if any(
                    metadata.get(key)
                    for key in (
                        "event_pattern",
                        "message_pattern",
                        "pattern",
                        "queue_name",
                        "cron",
                        "every_ms",
                        "interval_name",
                    )
                ):
                    rank -= 3.0
                if (transport in async_transport_terms) or (framework in async_transport_terms):
                    rank -= 3.5
                if transport == "rmq":
                    rank -= 1.0
                if role and role in query_tokens:
                    rank -= 2.0
                if "handler" in query_tokens and (sym_kind == "microservice_handler" or resource == "microservice_handler"):
                    rank -= 4.0
                if "worker" in query_tokens and (sym_kind == "queue_processor" or resource == "queue_processor"):
                    rank -= 4.0
                if ("cron" in query_tokens or "schedule" in query_tokens) and (
                    sym_kind == "scheduled_job" or resource == "scheduled_job"
                ):
                    rank -= 4.0
                if pattern_compact_matches or pattern_phrase_matches:
                    rank -= 10.0
                elif pattern_values and ({"message", "pattern"} <= query_tokens or {"event", "pattern"} <= query_tokens):
                    rank -= 4.0
                if async_transport_query and not has_async_metadata and is_documentation_path(file_path):
                    rank += 6.0

            if async_config_intent:
                async_config_candidate = (
                    (transport in async_transport_terms)
                    or (framework in async_transport_terms)
                    or any(term in file_path for term in async_transport_terms)
                )
                if async_config_candidate:
                    if resource in {"module", "transport"} or sym_kind in {"module", "transport"}:
                        rank -= 8.0
                    if any(marker in file_path for marker in async_config_paths):
                        rank -= 6.0
                    if metadata.get("connection_url"):
                        rank -= 3.5
                    if role in {"bootstrap", "client", "config", "producer"}:
                        rank -= 2.5
                elif is_documentation_path(file_path):
                    rank += 4.0

            if framework_terms and framework in framework_terms:
                rank -= 10.0

            if sym_kind in entrypoint_kinds or resource in entrypoint_kinds:
                rank -= 2.5

            if edge_count > 0:
                rank -= min(edge_count, 8) * 0.35

            row["rank"] = rank

    def get_symbol_by_id(self, symbol_id: int) -> SymbolRecord | None:
        assert self.conn is not None
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.id = ?""",
            (symbol_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def symbols_in_file(self, path: str) -> list[SymbolRecord]:
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE f.path = ?
               ORDER BY s.start_line""",
            (path,),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def _row_to_symbol(self, row: sqlite3.Row) -> SymbolRecord:
        d = {k: row[k] for k in row.keys()}
        d["is_async"] = bool(d.get("is_async", 0))
        d["is_static"] = bool(d.get("is_static", 0))
        if isinstance(d.get("parameters"), str):
            d["parameters"] = json.loads(d["parameters"])
        if isinstance(d.get("metadata"), str):
            d["metadata"] = json.loads(d["metadata"])
        if isinstance(d.get("file_summary_metadata"), str):
            d["file_summary_metadata"] = json.loads(d["file_summary_metadata"])
        # Filter to only SymbolRecord fields (joined queries may add extras)
        valid_fields = {f.name for f in fields(SymbolRecord)}
        d = {k: v for k, v in d.items() if k in valid_fields}
        return SymbolRecord(**d)

    # --- Search ---

    def search_symbols(
        self, query: str, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search symbols using FTS5 + LIKE fallback.

        Search tiers:
        1. FTS5 on symbol names (exact + tokenized CamelCase/snake_case)
        2. LIKE fallback on symbol names (substring match)
        3. FTS5 on source code content (trigram)
        4. FTS5 on documentation (porter stemmed)

        Kind filtering is applied inside each tier's query to avoid
        consuming the limit with non-matching results.
        """
        assert self.conn is not None
        results = []
        seen_ids: set[int] = set()

        # High-value kinds that agents typically search for
        _PRIMARY_KINDS = {"class", "struct", "interface", "enum", "function", "method"}
        query_lower = query.lower()
        code_like_query = is_code_like_query(query)

        def _add_row(row_dict: dict) -> None:
            sid = row_dict["symbol_id"]
            if sid not in seen_ids:
                if kind and row_dict["kind"] != kind:
                    return
                rank = row_dict.get("rank", 0)
                name = row_dict.get("name", "")
                sym_kind = row_dict.get("kind", "")
                # Boost exact name matches
                if name == query:
                    rank -= 50.0
                elif name and query_lower in name.lower():
                    rank -= 10.0
                # Boost primary symbol kinds (class/struct > prototype/namespace)
                if sym_kind in _PRIMARY_KINDS:
                    rank -= 5.0
                if code_like_query and sym_kind == "component":
                    rank -= 4.0
                # Name length normalization: shorter names closer to query length
                # are more relevant (ICaptureService > CaptureServiceImpl::OnHover)
                query_len = len(query)
                name_len = len(name)
                if name_len > query_len:
                    rank += min((name_len - query_len) * 0.3, 5.0)
                # Path-based ranking: core > bindings > vendored
                file_path = row_dict.get("file", "")
                if is_vendored_path(file_path):
                    rank += 20.0
                    row_dict["vendored"] = True
                elif file_path.startswith("bindings/"):
                    rank += 3.0  # Slight penalty vs core src/
                if code_like_query and (
                    sym_kind in {"section", "document"} or is_documentation_path(file_path)
                ):
                    rank += 18.0
                row_dict["rank"] = rank
                results.append(row_dict)
                seen_ids.add(sid)

        # Overfetch so we can sort across tiers, not first-come-first-served.
        OVERFETCH = 3
        fetch_limit = limit * 5 if kind else limit * OVERFETCH

        # Tier 1: FTS5 on symbol names (includes tokenized name_tokens column)
        try:
            for fts_query in _search_query_variants(query):
                if not fts_query or len(results) >= fetch_limit:
                    break
                rows = self.conn.execute(
                    """SELECT symbol_id, name, file_path, kind,
                              rank, snippet(symbol_names_fts, 1, '>>>', '<<<', '...', 20) as snippet
                       FROM symbol_names_fts
                       WHERE symbol_names_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, fetch_limit),
                ).fetchall()
                for row in rows:
                    if len(results) >= fetch_limit:
                        break
                    _add_row({
                        "symbol_id": int(row["symbol_id"]),
                        "name": row["name"],
                        "file": row["file_path"],
                        "kind": row["kind"],
                        "snippet": row["snippet"],
                        "source": "name",
                        "rank": row["rank"],
                    })
        except sqlite3.OperationalError:
            pass  # Invalid FTS query syntax

        # Tier 2: LIKE on symbol names — ALWAYS runs (even if Tier 1 filled up).
        # This is critical: FTS5 tokenization can bury exact substring matches
        # (e.g., "CaptureService" finds capture_service_* tokens before ICaptureService).
        # LIKE guarantees substring matches surface.
        # ORDER BY: exact → definition kinds (class/struct/interface) → prefix → length.
        # This ensures ICaptureService (class, non-prefix) beats
        # CaptureServiceImpl::OnHover (method, prefix).
        like_limit = limit * 4  # Generous budget — dedup handles overlap
        _DEFN_KINDS_SQL = "('class','struct','interface','enum')"
        if kind:
            rows = self.conn.execute(
                f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.name LIKE ? COLLATE NOCASE AND s.kind = ?
                   ORDER BY
                       CASE WHEN s.name = ? THEN 0 ELSE 1 END,
                       CASE WHEN s.kind IN {_DEFN_KINDS_SQL} THEN 0
                            WHEN s.kind IN ('function','prototype') THEN 1
                            ELSE 2 END,
                       CASE WHEN s.name LIKE ? THEN 0 ELSE 1 END,
                       length(s.name),
                       s.name
                   LIMIT ?""",
                (f"%{query}%", kind, query, f"{query}%", like_limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE s.name LIKE ? COLLATE NOCASE
                   ORDER BY
                       CASE WHEN s.name = ? THEN 0 ELSE 1 END,
                       CASE WHEN s.kind IN {_DEFN_KINDS_SQL} THEN 0
                            WHEN s.kind IN ('function','prototype') THEN 1
                            ELSE 2 END,
                       CASE WHEN s.name LIKE ? THEN 0 ELSE 1 END,
                       length(s.name),
                       s.name
                   LIMIT ?""",
                (f"%{query}%", query, f"{query}%", like_limit),
            ).fetchall()
        for row in rows:
            # LIKE results are name matches — give them a competitive base rank.
            # FTS5 BM25 ranks are typically -10 to -25 for name matches.
            # Using -15 here lets _add_row's substring/kind boosts push good
            # matches above FTS token-decomposition noise.
            _add_row({
                "symbol_id": int(row["symbol_id"]),
                "name": row["name"],
                "file": row["file_path"],
                "kind": row["kind"],
                "snippet": row["name"],
                "source": "name_like",
                "rank": -15.0,
            })

        tokenized_hint = tokenized_query_hint(query)
        if tokenized_hint:
            like_limit = limit * 3
            tokenized_pattern = f"%{tokenized_hint}%"
            if kind:
                rows = self.conn.execute(
                    """SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                       FROM symbols s JOIN files f ON s.file_id = f.id
                       WHERE s.kind = ?
                         AND (
                             COALESCE(s.signature, '') LIKE ? COLLATE NOCASE
                             OR s.content LIKE ? COLLATE NOCASE
                             OR COALESCE(s.doc_comment, '') LIKE ? COLLATE NOCASE
                         )
                       ORDER BY
                           CASE WHEN s.kind IN ('microservice_handler', 'queue_processor', 'scheduled_job') THEN 0
                                ELSE 1 END,
                           length(s.name),
                           s.name
                       LIMIT ?""",
                    (kind, tokenized_pattern, tokenized_pattern, tokenized_pattern, like_limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                       FROM symbols s JOIN files f ON s.file_id = f.id
                       WHERE COALESCE(s.signature, '') LIKE ? COLLATE NOCASE
                          OR s.content LIKE ? COLLATE NOCASE
                          OR COALESCE(s.doc_comment, '') LIKE ? COLLATE NOCASE
                       ORDER BY
                           CASE WHEN s.kind IN ('microservice_handler', 'queue_processor', 'scheduled_job') THEN 0
                                ELSE 1 END,
                           length(s.name),
                           s.name
                       LIMIT ?""",
                    (tokenized_pattern, tokenized_pattern, tokenized_pattern, like_limit),
                ).fetchall()
            for row in rows:
                _add_row({
                    "symbol_id": int(row["symbol_id"]),
                    "name": row["name"],
                    "file": row["file_path"],
                    "kind": row["kind"],
                    "snippet": tokenized_hint,
                    "source": "tokenized_like",
                    "rank": -14.0,
                })

        metadata_patterns = _metadata_like_patterns(query)
        if metadata_patterns:
            like_limit = limit * 3
            where_clause = " OR ".join("COALESCE(s.metadata, '') LIKE ? COLLATE NOCASE" for _ in metadata_patterns)
            params: list[Any] = []
            if kind:
                params.append(kind)
            params.extend(metadata_patterns)
            params.append(like_limit)
            if kind:
                rows = self.conn.execute(
                    f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                        FROM symbols s JOIN files f ON s.file_id = f.id
                        WHERE s.kind = ? AND ({where_clause})
                        ORDER BY
                            CASE WHEN s.kind IN ('microservice_handler', 'queue_processor', 'scheduled_job', 'transport', 'module') THEN 0
                                 ELSE 1 END,
                            length(s.name),
                            s.name
                        LIMIT ?""",
                    params,
                ).fetchall()
            else:
                rows = self.conn.execute(
                    f"""SELECT s.id as symbol_id, s.name, f.path as file_path, s.kind
                        FROM symbols s JOIN files f ON s.file_id = f.id
                        WHERE {where_clause}
                        ORDER BY
                            CASE WHEN s.kind IN ('microservice_handler', 'queue_processor', 'scheduled_job', 'transport', 'module') THEN 0
                                 ELSE 1 END,
                            length(s.name),
                            s.name
                        LIMIT ?""",
                    params,
                ).fetchall()
            for row in rows:
                _add_row({
                    "symbol_id": int(row["symbol_id"]),
                    "name": row["name"],
                    "file": row["file_path"],
                    "kind": row["kind"],
                    "snippet": query,
                    "source": "metadata_like",
                    "rank": -13.0,
                })

        # Tier 3: FTS5 on source code content (trigram)
        if len(results) < fetch_limit:
            try:
                for content_query in _search_query_variants(query):
                    if len(results) >= fetch_limit:
                        break
                    rows = self.conn.execute(
                        """SELECT symbol_id, name, file_path, kind,
                                  rank, snippet(symbol_content_fts, 0, '>>>', '<<<', '...', 30) as snippet
                           FROM symbol_content_fts
                           WHERE symbol_content_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (content_query, fetch_limit),
                    ).fetchall()
                    for row in rows:
                        if len(results) >= fetch_limit:
                            break
                        _add_row({
                            "symbol_id": int(row["symbol_id"]),
                            "name": row["name"],
                            "file": row["file_path"],
                            "kind": row["kind"],
                            "snippet": row["snippet"],
                            "source": "content",
                            "rank": row["rank"],
                        })
            except sqlite3.OperationalError:
                pass

        # Tier 4: FTS5 on documentation
        if len(results) < fetch_limit:
            try:
                for docs_query in _search_query_variants(query):
                    if len(results) >= fetch_limit:
                        break
                    rows = self.conn.execute(
                        """SELECT symbol_id, name, file_path, kind,
                                  rank, snippet(symbol_docs_fts, 0, '>>>', '<<<', '...', 30) as snippet
                           FROM symbol_docs_fts
                           WHERE symbol_docs_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (docs_query, fetch_limit),
                    ).fetchall()
                    for row in rows:
                        if len(results) >= fetch_limit:
                            break
                        _add_row({
                            "symbol_id": int(row["symbol_id"]),
                            "name": row["name"],
                            "file": row["file_path"],
                            "kind": row["kind"],
                            "snippet": row["snippet"],
                            "source": "docs",
                            "rank": row["rank"],
                        })
            except sqlite3.OperationalError:
                pass

        self._rerank_search_results(results, query, code_like_query)

        # Sort: project code first, then by rank within each group
        results.sort(key=lambda r: (r.get("vendored", False), r.get("rank", 0)))
        return results[:limit]

    # --- Edges ---

    def delete_edges_for_symbols(self, symbol_ids: list[int]) -> None:
        """Delete all edges where source or target is in the given symbol IDs."""
        assert self.conn is not None
        if not symbol_ids:
            return
        placeholders = ",".join("?" * len(symbol_ids))
        self.conn.execute(
            f"DELETE FROM symbol_edges WHERE source_id IN ({placeholders})"
            f" OR target_id IN ({placeholders})",
            symbol_ids + symbol_ids,
        )

    def all_symbol_names(self) -> dict[str, list[int]]:
        """Return a mapping of symbol name -> list of symbol IDs.

        Used for building the call graph: scan symbol content for references
        to known symbol names.
        """
        assert self.conn is not None
        result: dict[str, list[int]] = {}
        for row in self.conn.execute("SELECT id, name FROM symbols WHERE name IS NOT NULL"):
            name = row["name"]
            if name not in result:
                result[name] = []
            result[name].append(row["id"])
        return result

    def insert_edge(self, rec: EdgeRecord) -> int:
        assert self.conn is not None
        meta_json = json.dumps(rec.metadata) if rec.metadata else None
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO symbol_edges
               (source_id, target_id, edge_type, confidence, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (rec.source_id, rec.target_id, rec.edge_type, rec.confidence, meta_json),
        )
        return cur.lastrowid

    def get_callers(self, symbol_id: int) -> list[dict]:
        """Get all symbols that call/reference the given symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.source_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.target_id = ?
               ORDER BY e.edge_type, s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_callees(self, symbol_id: int) -> list[dict]:
        """Get all symbols that the given symbol calls/references."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.target_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.source_id = ?
               ORDER BY e.edge_type, s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_subclasses(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that inherit from the given symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.source_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.target_id = ? AND e.edge_type = 'inherits'
               ORDER BY s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_base_classes(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that the given symbol inherits from."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
               FROM symbol_edges e
               JOIN symbols s ON e.target_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE e.source_id = ? AND e.edge_type = 'inherits'
               ORDER BY s.name""",
            (symbol_id,),
        ).fetchall()
        return [
            {
                "symbol": self._row_to_symbol(r),
                "edge_type": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def get_dependents(self, symbol_id: int, transitive: bool = False, max_depth: int = 5) -> list[dict]:
        """Get all symbols that depend on (call/reference) the given symbol.

        If transitive=True, walks the caller graph recursively up to max_depth.
        """
        assert self.conn is not None
        if not transitive:
            return self.get_callers(symbol_id)

        visited: set[int] = set()
        result = []

        def _walk(sid: int, depth: int):
            if depth > max_depth or sid in visited:
                return
            visited.add(sid)
            callers = self.get_callers(sid)
            for c in callers:
                caller_id = c["symbol"].id
                if caller_id not in visited:
                    c["depth"] = depth
                    result.append(c)
                    _walk(caller_id, depth + 1)

        _walk(symbol_id, 1)
        return result

    def get_implementors(self, symbol_id: int) -> list[dict]:
        """Get all classes/structs that inherit from or implement the given interface/class.

        Same as get_subclasses but named for the interface pattern.
        """
        return self.get_subclasses(symbol_id)

    def get_tests_for(self, symbol_name: str) -> list[dict]:
        """Find test functions that likely test the given symbol.

        Heuristic: looks for symbols in test files whose names contain the target name.
        Matches patterns like test_X, X_test, Test_X, test_X_something.
        """
        assert self.conn is not None
        # Find test files (common patterns)
        test_rows = self.conn.execute(
            """SELECT s.*, f.path as file_path
               FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE (f.path LIKE '%test%' OR f.path LIKE '%spec%')
                 AND s.kind IN ('function', 'method')
                 AND (s.name LIKE ? OR s.name LIKE ? OR s.name LIKE ?)
               ORDER BY f.path, s.start_line""",
            (f"%test%{symbol_name}%", f"%{symbol_name}%test%", f"%{symbol_name}%spec%"),
        ).fetchall()

        results = []
        for r in test_rows:
            sym = self._row_to_symbol(r)
            results.append({
                "symbol": sym,
                "edge_type": "tests",
                "confidence": 0.7,
            })

        # Also check for explicit 'tests' edges if any exist
        sym = self.get_symbol_by_name(symbol_name)
        if sym and sym.id:
            edge_rows = self.conn.execute(
                """SELECT s.*, f.path as file_path, e.edge_type, e.confidence
                   FROM symbol_edges e
                   JOIN symbols s ON e.source_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.target_id = ? AND e.edge_type = 'tests'
                   ORDER BY s.name""",
                (sym.id,),
            ).fetchall()
            seen = {r["symbol"].id for r in results if r["symbol"].id}
            for r in edge_rows:
                s = self._row_to_symbol(r)
                if s.id not in seen:
                    results.append({
                        "symbol": s,
                        "edge_type": r["edge_type"],
                        "confidence": r["confidence"],
                    })

        return results

    # --- Embeddings ---

    def upsert_embedding(self, symbol_id: int, model: str, dimensions: int,
                         embedding_bytes: bytes, body_hash: str | None = None) -> None:
        """Insert or update an embedding for a symbol."""
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO symbol_embeddings (symbol_id, model, dimensions, embedding, body_hash)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(symbol_id) DO UPDATE SET
                   model=excluded.model,
                   dimensions=excluded.dimensions,
                   embedding=excluded.embedding,
                   body_hash=excluded.body_hash,
                   embedded_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (symbol_id, model, dimensions, embedding_bytes, body_hash),
        )
        # Bump embedding cache version so VectorCache knows to reload
        self.conn.execute(
            """INSERT INTO schema_info (key, value) VALUES ('embedding_cache_version', '1')
               ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)""",
        )

    def get_symbols_needing_embeddings(self, model: str, limit: int = 100000) -> list[dict]:
        """Get symbols that need embeddings, filtered in SQL."""
        assert self.conn is not None
        freshness_expr = (
            "CASE WHEN COALESCE(f.embedding_context_hash, '') = '' THEN s.body_hash "
            "ELSE s.body_hash || ':' || f.embedding_context_hash END"
        )
        rows = self.conn.execute(
            f"""SELECT s.*, f.path as file_path, f.summary as file_summary,
                      f.metadata as file_summary_metadata,
                      {freshness_expr} as embedding_body_hash,
                      e.body_hash as embedded_body_hash
               FROM symbols s
               JOIN files f ON s.file_id = f.id
               LEFT JOIN symbol_embeddings e ON s.id = e.symbol_id AND e.model = ?
               WHERE e.symbol_id IS NULL OR e.body_hash != {freshness_expr}
               ORDER BY s.id
               LIMIT ?""",
            (model, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            symbol = asdict(self._row_to_symbol(row))
            symbol["embedding_body_hash"] = row["embedding_body_hash"]
            results.append(symbol)
        return results

    def vector_search(self, query_embedding: bytes, dimensions: int,
                      limit: int = 10, kind: str | None = None,
                      cache=None) -> list[dict]:
        """Search symbols by cosine similarity to the query embedding.

        If a valid VectorCache is provided, uses the GPU-resident matrix (~3ms).
        Otherwise falls back to fetching all embeddings from SQLite (~585ms).
        """
        assert self.conn is not None

        # Fast path: use GPU-resident VectorCache
        if cache is not None and cache.is_valid(self.conn):
            top_k = cache.search(query_embedding, dimensions, limit, kind)
            return self._enrich_results(top_k)

        # Slow path: fetch all embeddings from SQLite
        n_floats = len(query_embedding) // 4
        query_vec = struct.unpack(f'{n_floats}f', query_embedding)

        if kind:
            rows = self.conn.execute(
                """SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                          s.kind, s.signature, f.path as file_path,
                          s.start_line, s.end_line, s.line_count, s.doc_comment
                   FROM symbol_embeddings e
                   JOIN symbols s ON e.symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.dimensions = ? AND s.kind = ?""",
                (dimensions, kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT e.symbol_id, e.embedding, s.name, s.qualified_name,
                          s.kind, s.signature, f.path as file_path,
                          s.start_line, s.end_line, s.line_count, s.doc_comment
                   FROM symbol_embeddings e
                   JOIN symbols s ON e.symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.dimensions = ?""",
                (dimensions,),
            ).fetchall()

        if not rows:
            return []

        blobs = [row["embedding"] for row in rows]
        matrix = decode_matrix(blobs, n_floats)
        top_k = cosine_top_k(query_vec, matrix, limit)

        results = []
        for idx, sim in top_k:
            row = rows[idx]
            results.append({
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

    def _enrich_results(self, top_k: list[tuple[int, float, int]]) -> list[dict]:
        """Fetch full metadata for top-k results from VectorCache.

        Args:
            top_k: list of (row_index, similarity, symbol_id) from VectorCache.search()
        Returns:
            list of result dicts with full symbol metadata.
        """
        assert self.conn is not None
        if not top_k:
            return []

        results = []
        for _row_idx, sim, symbol_id in top_k:
            row = self.conn.execute(
                """SELECT s.name, s.qualified_name, s.kind, s.signature,
                          f.path as file_path, s.start_line, s.end_line,
                          s.line_count, s.doc_comment
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.id = ?""",
                (symbol_id,),
            ).fetchone()
            if row is None:
                continue
            results.append({
                "symbol_id": symbol_id,
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

    def embedding_stats(self) -> dict:
        """Get embedding statistics."""
        assert self.conn is not None
        total_symbols = self.conn.execute("SELECT COUNT(*) as n FROM symbols").fetchone()["n"]
        embedded = self.conn.execute(
            "SELECT COUNT(*) as n FROM symbol_embeddings"
        ).fetchone()["n"]
        model_row = self.conn.execute(
            "SELECT model, dimensions FROM symbol_embeddings LIMIT 1"
        ).fetchone()
        return {
            "total_symbols": total_symbols,
            "embedded_symbols": embedded,
            "coverage_pct": round(embedded / total_symbols * 100, 1) if total_symbols else 0,
            "model": model_row["model"] if model_row else None,
            "dimensions": model_row["dimensions"] if model_row else None,
        }

    # --- Index State ---

    def get_index_state(self, repo_root: str) -> dict | None:
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT * FROM index_state WHERE repo_root = ?", (repo_root,)
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def update_index_state(
        self, repo_root: str, last_commit: str | None = None,
        files_indexed: int = 0, symbols_indexed: int = 0,
        indexer_version: str | None = None,
    ) -> None:
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO index_state (repo_root, last_commit, files_indexed, symbols_indexed, indexer_version)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(repo_root) DO UPDATE SET
                   last_commit=excluded.last_commit,
                   files_indexed=excluded.files_indexed,
                   symbols_indexed=excluded.symbols_indexed,
                   indexer_version=excluded.indexer_version,
                   indexed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (repo_root, last_commit, files_indexed, symbols_indexed, indexer_version),
        )
        self.conn.commit()

    # --- Stats ---

    def directory_summary(self, max_depth: int = 3) -> list[dict]:
        """Get symbol counts per directory, up to max_depth levels deep."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT f.path, COUNT(s.id) as symbol_count, f.language
               FROM files f
               LEFT JOIN symbols s ON f.id = s.file_id
               GROUP BY f.id
               ORDER BY symbol_count DESC"""
        ).fetchall()

        dir_counts: dict[str, dict] = {}
        for row in rows:
            parts = row["path"].split("/")
            # Build directory paths at each depth level
            for depth in range(1, min(len(parts), max_depth + 1)):
                dir_path = "/".join(parts[:depth])
                if dir_path not in dir_counts:
                    dir_counts[dir_path] = {"files": 0, "symbols": 0, "languages": set()}
                dir_counts[dir_path]["files"] += 1
                dir_counts[dir_path]["symbols"] += row["symbol_count"]
                if row["language"]:
                    dir_counts[dir_path]["languages"].add(row["language"])

        result = []
        for path in sorted(dir_counts.keys()):
            info = dir_counts[path]
            result.append({
                "path": path,
                "files": info["files"],
                "symbols": info["symbols"],
                "languages": sorted(info["languages"]),
            })
        return result

    def hotspot_files(self, limit: int = 10) -> list[dict]:
        """Get files with the most symbols (complexity hotspots)."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT f.path, f.language, f.line_count, COUNT(s.id) as symbol_count
               FROM files f
               JOIN symbols s ON f.id = s.file_id
               GROUP BY f.id
               ORDER BY symbol_count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "path": row["path"],
                "language": row["language"],
                "lines": row["line_count"],
                "symbols": row["symbol_count"],
            }
            for row in rows
        ]

    # --- Communities ---

    def store_communities(self, communities: list[dict]) -> None:
        """Store detected communities and their symbol memberships."""
        assert self.conn is not None
        self.conn.execute("DELETE FROM symbol_communities")
        self.conn.execute("DELETE FROM communities")

        for c in communities:
            self.conn.execute(
                """INSERT INTO communities (id, label, symbol_count, cohesion, keywords, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (c["id"], c["label"], c["symbol_count"], c["cohesion"],
                 json.dumps(c.get("keywords", [])), json.dumps(c.get("metadata"))),
            )
            for member in c.get("members", []):
                self.conn.execute(
                    "INSERT OR IGNORE INTO symbol_communities (symbol_id, community_id) VALUES (?, ?)",
                    (member["id"], c["id"]),
                )

    def _file_filter_sql(
        self,
        *,
        path_prefix: str | None = None,
        layer: str | None = None,
        file_alias: str = "f",
    ) -> tuple[str, list[str]]:
        """Build a SQL fragment for optional file path filters."""
        clauses: list[str] = []
        params: list[str] = []
        if path_prefix:
            clauses.append(f"{file_alias}.path LIKE ?")
            params.append(f"{path_prefix}%")
        if layer:
            clauses.append(f"{file_alias}.path LIKE ?")
            params.append(f"{layer}/%")
        if not clauses:
            return "", []
        return " AND " + " AND ".join(clauses), params

    def get_community_records(
        self,
        *,
        limit: int | None = None,
        path_prefix: str | None = None,
        layer: str | None = None,
    ) -> list[dict]:
        """Get community metadata with optional member-based file filters."""
        assert self.conn is not None
        filter_sql, filter_params = self._file_filter_sql(path_prefix=path_prefix, layer=layer)
        query = (
            """SELECT c.id, c.label, c.symbol_count, c.cohesion, c.keywords,
                      COUNT(sc.symbol_id) AS member_count
               FROM communities c
               JOIN symbol_communities sc ON sc.community_id = c.id
               JOIN symbols s ON sc.symbol_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE 1 = 1"""
            + filter_sql
            + """
               GROUP BY c.id, c.label, c.symbol_count, c.cohesion, c.keywords
               HAVING COUNT(sc.symbol_id) > 0
               ORDER BY member_count DESC, c.symbol_count DESC, c.id"""
        )
        params: list[Any] = list(filter_params)
        if limit is not None:
            query += "\n               LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "label": row["label"],
                "symbol_count": row["symbol_count"],
                "member_count": row["member_count"],
                "cohesion": row["cohesion"],
                "keywords": json.loads(row["keywords"] or "[]"),
            }
            for row in rows
        ]

    def get_communities(
        self,
        *,
        verbose: bool = False,
        limit: int | None = 25,
        member_limit: int | None = 5,
        path_prefix: str | None = None,
        layer: str | None = None,
    ) -> list[dict]:
        """Get shaped community results for summary or verbose consumers."""
        communities = self.get_community_records(
            limit=limit,
            path_prefix=path_prefix,
            layer=layer,
        )
        results: list[dict] = []
        for community in communities:
            members = self.get_community_members(
                community["id"],
                limit=member_limit,
                path_prefix=path_prefix,
                layer=layer,
                verbose=verbose,
            )
            results.append(
                {
                    "id": community["id"],
                    "label": community["label"],
                    "member_count": community["member_count"],
                    "cohesion": community["cohesion"],
                    "keywords": community["keywords"],
                    "truncated": member_limit is not None and community["member_count"] > len(members),
                    "member_limit_applied": member_limit,
                    "members": members,
                }
            )
        return results

    def get_community_for_symbol(self, symbol_id: int) -> int | None:
        """Get the community ID for a symbol, or None."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT community_id FROM symbol_communities WHERE symbol_id = ?",
            (symbol_id,),
        ).fetchone()
        return row["community_id"] if row else None

    def get_community_members(
        self,
        community_id: int,
        *,
        limit: int | None = None,
        path_prefix: str | None = None,
        layer: str | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Get community members with optional file filters and compact shaping."""
        assert self.conn is not None
        filter_sql, filter_params = self._file_filter_sql(path_prefix=path_prefix, layer=layer)
        query = (
            """SELECT s.id, s.name, s.qualified_name, s.kind, f.path as file_path
               FROM symbol_communities sc
               JOIN symbols s ON sc.symbol_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE sc.community_id = ?"""
            + filter_sql
            + """
               ORDER BY s.name"""
        )
        params: list[Any] = [community_id, *filter_params]
        if limit is not None:
            query += "\n               LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        members: list[dict] = []
        for row in rows:
            member = {
                "name": row["name"],
                "kind": row["kind"],
                "file_path": row["file_path"],
            }
            if verbose:
                member["id"] = row["id"]
                member["qualified_name"] = row["qualified_name"]
            members.append(member)
        return members

    # --- Execution Flows ---

    def store_execution_flows(self, flows: list[dict]) -> None:
        """Store execution flows and their steps."""
        assert self.conn is not None
        self.conn.execute("DELETE FROM flow_steps")
        self.conn.execute("DELETE FROM execution_flows")

        for flow in flows:
            cursor = self.conn.execute(
                """INSERT INTO execution_flows
                   (label, entry_symbol_id, terminal_symbol_id, step_count, communities_crossed)
                   VALUES (?, ?, ?, ?, ?)""",
                (flow["label"], flow["entry_symbol_id"], flow["terminal_symbol_id"],
                 flow["step_count"], flow["communities_crossed"]),
            )
            flow_id = cursor.lastrowid
            for step in flow["steps"]:
                self.conn.execute(
                    """INSERT INTO flow_steps (flow_id, step_order, symbol_id, community_id)
                       VALUES (?, ?, ?, ?)""",
                    (flow_id, step["order"], step["symbol_id"], step.get("community_id")),
                )

    def get_execution_flow_records(
        self,
        limit: int | None = 50,
        *,
        path_prefix: str | None = None,
        layer: str | None = None,
    ) -> list[dict]:
        """Get raw execution flow rows with optional step-based file filters."""
        assert self.conn is not None
        filter_sql, filter_params = self._file_filter_sql(path_prefix=path_prefix, layer=layer)
        query = (
            """SELECT ef.*, s1.name as entry_name, s2.name as terminal_name
               FROM execution_flows ef
               LEFT JOIN symbols s1 ON ef.entry_symbol_id = s1.id
               LEFT JOIN symbols s2 ON ef.terminal_symbol_id = s2.id"""
        )
        params: list[Any] = []
        if filter_sql:
            query += (
                """
               WHERE EXISTS (
                   SELECT 1
                   FROM flow_steps fs
                   JOIN symbols s ON fs.symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE fs.flow_id = ef.id"""
                + filter_sql
                + """
               )"""
            )
            params.extend(filter_params)
        query += "\n               ORDER BY ef.step_count DESC, ef.id"
        if limit is not None:
            query += "\n               LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_filtered_flow_metadata(
        self,
        flow_id: int,
        *,
        path_prefix: str | None = None,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Get filtered flow metadata without materializing the full step payload."""
        assert self.conn is not None
        filter_sql, filter_params = self._file_filter_sql(path_prefix=path_prefix, layer=layer)
        row = self.conn.execute(
            (
                """WITH filtered_steps AS (
                       SELECT fs.step_order, s.name, fs.community_id
                       FROM flow_steps fs
                       JOIN symbols s ON fs.symbol_id = s.id
                       JOIN files f ON s.file_id = f.id
                       WHERE fs.flow_id = ?"""
                + filter_sql
                + """
                   ),
                   ordered_steps AS (
                       SELECT step_order, name, community_id,
                              LAG(community_id) OVER (ORDER BY step_order) AS prev_community_id
                       FROM filtered_steps
                   )
                   SELECT
                       COUNT(*) AS step_count,
                       (SELECT name FROM filtered_steps ORDER BY step_order LIMIT 1) AS entry_name,
                       (SELECT name FROM filtered_steps ORDER BY step_order DESC LIMIT 1) AS terminal_name,
                       COALESCE(SUM(
                           CASE
                               WHEN prev_community_id IS NOT NULL AND community_id != prev_community_id
                               THEN 1
                               ELSE 0
                           END
                       ), 0) AS communities_crossed
                   FROM ordered_steps"""
            ),
            (flow_id, *filter_params),
        ).fetchone()
        return dict(row) if row else {
            "step_count": 0,
            "entry_name": None,
            "terminal_name": None,
            "communities_crossed": 0,
        }

    def get_execution_flows(
        self,
        limit: int | None = 50,
        *,
        verbose: bool = False,
        max_depth: int | None = None,
        path_prefix: str | None = None,
        layer: str | None = None,
    ) -> list[dict]:
        """Get shaped execution flow results for summary or verbose consumers."""
        flows = self.get_execution_flow_records(
            limit=limit,
            path_prefix=path_prefix,
            layer=layer,
        )
        results: list[dict] = []
        step_limit = max_depth if max_depth is not None else (None if verbose else 3)
        for flow in flows:
            metadata = self.get_filtered_flow_metadata(
                flow["id"],
                path_prefix=path_prefix,
                layer=layer,
            )
            steps = self.get_flow_steps(
                flow["id"],
                limit=step_limit,
                path_prefix=path_prefix,
                layer=layer,
                verbose=verbose,
            )
            total_steps = metadata["step_count"]
            entry_name = metadata["entry_name"] or flow["entry_name"]
            terminal_name = metadata["terminal_name"] or flow["terminal_name"]
            item = {
                "id": flow["id"],
                "label": f"{entry_name} -> {terminal_name}" if entry_name and terminal_name else flow["label"],
                "entry": entry_name,
                "terminal": terminal_name,
                "step_count": total_steps,
                "communities_crossed": metadata["communities_crossed"],
                "truncated": total_steps > len(steps),
            }
            if max_depth is not None:
                item["max_depth_applied"] = max_depth
            if verbose:
                item["steps"] = steps
            elif steps:
                item["key_steps"] = steps
            results.append(item)
        return results

    def get_flows_for_symbol(self, symbol_id: int) -> list[dict]:
        """Get all execution flows that pass through a symbol."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT DISTINCT ef.*, s1.name as entry_name, s2.name as terminal_name
               FROM flow_steps fs
               JOIN execution_flows ef ON fs.flow_id = ef.id
               LEFT JOIN symbols s1 ON ef.entry_symbol_id = s1.id
               LEFT JOIN symbols s2 ON ef.terminal_symbol_id = s2.id
               WHERE fs.symbol_id = ?
               ORDER BY ef.step_count DESC""",
            (symbol_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_flow_steps(
        self,
        flow_id: int,
        *,
        limit: int | None = None,
        path_prefix: str | None = None,
        layer: str | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Get ordered steps for a specific flow with optional compact shaping."""
        assert self.conn is not None
        filter_sql, filter_params = self._file_filter_sql(path_prefix=path_prefix, layer=layer)
        query = (
            """SELECT fs.step_order, fs.symbol_id, fs.community_id,
                      s.name, s.qualified_name, s.kind, f.path as file_path
               FROM flow_steps fs
               JOIN symbols s ON fs.symbol_id = s.id
               JOIN files f ON s.file_id = f.id
               WHERE fs.flow_id = ?"""
            + filter_sql
            + """
               ORDER BY fs.step_order"""
        )
        params: list[Any] = [flow_id, *filter_params]
        if limit is not None:
            query += "\n               LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        steps: list[dict] = []
        for row in rows:
            step = {
                "step_order": row["step_order"],
                "name": row["name"],
                "kind": row["kind"],
                "file_path": row["file_path"],
                "community_id": row["community_id"],
            }
            if verbose:
                step["symbol_id"] = row["symbol_id"]
                step["qualified_name"] = row["qualified_name"]
            steps.append(step)
        return steps

    # --- Stats ---

    def stats(self) -> dict:
        assert self.conn is not None
        files = self.conn.execute("SELECT COUNT(*) as n FROM files").fetchone()["n"]
        symbols = self.conn.execute("SELECT COUNT(*) as n FROM symbols").fetchone()["n"]
        edges = self.conn.execute("SELECT COUNT(*) as n FROM symbol_edges").fetchone()["n"]

        lang_counts = {}
        for row in self.conn.execute(
            "SELECT language, COUNT(*) as n FROM files GROUP BY language ORDER BY n DESC"
        ):
            lang_counts[row["language"] or "unknown"] = row["n"]

        kind_counts = {}
        for row in self.conn.execute(
            "SELECT kind, COUNT(*) as n FROM symbols GROUP BY kind ORDER BY n DESC"
        ):
            kind_counts[row["kind"]] = row["n"]

        db_size = self.path.stat().st_size if self.path.exists() else 0

        return {
            "files": files,
            "symbols": symbols,
            "edges": edges,
            "languages": lang_counts,
            "symbol_kinds": kind_counts,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }

    def resolve_import(self, name: str, hint_path: str | None = None) -> dict | None:
        """Try to resolve an import name to an indexed symbol or file.

        Resolution strategy:
        1. Exact symbol name match
        2. Qualified name match (e.g., 'os.path' matches qualified_name)
        3. File path match (convert module path to file path)

        Args:
            name: Import name (e.g., 'Database', 'os.path', './utils')
            hint_path: Optional path hint for relative imports

        Returns:
            Dict with resolved symbol/file info, or None if unresolved.
        """
        assert self.conn is not None

        # 1. Exact symbol name match
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name = ?
               ORDER BY s.kind, f.path
               LIMIT 1""",
            (name,),
        ).fetchone()
        if row:
            sym = self._row_to_symbol(row)
            return {
                "name": sym.name,
                "file": sym.file_path,
                "line": sym.start_line,
                "kind": sym.kind,
                "match_type": "symbol",
            }

        # 2. Qualified name match
        row = self.conn.execute(
            """SELECT s.*, f.path as file_path FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.qualified_name = ?
               ORDER BY s.kind, f.path
               LIMIT 1""",
            (name,),
        ).fetchone()
        if row:
            sym = self._row_to_symbol(row)
            return {
                "name": sym.name,
                "file": sym.file_path,
                "line": sym.start_line,
                "kind": sym.kind,
                "match_type": "qualified_name",
            }

        # 3. File path match — convert dotted module to path patterns
        candidates = []
        dotted = name.replace(".", "/")
        candidates.append(f"{dotted}.py")
        candidates.append(f"{dotted}/__init__.py")
        candidates.append(name)
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            candidates.append(f"{name}{ext}")
            candidates.append(f"{dotted}{ext}")

        for candidate in candidates:
            row = self.conn.execute(
                "SELECT * FROM files WHERE path = ? OR path LIKE ?",
                (candidate, f"%/{candidate}"),
            ).fetchone()
            if row:
                return {
                    "name": name,
                    "file": row["path"],
                    "line": 1,
                    "kind": "module",
                    "match_type": "file_path",
                }

        return None

    def get_dead_symbols(self, kind: str | None = None) -> list[SymbolRecord]:
        """Find symbols with no incoming edges — potential dead code.

        Returns symbols that are never referenced as a target in symbol_edges.
        Excludes entry points, test code, vendored paths, and non-code kinds.
        """
        assert self.conn is not None

        allowed_kinds = ("function", "method", "class", "struct", "enum", "interface")
        sql = """
            SELECT s.*, f.path as file_path
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            LEFT JOIN symbol_edges e ON e.target_id = s.id
            WHERE e.id IS NULL
              AND s.kind IN ({kinds})
              AND s.name NOT LIKE 'test\\_%' ESCAPE '\\'
              AND s.name NOT LIKE 'Test%'
              AND s.name NOT IN ('main', '__init__', '__main__', '__new__', '__del__')
              AND f.path NOT LIKE '%test%'
              AND f.path NOT LIKE '%vendor%'
              AND f.path NOT LIKE '%third_party%'
              AND f.path NOT LIKE '%third-party%'
        """.format(kinds=",".join("?" for _ in allowed_kinds))

        params: list[Any] = list(allowed_kinds)

        if kind:
            sql += "  AND s.kind = ?\n"
            params.append(kind)

        sql += "ORDER BY f.path, s.start_line"

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def search_pattern(
        self,
        pattern: str,
        language: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search for a regex pattern within symbol source code.

        Fetches candidate symbols from the database (filtered by language/kind),
        then applies the regex in Python against each symbol's content field.
        Returns matching symbols with match context.
        """
        assert self.conn is not None

        sql = """
            SELECT s.*, f.path as file_path, f.language
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.content IS NOT NULL AND s.content != ''
        """
        params: list[Any] = []

        if language:
            sql += " AND f.language = ?"
            params.append(language)

        if kind:
            sql += " AND s.kind = ?"
            params.append(kind)

        # Exclude non-code symbol kinds
        sql += " AND s.kind NOT IN ('namespace', 'module', 'macro', 'section', 'document')"
        sql += " ORDER BY f.path, s.start_line"

        compiled = re.compile(pattern)
        results: list[dict[str, Any]] = []

        for row in self.conn.execute(sql, params):
            content = row["content"]
            if not content:
                continue

            matches = []
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if compiled.search(line):
                    # Build snippet: matching line + 1 line context each side
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    snippet = "\n".join(lines[start:end])
                    matches.append({
                        "line_offset": i + 1,  # 1-based within symbol
                        "absolute_line": row["start_line"] + i,
                        "snippet": snippet,
                    })

            if matches:
                sym = self._row_to_symbol(row)
                results.append({
                    "name": sym.name,
                    "qualified_name": sym.qualified_name,
                    "kind": sym.kind,
                    "file": sym.file_path,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "language": row["language"],
                    "matches": matches,
                })
                if len(results) >= limit:
                    break

        return results

    def commit(self) -> None:
        assert self.conn is not None
        self.conn.commit()


def content_hash(data: bytes) -> str:
    """SHA256 hash of file content."""
    return hashlib.sha256(data).hexdigest()
