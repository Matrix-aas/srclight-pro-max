"""Workspace-scoped learnings storage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LearningRecord:
    kind: str
    content: str
    reasoning: str | None = None
    scope: str = "workspace"
    project: str | None = None
    confidence: float = 1.0
    ttl_days: int | None = None


@dataclass
class ConversationRecord:
    session_id: str
    task_summary: str
    project: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


class LearningsDB:
    """SQLite-backed workspace learnings database."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def initialize(self) -> None:
        assert self.conn is not None
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                reasoning TEXT,
                scope TEXT NOT NULL,
                project TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                ttl_days INTEGER,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                expires_at TEXT
            );
            CREATE TABLE IF NOT EXISTS learning_symbols (
                learning_id INTEGER NOT NULL REFERENCES learnings(id) ON DELETE CASCADE,
                symbol_name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_sources (
                learning_id INTEGER NOT NULL REFERENCES learnings(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS learnings_fts USING fts5(
                content,
                reasoning,
                project,
                kind,
                tokenize = 'porter unicode61'
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                project TEXT,
                task_summary TEXT NOT NULL,
                model TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                cost_usd REAL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            """
        )
        self.conn.commit()

    def record_learning(
        self,
        rec: LearningRecord,
        *,
        symbols: list[str] | None = None,
        sources: list[dict[str, str]] | None = None,
    ) -> int:
        assert self.conn is not None
        expires_at = None
        if rec.ttl_days is not None:
            row = self.conn.execute(
                "SELECT datetime('now', ?)",
                (f"+{int(rec.ttl_days)} days",),
            ).fetchone()
            expires_at = row[0] if row else None

        cur = self.conn.execute(
            """INSERT INTO learnings (
                   kind, content, reasoning, scope, project, confidence, ttl_days, expires_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.kind,
                rec.content,
                rec.reasoning,
                rec.scope,
                rec.project,
                rec.confidence,
                rec.ttl_days,
                expires_at,
            ),
        )
        learning_id = int(cur.lastrowid)
        self.conn.execute(
            (
                "INSERT INTO learnings_fts("
                "rowid, content, reasoning, project, kind"
                ") VALUES (?, ?, ?, ?, ?)"
            ),
            (learning_id, rec.content, rec.reasoning or "", rec.project or "", rec.kind),
        )
        for symbol_name in symbols or []:
            self.conn.execute(
                "INSERT INTO learning_symbols(learning_id, symbol_name) VALUES (?, ?)",
                (learning_id, symbol_name),
            )
        for source in sources or []:
            self.conn.execute(
                (
                    "INSERT INTO learning_sources("
                    "learning_id, source_type, source_ref"
                    ") VALUES (?, ?, ?)"
                ),
                (learning_id, source.get("type", ""), source.get("ref", "")),
            )
        self.conn.commit()
        return learning_id

    def record_conversation(self, rec: ConversationRecord) -> int:
        assert self.conn is not None
        cur = self.conn.execute(
            """INSERT INTO conversations (
                   session_id, project, task_summary, model, tokens_in, tokens_out, cost_usd
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.session_id,
                rec.project,
                rec.task_summary,
                rec.model,
                rec.tokens_in,
                rec.tokens_out,
                rec.cost_usd,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def search_fts(
        self,
        query: str,
        *,
        kind: str | None = None,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        assert self.conn is not None
        sql = """
            SELECT l.*, bm25(learnings_fts) AS score
            FROM learnings_fts
            JOIN learnings l ON l.id = learnings_fts.rowid
            WHERE learnings_fts MATCH ?
              AND (l.expires_at IS NULL OR l.expires_at > datetime('now'))
        """
        params: list[Any] = [query]
        if kind:
            sql += " AND l.kind = ?"
            params.append(kind)
        if project:
            sql += " AND l.project = ?"
            params.append(project)
        sql += " ORDER BY score ASC, l.created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def hybrid_search(
        self,
        fts_results: list[dict[str, Any]],
        embedding_results: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for rank, row in enumerate(fts_results, start=1):
            entry = dict(row)
            entry["rrf_score"] = 1.0 / (60 + rank)
            merged[int(row["id"])] = entry
        for rank, row in enumerate(embedding_results, start=1):
            row_id = int(row["id"])
            if row_id in merged:
                merged[row_id]["rrf_score"] += 1.0 / (60 + rank)
            else:
                entry = dict(row)
                entry["rrf_score"] = 1.0 / (60 + rank)
                merged[row_id] = entry
        ranked = sorted(
            merged.values(),
            key=lambda item: item.get("rrf_score", 0.0),
            reverse=True,
        )
        return ranked[:limit]

    def stats(self, *, project: str | None = None, days: int | None = None) -> dict[str, Any]:
        assert self.conn is not None
        where = ["(expires_at IS NULL OR expires_at > datetime('now'))"]
        params: list[Any] = []
        if project:
            where.append("project = ?")
            params.append(project)
        if days is not None:
            where.append("created_at >= datetime('now', ?)")
            params.append(f"-{int(days)} days")
        where_sql = " WHERE " + " AND ".join(where)

        total = int(self.conn.execute(
            "SELECT COUNT(*) FROM learnings" + where_sql,
            params,
        ).fetchone()[0])
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) AS count FROM learnings"
            + where_sql
            + " GROUP BY kind ORDER BY count DESC, kind",
            params,
        ).fetchall()
        return {
            "project": project,
            "days": days,
            "total": total,
            "by_kind": {row["kind"]: row["count"] for row in rows},
        }
