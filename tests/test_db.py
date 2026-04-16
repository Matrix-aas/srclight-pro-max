"""Tests for the database layer."""


import sqlite3

import pytest

from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord, content_hash


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def test_initialize(db):
    """Database initializes with all tables."""
    stats = db.stats()
    assert stats["files"] == 0
    assert stats["symbols"] == 0
    assert stats["edges"] == 0


def test_upsert_file(db):
    """Can insert and update file records."""
    rec = FileRecord(
        path="src/main.py",
        content_hash="abc123",
        mtime=1000.0,
        language="python",
        size=500,
        line_count=25,
    )
    file_id = db.upsert_file(rec)
    assert file_id > 0

    # Retrieve it
    got = db.get_file("src/main.py")
    assert got is not None
    assert got.content_hash == "abc123"
    assert got.language == "python"

    # Update it
    rec.content_hash = "def456"
    db.upsert_file(rec)
    got = db.get_file("src/main.py")
    assert got.content_hash == "def456"


def test_initialize_migrates_file_summary_columns(tmp_path):
    """Older DBs gain summary and metadata columns during initialize."""
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO schema_info (key, value) VALUES ('schema_version', '5');
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            content_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            language TEXT,
            size INTEGER,
            line_count INTEGER,
            indexed_at TEXT
        );
        INSERT INTO files (path, content_hash, mtime, language, size, line_count, indexed_at)
        VALUES ('legacy.py', 'abc123', 1.0, 'python', 42, 3, '2026-01-01T00:00:00.000Z');
    """)
    conn.commit()
    conn.close()

    db = Database(db_path)
    db.open()
    try:
        db.initialize()
        migrated = db.get_file("legacy.py")
        assert migrated is not None
        assert migrated.summary is None
        assert migrated.metadata is None

        columns = {
            row["name"] for row in db.conn.execute("PRAGMA table_info(files)").fetchall()
        }
        assert "summary" in columns
        assert "metadata" in columns
        assert "embedding_context_hash" in columns
    finally:
        db.close()


def test_file_needs_reindex(db):
    """Change detection works via content hash."""
    rec = FileRecord(
        path="src/main.py", content_hash="abc123",
        mtime=1000.0, language="python", size=100, line_count=10,
    )
    db.upsert_file(rec)

    assert not db.file_needs_reindex("src/main.py", "abc123")
    assert db.file_needs_reindex("src/main.py", "different_hash")
    assert db.file_needs_reindex("nonexistent.py", "abc123")


def test_list_files_filters_by_prefix_and_recursion(db):
    """list_files supports shallow and recursive prefix filtering."""
    for path in [
        "shared/src/domain/aggregate.ts",
        "shared/src/domain/level/nested.ts",
        "shared/src/domain/level/deeper/more.ts",
        "shared/src/other/util.ts",
    ]:
        db.upsert_file(FileRecord(
            path=path,
            content_hash=path,
            mtime=1000.0,
            language="typescript",
            size=100,
            line_count=10,
        ))
    db.commit()

    shallow = db.list_files(path_prefix="shared/src/domain", recursive=False)
    assert [item["path"] for item in shallow] == ["shared/src/domain/aggregate.ts"]

    deep = db.list_files(path_prefix="shared/src/domain", recursive=True, limit=2)
    assert len(deep) == 2
    assert all(item["path"].startswith("shared/src/domain/") for item in deep)

    level = db.list_files(path_prefix="shared/src/domain/level", recursive=False)
    assert [item["path"] for item in level] == ["shared/src/domain/level/nested.ts"]


def test_list_files_treats_like_metacharacters_in_prefix_as_literals(db):
    """Underscore and percent in path_prefix are matched literally."""
    for path in [
        "shared/src/domain_100%/exact.ts",
        "shared/src/domainA100x/wrong.ts",
        "shared/src/domain_100x/also-wrong.ts",
    ]:
        db.upsert_file(FileRecord(
            path=path,
            content_hash=path,
            mtime=1000.0,
            language="typescript",
            size=100,
            line_count=10,
        ))
    db.commit()

    results = db.list_files(path_prefix="shared/src/domain_100%", recursive=True)
    assert [item["path"] for item in results] == ["shared/src/domain_100%/exact.ts"]


def test_list_files_non_recursive_excludes_nested_descendants_in_sql(db):
    """Non-recursive listing returns only immediate children under the prefix."""
    for path in [
        "shared/src/domain/direct.ts",
        "shared/src/domain/level/nested.ts",
        "shared/src/domain/level/deeper/more.ts",
    ]:
        db.upsert_file(FileRecord(
            path=path,
            content_hash=path,
            mtime=1000.0,
            language="typescript",
            size=100,
            line_count=10,
        ))
    db.commit()

    shallow = db.list_files(path_prefix="shared/src/domain", recursive=False, limit=10)
    assert [item["path"] for item in shallow] == ["shared/src/domain/direct.ts"]


def test_update_and_get_file_summary(db):
    """File summaries persist lightweight metadata and expose file TOC data."""
    file_id = db.upsert_file(FileRecord(
        path="client/src/components/ProfileCard.vue",
        content_hash="profile-card",
        mtime=1000.0,
        language="vue",
        size=240,
        line_count=20,
    ))
    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="component",
        name="ProfileCard",
        qualified_name="ProfileCard",
        signature="<script setup>",
        start_line=1,
        end_line=20,
        content="<template><div /></template>",
        line_count=20,
    ), "client/src/components/ProfileCard.vue")
    db.commit()

    db.update_file_summary(
        "client/src/components/ProfileCard.vue",
        summary="Renders the account profile card shell.",
        metadata={"framework": "vue", "tags": ["ui", "profile"]},
    )
    db.commit()

    file_rec = db.get_file("client/src/components/ProfileCard.vue")
    assert file_rec is not None
    assert file_rec.summary == "Renders the account profile card shell."
    assert file_rec.metadata == {"framework": "vue", "tags": ["ui", "profile"]}

    summary = db.get_file_summary("client/src/components/ProfileCard.vue")
    assert summary is not None
    assert summary["file"] == "client/src/components/ProfileCard.vue"
    assert summary["summary"] == "Renders the account profile card shell."
    assert summary["metadata"] == {"framework": "vue", "tags": ["ui", "profile"]}
    assert summary["top_level_symbols"][0]["name"] == "ProfileCard"


def test_update_file_summary_preserves_untouched_field_on_partial_update(db):
    """Partial file-summary updates do not null the omitted field."""
    path = "client/src/components/ProfileCard.vue"
    db.upsert_file(FileRecord(
        path=path,
        content_hash="profile-card",
        mtime=1000.0,
        language="vue",
        size=240,
        line_count=20,
    ))
    db.update_file_summary(
        path,
        summary="Original summary.",
        metadata={"framework": "vue", "tags": ["profile"]},
    )
    db.commit()

    db.update_file_summary(path, summary="Updated summary only.")
    db.commit()

    updated = db.get_file(path)
    assert updated is not None
    assert updated.summary == "Updated summary only."
    assert updated.metadata == {"framework": "vue", "tags": ["profile"]}

    db.update_file_summary(path, metadata={"framework": "vue", "tags": ["card"]})
    db.commit()

    updated_again = db.get_file(path)
    assert updated_again is not None
    assert updated_again.summary == "Updated summary only."
    assert updated_again.metadata == {"framework": "vue", "tags": ["card"]}


def test_get_symbols_needing_embeddings_limits_materialization_to_stale_rows(db):
    """Freshness filtering should happen in SQL, not by scanning every row in Python."""
    path = "client/src/components/ProfileCard.vue"
    file_id = db.upsert_file(FileRecord(
        path=path,
        content_hash="profile-card",
        mtime=1000.0,
        language="vue",
        size=240,
        line_count=20,
        summary="Renders the account profile card shell.",
        metadata={"framework": "vue", "resource": "component", "props": ["msg"]},
    ))

    symbol_ids = []
    for i in range(30):
        symbol_ids.append(
            db.insert_symbol(
                SymbolRecord(
                    file_id=file_id,
                    kind="component",
                    name=f"Symbol{i}",
                    qualified_name=f"ProfileCard.Symbol{i}",
                    start_line=i + 1,
                    end_line=i + 2,
                    content=f"<div>{i}</div>",
                    body_hash=f"h{i}",
                ),
                path,
            )
        )
    db.commit()

    needing = db.get_symbols_needing_embeddings("mock:test-model")
    assert len(needing) == 30

    for row in needing:
        db.upsert_embedding(
            row["id"],
            "mock:test-model",
            4,
            b"\x00" * 16,
            row["embedding_body_hash"],
        )
    db.commit()

    db.update_file_summary(
        path,
        summary="Renders the updated account profile card shell.",
        metadata={"framework": "vue", "resource": "component", "props": ["msg"]},
    )
    db.commit()

    calls = 0
    original = db._row_to_symbol

    def wrapped(row):
        nonlocal calls
        calls += 1
        return original(row)

    db._row_to_symbol = wrapped  # type: ignore[assignment]
    try:
        stale = db.get_symbols_needing_embeddings("mock:test-model", limit=1)
    finally:
        db._row_to_symbol = original  # type: ignore[assignment]

    assert len(stale) == 1
    assert stale[0]["id"] in symbol_ids
    assert calls == 1


def test_upsert_file_preserves_existing_summary_metadata_when_incoming_values_are_none(db):
    """Reindex upserts do not wipe persisted summary metadata unless explicitly replaced."""
    path = "client/src/components/ProfileCard.vue"
    db.upsert_file(FileRecord(
        path=path,
        content_hash="profile-card-v1",
        mtime=1000.0,
        language="vue",
        size=240,
        line_count=20,
    ))
    db.update_file_summary(
        path,
        summary="Original profile card summary.",
        metadata={"framework": "vue", "tags": ["profile"]},
    )
    db.commit()

    db.upsert_file(FileRecord(
        path=path,
        content_hash="profile-card-v2",
        mtime=1001.0,
        language="vue",
        size=245,
        line_count=21,
        summary=None,
        metadata=None,
    ))
    db.commit()

    preserved = db.get_file(path)
    assert preserved is not None
    assert preserved.content_hash == "profile-card-v2"
    assert preserved.summary == "Original profile card summary."
    assert preserved.metadata == {"framework": "vue", "tags": ["profile"]}

    db.upsert_file(FileRecord(
        path=path,
        content_hash="profile-card-v3",
        mtime=1002.0,
        language="vue",
        size=250,
        line_count=22,
        summary="Updated profile card summary.",
        metadata={"framework": "vue", "tags": ["profile", "card"]},
    ))
    db.commit()

    replaced = db.get_file(path)
    assert replaced is not None
    assert replaced.summary == "Updated profile card summary."
    assert replaced.metadata == {"framework": "vue", "tags": ["profile", "card"]}


def test_insert_symbol_and_search(db):
    """Can insert symbols and find them via FTS5."""
    # Insert a file first
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    # Insert a symbol
    sym = SymbolRecord(
        file_id=file_id,
        kind="function",
        name="calculate_total",
        qualified_name="src/main.py::calculate_total",
        signature="def calculate_total(items: list) -> float",
        start_line=10,
        end_line=20,
        content="def calculate_total(items: list) -> float:\n    return sum(i.price for i in items)",
        doc_comment="Calculate the total price of all items.",
        line_count=11,
    )
    sym_id = db.insert_symbol(sym, "src/main.py")
    assert sym_id > 0

    # Search by name
    results = db.search_symbols("calculate_total")
    assert len(results) > 0
    assert results[0]["name"] == "calculate_total"

    # Search by content (trigram)
    results = db.search_symbols("price")
    assert len(results) > 0

    # Search by doc (porter stemmed)
    results = db.search_symbols("calculating prices")
    assert len(results) > 0


def test_symbols_in_file(db):
    """Can list all symbols in a file."""
    file_id = db.upsert_file(FileRecord(
        path="src/lib.py", content_hash="abc",
        mtime=1000.0, language="python", size=200, line_count=30,
    ))

    for i, name in enumerate(["foo", "bar", "baz"]):
        db.insert_symbol(SymbolRecord(
            file_id=file_id, kind="function", name=name,
            start_line=i * 10 + 1, end_line=i * 10 + 8,
            content=f"def {name}(): pass", line_count=8,
        ), "src/lib.py")

    db.commit()
    syms = db.symbols_in_file("src/lib.py")
    assert len(syms) == 3
    assert [s.name for s in syms] == ["foo", "bar", "baz"]


def test_edges(db):
    """Can insert and query symbol relationships."""
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    caller_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="main",
        start_line=1, end_line=5, content="def main(): calc()", line_count=5,
    ), "src/main.py")

    callee_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="calc",
        start_line=10, end_line=15, content="def calc(): pass", line_count=6,
    ), "src/main.py")

    db.insert_edge(EdgeRecord(
        source_id=caller_id, target_id=callee_id, edge_type="calls",
    ))
    db.commit()

    callers = db.get_callers(callee_id)
    assert len(callers) == 1
    assert callers[0]["symbol"].name == "main"
    assert callers[0]["edge_type"] == "calls"

    callees = db.get_callees(caller_id)
    assert len(callees) == 1
    assert callees[0]["symbol"].name == "calc"


def test_content_hash():
    """SHA256 content hashing works."""
    h1 = content_hash(b"hello world")
    h2 = content_hash(b"hello world")
    h3 = content_hash(b"different content")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA256 hex


def test_stats(db):
    """Stats reflect database contents."""
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc",
        mtime=1000.0, language="python", size=100, line_count=10,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="foo",
        start_line=1, end_line=5, content="def foo(): pass", line_count=5,
    ), "src/main.py")

    db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="class", name="Bar",
        start_line=10, end_line=20, content="class Bar: pass", line_count=11,
    ), "src/main.py")

    db.commit()
    stats = db.stats()
    assert stats["files"] == 1
    assert stats["symbols"] == 2
    assert stats["languages"]["python"] == 1
    assert stats["symbol_kinds"]["function"] == 1
    assert stats["symbol_kinds"]["class"] == 1
