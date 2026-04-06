"""Tests for community detection, execution flow tracing, and impact analysis."""

import pytest

from srclight.db import Database, FileRecord, SymbolRecord, EdgeRecord


@pytest.fixture
def db(tmp_path):
    """Create a temporary database with schema v5."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def _build_test_graph(db):
    """Build a small call graph with two clear clusters.

    Cluster A (auth): login -> validate_password -> hash_password
    Cluster B (db):   query_users -> connect_db -> execute_sql
    Cross-cluster:    login -> query_users (bridge edge)
    """
    f1 = db.upsert_file(FileRecord(
        path="src/auth.py", content_hash="a1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))
    f2 = db.upsert_file(FileRecord(
        path="src/database.py", content_hash="b1", mtime=1.0,
        language="python", size=100, line_count=20,
    ))

    sym_login = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="login",
        qualified_name="auth.login", start_line=1, end_line=10,
        content="def login(): validate_password(); query_users()",
    ), file_path="src/auth.py")
    sym_validate = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="validate_password",
        qualified_name="auth.validate_password", start_line=11, end_line=20,
        content="def validate_password(): hash_password()",
    ), file_path="src/auth.py")
    sym_hash = db.insert_symbol(SymbolRecord(
        file_id=f1, kind="function", name="hash_password",
        qualified_name="auth.hash_password", start_line=21, end_line=30,
        content="def hash_password(): pass",
    ), file_path="src/auth.py")

    sym_query = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="query_users",
        qualified_name="database.query_users", start_line=1, end_line=10,
        content="def query_users(): connect_db()",
    ), file_path="src/database.py")
    sym_connect = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="connect_db",
        qualified_name="database.connect_db", start_line=11, end_line=20,
        content="def connect_db(): execute_sql()",
    ), file_path="src/database.py")
    sym_exec = db.insert_symbol(SymbolRecord(
        file_id=f2, kind="function", name="execute_sql",
        qualified_name="database.execute_sql", start_line=21, end_line=30,
        content="def execute_sql(): pass",
    ), file_path="src/database.py")

    # Cluster A internal (triangle for strong internal connectivity)
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_validate, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_validate, target_id=sym_hash, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_hash, edge_type="calls"))
    # Cluster B internal (triangle)
    db.insert_edge(EdgeRecord(source_id=sym_query, target_id=sym_connect, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_connect, target_id=sym_exec, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=sym_query, target_id=sym_exec, edge_type="calls"))
    # Bridge (single weak link)
    db.insert_edge(EdgeRecord(source_id=sym_login, target_id=sym_query, edge_type="calls"))

    db.commit()

    return {
        "login": sym_login, "validate_password": sym_validate,
        "hash_password": sym_hash, "query_users": sym_query,
        "connect_db": sym_connect, "execute_sql": sym_exec,
    }


def test_detect_communities_two_clusters(db):
    """Louvain should detect two communities in a graph with two clear clusters."""
    from srclight.community import detect_communities

    syms = _build_test_graph(db)
    communities = detect_communities(db)

    assert len(communities) >= 2

    login_comm = None
    query_comm = None
    for c in communities:
        member_ids = {m["id"] for m in c["members"]}
        if syms["login"] in member_ids:
            login_comm = c
        if syms["query_users"] in member_ids:
            query_comm = c

    assert login_comm is not None
    assert query_comm is not None
    assert login_comm["id"] != query_comm["id"]
