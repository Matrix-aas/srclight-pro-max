"""Tests for new features: search quality, edges, templates, parent-child, qualified names."""

import json

import pytest

import srclight.server as server
from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord, split_identifier
from srclight.indexer import IndexConfig, Indexer


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def _insert_search_symbol(
    db: Database,
    *,
    path: str,
    kind: str,
    name: str,
    signature: str | None = None,
    content: str | None = None,
    doc_comment: str | None = None,
    metadata: dict | None = None,
) -> None:
    file_id = db.upsert_file(
        FileRecord(
            path=path,
            content_hash=f"{path}:{name}",
            mtime=1.0,
            language="typescript",
            size=200,
            line_count=20,
        )
    )
    db.insert_symbol(
        SymbolRecord(
            file_id=file_id,
            kind=kind,
            name=name,
            qualified_name=name,
            signature=signature,
            start_line=1,
            end_line=10,
            content=content or name,
            doc_comment=doc_comment,
            line_count=10,
            metadata=metadata,
        ),
        path,
    )


def _store_server_flow_graph(db: Database) -> None:
    from srclight.community import detect_communities, trace_execution_flows

    server_file = db.upsert_file(
        FileRecord(
            path="server/app.ts",
            content_hash="server/app.ts",
            mtime=1.0,
            language="typescript",
            size=200,
            line_count=20,
        )
    )
    server_db_file = db.upsert_file(
        FileRecord(
            path="server/data.ts",
            content_hash="server/data.ts",
            mtime=1.0,
            language="typescript",
            size=200,
            line_count=20,
        )
    )
    worker_file = db.upsert_file(
        FileRecord(
            path="worker/jobs.ts",
            content_hash="worker/jobs.ts",
            mtime=1.0,
            language="typescript",
            size=200,
            line_count=20,
        )
    )

    bootstrap = db.insert_symbol(
        SymbolRecord(
            file_id=server_file,
            kind="function",
            name="bootstrap",
            qualified_name="server.bootstrap",
            start_line=1,
            end_line=5,
            content="bootstrap calls routeRequest",
            line_count=5,
        ),
        "server/app.ts",
    )
    route_request = db.insert_symbol(
        SymbolRecord(
            file_id=server_file,
            kind="function",
            name="routeRequest",
            qualified_name="server.routeRequest",
            start_line=6,
            end_line=10,
            content="routeRequest calls fetchUser",
            line_count=5,
        ),
        "server/app.ts",
    )
    fetch_user = db.insert_symbol(
        SymbolRecord(
            file_id=server_db_file,
            kind="function",
            name="fetchUser",
            qualified_name="server.fetchUser",
            start_line=1,
            end_line=5,
            content="fetchUser calls hydrateUser",
            line_count=5,
        ),
        "server/data.ts",
    )
    hydrate_user = db.insert_symbol(
        SymbolRecord(
            file_id=server_db_file,
            kind="function",
            name="hydrateUser",
            qualified_name="server.hydrateUser",
            start_line=6,
            end_line=10,
            content="hydrateUser terminal",
            line_count=5,
        ),
        "server/data.ts",
    )
    run_jobs = db.insert_symbol(
        SymbolRecord(
            file_id=worker_file,
            kind="function",
            name="runJobs",
            qualified_name="worker.runJobs",
            start_line=1,
            end_line=5,
            content="runJobs calls sendEmail",
            line_count=5,
        ),
        "worker/jobs.ts",
    )
    send_email = db.insert_symbol(
        SymbolRecord(
            file_id=worker_file,
            kind="function",
            name="sendEmail",
            qualified_name="worker.sendEmail",
            start_line=6,
            end_line=10,
            content="sendEmail terminal",
            line_count=5,
        ),
        "worker/jobs.ts",
    )

    db.insert_edge(EdgeRecord(source_id=bootstrap, target_id=route_request, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=route_request, target_id=fetch_user, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=fetch_user, target_id=hydrate_user, edge_type="calls"))
    db.insert_edge(EdgeRecord(source_id=run_jobs, target_id=send_email, edge_type="calls"))
    db.commit()

    communities = detect_communities(db)
    sym_to_comm = {
        member["id"]: community["id"]
        for community in communities
        for member in community["members"]
    }
    flows = trace_execution_flows(db, sym_to_comm)

    db.store_communities(communities)
    db.store_execution_flows(flows)
    db.commit()


# --- 1. CamelCase / identifier splitting ---


class TestSplitIdentifier:
    def test_camel_case(self):
        tokens = split_identifier("SQLiteDictionary")
        assert "Dictionary" in tokens
        assert "dictionary" in tokens

    def test_snake_case(self):
        tokens = split_identifier("get_callers")
        assert "get" in tokens
        assert "callers" in tokens

    def test_acronym(self):
        tokens = split_identifier("OCRManager")
        assert "OCR" in tokens
        assert "Manager" in tokens

    def test_cpp_qualified(self):
        tokens = split_identifier("myapp::util::ConfigManager")
        assert "myapp" in tokens
        assert "util" in tokens
        assert "Config" in tokens
        assert "Manager" in tokens

    def test_empty(self):
        assert split_identifier("") == ""
        assert split_identifier(None) == ""


class TestCamelCaseSearch:
    def test_find_camel_case_by_suffix(self, db):
        """Searching 'Dictionary' should find 'SQLiteDictionary'."""
        fid = db.upsert_file(FileRecord(
            path="test.cpp", content_hash="abc", mtime=1.0,
            language="cpp", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="class", name="SQLiteDictionary",
            qualified_name="SQLiteDictionary",
            start_line=1, end_line=10,
            content="class SQLiteDictionary {};", line_count=10,
        ), "test.cpp")
        db.commit()

        results = db.search_symbols("Dictionary")
        assert any(r["name"] == "SQLiteDictionary" for r in results)

    def test_find_camel_case_by_prefix(self, db):
        """Searching 'Broker' should find 'BrokerClientImpl'."""
        fid = db.upsert_file(FileRecord(
            path="test.cpp", content_hash="abc", mtime=1.0,
            language="cpp", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="class", name="BrokerClientImpl",
            qualified_name="BrokerClientImpl",
            start_line=1, end_line=10,
            content="class BrokerClientImpl {};", line_count=10,
        ), "test.cpp")
        db.commit()

        results = db.search_symbols("Broker")
        assert any(r["name"] == "BrokerClientImpl" for r in results)


class TestSearchRanking:
    def test_route_query_ranking_prefers_route_surfaces(self, db):
        _insert_search_symbol(
            db,
            path="server/src/routes/auth.ts",
            kind="router",
            name="authRoutes",
            signature="Elysia router | /api/auth",
            content="auth routes router refresh login",
            doc_comment="Auth routes for login and refresh.",
            metadata={"framework": "elysia", "resource": "router", "route_prefix": "/api/auth"},
        )
        _insert_search_symbol(
            db,
            path="server/src/services/auth-routes-service.ts",
            kind="class",
            name="AuthRoutesService",
            signature="class AuthRoutesService",
            content="auth routes service helpers",
            doc_comment="Service object for auth routes coordination.",
        )
        db.commit()

        results = db.search_symbols("auth routes")

        assert results
        assert results[0]["kind"] in {"route", "router", "route_handler"}

    def test_mikro_orm_ranking_prefers_persistence_symbols(self, db):
        _insert_search_symbol(
            db,
            path="server/src/db/mikroorm.ts",
            kind="entity",
            name="User",
            signature="mikro orm entity | User | users",
            content="mikro orm entity user users table",
            doc_comment="Mikro ORM entity for users.",
            metadata={"framework": "mikroorm", "resource": "entity", "table_name": "users"},
        )
        _insert_search_symbol(
            db,
            path="server/src/db/mikroorm.ts",
            kind="repository",
            name="UserRepository",
            signature="mikro orm repository | UserRepository | User",
            content="mikro orm repository user entities",
            doc_comment="Mikro ORM repository for User entities.",
            metadata={"framework": "mikroorm", "resource": "repository", "entity_name": "User"},
        )
        _insert_search_symbol(
            db,
            path="server/src/db/mikroorm.ts",
            kind="database",
            name="orm",
            signature="mikro orm database",
            content="mikro orm database entities init",
            doc_comment="Mikro ORM database initialization for entities.",
            metadata={"framework": "mikroorm", "resource": "database", "entity_names": ["User"]},
        )
        _insert_search_symbol(
            db,
            path="docs/mikro-orm-guide.md",
            kind="class",
            name="MikroOrmEntitiesGuide",
            signature="MikroOrmEntitiesGuide",
            content="mikro orm entities guide overview",
            doc_comment="Mikro ORM entities guide and overview.",
        )
        db.commit()

        results = db.search_symbols("mikro orm entities")

        assert results
        assert results[0]["kind"] in {"entity", "repository", "database"}

    def test_hybrid_search_rmq_handlers_prefers_async_symbols(self, monkeypatch, db):
        _insert_search_symbol(
            db,
            path="docs/rmq-handlers-guide.md",
            kind="class",
            name="RmqHandlersGuide",
            signature="RmqHandlersGuide",
            content="rmq handlers guide architecture overview",
            doc_comment="Guide to rmq handlers and background workers.",
        )
        _insert_search_symbol(
            db,
            path="server/src/messaging/diary.consumer.ts",
            kind="microservice_handler",
            name="handleDiaryPush",
            signature="RabbitMQ consumer diary note push handler",
            content="consumer handler for diary note push events",
            metadata={
                "framework": "nestjs",
                "resource": "microservice_handler",
                "message_pattern": "diary.note.push",
                "pattern": "diary.note.push",
                "transport": "rmq",
                "role": "consumer",
            },
        )
        db.commit()

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)

        payload = json.loads(server.hybrid_search("rmq handlers"))

        assert payload["results"]
        assert payload["results"][0]["kind"] == "microservice_handler"

    def test_hybrid_search_message_patterns_prefers_async_symbols(self, monkeypatch, db):
        _insert_search_symbol(
            db,
            path="docs/message-patterns-overview.md",
            kind="class",
            name="MessagePatternsOverview",
            signature="MessagePatternsOverview",
            content="message patterns overview for distributed systems",
            doc_comment="High level guide to message patterns.",
        )
        _insert_search_symbol(
            db,
            path="server/src/messaging/diary.consumer.ts",
            kind="microservice_handler",
            name="handleDiaryPush",
            signature="MessagePattern diary note push consumer",
            content="consumer for diary note push messages",
            metadata={
                "framework": "nestjs",
                "resource": "microservice_handler",
                "message_pattern": "diary.note.push",
                "pattern": "diary.note.push",
                "transport": "rmq",
                "role": "consumer",
            },
        )
        db.commit()

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)

        payload = json.loads(server.hybrid_search("message patterns"))

        assert payload["results"]
        assert payload["results"][0]["kind"] == "microservice_handler"

    def test_hybrid_search_rabbitmq_bootstrap_prefers_async_config_surfaces(self, monkeypatch, db):
        _insert_search_symbol(
            db,
            path="docs/rabbitmq-bootstrap.md",
            kind="class",
            name="RabbitMqBootstrapGuide",
            signature="RabbitMqBootstrapGuide",
            content="rabbitmq bootstrap guide and setup notes",
            doc_comment="Guide to rabbitmq bootstrap and setup.",
        )
        _insert_search_symbol(
            db,
            path="server/src/config/rabbitmq.config.ts",
            kind="module",
            name="RabbitMqModule",
            signature="RabbitMq bootstrap module",
            content="register rabbitmq bootstrap config transport",
            metadata={
                "framework": "nestjs",
                "resource": "transport",
                "transport": "rabbitmq",
                "role": "bootstrap",
                "connection_url": "amqp://guest:guest@localhost:5672",
            },
        )
        db.commit()

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)

        payload = json.loads(server.hybrid_search("rabbitmq bootstrap"))

        assert payload["results"]
        assert payload["results"][0]["name"] == "RabbitMqModule"

    def test_hybrid_search_with_embeddings_surfaces_async_symbol_first(self, monkeypatch, db):
        _insert_search_symbol(
            db,
            path="docs/message-patterns-overview.md",
            kind="class",
            name="MessagePatternsOverview",
            signature="MessagePatternsOverview",
            content="message patterns overview for distributed systems",
            doc_comment="High level guide to message patterns.",
        )
        handler_file_id = db.upsert_file(FileRecord(
            path="server/src/messaging/diary.consumer.ts",
            content_hash="server/src/messaging/diary.consumer.ts:handleDiaryPush",
            mtime=1.0,
            language="typescript",
            size=200,
            line_count=20,
        ))
        handler_id = db.insert_symbol(
            SymbolRecord(
                file_id=handler_file_id,
                kind="microservice_handler",
                name="handleDiaryPush",
                qualified_name="handleDiaryPush",
                signature="RpcRequest consumer handler",
                start_line=1,
                end_line=10,
                content="consumer handler for diary note push messages",
                doc_comment="Consumes diary note push messages over RMQ.",
                line_count=10,
                metadata={
                    "framework": "nestjs",
                    "resource": "microservice_handler",
                    "message_pattern": "diary.note.push",
                    "pattern": "diary.note.push",
                    "transport": "rmq",
                    "role": "consumer",
                },
            ),
            "server/src/messaging/diary.consumer.ts",
        )
        db.upsert_embedding(handler_id, "mock:test-model", 4, b"\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        db.commit()

        class _FakeProvider:
            def embed_one(self, query):
                assert query == "message patterns"
                return [1.0, 0.0, 0.0, 0.0]

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr("srclight.embeddings.get_provider", lambda model_name: _FakeProvider())

        payload = json.loads(server.hybrid_search("message patterns"))

        assert payload["mode"] == "hybrid (FTS5 + embeddings)"
        assert payload["model"] == "mock:test-model"
        assert payload["results"]
        assert payload["results"][0]["kind"] == "microservice_handler"
        assert "rrf_score" not in payload["results"][0]
        assert payload["results"][0]["sources"] == ["fts", "embedding"]
        assert payload["results"][0]["rank_source"] in {"keyword", "semantic", "hybrid"}
        assert payload["results"][0]["match_reasons"]

    def test_search_symbols_tokenizes_dotted_async_patterns_for_handlers(self, db):
        _insert_search_symbol(
            db,
            path="docs/diary-note-push.md",
            kind="class",
            name="DiaryNotePushGuide",
            signature="DiaryNotePushGuide",
            content="diary note push documentation and examples",
            doc_comment="Guide to diary note push events.",
        )
        _insert_search_symbol(
            db,
            path="server/src/messaging/diary.consumer.ts",
            kind="microservice_handler",
            name="handleDiaryPush",
            signature="RpcRequest consumer handler",
            content="consumer handler for diary note push messages",
            metadata={
                "framework": "nestjs",
                "resource": "microservice_handler",
                "message_pattern": "diary.note.push",
                "pattern": "diary.note.push",
                "transport": "rmq",
                "role": "consumer",
            },
        )
        db.commit()

        results = db.search_symbols("diary.note.push", limit=5)

        assert results
        assert results[0]["kind"] == "microservice_handler"
        assert results[0]["name"] == "handleDiaryPush"


def test_fallback_hint_suggests_tokenized_identifier_search(monkeypatch, db):
    _insert_search_symbol(
        db,
        path="server/src/services/auth-routes-service.ts",
        kind="class",
        name="AuthRoutesService",
        signature="class AuthRoutesService",
        content="auth routes service helpers",
        doc_comment="Service object for auth routes coordination.",
    )
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server._symbol_not_found_error("authRoutes"))

    assert "AuthRoutesService" in payload["did_you_mean"]
    assert (
        'Try search_symbols("auth routes") for tokenized identifier search'
        in payload["suggestions"]
    )


def test_search_symbols_preserves_source_while_adding_rank_hints(monkeypatch, db):
    _insert_search_symbol(
        db,
        path="server/src/services/auth-routes-service.ts",
        kind="class",
        name="AuthRoutesService",
        signature="class AuthRoutesService",
        content="auth routes service helpers",
        doc_comment="Service object for auth routes coordination.",
    )
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server.search_symbols("auth routes"))

    assert payload
    assert payload[0]["source"] in {"name", "name_like", "tokenized_like", "metadata_like", "content", "docs"}
    assert payload[0]["rank_source"] == "keyword"
    assert payload[0]["match_reasons"]


def test_get_communities_summary_first_by_default(monkeypatch, db):
    _store_server_flow_graph(db)

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    communities = json.loads(
        server.get_communities(member_limit=1, path_prefix="server/", layer="server")
    )

    assert communities["communities"]
    assert communities["communities"][0]["member_count"] >= len(communities["communities"][0]["members"])
    assert len(communities["communities"][0]["members"]) <= 1
    assert communities["communities"][0]["truncated"] is True
    assert communities["communities"][0]["member_limit_applied"] == 1


def test_get_communities_verbose_returns_detailed_members(monkeypatch, db):
    _store_server_flow_graph(db)

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    communities = json.loads(server.get_communities(verbose=True))

    assert communities["communities"]
    assert communities["communities"][0]["members"]
    assert "qualified_name" in communities["communities"][0]["members"][0]


def test_get_execution_flows_summary_first_by_default(monkeypatch, db):
    _store_server_flow_graph(db)

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server.get_execution_flows())

    assert payload["flows"]
    assert payload["flows"][0]["truncated"] is True
    assert "steps" not in payload["flows"][0]


def test_get_execution_flows_verbose_and_filtered(monkeypatch, db):
    _store_server_flow_graph(db)

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    verbose = json.loads(server.get_execution_flows(verbose=True, max_depth=4))
    filtered = json.loads(server.get_execution_flows(path_prefix="server/", layer="server"))

    assert verbose["flows"]
    assert verbose["flows"][0]["steps"]
    assert verbose["flows"][0]["max_depth_applied"] == 4
    assert filtered["flows"]


def test_get_community_missing_symbol_returns_file_level_fallback(monkeypatch, db):
    _insert_search_symbol(
        db,
        path="src/ui/LayoutEngine.ts",
        kind="function",
        name="measureNode",
        signature="function measureNode(node)",
        content="measure layout node tree",
        doc_comment="Measures layout nodes for rendering.",
    )
    _insert_search_symbol(
        db,
        path="src/ui/LayoutEngine.ts",
        kind="function",
        name="renderFrame",
        signature="function renderFrame(frame)",
        content="render frame after layout measurement",
        doc_comment="Renders a frame after layout measurement.",
    )
    db.commit()
    _store_server_flow_graph(db)

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server.get_community("LayoutEngine"))

    assert payload["community"] is None
    assert payload["fallback_stage"] == "file_candidate"
    assert payload["file_candidates"]
    assert payload["file_candidates"][0]["path"] == "src/ui/LayoutEngine.ts"
    assert payload["next_step"]["tool"] == "symbols_in_file"


def test_get_community_missing_symbol_prefers_nearest_symbol_stage(monkeypatch, db):
    _insert_search_symbol(
        db,
        path="src/ui/layout-engine.ts",
        kind="class",
        name="LayoutEngineService",
        signature="class LayoutEngineService",
        content="layout engine service helpers",
        doc_comment="Coordinates layout engine helpers.",
    )
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server.get_community("LayoutEngine"))

    assert payload["community"] is None
    assert payload["fallback_stage"] == "nearest_symbol"
    assert payload["nearest_symbol"]["name"] == "LayoutEngineService"
    assert payload["next_step"]["tool"] == "get_symbol"
    assert 'get_symbol("LayoutEngineService")' == payload["next_step"]["call"]


def test_suggest_file_candidates_prefers_exact_filename_before_limit(db):
    for index in range(30):
        _insert_search_symbol(
            db,
            path=f"src/layout/LayoutEngineHelper{index}.ts",
            kind="function",
            name=f"helper{index}",
            signature=f"function helper{index}()",
            content="layout engine helper utilities",
        )
    _insert_search_symbol(
        db,
        path="src/ui/LayoutEngine.ts",
        kind="function",
        name="measureNode",
        signature="function measureNode()",
        content="measure layout engine nodes",
    )
    db.commit()

    candidates = db.suggest_file_candidates("LayoutEngine", limit=1)

    assert candidates
    assert candidates[0]["path"] == "src/ui/LayoutEngine.ts"
    assert candidates[0]["match_reason"] == "exact filename match"


# --- 2. Call graph edges ---


@pytest.fixture
def python_with_calls(tmp_path):
    """Project where functions call each other."""
    src = tmp_path / "proj"
    src.mkdir()
    (src / "main.py").write_text('''\
def helper():
    return 42

def process():
    result = helper()
    return result

class Worker:
    def run(self):
        return process()
''')
    return src


class TestEdges:
    def test_edges_populated(self, db, python_with_calls):
        """Indexing creates call graph edges."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        stats = indexer.index(python_with_calls)
        assert stats.edges_created > 0
        assert db.stats()["edges"] > 0

    def test_callers(self, db, python_with_calls):
        """get_callers finds the calling function."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        indexer.index(python_with_calls)

        # helper is called by process
        helper = db.get_symbol_by_name("helper")
        assert helper is not None
        callers = db.get_callers(helper.id)
        caller_names = [c["symbol"].name for c in callers]
        assert "process" in caller_names

    def test_callees(self, db, python_with_calls):
        """get_callees finds the called function."""
        indexer = Indexer(db, IndexConfig(root=python_with_calls))
        indexer.index(python_with_calls)

        process = db.get_symbol_by_name("process")
        assert process is not None
        callees = db.get_callees(process.id)
        callee_names = [c["symbol"].name for c in callees]
        assert "helper" in callee_names


# --- 3. C++ template name extraction ---


@pytest.fixture
def cpp_templates(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "templates.cpp").write_text('''\
template <typename T>
class Container {
    T value;
};

template <typename T>
T max_value(T a, T b) {
    return a > b ? a : b;
}

template <typename K, typename V>
struct Pair {
    K key;
    V val;
};
''')
    return src


class TestTemplateNames:
    def test_template_names_extracted(self, db, cpp_templates):
        """Template symbols have their inner names extracted."""
        indexer = Indexer(db, IndexConfig(root=cpp_templates))
        indexer.index(cpp_templates)

        syms = db.symbols_in_file("templates.cpp")
        names = [s.name for s in syms if s.name is not None]
        assert "Container" in names
        assert "Pair" in names
        # max_value might appear depending on query matching
        template_syms = [s for s in syms if s.kind == "template"]
        assert len(template_syms) > 0
        # All template symbols should have names
        for s in template_syms:
            assert s.name is not None, f"Template at line {s.start_line} has no name"


# --- 4. Parent-child relationships ---


class TestParentChild:
    def test_python_methods_have_parent(self, db, tmp_path):
        """Python methods inside a class have parent_symbol_id set."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "calc.py").write_text('''\
class Calculator:
    def add(self, a, b):
        return a + b
    def multiply(self, a, b):
        return a * b
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("calc.py")
        calc = next(s for s in syms if s.name == "Calculator")
        add = next(s for s in syms if s.name == "add")
        mul = next(s for s in syms if s.name == "multiply")

        assert add.parent_symbol_id == calc.id
        assert mul.parent_symbol_id == calc.id
        assert calc.parent_symbol_id is None


# --- 5. Multi-match get_symbol ---


class TestMultiMatch:
    def test_get_symbols_by_name_returns_all(self, db):
        """get_symbols_by_name returns all matching symbols."""
        fid1 = db.upsert_file(FileRecord(
            path="a.py", content_hash="a", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        fid2 = db.upsert_file(FileRecord(
            path="b.py", content_hash="b", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid1, kind="function", name="main",
            start_line=1, end_line=5, content="def main(): pass", line_count=5,
        ), "a.py")
        db.insert_symbol(SymbolRecord(
            file_id=fid2, kind="function", name="main",
            start_line=1, end_line=5, content="def main(): pass", line_count=5,
        ), "b.py")
        db.commit()

        results = db.get_symbols_by_name("main")
        assert len(results) == 2

    def test_get_symbols_fuzzy_fallback(self, db):
        """get_symbols_by_name falls back to LIKE matching."""
        fid = db.upsert_file(FileRecord(
            path="test.py", content_hash="a", mtime=1.0,
            language="python", size=100, line_count=10,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=fid, kind="function", name="calculate_total",
            start_line=1, end_line=5, content="def calculate_total(): pass",
            line_count=5,
        ), "test.py")
        db.commit()

        # Exact match returns nothing, but LIKE should find it
        results = db.get_symbols_by_name("calculate")
        assert len(results) == 1
        assert results[0].name == "calculate_total"


# --- 6. Enriched codebase_map ---


class TestCodebaseMap:
    def test_directory_summary(self, db):
        """directory_summary groups files by directory."""
        for name in ["src/a.py", "src/b.py", "lib/c.py"]:
            db.upsert_file(FileRecord(
                path=name, content_hash=name, mtime=1.0,
                language="python", size=100, line_count=10,
            ))
        db.commit()

        dirs = db.directory_summary()
        dir_paths = [d["path"] for d in dirs]
        assert "src" in dir_paths
        assert "lib" in dir_paths
        src_dir = next(d for d in dirs if d["path"] == "src")
        assert src_dir["files"] == 2

    def test_hotspot_files(self, db):
        """hotspot_files returns files with most symbols."""
        fid = db.upsert_file(FileRecord(
            path="big.py", content_hash="big", mtime=1.0,
            language="python", size=1000, line_count=100,
        ))
        for i in range(10):
            db.insert_symbol(SymbolRecord(
                file_id=fid, kind="function", name=f"fn_{i}",
                start_line=i * 10, end_line=i * 10 + 5,
                content=f"def fn_{i}(): pass", line_count=5,
            ), "big.py")
        db.commit()

        hotspots = db.hotspot_files(limit=5)
        assert len(hotspots) == 1
        assert hotspots[0]["path"] == "big.py"
        assert hotspots[0]["symbols"] == 10


# --- 7. C++ qualified names ---


class TestQualifiedNames:
    def test_cpp_namespace_class(self, db, tmp_path):
        """C++ symbols get proper namespace::class::name qualified names."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "test.cpp").write_text('''\
namespace outer {
namespace inner {

class MyClass {
};

} // namespace inner
} // namespace outer

void standalone() {}
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("test.cpp")
        my_class = next((s for s in syms if s.name == "MyClass"), None)
        assert my_class is not None
        assert my_class.qualified_name == "outer::inner::MyClass"

        standalone = next((s for s in syms if s.name == "standalone"), None)
        assert standalone is not None
        assert standalone.qualified_name == "standalone"

    def test_python_class_method(self, db, tmp_path):
        """Python methods get Class.method qualified names."""
        src = tmp_path / "proj"
        src.mkdir()
        (src / "test.py").write_text('''\
class Foo:
    def bar(self):
        pass
''')
        indexer = Indexer(db, IndexConfig(root=src))
        indexer.index(src)

        syms = db.symbols_in_file("test.py")
        bar = next((s for s in syms if s.name == "bar"), None)
        assert bar is not None
        assert bar.qualified_name == "Foo.bar"


@pytest.fixture
def backend_ownership_project(tmp_path):
    """Create a Nest backend where traversal relies on ownership edges."""
    src = tmp_path / "backend-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)
    (src / "server/src/config").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import { AuthService } from './auth.infra';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: AuthService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/auth.infra.ts").write_text('''\
import { Inject, Injectable } from '@nestjs/common';
import { ConfigType } from '@nestjs/config';
import { AuthConfig } from '../../config/auth.config';
import { UserRepository } from '../persistence/persistence.infra';

@Injectable()
export class AuthService {
  constructor(
    @Inject(AuthConfig.KEY)
    private readonly authConfig: ConfigType<typeof AuthConfig>,
    private readonly users: UserRepository,
  ) {}

  validate() {
    return this.authConfig.jwtSecret && this.users.findById();
  }
}

@Injectable()
export class JwtAuthGuard {
  canActivate() {
    return true;
  }
}
''')

    (src / "server/src/modules/persistence/persistence.infra.ts").write_text('''\
import { EntityRepository } from '@mikro-orm/core';

export class User {}

export class UserRepository extends EntityRepository<User> {
  findById() {
    return { id: 1 };
  }
}
''')

    (src / "server/src/modules/persistence/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { UserRepository } from './persistence.infra';

@Module({
  providers: [UserRepository],
  exports: [UserRepository],
})
export class PersistenceModule {}
''')

    (src / "server/src/config/auth.config.ts").write_text('''\
import { registerAs } from '@nestjs/config';

export const AuthConfig = registerAs('auth', () => ({
  jwtSecret: 'secret',
}));

export const authConfig = registerAs('auth.secondary', () => ({
  jwtAudience: 'users',
}));
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AuthConfig, authConfig } from '../../config/auth.config';
import { AuthController } from './auth.controller';
import { AuthService, JwtAuthGuard } from './auth.infra';
import { PersistenceModule } from '../persistence/persistence.module';

@Module({
  imports: [ConfigModule.forFeature(AuthConfig), ConfigModule.forFeature(authConfig), PersistenceModule],
  controllers: [AuthController],
  providers: [AuthService, JwtAuthGuard],
  exports: [AuthService],
})
export class AuthModule {}
''')

    return src


def test_backend_ownership_dependents_follow_module_exports(db, backend_ownership_project):
    """Dependents should traverse controller -> service -> imported module exports."""
    indexer = Indexer(db, IndexConfig(root=backend_ownership_project))
    indexer.index(backend_ownership_project)

    auth_service = db.get_symbol_by_name("AuthService")
    jwt_auth_guard = db.get_symbol_by_name("JwtAuthGuard")
    user_repository = db.get_symbol_by_name("UserRepository")
    auth_module = db.get_symbol_by_name("AuthModule")
    auth_config = db.get_symbol_by_name("AuthConfig")
    auth_config_secondary = db.get_symbol_by_name("authConfig")

    assert auth_service is not None
    assert jwt_auth_guard is not None
    assert user_repository is not None
    assert auth_module is not None
    assert auth_config is not None
    assert auth_config_secondary is not None

    auth_service_dependents = [item["symbol"].name for item in db.get_dependents(auth_service.id)]
    assert "AuthController" in auth_service_dependents

    jwt_auth_guard_dependents = [item["symbol"].name for item in db.get_dependents(jwt_auth_guard.id)]
    assert "AuthController" not in jwt_auth_guard_dependents

    user_repository_callers = [item["symbol"].name for item in db.get_callers(user_repository.id)]
    assert "AuthService" in user_repository_callers
    assert "AuthController" not in user_repository_callers

    transitive_user_repository_dependents = [
        item["symbol"].name for item in db.get_dependents(user_repository.id, transitive=True)
    ]
    assert "getMe" in transitive_user_repository_dependents

    auth_config_callees = [item["symbol"].name for item in db.get_callees(auth_config.id)]
    assert "AuthModule" in auth_config_callees
    assert "AuthService" in auth_config_callees

    auth_config_secondary_callees = [
        item["symbol"].name for item in db.get_callees(auth_config_secondary.id)
    ]
    assert "AuthModule" in auth_config_secondary_callees

    auth_module_callers = [item["symbol"].name for item in db.get_callers(auth_module.id)]
    assert "AuthConfig" in auth_module_callers

    auth_module_dependents = [item["symbol"].name for item in db.get_dependents(auth_module.id)]
    assert "AuthConfig" in auth_module_dependents
    assert "authConfig" in auth_module_dependents


def test_async_traversal_ownership_links_handlers_and_queue_resources(db, tmp_path):
    """Synthetic async symbols should still participate in ownership traversal."""
    src = tmp_path / "async-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/async/predictions.ts",
        content_hash="async-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="event_pattern",
        name="prediction.generated",
        qualified_name="prediction.generated",
        start_line=1,
        end_line=1,
        content="prediction.generated",
        line_count=1,
        metadata={"framework": "nestjs", "resource": "pattern"},
    ), "server/src/async/predictions.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="transport",
        name="PredictionsTransport",
        qualified_name="PredictionsTransport",
        start_line=2,
        end_line=2,
        content="PredictionsTransport",
        line_count=1,
        metadata={"framework": "nestjs", "resource": "transport", "transport": "rmq"},
    ), "server/src/async/predictions.ts")

    producer_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="microservice_handler",
        name="publishPredictionGenerated",
        qualified_name="publishPredictionGenerated",
        signature="prediction.generated",
        start_line=3,
        end_line=6,
        content="publishPredictionGenerated",
        line_count=4,
        metadata={
            "framework": "nestjs",
            "resource": "microservice_handler",
            "pattern": "prediction.generated",
            "transport": "PredictionsTransport",
        },
    ), "server/src/async/predictions.ts")

    consumer_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="microservice_handler",
        name="handlePredictionGenerated",
        qualified_name="handlePredictionGenerated",
        signature="prediction.generated",
        start_line=7,
        end_line=10,
        content="handlePredictionGenerated",
        line_count=4,
        metadata={
            "framework": "nestjs",
            "resource": "microservice_handler",
            "pattern": "prediction.generated",
            "transport": "PredictionsTransport",
        },
    ), "server/src/async/predictions.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="queue",
        name="notifications",
        qualified_name="notifications",
        start_line=11,
        end_line=11,
        content="notifications",
        line_count=1,
        metadata={"resource": "queue", "queue_name": "notifications"},
    ), "server/src/async/predictions.ts")

    worker_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="queue_processor",
        name="notificationWorker",
        qualified_name="notificationWorker",
        start_line=12,
        end_line=18,
        content="notificationWorker",
        line_count=7,
        metadata={
            "resource": "queue_processor",
            "queue_name": "notifications",
        },
    ), "server/src/async/predictions.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    producer_dependents = [item["symbol"].name for item in db.get_dependents(producer_id)]
    assert "handlePredictionGenerated" not in producer_dependents

    consumer_callees = [item["symbol"].name for item in db.get_callees(consumer_id)]
    assert "prediction.generated" in consumer_callees
    assert "PredictionsTransport" in consumer_callees

    worker_callees = [item["symbol"].name for item in db.get_callees(worker_id)]
    assert "notifications" in worker_callees


def test_module_ownership_does_not_emit_direct_entity_edges(db, tmp_path):
    """Module ownership should not create direct module -> entity traversal edges."""
    src = tmp_path / "module-entity-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/modules/persistence/persistence.module.ts",
        content_hash="module-entity-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    module_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="PersistenceModule",
        qualified_name="PersistenceModule",
        start_line=1,
        end_line=10,
        content="PersistenceModule",
        line_count=10,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "mikroorm_feature_entities": ["User"],
            "providers": ["UserRepository"],
        },
    ), "server/src/modules/persistence/persistence.module.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="entity",
        name="User",
        qualified_name="User",
        start_line=11,
        end_line=14,
        content="User",
        line_count=4,
        metadata={"framework": "mikroorm", "resource": "entity"},
    ), "server/src/modules/persistence/persistence.module.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="repository",
        name="UserRepository",
        qualified_name="UserRepository",
        start_line=15,
        end_line=18,
        content="UserRepository",
        line_count=4,
        metadata={"framework": "mikroorm", "resource": "repository", "entity_name": "User"},
    ), "server/src/modules/persistence/persistence.module.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    module_callees = [item["symbol"].name for item in db.get_callees(module_id)]
    assert "UserRepository" in module_callees
    assert "User" not in module_callees


def test_non_task6_entrypoints_do_not_gain_parent_owner_edges(db, tmp_path):
    """Only route handlers should get parent-owner ownership edges in Task 6."""
    src = tmp_path / "entrypoint-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/entrypoints.ts",
        content_hash="entrypoint-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    resolver_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="resolver",
        name="ViewerResolver",
        qualified_name="ViewerResolver",
        start_line=1,
        end_line=4,
        content="ViewerResolver",
        line_count=4,
    ), "server/src/entrypoints.ts")

    job_owner_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="SyncService",
        qualified_name="SyncService",
        start_line=5,
        end_line=8,
        content="SyncService",
        line_count=4,
    ), "server/src/entrypoints.ts")

    query_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="query",
        name="viewer",
        qualified_name="ViewerResolver.viewer",
        start_line=9,
        end_line=10,
        content="viewer",
        line_count=2,
        parent_symbol_id=resolver_id,
    ), "server/src/entrypoints.ts")

    scheduled_job_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="scheduled_job",
        name="dailySync",
        qualified_name="SyncService.dailySync",
        start_line=11,
        end_line=12,
        content="dailySync",
        line_count=2,
        parent_symbol_id=job_owner_id,
    ), "server/src/entrypoints.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    query_callees = [item["symbol"].name for item in db.get_callees(query_id)]
    scheduled_job_callees = [item["symbol"].name for item in db.get_callees(scheduled_job_id)]
    assert "ViewerResolver" not in query_callees
    assert "SyncService" not in scheduled_job_callees


def test_config_ownership_is_limited_to_module_and_service(db, tmp_path):
    """Config ownership edges should only apply to modules and services."""
    src = tmp_path / "config-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/config-ownership.ts",
        content_hash="config-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    config_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="config",
        name="AuthConfig",
        qualified_name="AuthConfig",
        start_line=1,
        end_line=2,
        content="AuthConfig",
        line_count=2,
        metadata={"framework": "nestjs", "resource": "config", "config_namespace": "auth"},
    ), "server/src/config-ownership.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AuthModule",
        qualified_name="AuthModule",
        start_line=3,
        end_line=6,
        content="ConfigModule.forFeature(AuthConfig)",
        line_count=4,
        metadata={"framework": "nestjs", "resource": "module", "config_refs": ["AuthConfig"]},
    ), "server/src/config-ownership.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=7,
        end_line=10,
        content="this.config = AuthConfig;",
        line_count=4,
        metadata={"framework": "nestjs", "resource": "service", "config_refs": ["AuthConfig"]},
    ), "server/src/config-ownership.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="controller",
        name="AuthController",
        qualified_name="AuthController",
        start_line=11,
        end_line=14,
        content="AuthConfig 'auth'",
        line_count=4,
    ), "server/src/config-ownership.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    config_callees = [item["symbol"].name for item in db.get_callees(config_id)]
    assert "AuthModule" in config_callees
    assert "AuthService" in config_callees
    assert "AuthController" not in config_callees


def test_name_suffix_service_does_not_gain_task6_service_edges(db, tmp_path):
    """Plain *Service names should not count as Task 6 services without service kind."""
    src = tmp_path / "suffix-service-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/suffix-service.ts",
        content_hash="suffix-service-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    config_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="config",
        name="AuthConfig",
        qualified_name="AuthConfig",
        start_line=1,
        end_line=2,
        content="AuthConfig",
        line_count=2,
        metadata={"framework": "nestjs", "resource": "config", "config_namespace": "auth"},
    ), "server/src/suffix-service.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AuthModule",
        qualified_name="AuthModule",
        start_line=3,
        end_line=6,
        content="AuthModule",
        line_count=4,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "controllers": ["AuthController"],
            "providers": ["AuthService", "BillingService"],
            "config_refs": ["AuthConfig"],
        },
    ), "server/src/suffix-service.ts")

    controller_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="controller",
        name="AuthController",
        qualified_name="AuthController",
        start_line=7,
        end_line=10,
        content="constructor(private readonly auth: AuthService) {}",
        line_count=4,
    ), "server/src/suffix-service.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=11,
        end_line=14,
        content="AuthConfig",
        line_count=4,
        metadata={"framework": "nestjs", "resource": "service", "config_refs": ["AuthConfig"]},
    ), "server/src/suffix-service.ts")

    billing_service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="class",
        name="BillingService",
        qualified_name="BillingService",
        start_line=15,
        end_line=18,
        content="AuthConfig",
        line_count=4,
    ), "server/src/suffix-service.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    controller_callees = [item["symbol"].name for item in db.get_callees(controller_id)]
    assert "AuthService" in controller_callees
    assert "BillingService" not in controller_callees

    config_callees = [item["symbol"].name for item in db.get_callees(config_id)]
    assert "AuthModule" in config_callees
    assert "AuthService" in config_callees
    assert "BillingService" not in config_callees

    billing_service_dependents = [item["symbol"].name for item in db.get_dependents(billing_service_id)]
    assert "AuthController" not in billing_service_dependents


def test_name_suffix_repository_does_not_gain_task6_persistence_edges(db, tmp_path):
    """Plain *Repository names should not count as Task 6 persistence targets without persistence kind."""
    src = tmp_path / "suffix-repository-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/suffix-repository.ts",
        content_hash="suffix-repository-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AuthModule",
        qualified_name="AuthModule",
        start_line=1,
        end_line=4,
        content="AuthModule",
        line_count=4,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "providers": ["AuthService", "LegacyRepository"],
        },
    ), "server/src/suffix-repository.ts")

    service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=5,
        end_line=8,
        content="AuthService",
        line_count=4,
    ), "server/src/suffix-repository.ts")

    legacy_repository_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="class",
        name="LegacyRepository",
        qualified_name="LegacyRepository",
        start_line=9,
        end_line=12,
        content="LegacyRepository",
        line_count=4,
    ), "server/src/suffix-repository.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    service_callees = [item["symbol"].name for item in db.get_callees(service_id)]
    assert "LegacyRepository" not in service_callees

    legacy_repository_callers = [item["symbol"].name for item in db.get_callers(legacy_repository_id)]
    assert "AuthService" not in legacy_repository_callers


def test_service_ownership_excludes_schema_targets(db, tmp_path):
    """Task 6 service ownership should stop at repository/entity, not schema."""
    src = tmp_path / "schema-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/schema-ownership.ts",
        content_hash="schema-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=30,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="PersistenceModule",
        qualified_name="PersistenceModule",
        start_line=1,
        end_line=6,
        content="PersistenceModule",
        line_count=6,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "providers": ["AuthService", "UserRepository"],
            "entity_names": ["User", "UserSchema"],
        },
    ), "server/src/schema-ownership.ts")

    service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=7,
        end_line=10,
        content="constructor(private readonly repo: UserRepository) { return User; }",
        line_count=4,
    ), "server/src/schema-ownership.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="repository",
        name="UserRepository",
        qualified_name="UserRepository",
        start_line=11,
        end_line=14,
        content="UserRepository",
        line_count=4,
    ), "server/src/schema-ownership.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="entity",
        name="User",
        qualified_name="User",
        start_line=15,
        end_line=18,
        content="User",
        line_count=4,
    ), "server/src/schema-ownership.ts")

    schema_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="schema",
        name="UserSchema",
        qualified_name="UserSchema",
        start_line=19,
        end_line=22,
        content="UserSchema",
        line_count=4,
    ), "server/src/schema-ownership.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    service_callees = [item["symbol"].name for item in db.get_callees(service_id)]
    assert "UserRepository" in service_callees
    assert "User" in service_callees
    assert "UserSchema" not in service_callees

    schema_callers = [item["symbol"].name for item in db.get_callers(schema_id)]
    assert "AuthService" not in schema_callers


def test_service_ownership_does_not_cross_link_all_module_persistence_targets(db, tmp_path):
    """Services should only own persistence targets that are evidenced locally."""
    src = tmp_path / "service-persistence-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/service-persistence.ts",
        content_hash="service-persistence-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=40,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AppModule",
        qualified_name="AppModule",
        start_line=1,
        end_line=8,
        content="AppModule",
        line_count=8,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "providers": [
                "AuthService",
                "BillingService",
                "AuthRepository",
                "BillingRepository",
            ],
        },
    ), "server/src/service-persistence.ts")

    auth_service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=9,
        end_line=14,
        content="constructor(private readonly repo: AuthRepository) {}",
        line_count=6,
    ), "server/src/service-persistence.ts")

    billing_service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="BillingService",
        qualified_name="BillingService",
        start_line=15,
        end_line=20,
        content="constructor(private readonly repo: BillingRepository) {}",
        line_count=6,
    ), "server/src/service-persistence.ts")

    auth_repository_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="repository",
        name="AuthRepository",
        qualified_name="AuthRepository",
        start_line=21,
        end_line=24,
        content="AuthRepository",
        line_count=4,
    ), "server/src/service-persistence.ts")

    billing_repository_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="repository",
        name="BillingRepository",
        qualified_name="BillingRepository",
        start_line=25,
        end_line=28,
        content="BillingRepository",
        line_count=4,
    ), "server/src/service-persistence.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    auth_service_callees = [item["symbol"].name for item in db.get_callees(auth_service_id)]
    assert "AuthRepository" in auth_service_callees
    assert "BillingRepository" not in auth_service_callees

    billing_service_callees = [item["symbol"].name for item in db.get_callees(billing_service_id)]
    assert "BillingRepository" in billing_service_callees
    assert "AuthRepository" not in billing_service_callees

    auth_repository_callers = [item["symbol"].id for item in db.get_callers(auth_repository_id)]
    billing_repository_callers = [item["symbol"].id for item in db.get_callers(billing_repository_id)]
    assert billing_service_id not in auth_repository_callers
    assert auth_service_id not in billing_repository_callers


def test_controller_ownership_uses_controller_local_service_evidence(db, tmp_path):
    """Controllers should only link to services they actually reference."""
    src = tmp_path / "controller-service-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/controller-service.ts",
        content_hash="controller-service-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=40,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AppModule",
        qualified_name="AppModule",
        start_line=1,
        end_line=8,
        content="AppModule",
        line_count=8,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "controllers": ["AuthController"],
            "providers": ["AuthService", "BillingService"],
        },
    ), "server/src/controller-service.ts")

    controller_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="controller",
        name="AuthController",
        qualified_name="AuthController",
        start_line=9,
        end_line=14,
        content="constructor(private readonly auth: AuthService) {}",
        line_count=6,
    ), "server/src/controller-service.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuthService",
        qualified_name="AuthService",
        start_line=15,
        end_line=18,
        content="AuthService",
        line_count=4,
    ), "server/src/controller-service.ts")

    billing_service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="BillingService",
        qualified_name="BillingService",
        start_line=19,
        end_line=22,
        content="BillingService",
        line_count=4,
    ), "server/src/controller-service.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    controller_callees = [item["symbol"].name for item in db.get_callees(controller_id)]
    billing_service_callers = [item["symbol"].id for item in db.get_callers(billing_service_id)]
    assert "AuthService" in controller_callees
    assert "BillingService" not in controller_callees
    assert controller_id not in billing_service_callers


def test_module_ownership_uses_import_aware_resolution_for_duplicate_module_names(db, tmp_path):
    """Duplicate module names in different contexts should not cross-link ownership edges."""
    src = tmp_path / "bounded-context-ownership"
    (src / "server/src/contexts/auth").mkdir(parents=True)
    (src / "server/src/contexts/billing").mkdir(parents=True)
    (src / "server/src/contexts/app").mkdir(parents=True)

    (src / "server/src/contexts/auth/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/billing/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/app/app.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { PersistenceModule } from '../auth/persistence.module';

@Module({
  imports: [PersistenceModule],
})
export class AppModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    app_module = next(
        sym for sym in db.symbols_in_file("server/src/contexts/app/app.module.ts")
        if sym.name == "AppModule"
    )
    auth_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/auth/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )
    billing_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/billing/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )

    app_module_callees = [item["symbol"].id for item in db.get_callees(app_module.id)]
    assert auth_persistence.id in app_module_callees
    assert billing_persistence.id not in app_module_callees


def test_module_ownership_uses_import_aware_resolution_for_aliased_imports(db, tmp_path):
    """Aliased imports should still resolve to the correct bounded-context target only."""
    src = tmp_path / "aliased-context-ownership"
    (src / "server/src/contexts/auth").mkdir(parents=True)
    (src / "server/src/contexts/billing").mkdir(parents=True)
    (src / "server/src/contexts/app").mkdir(parents=True)

    (src / "server/src/contexts/auth/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/billing/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/app/app.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { PersistenceModule as AuthPersistenceModule } from '../auth/persistence.module';

@Module({
  imports: [AuthPersistenceModule],
})
export class AppModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    app_module = next(
        sym for sym in db.symbols_in_file("server/src/contexts/app/app.module.ts")
        if sym.name == "AppModule"
    )
    auth_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/auth/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )
    billing_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/billing/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )

    app_module_callees = [item["symbol"].id for item in db.get_callees(app_module.id)]
    assert auth_persistence.id in app_module_callees
    assert billing_persistence.id not in app_module_callees


def test_module_ownership_barrel_imports_do_not_broad_link_duplicate_modules(db, tmp_path):
    """Barrel imports may under-link, but must not broad-link duplicate module names."""
    src = tmp_path / "barrel-context-ownership"
    (src / "server/src/contexts/auth").mkdir(parents=True)
    (src / "server/src/contexts/billing").mkdir(parents=True)
    (src / "server/src/contexts/app").mkdir(parents=True)

    (src / "server/src/contexts/auth/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/auth/index.ts").write_text('''\
export { PersistenceModule } from './persistence.module';
''')

    (src / "server/src/contexts/billing/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/app/app.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { PersistenceModule } from '../auth';

@Module({
  imports: [PersistenceModule],
})
export class AppModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    app_module = next(
        sym for sym in db.symbols_in_file("server/src/contexts/app/app.module.ts")
        if sym.name == "AppModule"
    )
    billing_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/billing/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )

    app_module_callees = [item["symbol"].id for item in db.get_callees(app_module.id)]
    assert billing_persistence.id not in app_module_callees


def test_aliased_ownership_evidence_links_controller_and_service_dependencies(db, tmp_path):
    """Aliased imports should still count as local evidence for Task 6 ownership edges."""
    src = tmp_path / "aliased-evidence-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import { AuthService as LoginService } from './auth.infra';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/auth.infra.ts").write_text('''\
import { Injectable } from '@nestjs/common';
import { UserRepository as AccountsRepo } from '../persistence/persistence.infra';

@Injectable()
export class AuthService {
  constructor(private readonly users: AccountsRepo) {}

  validate() {
    return this.users.findById();
  }
}
''')

    (src / "server/src/modules/persistence/persistence.infra.ts").write_text('''\
import { EntityRepository } from '@mikro-orm/core';

export class User {}

export class UserRepository extends EntityRepository<User> {
  findById() {
    return { id: 1 };
  }
}
''')

    (src / "server/src/modules/persistence/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { UserRepository } from './persistence.infra';

@Module({
  providers: [UserRepository],
  exports: [UserRepository],
})
export class PersistenceModule {}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.infra';
import { PersistenceModule } from '../persistence/persistence.module';

@Module({
  imports: [PersistenceModule],
  controllers: [AuthController],
  providers: [AuthService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    service = db.get_symbol_by_name("AuthService")
    repository = db.get_symbol_by_name("UserRepository")

    assert controller is not None
    assert service is not None
    assert repository is not None

    controller_callees = [item["symbol"].name for item in db.get_callees(controller.id)]
    service_callees = [item["symbol"].name for item in db.get_callees(service.id)]
    assert "AuthService" in controller_callees
    assert "UserRepository" in service_callees


def test_module_ownership_non_relative_alias_import_resolves_unique_candidate(db, tmp_path):
    """Unique non-relative alias imports should preserve Task 6 ownership edges."""
    src = tmp_path / "non-relative-alias-ownership"
    (src / "server/src/contexts/auth").mkdir(parents=True)
    (src / "server/src/contexts/app").mkdir(parents=True)

    (src / "server/src/contexts/auth/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class PersistenceModule {}
''')

    (src / "server/src/contexts/app/app.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { PersistenceModule } from '@app/contexts/auth/persistence.module';

@Module({
  imports: [PersistenceModule],
})
export class AppModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    app_module = next(
        sym for sym in db.symbols_in_file("server/src/contexts/app/app.module.ts")
        if sym.name == "AppModule"
    )
    auth_persistence = next(
        sym for sym in db.symbols_in_file("server/src/contexts/auth/persistence.module.ts")
        if sym.name == "PersistenceModule"
    )

    app_module_callees = [item["symbol"].id for item in db.get_callees(app_module.id)]
    assert auth_persistence.id in app_module_callees


def test_module_ownership_external_package_import_does_not_resolve_local_symbol(db, tmp_path):
    """External package imports must not create local Task 6 ownership edges."""
    src = tmp_path / "external-package-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/app").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';

@Module({})
export class AuthModule {}
''')

    (src / "server/src/modules/app/app.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthModule } from '@acme/shared/auth.module';

@Module({
  imports: [AuthModule],
})
export class AppModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    app_module = next(
        sym for sym in db.symbols_in_file("server/src/modules/app/app.module.ts")
        if sym.name == "AppModule"
    )
    local_auth_module = next(
        sym for sym in db.symbols_in_file("server/src/modules/auth/auth.module.ts")
        if sym.name == "AuthModule"
    )

    app_module_callees = [item["symbol"].id for item in db.get_callees(app_module.id)]
    assert local_auth_module.id not in app_module_callees


def test_default_import_alias_ownership_links_controller_and_service_dependencies(db, tmp_path):
    """Default-import aliases should still count as ownership evidence."""
    src = tmp_path / "default-import-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import LoginService from './auth.infra';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/auth.infra.ts").write_text('''\
import { Injectable } from '@nestjs/common';
import AccountsRepo from '../persistence/persistence.infra';

@Injectable()
export default class AuthService {
  constructor(private readonly users: AccountsRepo) {}

  validate() {
    return this.users.findById();
  }
}

@Injectable()
export class BillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/persistence/persistence.infra.ts").write_text('''\
import { EntityRepository } from '@mikro-orm/core';

export class User {}

export default class UserRepository extends EntityRepository<User> {
  findById() {
    return { id: 1 };
  }
}

export class AuditRepository extends EntityRepository<User> {
  findAll() {
    return [];
  }
}
''')

    (src / "server/src/modules/persistence/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import UserRepository from './persistence.infra';

@Module({
  providers: [UserRepository],
  exports: [UserRepository],
})
export class PersistenceModule {}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import AuthService from './auth.infra';
import { PersistenceModule } from '../persistence/persistence.module';

@Module({
  imports: [PersistenceModule],
  controllers: [AuthController],
  providers: [AuthService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    repository = db.get_symbol_by_name("UserRepository")
    audit_repository = db.get_symbol_by_name("AuditRepository")
    charge = db.get_symbol_by_name("charge")
    find_all = db.get_symbol_by_name("findAll")

    assert controller is not None
    assert service is not None
    assert billing_service is not None
    assert repository is not None
    assert audit_repository is not None
    assert charge is not None
    assert find_all is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    service_callees = [item["symbol"].id for item in db.get_callees(service.id)]
    assert service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert charge.id not in controller_callees
    assert repository.id in service_callees
    assert audit_repository.id not in service_callees
    assert find_all.id not in service_callees


def test_workspace_alias_default_import_uses_default_export_only(db, tmp_path):
    """Workspace-alias default imports should resolve only to the proved default export."""
    src = tmp_path / "workspace-default-import-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import LoginService from '@/modules/auth/auth.infra';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/auth.infra.ts").write_text('''\
import { Injectable } from '@nestjs/common';
import AccountsRepo from '@/modules/persistence/persistence.infra';

@Injectable()
export default class AuthService {
  constructor(private readonly users: AccountsRepo) {}

  validate() {
    return this.users.findById();
  }
}

@Injectable()
export class BillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/persistence/persistence.infra.ts").write_text('''\
import { EntityRepository } from '@mikro-orm/core';

export class User {}

export default class UserRepository extends EntityRepository<User> {
  findById() {
    return { id: 1 };
  }
}

export class AuditRepository extends EntityRepository<User> {
  findAll() {
    return [];
  }
}
''')

    (src / "server/src/modules/persistence/persistence.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import UserRepository from '@/modules/persistence/persistence.infra';

@Module({
  providers: [UserRepository],
  exports: [UserRepository],
})
export class PersistenceModule {}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import AuthService from './auth.infra';
import { PersistenceModule } from '../persistence/persistence.module';

@Module({
  imports: [PersistenceModule],
  controllers: [AuthController],
  providers: [AuthService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    repository = db.get_symbol_by_name("UserRepository")
    audit_repository = db.get_symbol_by_name("AuditRepository")

    assert controller is not None
    assert service is not None
    assert billing_service is not None
    assert repository is not None
    assert audit_repository is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    service_callees = [item["symbol"].id for item in db.get_callees(service.id)]
    assert service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert repository.id in service_callees
    assert audit_repository.id not in service_callees


def test_semicolonless_default_export_name_is_resolved_precisely(db, tmp_path):
    """Semicolonless default export aliases should still resolve only to the default symbol."""
    src = tmp_path / "semicolonless-default-import-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common'
import LoginService from './auth.infra'

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate()
  }
}
''')

    (src / "server/src/modules/auth/auth.infra.ts").write_text('''\
import { Injectable } from '@nestjs/common'

@Injectable()
export class BillingService {
  charge() {
    return true
  }
}

@Injectable()
export class AuthService {
  validate() {
    return true
  }
}

export default AuthService
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common'
import { AuthController } from './auth.controller'
import LoginService from './auth.infra'

@Module({
  controllers: [AuthController],
  providers: [LoginService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    auth_service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    charge = db.get_symbol_by_name("charge")

    assert controller is not None
    assert auth_service is not None
    assert billing_service is not None
    assert charge is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    assert auth_service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert charge.id not in controller_callees


def test_relative_default_import_prefers_file_over_index(db, tmp_path):
    """Relative imports should resolve one TS-style file path before default-export filtering."""
    src = tmp_path / "relative-default-import-precedence"
    (src / "server/src/modules/auth/foo").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import LoginService from './foo';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/foo.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class AuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class BillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/foo/index.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class IndexAuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class IndexBillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import LoginService from './foo';

@Module({
  controllers: [AuthController],
  providers: [LoginService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    auth_service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    index_auth_service = db.get_symbol_by_name("IndexAuthService")
    index_billing_service = db.get_symbol_by_name("IndexBillingService")
    charge = db.get_symbol_by_name("charge")

    assert controller is not None
    assert auth_service is not None
    assert billing_service is not None
    assert index_auth_service is not None
    assert index_billing_service is not None
    assert charge is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    assert auth_service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert index_auth_service.id not in controller_callees
    assert index_billing_service.id not in controller_callees
    assert charge.id not in controller_callees


def test_workspace_alias_default_import_prefers_file_over_index(db, tmp_path):
    """Workspace aliases should use the same one-file precedence as relative imports."""
    src = tmp_path / "workspace-default-import-precedence"
    (src / "server/src/modules/auth/foo").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import LoginService from '@/modules/auth/foo';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/foo.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class AuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class BillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/foo/index.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class IndexAuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class IndexBillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import LoginService from '@/modules/auth/foo';

@Module({
  controllers: [AuthController],
  providers: [LoginService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    auth_service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    index_auth_service = db.get_symbol_by_name("IndexAuthService")
    index_billing_service = db.get_symbol_by_name("IndexBillingService")
    charge = db.get_symbol_by_name("charge")

    assert controller is not None
    assert auth_service is not None
    assert billing_service is not None
    assert index_auth_service is not None
    assert index_billing_service is not None
    assert charge is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    assert auth_service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert index_auth_service.id not in controller_callees
    assert index_billing_service.id not in controller_callees
    assert charge.id not in controller_callees


def test_tsconfig_alias_default_import_resolves_with_ordered_precedence(db, tmp_path):
    """Repo-local tsconfig aliases should resolve through ordered file-before-index precedence."""
    src = tmp_path / "tsconfig-alias-default-import"
    (src / "server/src/modules/auth/foo").mkdir(parents=True)

    (src / "tsconfig.json").write_text('''\
{
  "compilerOptions": {
    "baseUrl": "server/src",
    "paths": {
      "#/*": ["*"]
    }
  }
}
''')

    (src / "server/src/modules/auth/auth.controller.ts").write_text('''\
import { Controller, Get } from '@nestjs/common';
import LoginService from '#/modules/auth/foo';

@Controller('auth')
export class AuthController {
  constructor(private readonly auth: LoginService) {}

  @Get('me')
  getMe() {
    return this.auth.validate();
  }
}
''')

    (src / "server/src/modules/auth/foo.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class AuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class BillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/foo/index.ts").write_text('''\
import { Injectable } from '@nestjs/common';

@Injectable()
export default class IndexAuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class IndexBillingService {
  charge() {
    return true;
  }
}
''')

    (src / "server/src/modules/auth/auth.module.ts").write_text('''\
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import LoginService from '#/modules/auth/foo';

@Module({
  controllers: [AuthController],
  providers: [LoginService],
})
export class AuthModule {}
''')

    indexer = Indexer(db, IndexConfig(root=src))
    indexer.index(src)

    controller = db.get_symbol_by_name("AuthController")
    auth_service = db.get_symbol_by_name("AuthService")
    billing_service = db.get_symbol_by_name("BillingService")
    index_auth_service = db.get_symbol_by_name("IndexAuthService")
    index_billing_service = db.get_symbol_by_name("IndexBillingService")
    charge = db.get_symbol_by_name("charge")

    assert controller is not None
    assert auth_service is not None
    assert billing_service is not None
    assert index_auth_service is not None
    assert index_billing_service is not None
    assert charge is not None

    controller_callees = [item["symbol"].id for item in db.get_callees(controller.id)]
    assert auth_service.id in controller_callees
    assert billing_service.id not in controller_callees
    assert index_auth_service.id not in controller_callees
    assert index_billing_service.id not in controller_callees
    assert charge.id not in controller_callees


def test_unrelated_config_namespace_literal_does_not_create_config_edge(db, tmp_path):
    """Arbitrary namespace string literals should not create Task 6 config ownership edges."""
    src = tmp_path / "config-literal-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/config-literal.ts",
        content_hash="config-literal-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=20,
    ))

    config_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="config",
        name="AuthConfig",
        qualified_name="AuthConfig",
        start_line=1,
        end_line=2,
        content="AuthConfig",
        line_count=2,
        metadata={"framework": "nestjs", "resource": "config", "config_namespace": "auth"},
    ), "server/src/config-literal.ts")

    service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuditService",
        qualified_name="AuditService",
        start_line=3,
        end_line=6,
        content="logger.info('auth')",
        line_count=4,
        metadata={"framework": "nestjs", "resource": "service"},
    ), "server/src/config-literal.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    config_callees = [item["symbol"].id for item in db.get_callees(config_id)]
    service_callers = [item["symbol"].id for item in db.get_callers(service_id)]
    assert service_id not in config_callees
    assert config_id not in service_callers


def test_service_ownership_ignores_repository_names_in_strings_and_comments(db, tmp_path):
    """Repository names inside strings/comments should not count as Task 6 persistence evidence."""
    src = tmp_path / "service-string-comment-ownership"
    src.mkdir()

    file_id = db.upsert_file(FileRecord(
        path="server/src/service-string-comment.ts",
        content_hash="service-string-comment-ownership",
        mtime=1.0,
        language="typescript",
        size=100,
        line_count=30,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AppModule",
        qualified_name="AppModule",
        start_line=1,
        end_line=6,
        content="AppModule",
        line_count=6,
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "providers": ["AuditService", "UserRepository"],
        },
    ), "server/src/service-string-comment.ts")

    service_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="service",
        name="AuditService",
        qualified_name="AuditService",
        start_line=7,
        end_line=14,
        content='// UserRepository should not count\nconst label = "UserRepository";',
        line_count=8,
    ), "server/src/service-string-comment.ts")

    repository_id = db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="repository",
        name="UserRepository",
        qualified_name="UserRepository",
        start_line=15,
        end_line=18,
        content="UserRepository",
        line_count=4,
    ), "server/src/service-string-comment.ts")

    db.commit()

    indexer = Indexer(db, IndexConfig(root=src))
    indexer._build_edges()

    service_callees = [item["symbol"].id for item in db.get_callees(service_id)]
    repository_callers = [item["symbol"].id for item in db.get_callers(repository_id)]
    assert repository_id not in service_callees
    assert service_id not in repository_callers
