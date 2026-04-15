"""Tests for workspace (multi-repo) functionality."""

import json
from pathlib import Path

import pytest

import srclight.server as server
from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord
from srclight.embeddings import vector_to_bytes
from srclight.vector_cache import VectorCache
from srclight.workspace import WorkspaceConfig, WorkspaceDB, _sanitize_schema_name


@pytest.fixture
def ws_dir(tmp_path):
    """Override workspaces dir for testing."""
    import srclight.workspace as ws_mod
    orig = ws_mod.WORKSPACES_DIR
    ws_mod.WORKSPACES_DIR = tmp_path / "workspaces"
    yield tmp_path / "workspaces"
    ws_mod.WORKSPACES_DIR = orig


def _create_indexed_project(tmp_path: Path, name: str, symbols: list[tuple[str, str] | dict]):
    """Create a project dir with a .srclight/index.db populated with symbols.

    symbols: list of (name, kind) tuples or symbol dictionaries
    """
    project_dir = tmp_path / name
    project_dir.mkdir()
    db_dir = project_dir / ".srclight"
    db_dir.mkdir()
    db_path = db_dir / "index.db"

    db = Database(db_path)
    db.open()
    db.initialize()

    file_ids: dict[str, int] = {}

    def _file_id_for(path: str) -> int:
        if path not in file_ids:
            file_ids[path] = db.upsert_file(
                FileRecord(
                    path=path,
                    content_hash=f"{name}:{path}",
                    mtime=1000.0,
                    language="typescript" if path.endswith(".ts") else "csharp",
                    size=500,
                    line_count=50,
                )
            )
        return file_ids[path]

    for i, item in enumerate(symbols):
        if isinstance(item, tuple):
            sym_name, sym_kind = item
            path = f"src/{name}.cs"
            signature = f"{sym_kind} {sym_name}()" if sym_kind in ("method", "function") else sym_name
            content = f"{sym_kind} {sym_name} {{ }}"
            doc_comment = None
            metadata = None
        else:
            sym_name = item["name"]
            sym_kind = item["kind"]
            path = item.get("path", f"src/{name}.cs")
            signature = item.get("signature")
            content = item.get("content", f"{sym_kind} {sym_name} {{ }}")
            doc_comment = item.get("doc_comment")
            metadata = item.get("metadata")

        db.insert_symbol(
            SymbolRecord(
                file_id=_file_id_for(path),
                kind=sym_kind,
                name=sym_name,
                qualified_name=f"{name}.{sym_name}",
                signature=signature,
                start_line=i * 10 + 1,
                end_line=i * 10 + 8,
                content=content,
                doc_comment=doc_comment,
                line_count=8,
                metadata=metadata,
            ),
            path,
        )

    db.commit()
    db.close()
    return project_dir


def _build_project_sidecar(project_dir: Path, vector: list[float]) -> None:
    """Add a single embedding and build the project's vector sidecar."""
    db_path = project_dir / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    try:
        symbol_id = db.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()[0]
        db.upsert_embedding(symbol_id, "mock:test", len(vector), vector_to_bytes(vector), "hash-1")
        db.commit()
        VectorCache(project_dir / ".srclight").build_from_db(db.conn)
    finally:
        db.close()


def _store_workspace_communities_and_flows(project_dir: Path) -> None:
    from srclight.community import detect_communities, trace_execution_flows

    db_path = project_dir / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    try:
        rows = db.conn.execute(
            """SELECT s.id, s.name
               FROM symbols s
               ORDER BY s.start_line, s.id"""
        ).fetchall()
        symbol_ids = {row["name"]: row["id"] for row in rows}

        db.insert_edge(EdgeRecord(
            source_id=symbol_ids["bootstrap"],
            target_id=symbol_ids["routeRequest"],
            edge_type="calls",
        ))
        db.insert_edge(EdgeRecord(
            source_id=symbol_ids["routeRequest"],
            target_id=symbol_ids["fetchUser"],
            edge_type="calls",
        ))
        db.insert_edge(EdgeRecord(
            source_id=symbol_ids["fetchUser"],
            target_id=symbol_ids["serializeUser"],
            edge_type="calls",
        ))
        db.insert_edge(EdgeRecord(
            source_id=symbol_ids["runJobs"],
            target_id=symbol_ids["sendEmail"],
            edge_type="calls",
        ))
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
    finally:
        db.close()


def test_sanitize_schema_name():
    assert _sanitize_schema_name("nomad-builder") == "nomad_builder"
    assert _sanitize_schema_name("qi") == "qi"
    assert _sanitize_schema_name("123bad") == "_123bad"
    assert _sanitize_schema_name("hello.world") == "hello_world"
    # Reserved names get prefixed
    assert _sanitize_schema_name("main") == "p_main"
    assert _sanitize_schema_name("temp") == "p_temp"
    assert _sanitize_schema_name("") == "_unnamed"


def test_workspace_config_crud(ws_dir):
    """Create, save, load, and modify workspace config."""
    config = WorkspaceConfig(name="test")
    config.save()
    assert config.config_path.exists()

    config.add_project("repo1", "/tmp/repo1")
    config.add_project("repo2", "/tmp/repo2")

    loaded = WorkspaceConfig.load("test")
    assert loaded.name == "test"
    assert len(loaded.projects) == 2
    assert "repo1" in loaded.projects

    config.remove_project("repo1")
    loaded = WorkspaceConfig.load("test")
    assert len(loaded.projects) == 1

    names = WorkspaceConfig.list_all()
    assert "test" in names


def test_workspace_db_attach_and_search(tmp_path, ws_dir):
    """WorkspaceDB attaches multiple project DBs and searches across them."""
    # Create two indexed projects
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Dictionary", "class"),
        ("lookup", "method"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Dictionary", "class"),
        ("translate", "method"),
        ("Parser", "class"),
    ])

    # Create workspace config
    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        assert wdb.project_count == 2

        # Search across both projects
        results = wdb.search_symbols("Dictionary")
        assert len(results) >= 2
        projects = {r["project"] for r in results}
        assert "alpha" in projects
        assert "beta" in projects

        # Search with project filter
        results = wdb.search_symbols("Dictionary", project="alpha")
        assert all(r["project"] == "alpha" for r in results)

        # Search for something only in beta
        results = wdb.search_symbols("Parser")
        assert any(r["name"] == "Parser" for r in results)
        assert all(r["project"] == "beta" for r in results if r["name"] == "Parser")


def test_workspace_db_codebase_map(tmp_path, ws_dir):
    """codebase_map aggregates stats across projects."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Foo", "class"), ("bar", "method"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Baz", "class"), ("qux", "function"), ("quux", "function"),
    ])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        stats = wdb.codebase_map()
        assert stats["workspace"] == "test"
        assert stats["projects_attached"] == 2
        assert stats["totals"]["files"] == 2
        assert stats["totals"]["symbols"] == 5


def test_workspace_db_codebase_map_unknown_project_raises_lookup_error(tmp_path, ws_dir):
    proj1 = _create_indexed_project(tmp_path, "alpha", [("Foo", "class")])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))

    with WorkspaceDB(config) as wdb:
        with pytest.raises(LookupError, match="typo"):
            wdb.codebase_map(project="typo")


def test_workspace_db_list_projects(tmp_path, ws_dir):
    """list_projects shows stats for each project."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [("Foo", "class")])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("missing", "/nonexistent/path")

    with WorkspaceDB(config) as wdb:
        projects = wdb.list_projects()
        # alpha should be indexed with stats
        alpha = next(p for p in projects if p["project"] == "alpha")
        assert alpha["files"] == 1
        assert alpha["symbols"] == 1
        # missing should show as unindexed
        missing = next(p for p in projects if p["project"] == "missing")
        assert missing.get("indexed") is False or missing.get("files", 0) == 0


def test_workspace_db_get_symbol(tmp_path, ws_dir):
    """get_symbol returns details from across projects."""
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        ("Dictionary", "class"),
    ])
    proj2 = _create_indexed_project(tmp_path, "beta", [
        ("Dictionary", "class"),
    ])

    config = WorkspaceConfig(name="test")
    config.add_project("alpha", str(proj1))
    config.add_project("beta", str(proj2))

    with WorkspaceDB(config) as wdb:
        results = wdb.get_symbol("Dictionary")
        assert len(results) == 2
        projects = {r["project"] for r in results}
        assert projects == {"alpha", "beta"}

        # Filter by project
        results = wdb.get_symbol("Dictionary", project="beta")
        assert len(results) == 1
        assert results[0]["project"] == "beta"


def test_workspace_vector_search_keeps_projects_distinct_when_symbol_ids_collide(tmp_path, ws_dir):
    alpha = _create_indexed_project(tmp_path, "alpha", [("Match", "class")])
    beta = _create_indexed_project(tmp_path, "beta", [("Match", "class")])

    _build_project_sidecar(alpha, [1.0, 0.0])
    _build_project_sidecar(beta, [0.8, 0.2])

    config = WorkspaceConfig(name="vector-collision")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        results = wdb.vector_search(vector_to_bytes([1.0, 0.0]), 2, limit=2)

    assert [r["project"] for r in results] == ["alpha", "beta"]
    assert all(r["symbol_id"] == 1 for r in results)


def test_workspace_vector_search_falls_back_when_mixed_sidecar_coverage(tmp_path, ws_dir):
    alpha = _create_indexed_project(tmp_path, "alpha", [("Match", "class")])
    beta = _create_indexed_project(tmp_path, "beta", [("Match", "class")])

    _build_project_sidecar(alpha, [1.0, 0.0])
    db_path = beta / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    try:
        symbol_id = db.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()[0]
        db.upsert_embedding(symbol_id, "mock:test", 2, vector_to_bytes([0.9, 0.1]), "hash-1")
        db.commit()
    finally:
        db.close()

    config = WorkspaceConfig(name="mixed-sidecar")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        results = wdb.vector_search(vector_to_bytes([1.0, 0.0]), 2, limit=2)

    assert {r["project"] for r in results} == {"alpha", "beta"}
    assert len(results) == 2


def test_workspace_vector_search_ignores_plain_indexed_projects_without_embeddings(
    tmp_path, ws_dir, monkeypatch
):
    alpha = _create_indexed_project(tmp_path, "alpha", [("Match", "class")])
    beta = _create_indexed_project(tmp_path, "beta", [("Match", "class")])

    _build_project_sidecar(alpha, [1.0, 0.0])

    config = WorkspaceConfig(name="plain-indexed")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        monkeypatch.setattr(
            WorkspaceDB,
            "_vector_search_slow",
            lambda *args, **kwargs: pytest.fail("slow path should not run for empty indexed DBs"),
        )
        results = wdb.vector_search(vector_to_bytes([1.0, 0.0]), 2, limit=1)

    assert [r["project"] for r in results] == ["alpha"]


def test_workspace_db_batch_over_10_projects(tmp_path, ws_dir):
    """Batch iteration handles >10 projects (SQLite ATTACH limit)."""
    import srclight.workspace as ws_mod
    # Temporarily lower the limit to test batching with fewer projects
    orig_limit = ws_mod.MAX_ATTACH
    ws_mod.MAX_ATTACH = 3

    try:
        # Create 5 projects (will need 2 batches of 3)
        projects = {}
        for i in range(5):
            name = f"proj{i}"
            proj = _create_indexed_project(tmp_path, name, [
                (f"Class{i}", "class"),
                (f"method{i}", "method"),
            ])
            projects[name] = proj

        config = WorkspaceConfig(name="batch-test")
        for name, proj_dir in projects.items():
            config.add_project(name, str(proj_dir))

        with WorkspaceDB(config) as wdb:
            assert wdb.project_count == 5

            # list_projects should see all 5
            all_projects = wdb.list_projects()
            indexed = [p for p in all_projects if p.get("files", 0) > 0]
            assert len(indexed) == 5

            # codebase_map should aggregate across all
            stats = wdb.codebase_map()
            assert stats["totals"]["symbols"] == 10  # 2 per project * 5

            # search_symbols should find results across batches
            results = wdb.search_symbols("Class")
            assert len(results) >= 5
            found_projects = {r["project"] for r in results}
            assert len(found_projects) == 5

            # Project filter should work across batch boundaries
            results = wdb.search_symbols("Class4", project="proj4")
            assert len(results) >= 1
            assert all(r["project"] == "proj4" for r in results)

            # get_symbol across batches
            for i in range(5):
                results = wdb.get_symbol(f"Class{i}")
                assert len(results) >= 1
                assert results[0]["project"] == f"proj{i}"
    finally:
        ws_mod.MAX_ATTACH = orig_limit


def test_workspace_ranking_route_query_prefers_route_surfaces(tmp_path, ws_dir):
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "authRoutes",
            "kind": "router",
            "path": "server/src/routes/auth.ts",
            "signature": "Elysia router | /api/auth",
            "content": "auth routes router refresh login",
            "doc_comment": "Auth routes for login and refresh.",
            "metadata": {"framework": "elysia", "resource": "router", "route_prefix": "/api/auth"},
        },
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])

    config = WorkspaceConfig(name="ranking-test")
    config.add_project("alpha", str(proj1))

    with WorkspaceDB(config) as wdb:
        results = wdb.search_symbols("auth routes")
        assert results
        assert results[0]["project"] == "alpha"
        assert results[0]["kind"] in {"route", "router", "route_handler"}


def test_workspace_ranking_mikro_orm_prefers_persistence_symbols(tmp_path, ws_dir):
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "User",
            "kind": "entity",
            "path": "server/src/db/mikroorm.ts",
            "signature": "mikro orm entity | User | users",
            "content": "mikro orm entity user users table",
            "doc_comment": "Mikro ORM entity for users.",
            "metadata": {"framework": "mikroorm", "resource": "entity", "table_name": "users"},
        },
        {
            "name": "UserRepository",
            "kind": "repository",
            "path": "server/src/db/mikroorm.ts",
            "signature": "mikro orm repository | UserRepository | User",
            "content": "mikro orm repository user entities",
            "doc_comment": "Mikro ORM repository for User entities.",
            "metadata": {"framework": "mikroorm", "resource": "repository", "entity_name": "User"},
        },
        {
            "name": "orm",
            "kind": "database",
            "path": "server/src/db/mikroorm.ts",
            "signature": "mikro orm database",
            "content": "mikro orm database entities init",
            "doc_comment": "Mikro ORM database initialization for entities.",
            "metadata": {"framework": "mikroorm", "resource": "database", "entity_names": ["User"]},
        },
        {
            "name": "MikroOrmEntitiesGuide",
            "kind": "class",
            "path": "docs/mikro-orm-guide.md",
            "signature": "MikroOrmEntitiesGuide",
            "content": "mikro orm entities guide overview",
            "doc_comment": "Mikro ORM entities guide and overview.",
        },
    ])

    config = WorkspaceConfig(name="mikro-ranking-test")
    config.add_project("alpha", str(proj1))

    with WorkspaceDB(config) as wdb:
        results = wdb.search_symbols("mikro orm entities")
        assert results
        assert results[0]["kind"] in {"entity", "repository", "database"}


def test_workspace_fallback_hint_suggests_symbol_names(tmp_path, ws_dir):
    proj1 = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])

    config = WorkspaceConfig(name="workspace-hints")
    config.add_project("alpha", str(proj1))

    with WorkspaceDB(config) as wdb:
        suggestions = wdb.suggest_symbol_names("authRoutes")
        assert suggestions
        assert suggestions[0]["name"] == "AuthRoutesService"


def test_workspace_fallback_hint_orders_suggestions_globally(tmp_path, ws_dir):
    alpha = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "AuthRoute",
            "kind": "class",
            "path": "server/src/services/auth-route.ts",
            "signature": "class AuthRoute",
            "content": "auth route helper",
            "doc_comment": "Singular auth route helper.",
        },
    ])
    beta = _create_indexed_project(tmp_path, "beta", [
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])

    config = WorkspaceConfig(name="workspace-ordering")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        suggestions = wdb.suggest_symbol_names("authRoutes")
        assert suggestions
        assert suggestions[0] == {"project": "beta", "name": "AuthRoutesService"}


def test_workspace_get_symbol_miss_keeps_project_scoped_hints(tmp_path, ws_dir, monkeypatch):
    alpha = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "AuthRouteAlpha",
            "kind": "class",
            "path": "server/src/services/auth-route-alpha.ts",
            "signature": "class AuthRouteAlpha",
            "content": "auth route alpha helper",
            "doc_comment": "Alpha-local auth route helper.",
        },
    ])
    beta = _create_indexed_project(tmp_path, "beta", [
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])

    config = WorkspaceConfig(name="workspace-project-miss")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)
        monkeypatch.setattr(server, "_get_workspace_db", lambda: wdb)

        payload = json.loads(server.get_symbol("authRoutes", project="alpha"))

    assert payload["error"] == "Symbol 'authRoutes' not found in alpha"
    assert payload["did_you_mean"] == ["AuthRouteAlpha"]
    assert "AuthRoutesService" not in payload["did_you_mean"]


def test_workspace_db_codebase_map_orients_fullstack_backend_async_project(tmp_path, ws_dir):
    project_dir = _create_indexed_project(tmp_path, "fullstack", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "src/main.ts",
            "signature": "async function bootstrap()",
            "content": "nest bootstrap http server",
        },
        {
            "name": "UsersController",
            "kind": "controller",
            "path": "src/controllers/users.controller.ts",
            "signature": "class UsersController",
            "content": "nest controller users routes",
            "metadata": {"framework": "nest", "resource": "controller"},
        },
        {
            "name": "EmailProcessor",
            "kind": "queue_processor",
            "path": "src/queues/email.processor.ts",
            "signature": "class EmailProcessor",
            "content": "bullmq queue processor emails",
            "metadata": {"framework": "bullmq", "resource": "processor"},
        },
    ])

    (project_dir / "app/pages").mkdir(parents=True)
    (project_dir / "server/api").mkdir(parents=True)
    (project_dir / "src/controllers").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/queues").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/config").mkdir(parents=True)
    (project_dir / "prisma").mkdir(parents=True)

    (project_dir / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
            "@prisma/client": "^6.0.0",
            "bullmq": "^5.0.0",
        },
    }))
    (project_dir / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (project_dir / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (project_dir / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (project_dir / "src/config/runtime.ts").write_text("export const runtime = {}\n")
    (project_dir / "prisma/schema.prisma").write_text("model User { id Int @id }\n")

    config = WorkspaceConfig(name="fullstack-map")
    config.add_project("fullstack", str(project_dir))

    with WorkspaceDB(config) as wdb:
        payload = wdb.codebase_map(project="fullstack")

    assert payload["framework_hints"]["app_type"] == "fullstack"
    assert payload["representative_files"]["backend"] == [
        "server/api/health.get.ts",
        "src/main.ts",
        "src/controllers/users.controller.ts",
    ]
    assert payload["topology"]["backend"] == {
        "files": [
            "server/api/health.get.ts",
            "src/main.ts",
            "src/controllers/users.controller.ts",
        ],
        "summary": "Primary backend entrypoints, HTTP surfaces, and server modules.",
    }
    assert payload["representative_files"]["data"] == ["prisma/schema.prisma"]
    assert payload["representative_files"]["async"] == ["src/queues/email.processor.ts"]
    assert payload["topology"]["routes"]["systems"] == ["nest_controllers", "nitro_file_routes"]
    assert payload["topology"]["routes"]["summary"] == (
        "HTTP route and transport surfaces from Nest controllers and Nitro file routes."
    )
    assert payload["topology"]["data"]["systems"] == ["prisma"]
    assert payload["topology"]["async"]["systems"] == ["bullmq"]
    assert payload["topology"]["runtime"]["files"] == [
        "nuxt.config.ts",
        "package.json",
        "src/config/runtime.ts",
    ]
    assert any(item["path"] == "src/main.ts" for item in payload["start_here"])


def test_workspace_db_codebase_map_surfaces_nest_backend_data_and_async_start_here(
    tmp_path, ws_dir
):
    project_dir = _create_indexed_project(tmp_path, "nest-app", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "src/main.ts",
            "signature": "async function bootstrap()",
            "content": "nest bootstrap http server",
            "metadata": {"framework": "nest", "resource": "bootstrap"},
        },
        {
            "name": "UsersController",
            "kind": "controller",
            "path": "src/controllers/users.controller.ts",
            "signature": "class UsersController",
            "content": "nest controller users routes",
            "metadata": {"framework": "nest", "resource": "controller", "route_prefix": "/users"},
        },
        {
            "name": "AppModule",
            "kind": "class",
            "path": "src/modules/app.module.ts",
            "signature": "class AppModule",
            "content": "nest module app wiring",
            "metadata": {"framework": "nest", "resource": "module"},
        },
        {
            "name": "UserModel",
            "kind": "class",
            "path": "src/persistence/user.model.ts",
            "signature": "class UserModel",
            "content": "mongoose user model",
            "metadata": {"framework": "mongoose", "resource": "model"},
        },
        {
            "name": "OrdersConsumer",
            "kind": "microservice_handler",
            "path": "src/messaging/orders.consumer.ts",
            "signature": "function handleOrders()",
            "content": "rmq consumer orders.created",
            "metadata": {"framework": "rmq", "resource": "consumer", "transport": "rmq"},
        },
    ])

    (project_dir / "package.json").write_text(json.dumps({
        "dependencies": {
            "@nestjs/core": "^11.0.0",
        },
    }))
    (project_dir / "src/controllers").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/modules").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/persistence").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/messaging").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/main.ts").write_text("async function bootstrap() {}\n")
    (project_dir / "src/controllers/users.controller.ts").write_text("export class UsersController {}\n")
    (project_dir / "src/modules/app.module.ts").write_text("export class AppModule {}\n")
    (project_dir / "src/persistence/user.model.ts").write_text("export class UserModel {}\n")
    (project_dir / "src/messaging/orders.consumer.ts").write_text("export class OrdersConsumer {}\n")

    config = WorkspaceConfig(name="nest-start-here-regression")
    config.add_project("nest", str(project_dir))

    with WorkspaceDB(config) as wdb:
        payload = wdb.codebase_map(project="nest")

    start_paths = [item["path"] for item in payload["start_here"]]

    assert payload["framework_hints"]["app_type"] == "nest"
    assert "src/main.ts" in start_paths
    assert any(path == "src/controllers/users.controller.ts" for path in start_paths)
    assert any(path == "src/modules/app.module.ts" for path in start_paths)
    assert any(path == "src/persistence/user.model.ts" for path in start_paths)
    assert any(path == "src/messaging/orders.consumer.ts" for path in start_paths)


def test_workspace_db_codebase_map_keeps_fullstack_backend_entrypoints_with_two_nitro_routes(
    tmp_path, ws_dir
):
    project_dir = _create_indexed_project(tmp_path, "fullstack", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "src/main.ts",
            "signature": "async function bootstrap()",
            "content": "nest bootstrap http server",
        },
        {
            "name": "UsersController",
            "kind": "controller",
            "path": "src/controllers/users.controller.ts",
            "signature": "class UsersController",
            "content": "nest controller users routes",
            "metadata": {"framework": "nest", "resource": "controller"},
        },
        {
            "name": "AppModule",
            "kind": "class",
            "path": "src/modules/app.module.ts",
            "signature": "class AppModule",
            "content": "nest module app wiring",
            "metadata": {"framework": "nest", "resource": "module"},
        },
        {
            "name": "EmailProcessor",
            "kind": "queue_processor",
            "path": "src/queues/email.processor.ts",
            "signature": "class EmailProcessor",
            "content": "bullmq queue processor emails",
            "metadata": {"framework": "bullmq", "resource": "processor"},
        },
    ])

    (project_dir / "app/pages").mkdir(parents=True)
    (project_dir / "server/api").mkdir(parents=True)
    (project_dir / "server/routes/api").mkdir(parents=True)
    (project_dir / "src/controllers").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/modules").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/queues").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/config").mkdir(parents=True)
    (project_dir / "prisma").mkdir(parents=True)

    (project_dir / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
            "@prisma/client": "^6.0.0",
            "bullmq": "^5.0.0",
        },
    }))
    (project_dir / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (project_dir / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (project_dir / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (project_dir / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'slug')\n")
    (project_dir / "src/config/runtime.ts").write_text("export const runtime = {}\n")
    (project_dir / "prisma/schema.prisma").write_text("model User { id Int @id }\n")

    config = WorkspaceConfig(name="fullstack-map-two-routes")
    config.add_project("fullstack", str(project_dir))

    with WorkspaceDB(config) as wdb:
        payload = wdb.codebase_map(project="fullstack")

    assert payload["framework_hints"]["app_type"] == "fullstack"
    assert payload["representative_files"]["backend"] == [
        "server/api/health.get.ts",
        "src/main.ts",
        "src/controllers/users.controller.ts",
    ]
    assert payload["topology"]["routes"]["files"] == [
        "server/api/health.get.ts",
        "server/routes/api/[...slug].ts",
        "src/controllers/users.controller.ts",
    ]
    assert any(item["path"] == "src/main.ts" for item in payload["start_here"])
    assert payload["start_here"][0]["path"] == "nuxt.config.ts"
    assert sum(1 for item in payload["start_here"] if item["path"].startswith("server/")) == 1


def test_workspace_db_codebase_map_distinguishes_route_systems_and_generic_layers(tmp_path, ws_dir):
    nitro_project = tmp_path / "nitro-only"
    (nitro_project / "app/pages").mkdir(parents=True)
    (nitro_project / "server/api").mkdir(parents=True)
    (nitro_project / "server/routes/api").mkdir(parents=True)
    (nitro_project / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
        },
    }))
    (nitro_project / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (nitro_project / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (nitro_project / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (nitro_project / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'slug')\n")

    nest_project = tmp_path / "nest-modules-only"
    (nest_project / "src/modules").mkdir(parents=True)
    (nest_project / "package.json").write_text(json.dumps({
        "dependencies": {
            "@nestjs/core": "^11.0.0",
        },
    }))
    (nest_project / "nest-cli.json").write_text("{}\n")
    (nest_project / "src/modules/app.module.ts").write_text("export class AppModule {}\n")
    (nest_project / "src/modules/users.module.ts").write_text("export class UsersModule {}\n")

    generic_project = tmp_path / "generic-layers"
    (generic_project / "src/db").mkdir(parents=True)
    (generic_project / "src/workers").mkdir(parents=True)
    (generic_project / "src/db/client.ts").write_text("export const client = {}\n")
    (generic_project / "src/workers/email.ts").write_text("export const emailWorker = {}\n")

    config = WorkspaceConfig(name="topology-regressions")
    config.add_project("nitro", str(nitro_project))
    config.add_project("nest", str(nest_project))
    config.add_project("generic", str(generic_project))

    with WorkspaceDB(config) as wdb:
        nitro_payload = wdb.codebase_map(project="nitro")
        nest_payload = wdb.codebase_map(project="nest")
        generic_payload = wdb.codebase_map(project="generic")

    assert nitro_payload["topology"]["routes"]["systems"] == ["nitro_file_routes"]
    assert nitro_payload["topology"]["routes"]["summary"] == (
        "HTTP route and transport surfaces from Nitro file routes."
    )
    assert "routes" not in nest_payload["topology"]
    assert generic_payload["topology"]["data"]["systems"] == ["generic"]
    assert generic_payload["topology"]["data"]["files"] == ["src/db/client.ts"]
    assert generic_payload["topology"]["async"]["systems"] == ["generic"]
    assert generic_payload["topology"]["async"]["files"] == ["src/workers/email.ts"]


def test_workspace_db_codebase_map_uses_indexed_metadata_for_unconventional_backend_surfaces(
    tmp_path, ws_dir
):
    project_dir = _create_indexed_project(tmp_path, "fullstack", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "src/main.ts",
            "signature": "async function bootstrap()",
            "content": "nest bootstrap http server",
            "metadata": {"framework": "nest", "resource": "bootstrap"},
        },
        {
            "name": "OrdersController",
            "kind": "controller",
            "path": "src/http/orders.controller.ts",
            "signature": "class OrdersController",
            "content": "nest controller orders routes",
            "metadata": {"framework": "nest", "resource": "controller", "route_prefix": "/orders"},
        },
        {
            "name": "RuntimeModule",
            "kind": "class",
            "path": "src/bootstrap/runtime.module.ts",
            "signature": "class RuntimeModule",
            "content": "runtime module env wiring",
            "metadata": {"framework": "nest", "resource": "module"},
        },
        {
            "name": "UserModel",
            "kind": "class",
            "path": "src/persistence/user.model.ts",
            "signature": "class UserModel",
            "content": "mongoose user model",
            "metadata": {"framework": "mongoose", "resource": "model"},
        },
        {
            "name": "OrdersConsumer",
            "kind": "microservice_handler",
            "path": "src/messaging/orders.consumer.ts",
            "signature": "function handleOrders()",
            "content": "rabbitmq consumer orders.created",
            "metadata": {"framework": "rabbitmq", "resource": "consumer", "transport": "rabbitmq"},
        },
        {
            "name": "CacheModule",
            "kind": "class",
            "path": "src/bootstrap/cache.module.ts",
            "signature": "class CacheModule",
            "content": "redis cache configuration",
            "metadata": {"framework": "redis", "resource": "config", "transport": "redis"},
        },
    ])

    (project_dir / "app/pages").mkdir(parents=True)
    (project_dir / "server/api").mkdir(parents=True)
    (project_dir / "server/routes/api").mkdir(parents=True)
    (project_dir / "src/http").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/bootstrap").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/persistence").mkdir(parents=True, exist_ok=True)
    (project_dir / "src/messaging").mkdir(parents=True, exist_ok=True)

    (project_dir / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
        },
    }))
    (project_dir / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (project_dir / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (project_dir / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (project_dir / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'slug')\n")
    (project_dir / "src/main.ts").write_text("async function bootstrap() {}\n")
    (project_dir / "src/http/orders.controller.ts").write_text("export class OrdersController {}\n")
    (project_dir / "src/bootstrap/runtime.module.ts").write_text("export class RuntimeModule {}\n")
    (project_dir / "src/persistence/user.model.ts").write_text("export class UserModel {}\n")
    (project_dir / "src/messaging/orders.consumer.ts").write_text("export class OrdersConsumer {}\n")
    (project_dir / "src/bootstrap/cache.module.ts").write_text("export class CacheModule {}\n")

    config = WorkspaceConfig(name="metadata-oriented-fullstack")
    config.add_project("fullstack", str(project_dir))

    with WorkspaceDB(config) as wdb:
        payload = wdb.codebase_map(project="fullstack")

    assert payload["representative_files"]["backend"] == [
        "server/api/health.get.ts",
        "src/main.ts",
        "src/http/orders.controller.ts",
    ]
    assert payload["topology"]["routes"]["systems"] == ["nest_controllers", "nitro_file_routes"]
    assert payload["topology"]["routes"]["files"] == [
        "server/api/health.get.ts",
        "server/routes/api/[...slug].ts",
        "src/http/orders.controller.ts",
    ]
    assert payload["topology"]["data"]["systems"] == ["mongoose"]
    assert payload["topology"]["data"]["files"] == ["src/persistence/user.model.ts"]
    assert payload["topology"]["async"]["systems"] == ["rabbitmq", "redis"]
    assert payload["topology"]["async"]["files"] == [
        "src/messaging/orders.consumer.ts",
        "src/bootstrap/cache.module.ts",
    ]
    assert payload["topology"]["runtime"]["files"] == [
        "nuxt.config.ts",
        "package.json",
        "src/bootstrap/runtime.module.ts",
    ]
    assert any(item["path"] == "src/main.ts" for item in payload["start_here"])
    assert sum(1 for item in payload["start_here"] if item["path"].startswith("server/")) == 1


def test_workspace_project_scoped_miss_recovery_strings_include_project(
    tmp_path, ws_dir, monkeypatch
):
    alpha = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "AuthRouteAlpha",
            "kind": "class",
            "path": "server/src/services/auth-route-alpha.ts",
            "signature": "class AuthRouteAlpha",
            "content": "auth route alpha helper",
            "doc_comment": "Alpha-local auth route helper.",
        },
    ])
    beta = _create_indexed_project(tmp_path, "beta", [
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])

    config = WorkspaceConfig(name="workspace-project-scoped-recovery")
    config.add_project("alpha", str(alpha))
    config.add_project("beta", str(beta))

    with WorkspaceDB(config) as wdb:
        monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)
        monkeypatch.setattr(server, "_get_workspace_db", lambda: wdb)

        get_payload = json.loads(server.get_symbol("authRoutesService", project="alpha"))
        search_payload = json.loads(server.search_symbols("authRoutesService", project="alpha"))

    assert all('project="alpha"' in suggestion for suggestion in get_payload["suggestions"])
    assert any(
        'search_symbols("auth routes service", project="alpha")' in suggestion
        for suggestion in get_payload["suggestions"]
    )
    assert all('project="alpha"' in suggestion for suggestion in search_payload["suggestions"])
    assert any(
        'search_symbols("auth routes service", project="alpha")' in suggestion
        for suggestion in search_payload["suggestions"]
    )
    assert any(
        'hybrid_search("auth routes service", project="alpha")' in suggestion
        for suggestion in search_payload["suggestions"]
    )
    assert 'project="alpha"' in search_payload["hint"]


def test_workspace_get_communities_supports_summary_and_verbose_params(
    tmp_path, ws_dir, monkeypatch
):
    project_dir = _create_indexed_project(tmp_path, "flowproj", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "server/app.ts",
            "signature": "function bootstrap()",
            "content": "bootstrap routeRequest",
        },
        {
            "name": "routeRequest",
            "kind": "function",
            "path": "server/app.ts",
            "signature": "function routeRequest()",
            "content": "routeRequest fetchUser",
        },
        {
            "name": "fetchUser",
            "kind": "function",
            "path": "server/data.ts",
            "signature": "function fetchUser()",
            "content": "fetchUser serializeUser",
        },
        {
            "name": "serializeUser",
            "kind": "function",
            "path": "server/data.ts",
            "signature": "function serializeUser()",
            "content": "serializeUser return",
        },
        {
            "name": "runJobs",
            "kind": "function",
            "path": "worker/jobs.ts",
            "signature": "function runJobs()",
            "content": "runJobs sendEmail",
        },
        {
            "name": "sendEmail",
            "kind": "function",
            "path": "worker/jobs.ts",
            "signature": "function sendEmail()",
            "content": "sendEmail return",
        },
    ])
    _store_workspace_communities_and_flows(project_dir)

    config = WorkspaceConfig(name="workspace-flow-tools")
    config.add_project("flowproj", str(project_dir))
    config.save()

    monkeypatch.setattr(server, "_workspace_name", "workspace-flow-tools")
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)

    summary = json.loads(
        server.get_communities(
            project="flowproj",
            member_limit=1,
            path_prefix="server/",
            layer="server",
        )
    )
    verbose = json.loads(
        server.get_communities(
            project="flowproj",
            verbose=True,
            member_limit=3,
            path_prefix="server/",
            layer="server",
        )
    )

    assert summary["communities"]
    assert summary["project"] == "flowproj"
    assert len(summary["communities"][0]["members"]) <= 1
    assert any(item["truncated"] is True for item in summary["communities"])
    assert all(item["member_limit_applied"] == 1 for item in summary["communities"])
    assert all(member["file_path"].startswith("server/") for member in summary["communities"][0]["members"])
    assert verbose["communities"][0]["members"]
    assert "qualified_name" in verbose["communities"][0]["members"][0]


def test_workspace_get_execution_flows_supports_verbose_depth_and_filters(
    tmp_path, ws_dir, monkeypatch
):
    project_dir = _create_indexed_project(tmp_path, "flowproj", [
        {
            "name": "bootstrap",
            "kind": "function",
            "path": "server/app.ts",
            "signature": "function bootstrap()",
            "content": "bootstrap routeRequest",
        },
        {
            "name": "routeRequest",
            "kind": "function",
            "path": "server/app.ts",
            "signature": "function routeRequest()",
            "content": "routeRequest fetchUser",
        },
        {
            "name": "fetchUser",
            "kind": "function",
            "path": "server/data.ts",
            "signature": "function fetchUser()",
            "content": "fetchUser serializeUser",
        },
        {
            "name": "serializeUser",
            "kind": "function",
            "path": "server/data.ts",
            "signature": "function serializeUser()",
            "content": "serializeUser return",
        },
        {
            "name": "runJobs",
            "kind": "function",
            "path": "worker/jobs.ts",
            "signature": "function runJobs()",
            "content": "runJobs sendEmail",
        },
        {
            "name": "sendEmail",
            "kind": "function",
            "path": "worker/jobs.ts",
            "signature": "function sendEmail()",
            "content": "sendEmail return",
        },
    ])
    _store_workspace_communities_and_flows(project_dir)

    config = WorkspaceConfig(name="workspace-flow-tools-exec")
    config.add_project("flowproj", str(project_dir))
    config.save()

    monkeypatch.setattr(server, "_workspace_name", "workspace-flow-tools-exec")
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)

    summary = json.loads(
        server.get_execution_flows(
            project="flowproj",
            path_prefix="server/",
            layer="server",
        )
    )
    verbose = json.loads(
        server.get_execution_flows(
            project="flowproj",
            verbose=True,
            max_depth=3,
            path_prefix="server/",
            layer="server",
        )
    )

    assert summary["flows"]
    assert summary["project"] == "flowproj"
    assert summary["flows"][0]["terminal"] == "serializeUser"
    assert summary["flows"][0]["truncated"] is True
    assert "steps" not in summary["flows"][0]
    assert len(summary["flows"][0]["key_steps"]) == 3
    assert verbose["flows"][0]["steps"]
    assert verbose["flows"][0]["max_depth_applied"] == 3
    assert verbose["flows"][0]["terminal"] == "serializeUser"
    assert all(step["file_path"].startswith("server/") for step in verbose["flows"][0]["steps"])


def test_workspace_task7_paths_do_not_retain_project_db_handles_over_max_attach(
    tmp_path, ws_dir, monkeypatch
):
    import srclight.db as db_mod
    import srclight.workspace as ws_mod

    orig_limit = ws_mod.MAX_ATTACH
    ws_mod.MAX_ATTACH = 3

    try:
        projects = {}
        for i in range(5):
            name = f"proj{i}"
            projects[name] = _create_indexed_project(tmp_path, name, [
                {
                    "name": f"AuthRoutesService{i}",
                    "kind": "class",
                    "path": f"server/src/services/auth-routes-service-{i}.ts",
                    "signature": f"class AuthRoutesService{i}",
                    "content": "auth routes service helpers",
                    "doc_comment": f"Service object for auth routes coordination {i}.",
                },
            ])

        active_handles = 0
        max_active_handles = 0
        real_open = db_mod.Database.open
        real_close = db_mod.Database.close

        def tracked_open(self):
            nonlocal active_handles, max_active_handles
            real_open(self)
            active_handles += 1
            max_active_handles = max(max_active_handles, active_handles)

        def tracked_close(self):
            nonlocal active_handles
            try:
                real_close(self)
            finally:
                active_handles -= 1

        monkeypatch.setattr(db_mod.Database, "open", tracked_open)
        monkeypatch.setattr(db_mod.Database, "close", tracked_close)

        config = WorkspaceConfig(name="workspace-bounded-handles")
        for name, project_dir in projects.items():
            config.add_project(name, str(project_dir))

        with WorkspaceDB(config) as wdb:
            results = wdb.search_symbols("auth routes")
            assert results
            assert active_handles == 0

            suggestions = wdb.suggest_symbol_names("authRoutes")
            assert suggestions
            assert active_handles == 0

        assert max_active_handles <= 1
    finally:
        ws_mod.MAX_ATTACH = orig_limit


def test_workspace_task7_paths_skip_broken_project_indexes(tmp_path, ws_dir):
    good = _create_indexed_project(tmp_path, "alpha", [
        {
            "name": "AuthRoutesService",
            "kind": "class",
            "path": "server/src/services/auth-routes-service.ts",
            "signature": "class AuthRoutesService",
            "content": "auth routes service helpers",
            "doc_comment": "Service object for auth routes coordination.",
        },
    ])
    broken = tmp_path / "broken"
    (broken / ".srclight").mkdir(parents=True)
    (broken / ".srclight" / "index.db").write_bytes(b"")

    config = WorkspaceConfig(name="workspace-broken-index")
    config.add_project("alpha", str(good))
    config.add_project("broken", str(broken))

    with WorkspaceDB(config) as wdb:
        results = wdb.search_symbols("auth routes")
        assert results
        assert any(result["project"] == "alpha" for result in results)

        suggestions = wdb.suggest_symbol_names("authRoutes")
        assert suggestions
        assert any(item["project"] == "alpha" for item in suggestions)
