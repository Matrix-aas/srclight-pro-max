"""Tests for embedding UX and server-facing embedding hints."""

from __future__ import annotations

import io
import json
import shlex
import sqlite3
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from srclight import server
from srclight.cli import main as cli_main
from srclight.db import Database, FileRecord, SymbolRecord
from srclight.embeddings import DEFAULT_OLLAMA_EMBED_MODEL


class _FakeDatabase:
    def __init__(self, path):
        self.path = path

    def open(self):
        return None

    def initialize(self):
        return None

    def close(self):
        return None

    def stats(self):
        return {"db_size_mb": 0.1}

    def embedding_stats(self):
        return {
            "embedded_symbols": 1,
            "total_symbols": 1,
            "coverage_pct": 100.0,
        }


class _FakeIndexer:
    def __init__(self, db, config):
        self.db = db
        self.config = config

    def index(self, root, on_progress=None, on_event=None):
        return SimpleNamespace(
            files_scanned=1,
            files_indexed=1,
            files_unchanged=0,
            files_removed=0,
            symbols_extracted=1,
            errors=0,
            elapsed_seconds=0.01,
        )


def _patch_indexer_stack(monkeypatch, tmp_path, workspace_entries=None):
    monkeypatch.setattr("srclight.db.Database", _FakeDatabase)
    monkeypatch.setattr("srclight.indexer.Indexer", _FakeIndexer)

    if workspace_entries is not None:
        class _FakeWorkspaceConfig:
            def __init__(self, entries):
                self._entries = entries

            def get_entries(self):
                return self._entries

        monkeypatch.setattr(
            "srclight.workspace.WorkspaceConfig.load",
            staticmethod(lambda ws_name: _FakeWorkspaceConfig(workspace_entries)),
        )


def test_index_bare_embed_defaults_to_local_model(tmp_path, monkeypatch):
    _patch_indexer_stack(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        ["index", str(tmp_path), "--db", str(tmp_path / "index.db"), "--embed"],
    )

    assert result.exit_code == 0, result.output
    assert f"Embedding model: {DEFAULT_OLLAMA_EMBED_MODEL}" in result.output


def test_index_explicit_embed_model_is_preserved(tmp_path, monkeypatch):
    _patch_indexer_stack(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        [
            "index",
            str(tmp_path),
            "--db",
            str(tmp_path / "index.db"),
            "--embed",
            "openai:text-embedding-3-small",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Embedding model: openai:text-embedding-3-small" in result.output


def test_workspace_index_bare_embed_defaults_to_local_model(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    entries = [SimpleNamespace(name="repo", path=str(repo))]
    _patch_indexer_stack(monkeypatch, tmp_path, workspace_entries=entries)
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        ["workspace", "index", "-w", "test-ws", "--embed"],
    )

    assert result.exit_code == 0, result.output
    assert f"Embedding model: {DEFAULT_OLLAMA_EMBED_MODEL}" in result.output


def test_workspace_index_explicit_embed_model_is_preserved(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    entries = [SimpleNamespace(name="repo", path=str(repo))]
    _patch_indexer_stack(monkeypatch, tmp_path, workspace_entries=entries)
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        [
            "workspace",
            "index",
            "-w",
            "test-ws",
            "--embed",
            "openai:text-embedding-3-small",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Embedding model: openai:text-embedding-3-small" in result.output


def test_index_progress_finishes_line_before_following_logs(tmp_path, monkeypatch):
    class _LoggingIndexer:
        def __init__(self, db, config):
            self.db = db
            self.config = config

        def index(self, root, on_progress=None, on_event=None):
            if on_progress is not None:
                on_progress("shared/src/validation/users.ts", 477, 477)
            click.echo("INFO srclight.indexer: Detected 38 communities, 50 execution flows")
            return SimpleNamespace(
                files_scanned=477,
                files_indexed=477,
                files_unchanged=0,
                files_removed=0,
                symbols_extracted=2056,
                errors=0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr("srclight.db.Database", _FakeDatabase)
    monkeypatch.setattr("srclight.indexer.Indexer", _LoggingIndexer)

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["index", str(tmp_path), "--db", str(tmp_path / "index.db")],
    )

    assert result.exit_code == 0, result.output
    assert "INFO srclight.indexer: Detected 38 communities, 50 execution flows" in result.output
    assert (
        "shared/src/validation/users.ts                              INFO srclight.indexer"
        not in result.output
    )
    assert "\nINFO srclight.indexer: Detected 38 communities, 50 execution flows" in result.output


def test_workspace_index_progress_finishes_line_before_following_logs(tmp_path, monkeypatch):
    class _LoggingIndexer:
        def __init__(self, db, config):
            self.db = db
            self.config = config

        def index(self, root, on_progress=None, on_event=None):
            if on_progress is not None:
                on_progress("shared/src/validation/users.ts", 477, 477)
            click.echo("INFO srclight.indexer: Detected 38 communities, 50 execution flows")
            return SimpleNamespace(
                files_scanned=477,
                files_indexed=477,
                files_unchanged=0,
                files_removed=0,
                symbols_extracted=2056,
                errors=0,
                elapsed_seconds=0.01,
            )

    repo = tmp_path / "repo"
    repo.mkdir()
    entries = [SimpleNamespace(name="repo", path=str(repo))]

    monkeypatch.setattr("srclight.db.Database", _FakeDatabase)
    monkeypatch.setattr("srclight.indexer.Indexer", _LoggingIndexer)

    class _FakeWorkspaceConfig:
        def __init__(self, workspace_entries):
            self._entries = workspace_entries

        def get_entries(self):
            return self._entries

    monkeypatch.setattr(
        "srclight.workspace.WorkspaceConfig.load",
        staticmethod(lambda ws_name: _FakeWorkspaceConfig(entries)),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["workspace", "index", "-w", "test-ws"],
    )

    assert result.exit_code == 0, result.output
    assert "INFO srclight.indexer: Detected 38 communities, 50 execution flows" in result.output
    assert (
        "shared/src/validation/users.ts                         INFO srclight.indexer"
        not in result.output
    )
    assert "\nINFO srclight.indexer: Detected 38 communities, 50 execution flows" in result.output


def test_index_plain_progress_fallback_reports_phases_and_embedding_batches(tmp_path, monkeypatch):
    class _EventingIndexer:
        def __init__(self, db, config):
            self.db = db
            self.config = config

        def index(self, root, on_progress=None, on_event=None):
            if on_progress is not None:
                on_progress("shared/src/validation/users.ts", 2, 5)
            if on_event is not None:
                on_event(
                    {
                        "phase": "graph",
                        "message": "Building call graph and inheritance edges",
                    }
                )
                on_event(
                    {
                        "phase": "communities",
                        "message": "Detected 38 communities, 50 execution flows",
                    }
                )
                on_event(
                    {
                        "phase": "embeddings",
                        "current": 3,
                        "total": 65,
                        "detail": "2056 symbols",
                        "elapsed_seconds": 35,
                        "remaining_seconds": 723,
                    }
                )
            return SimpleNamespace(
                files_scanned=477,
                files_indexed=477,
                files_unchanged=0,
                files_removed=0,
                symbols_extracted=2056,
                errors=0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr("srclight.db.Database", _FakeDatabase)
    monkeypatch.setattr("srclight.indexer.Indexer", _EventingIndexer)

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["index", str(tmp_path), "--db", str(tmp_path / "index.db"), "--embed"],
    )

    assert result.exit_code == 0, result.output
    assert "Graph: Building call graph and inheritance edges" in result.output
    assert "Communities: Detected 38 communities, 50 execution flows" in result.output
    assert "Embeddings: 3/65" in result.output
    assert "2056 symbols" in result.output
    assert "35s elapsed" in result.output
    assert "~12m03s remaining" in result.output


def test_rich_progress_renderer_constructs_on_supported_terminals(monkeypatch):
    pytest.importorskip("rich")
    from rich.console import Console

    from srclight import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_get_rich_console",
        lambda: Console(file=io.StringIO(), force_terminal=True),
    )

    progress = cli_module._create_index_progress()

    assert progress.__class__.__name__ == "_RichIndexProgress"


def test_rich_progress_switches_to_indeterminate_phase(monkeypatch):
    pytest.importorskip("rich")
    from rich.console import Console

    from srclight import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_get_rich_console",
        lambda: Console(file=io.StringIO(), force_terminal=True),
    )

    progress = cli_module._create_index_progress()
    with progress:
        progress.update_scan("shared/src/validation/users.ts", 1, 1)
        progress.emit_event(
            {
                "phase": "communities",
                "message": "Detecting communities and execution flows",
            }
        )
        task = progress._progress.tasks[0]

    assert task.total is None
    assert task.fields["counts"] == ""


@pytest.mark.asyncio
async def test_server_embedding_messages_reflect_default_embed_ux(monkeypatch):
    class _EmptyDB:
        def embedding_stats(self):
            return {}

    monkeypatch.setattr(server, "_get_db", lambda: _EmptyDB())

    semantic = json.loads(server.semantic_search("needle"))
    assert semantic["error"].startswith("No embeddings found.")
    assert "srclight index --embed" in semantic["hint"]
    assert DEFAULT_OLLAMA_EMBED_MODEL in semantic["hint"]

    status = json.loads(server.embedding_status())
    assert "srclight index --embed" in status["hint"]
    assert DEFAULT_OLLAMA_EMBED_MODEL in status["hint"]

    health = json.loads(server.embedding_health())
    assert health["status"] == "no_embeddings"
    assert "srclight index --embed" in health["detail"]
    assert DEFAULT_OLLAMA_EMBED_MODEL in health["detail"]


@pytest.mark.asyncio
async def test_setup_guide_mentions_bare_embed_and_default_model():
    guide = json.loads(await server.setup_guide())
    step_two = guide["steps"][1]

    assert "quick_start" in guide
    assert "decision_tree" in guide
    assert "troubleshooting" in guide
    assert "next_action" in guide
    assert "stdio_command" in guide["client_snippets"]
    assert guide["client_snippets"]["stdio_command"] == "srclight serve --transport stdio"
    assert guide["client_snippets"]["cursor_mcp_url"] == "http://127.0.0.1:8742/mcp"
    assert "srclight workspace index -w WORKSPACE_NAME --embed" in step_two["commands"]
    assert DEFAULT_OLLAMA_EMBED_MODEL in step_two["notes"]
    assert "reindex()" in step_two["notes"] or "restart the server" in step_two["notes"]


def test_project_required_error_points_agents_to_list_projects(monkeypatch):
    fake_workspace = SimpleNamespace(
        _all_indexable=[
            SimpleNamespace(name="alpha"),
            SimpleNamespace(name="beta"),
        ]
    )
    monkeypatch.setattr(server, "_get_workspace_db", lambda: fake_workspace)

    payload = json.loads(server._project_required_error("detect_changes"))

    assert payload["available_projects"] == ["alpha", "beta"]
    assert "list_projects()" in payload["hint"]
    assert 'project="alpha"' in payload["hint"]


def test_codebase_map_unknown_workspace_project_returns_not_found_contract(tmp_path, monkeypatch):
    def _raise_if_typo(project=None):
        if project == "typo":
            raise LookupError(project)
        return {"workspace": "test", "projects": []}

    fake_workspace = SimpleNamespace(
        _all_indexable=[SimpleNamespace(name="alpha")],
        codebase_map=_raise_if_typo,
    )
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)
    monkeypatch.setattr(server, "_get_workspace_db", lambda: fake_workspace)

    payload = json.loads(server.codebase_map(project="typo"))

    assert payload["error"] == "Project 'typo' not found"
    assert payload["available_projects"] == ["alpha"]


@pytest.mark.parametrize(
    "tool_name, call",
    [
        ("search_symbols", lambda: server.search_symbols("needle", project="typo")),
        ("hybrid_search", lambda: server.hybrid_search("needle", project="typo")),
        ("semantic_search", lambda: server.semantic_search("needle", project="typo")),
        ("embedding_status", lambda: server.embedding_status(project="typo")),
        ("embedding_health", lambda: server.embedding_health(project="typo")),
    ],
)
def test_workspace_project_typo_returns_not_found_contract_for_search_and_embedding_tools(
    monkeypatch, tool_name, call
):
    fake_workspace = SimpleNamespace(
        _all_indexable=[SimpleNamespace(name="alpha")],
        search_symbols=lambda *args, **kwargs: [],
        embedding_stats=lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)
    monkeypatch.setattr(server, "_get_workspace_db", lambda: fake_workspace)

    payload = json.loads(call())

    assert payload["error"] == "Project 'typo' not found", tool_name
    assert payload["available_projects"] == ["alpha"], tool_name


def test_index_status_bootstraps_on_cold_repo_without_database(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    server._db = None
    server._db_path = None
    server._repo_root = None

    payload = json.loads(server.index_status())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["index"]["status"] == "not_indexed"
    assert payload["index"]["present"] is False


def test_build_dynamic_instructions_uses_project_names_from_list_projects(monkeypatch):
    fake_workspace = SimpleNamespace(
        list_projects=lambda: [
            {"project": "alpha", "files": 2, "symbols": 3, "edges": 4},
            {"project": "beta", "files": 1, "symbols": 1, "edges": 1},
        ]
    )
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: True)
    monkeypatch.setattr(server, "_get_workspace_db", lambda: fake_workspace)
    monkeypatch.setattr(server, "_workspace_name", "demo")

    instructions = server._build_dynamic_instructions()

    assert "Projects: alpha, beta" in instructions
    assert "?, ?" not in instructions


def test_symbol_not_found_error_prefers_hybrid_search_first():
    payload = json.loads(server._symbol_not_found_error("Needle"))

    assert payload["suggestions"][0] == 'Try hybrid_search("Needle") for concept + keyword search'
    assert payload["suggestions"][1] == 'Try search_symbols("Needle") for exact/fuzzy keyword search'


def test_serve_defaults_to_stdio_for_agent_workflows(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "srclight.server.configure",
        lambda db_path=None, repo_root=None: calls.update({
            "db_path": db_path,
            "repo_root": repo_root,
        }),
    )
    monkeypatch.setattr(
        "srclight.server.run_server",
        lambda transport="sse", port=8742: calls.update({
            "transport": transport,
            "port": port,
        }),
    )

    runner = CliRunner()
    result = runner.invoke(cli_main, ["serve"])

    assert result.exit_code == 0, result.output
    assert calls["transport"] == "stdio"
    assert calls["repo_root"] == repo


def test_serve_web_auto_switches_to_sse_when_stdio_is_default(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "srclight.server.configure",
        lambda db_path=None, repo_root=None: calls.update({
            "db_path": db_path,
            "repo_root": repo_root,
        }),
    )
    monkeypatch.setattr(
        "srclight.server.run_server",
        lambda transport="sse", port=8742: calls.update({
            "transport": transport,
            "port": port,
        }),
    )
    monkeypatch.setattr("srclight.server.make_sse_and_streamable_http_app", lambda mount_path="/": object())
    monkeypatch.setattr("srclight.web.add_web_routes", lambda app: calls.update({"web_routes": True}))
    monkeypatch.setattr("anyio.run", lambda func: calls.update({"web_runner": True}))

    runner = CliRunner()
    result = runner.invoke(cli_main, ["serve", "--web"])

    assert result.exit_code == 0, result.output
    assert calls["web_runner"] is True
    assert "--web requires SSE transport" in result.output


def _reset_single_repo_server_state(monkeypatch, repo_root=None):
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_path", None)
    monkeypatch.setattr(server, "_repo_root", repo_root)
    monkeypatch.setattr(server, "_workspace_name", None)
    monkeypatch.setattr(server, "_workspace_db", None)


def test_codebase_map_bootstraps_frontend_repo_without_index(tmp_path, monkeypatch):
    repo = tmp_path / "nuxt-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "app/components").mkdir(parents=True)
    (repo / "app/composables").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)
    (repo / "app/assets/styles").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "graphql": "^16.0.0",
        },
        "devDependencies": {
            "postcss": "^8.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app.vue").write_text("<template><NuxtPage /></template>\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "app/components/AppHeader.vue").write_text("<template>Header</template>\n")
    (repo / "app/composables/useAuth.ts").write_text("export const useAuth = () => ({})\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "app/assets/styles/main.postcss").write_text(".root { color: red; }\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["index"]["status"] == "not_indexed"
    assert "srclight index --embed" in payload["hint"]
    assert payload["framework_hints"]["app_type"] == "nuxt"
    assert {"nuxt", "vue", "graphql", "postcss"} <= set(payload["framework_hints"]["signals"])
    assert any(item["path"] == "nuxt.config.ts" for item in payload["start_here"])
    assert any(item["path"] == "app/pages/index.vue" for item in payload["start_here"])
    assert payload["representative_files"]["server"] == ["server/api/health.get.ts"]


def test_codebase_map_detects_elysia_signals_and_large_subsystems_without_index(tmp_path, monkeypatch):
    repo = tmp_path / "elysia-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "src/http").mkdir(parents=True)
    (repo / "shared/src/domain/level").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({"dependencies": {}}))
    (repo / "src/http/app.ts").write_text(
        "import { Elysia } from 'elysia';\n"
        "export const app = new Elysia().get('/health', () => ({ ok: true }));\n"
    )
    for index in range(22):
        (repo / "shared/src/domain/level" / f"LevelPart{index}.ts").write_text(
            f"export const levelPart{index} = {index};\n"
        )

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    assert "elysia" in payload["framework_hints"]["signals"]
    assert payload["framework_hints"]["app_type"] == "node"
    assert "shared/src/domain/level (22 files)" in payload["brief"]


def test_codebase_map_fullstack_backend_keeps_entrypoints_and_route_surfaces(tmp_path, monkeypatch):
    repo = tmp_path / "mixed-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)
    (repo / "server/routes/api").mkdir(parents=True)
    (repo / "src/controllers").mkdir(parents=True)
    (repo / "src/modules").mkdir(parents=True)
    (repo / "src/config").mkdir(parents=True)
    (repo / "prisma").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
            "@prisma/client": "^6.0.0",
            "bullmq": "^5.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app.vue").write_text("<template><NuxtPage /></template>\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'slug')\n")
    (repo / "src/main.ts").write_text("async function bootstrap() {}\n")
    (repo / "src/controllers/users.controller.ts").write_text("export class UsersController {}\n")
    (repo / "src/modules/app.module.ts").write_text("export class AppModule {}\n")
    (repo / "src/config/runtime.ts").write_text("export const runtime = {}\n")
    (repo / "prisma/schema.prisma").write_text("model User { id Int @id }\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

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


def test_codebase_map_topology_distinguishes_route_systems_and_generic_layers(tmp_path, monkeypatch):
    def _bootstrap_repo(name: str):
        repo = tmp_path / name
        (repo / ".git").mkdir(parents=True)
        monkeypatch.chdir(repo)
        _reset_single_repo_server_state(monkeypatch)
        monkeypatch.setattr(
            server,
            "_get_db",
            lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
        )
        return repo

    nitro_repo = _bootstrap_repo("nitro-only")
    (nitro_repo / "app/pages").mkdir(parents=True)
    (nitro_repo / "server/api").mkdir(parents=True)
    (nitro_repo / "server/routes/api").mkdir(parents=True)
    (nitro_repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
        },
    }))
    (nitro_repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (nitro_repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (nitro_repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (nitro_repo / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'slug')\n")

    nitro_payload = json.loads(server.codebase_map())

    assert nitro_payload["topology"]["routes"]["systems"] == ["nitro_file_routes"]
    assert "nest_controllers" not in nitro_payload["topology"]["routes"]["systems"]
    assert nitro_payload["topology"]["routes"]["summary"] == (
        "HTTP route and transport surfaces from Nitro file routes."
    )

    nest_repo = _bootstrap_repo("nest-modules-only")
    (nest_repo / "src/modules").mkdir(parents=True)
    (nest_repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "@nestjs/core": "^11.0.0",
        },
    }))
    (nest_repo / "nest-cli.json").write_text("{}\n")
    (nest_repo / "src/modules/app.module.ts").write_text("export class AppModule {}\n")
    (nest_repo / "src/modules/users.module.ts").write_text("export class UsersModule {}\n")

    nest_payload = json.loads(server.codebase_map())

    assert "routes" not in nest_payload["topology"]

    generic_repo = _bootstrap_repo("generic-layers")
    (generic_repo / "src/db").mkdir(parents=True)
    (generic_repo / "src/workers").mkdir(parents=True)
    (generic_repo / "src/db/client.ts").write_text("export const client = {}\n")
    (generic_repo / "src/workers/email.ts").write_text("export const emailWorker = {}\n")

    generic_payload = json.loads(server.codebase_map())

    assert generic_payload["topology"]["data"]["systems"] == ["generic"]
    assert generic_payload["topology"]["data"]["files"] == ["src/db/client.ts"]
    assert generic_payload["topology"]["async"]["systems"] == ["generic"]
    assert generic_payload["topology"]["async"]["files"] == ["src/workers/email.ts"]


def test_codebase_map_treats_empty_index_file_as_not_indexed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".srclight").mkdir(parents=True)
    (repo / ".srclight/index.db").write_text("")
    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
        },
    }))
    (repo / "app.vue").write_text("<template><NuxtPage /></template>\n")

    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(AssertionError("codebase_map should not open an empty index.db")),
    )

    payload = json.loads(server.codebase_map())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["index"]["status"] == "not_indexed"
    assert payload["index"]["present"] is True


def test_codebase_map_falls_back_when_index_db_is_unreadable(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".srclight").mkdir(parents=True)
    (repo / ".srclight/index.db").write_text("not-a-real-db")
    (repo / "README.md").write_text("# Repo\n")

    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("malformed database")),
    )

    payload = json.loads(server.codebase_map())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["index"]["status"] == "not_indexed"
    assert payload["index"]["present"] is True


def test_codebase_map_bootstrap_falls_back_to_non_empty_start_here(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "README.md").write_text("# Repo\n")
    (repo / "package.json").write_text(json.dumps({
        "name": "repo",
    }))

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["start_here"]
    assert payload["start_here"][0]["path"] in {"package.json", "README.md"}


def test_configure_does_not_create_index_for_unindexed_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "README.md").write_text("# Repo\n")
    db_path = repo / ".srclight" / "index.db"

    _reset_single_repo_server_state(monkeypatch)
    calls = {"get_db": 0}
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: calls.__setitem__("get_db", calls["get_db"] + 1),
    )

    server.configure(repo_root=repo, db_path=db_path)

    assert calls["get_db"] == 0
    assert not db_path.exists()
    payload = json.loads(server.codebase_map())
    assert payload["bootstrap_mode"] == "filesystem_only"
    assert not db_path.exists()


def test_codebase_map_quotes_bootstrap_index_hint_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo with spaces"
    (repo / ".git").mkdir(parents=True)
    (repo / "README.md").write_text("# Repo\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    quoted = shlex.quote(str(repo))
    assert quoted in payload["hint"]
    assert any(quoted in action for action in payload["next_actions"])


def test_codebase_map_enriches_indexed_frontend_repo_with_start_here(tmp_path, monkeypatch):
    repo = tmp_path / "frontend"
    (repo / ".srclight").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "app/components").mkdir(parents=True)
    (repo / "app/composables").mkdir(parents=True)
    (repo / "app/stores").mkdir(parents=True)
    (repo / "app/graphql").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@pinia/nuxt": "^1.0.0",
            "@apollo/client": "^3.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app.vue").write_text("<template><NuxtPage /></template>\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "app/components/AppHeader.vue").write_text("<template>Header</template>\n")
    (repo / "app/composables/useAuth.ts").write_text("export const useAuth = () => ({})\n")
    (repo / "app/stores/auth.ts").write_text("export const useAuthStore = defineStore('auth', {})\n")
    (repo / "app/graphql/GetViewer.gql").write_text("query GetViewer { viewer { id } }\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / ".srclight/index.db").write_text("not-empty")

    class _FakeRepoBriefDB:
        def stats(self):
            return {
                "files": 8,
                "symbols": 24,
                "edges": 13,
                "db_size_mb": 1.2,
                "languages": {"vue": 3, "typescript": 4, "graphql": 1},
                "symbol_kinds": {"component": 3, "function": 7},
            }

        def get_index_state(self, repo_root):
            return {
                "repo_root": repo_root,
                "last_commit": "abc123",
                "indexed_at": "2026-04-14T18:30:00Z",
            }

        def directory_summary(self, max_depth=2):
            return [
                {"path": "app", "files": 6, "symbols": 20, "languages": ["typescript", "vue"]},
                {"path": "server", "files": 1, "symbols": 2, "languages": ["typescript"]},
            ]

        def hotspot_files(self, limit=10):
            return [
                {"path": "app/pages/index.vue", "language": "vue", "lines": 20, "symbols": 5},
                {"path": "app/stores/auth.ts", "language": "typescript", "lines": 40, "symbols": 4},
            ]

    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeRepoBriefDB())
    monkeypatch.setattr(server, "_read_index_signal", lambda root: {"timestamp": "2026-04-14T18:31:00Z"})

    payload = json.loads(server.codebase_map())

    assert payload["index"]["status"] == "ready"
    assert payload["framework_hints"]["app_type"] == "nuxt"
    assert {"nuxt", "vue", "graphql", "pinia"} <= set(payload["framework_hints"]["signals"])
    assert payload["representative_files"]["stores"] == ["app/stores/auth.ts"]
    assert payload["representative_files"]["graphql"] == ["app/graphql/GetViewer.gql"]
    assert any(item["path"] == "app.vue" for item in payload["start_here"])
    assert any(item["path"] == "app/stores/auth.ts" for item in payload["start_here"])


def test_codebase_map_balances_frontend_heavy_fullstack_orientation_from_indexed_hints(tmp_path, monkeypatch):
    repo = tmp_path / "mean-like-app"
    (repo / ".srclight").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "client/src/views").mkdir(parents=True)
    (repo / "client/src/render").mkdir(parents=True)
    (repo / "server/src/http").mkdir(parents=True)
    (repo / "shared/src/db").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "private": True,
        "workspaces": ["client", "server"],
    }))
    (repo / "client/package.json").write_text(json.dumps({
        "dependencies": {
            "vue": "^3.5.0",
            "three": "^0.175.0",
        },
        "devDependencies": {
            "vite": "^6.0.0",
        },
    }))
    (repo / "server/package.json").write_text(json.dumps({
        "dependencies": {
            "elysia": "^1.0.0",
            "drizzle-orm": "^0.40.0",
        },
    }))
    (repo / "vite.config.ts").write_text("export default defineConfig({})\n")
    (repo / "client/src/main.ts").write_text("createApp(App).mount('#app')\n")
    (repo / "client/src/views/SoloView.vue").write_text("<template>Solo</template>\n")
    (repo / "client/src/render/BoardRenderer3D.ts").write_text("export class BoardRenderer3D {}\n")
    (repo / "server/src/http/auth-link.ts").write_text("export const authLink = () => ({})\n")
    (repo / "shared/src/db/schema.ts").write_text("export const schema = {};\n")
    (repo / ".srclight/index.db").write_text("not-empty")

    class _FakeRepoBriefDB:
        def stats(self):
            return {
                "files": 462,
                "symbols": 1574,
                "edges": 756,
                "db_size_mb": 18.4,
                "languages": {"typescript": 290, "vue": 145},
                "symbol_kinds": {"function": 500, "component": 120, "class": 90},
            }

        def get_index_state(self, repo_root):
            return {
                "repo_root": repo_root,
                "last_commit": "abc123",
                "indexed_at": "2026-04-16T01:00:00Z",
            }

        def directory_summary(self, max_depth=2):
            return [
                {"path": "client", "files": 245, "symbols": 910, "languages": ["typescript", "vue"]},
                {"path": "server", "files": 133, "symbols": 410, "languages": ["typescript"]},
                {"path": "shared", "files": 84, "symbols": 254, "languages": ["typescript"]},
                *[
                    {"path": f"client/src/feature{i}", "files": i + 1, "symbols": i + 2, "languages": ["typescript"]}
                    for i in range(12)
                ],
            ]

        def hotspot_files(self, limit=10):
            hotspots = [
                {"path": "client/src/views/SoloView.vue", "language": "vue", "lines": 180, "symbols": 15},
                {"path": "client/src/render/BoardRenderer3D.ts", "language": "typescript", "lines": 320, "symbols": 14},
                {"path": "server/src/http/auth-link.ts", "language": "typescript", "lines": 90, "symbols": 9},
                {"path": "shared/src/db/schema.ts", "language": "typescript", "lines": 70, "symbols": 6},
            ]
            return hotspots[:limit]

        def orientation_files(self, limit=100):
            return [
                {
                    "path": "client/src/main.ts",
                    "language": "typescript",
                    "size": 200,
                    "line_count": 20,
                    "summary": "Vue 3 + Vite client bootstrap.",
                    "metadata": {"framework": "vue", "resource": "bootstrap"},
                    "top_level_symbols": [{"name": "mountApp", "kind": "function"}],
                },
                {
                    "path": "client/src/views/SoloView.vue",
                    "language": "vue",
                    "size": 800,
                    "line_count": 180,
                    "summary": "Gameplay view with Three scene controls and board renderer.",
                    "metadata": {"framework": "vue", "resource": "route"},
                    "top_level_symbols": [{"name": "SoloView", "kind": "component"}],
                },
                {
                    "path": "client/src/render/BoardRenderer3D.ts",
                    "language": "typescript",
                    "size": 1200,
                    "line_count": 320,
                    "summary": "Three.js renderer for gameplay board scenes.",
                    "metadata": {"framework": "three", "resource": "renderer"},
                    "top_level_symbols": [{"name": "BoardRenderer3D", "kind": "class"}],
                },
                {
                    "path": "server/src/http/auth-link.ts",
                    "language": "typescript",
                    "size": 300,
                    "line_count": 90,
                    "summary": "Elysia auth route handlers.",
                    "metadata": {"framework": "elysia", "resource": "route", "route_path": "/auth/link", "http_method": "GET"},
                    "top_level_symbols": [{"name": "authLink", "kind": "route_handler"}],
                },
                {
                    "path": "shared/src/db/schema.ts",
                    "language": "typescript",
                    "size": 220,
                    "line_count": 70,
                    "summary": "Drizzle schema and persistence metadata.",
                    "metadata": {"framework": "drizzle", "resource": "schema"},
                    "top_level_symbols": [{"name": "matchSchema", "kind": "schema"}],
                },
            ][:limit]

        def orientation_symbols(self, limit=200):
            return [
                {
                    "kind": "function",
                    "name": "mountApp",
                    "signature": "function mountApp()",
                    "file_path": "client/src/main.ts",
                    "metadata": {"framework": "vue", "resource": "bootstrap"},
                },
                {
                    "kind": "component",
                    "name": "SoloView",
                    "signature": "component SoloView",
                    "file_path": "client/src/views/SoloView.vue",
                    "metadata": {"framework": "vue", "resource": "route"},
                },
                {
                    "kind": "class",
                    "name": "BoardRenderer3D",
                    "signature": "class BoardRenderer3D",
                    "file_path": "client/src/render/BoardRenderer3D.ts",
                    "metadata": {"framework": "three", "resource": "renderer"},
                },
                {
                    "kind": "route_handler",
                    "name": "authLink",
                    "signature": "GET /auth/link",
                    "file_path": "server/src/http/auth-link.ts",
                    "metadata": {"framework": "elysia", "resource": "route", "route_path": "/auth/link", "http_method": "GET"},
                },
                {
                    "kind": "schema",
                    "name": "matchSchema",
                    "signature": "const matchSchema",
                    "file_path": "shared/src/db/schema.ts",
                    "metadata": {"framework": "drizzle", "resource": "schema"},
                },
            ][:limit]

    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeRepoBriefDB())
    monkeypatch.setattr(server, "_read_index_signal", lambda root: {"timestamp": "2026-04-16T01:02:00Z"})

    payload = json.loads(server.codebase_map())

    signals = set(payload["framework_hints"]["signals"])
    start_paths = [item["path"] for item in payload["start_here"]]

    assert payload["framework_hints"]["app_type"] == "fullstack"
    assert {"vue", "vite", "three", "elysia", "drizzle"} <= signals
    assert "client/src/main.ts" in start_paths
    assert any(path.startswith("client/src/") for path in start_paths[:4])
    assert any(path.startswith("server/src/") for path in start_paths[:6])
    assert "client (245 files)" in payload["brief"]
    assert payload["topology"]["frontend"]["files"][0] == "client/src/main.ts"


def test_codebase_map_uses_workspace_package_roots_for_generic_frontend_backend_discovery(tmp_path, monkeypatch):
    repo = tmp_path / "workspace-oriented-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "apps/web/src/views").mkdir(parents=True)
    (repo / "apps/api/src/http").mkdir(parents=True)
    (repo / "packages/shared/src").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "private": True,
        "workspaces": ["apps/*", "packages/*"],
    }))
    (repo / "apps/web/package.json").write_text(json.dumps({
        "dependencies": {"vue": "^3.5.0"},
        "devDependencies": {"vite": "^6.0.0"},
    }))
    (repo / "apps/api/package.json").write_text(json.dumps({
        "dependencies": {"elysia": "^1.0.0"},
    }))
    (repo / "apps/web/vite.config.ts").write_text("export default defineConfig({})\n")
    (repo / "apps/web/src/main.ts").write_text("createApp(App).mount('#app')\n")
    (repo / "apps/web/src/views/HomeView.vue").write_text("<template>Home</template>\n")
    (repo / "apps/api/src/http/routes.ts").write_text("new Elysia()\n")
    (repo / "packages/shared/src/index.ts").write_text("export const shared = true;\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    signals = set(payload["framework_hints"]["signals"])
    start_paths = [item["path"] for item in payload["start_here"]]

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["framework_hints"]["app_type"] == "fullstack"
    assert {"vue", "vite", "elysia"} <= signals
    assert "apps/web/src/main.ts" in payload["representative_files"]["entrypoints"]
    assert "apps/api/src/http/routes.ts" in payload["representative_files"]["backend"]
    assert "apps/web/src/main.ts" in start_paths
    assert "apps/api/src/http/routes.ts" in start_paths


def test_codebase_map_surfaces_nitro_server_files_for_indexed_and_unindexed_repos(tmp_path, monkeypatch):
    repo = tmp_path / "nuxt-nitro-app"
    (repo / ".git").mkdir(parents=True)
    (repo / ".srclight").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)
    (repo / "server/routes/api").mkdir(parents=True)
    (repo / "server/plugins").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "server/routes/api/[...slug].ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "server/plugins/auth.ts").write_text("export default defineNitroPlugin(() => {})\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    unindexed_payload = json.loads(server.codebase_map())

    assert unindexed_payload["bootstrap_mode"] == "filesystem_only"
    assert "nitro" in unindexed_payload["framework_hints"]["signals"]
    assert "server/api/health.get.ts" in unindexed_payload["representative_files"]["server"]
    assert "server/routes/api/[...slug].ts" in unindexed_payload["representative_files"]["server"]
    assert any(item["path"] == "server/api/health.get.ts" for item in unindexed_payload["start_here"])

    (repo / ".srclight/index.db").write_text("not-empty")

    class _FakeNitroRepoBriefDB:
        def stats(self):
            return {
                "files": 5,
                "symbols": 9,
                "edges": 3,
                "db_size_mb": 0.4,
                "languages": {"typescript": 3, "vue": 1},
                "symbol_kinds": {"route": 2, "plugin": 1, "middleware": 1},
            }

        def get_index_state(self, repo_root):
            return {
                "repo_root": repo_root,
                "last_commit": "abc123",
                "indexed_at": "2026-04-14T18:30:00Z",
            }

        def directory_summary(self, max_depth=2):
            return [
                {"path": "server", "files": 3, "symbols": 4, "languages": ["typescript"]},
                {"path": "app", "files": 1, "symbols": 2, "languages": ["vue"]},
            ]

        def hotspot_files(self, limit=10):
            return [
                {"path": "server/api/health.get.ts", "language": "typescript", "lines": 5, "symbols": 1},
                {"path": "server/routes/api/[...slug].ts", "language": "typescript", "lines": 5, "symbols": 1},
            ]

    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeNitroRepoBriefDB())
    monkeypatch.setattr(server, "_read_index_signal", lambda root: {"timestamp": "2026-04-14T18:31:00Z"})

    indexed_payload = json.loads(server.codebase_map())

    assert indexed_payload["index"]["status"] == "ready"
    assert "nitro" in indexed_payload["framework_hints"]["signals"]
    assert "server/api/health.get.ts" in indexed_payload["representative_files"]["server"]
    assert "server/routes/api/[...slug].ts" in indexed_payload["representative_files"]["server"]
    assert any(item["path"] == "server/api/health.get.ts" for item in indexed_payload["start_here"])


def test_codebase_map_bootstrap_orients_fullstack_backend_async_layers(tmp_path, monkeypatch):
    repo = tmp_path / "fullstack-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)
    (repo / "src/controllers").mkdir(parents=True)
    (repo / "src/modules").mkdir(parents=True)
    (repo / "src/queues").mkdir(parents=True)
    (repo / "src/config").mkdir(parents=True)
    (repo / "prisma").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
            "@prisma/client": "^6.0.0",
            "bullmq": "^5.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "src/main.ts").write_text("async function bootstrap() {}\n")
    (repo / "src/controllers/users.controller.ts").write_text("export class UsersController {}\n")
    (repo / "src/modules/app.module.ts").write_text("export class AppModule {}\n")
    (repo / "src/queues/email.processor.ts").write_text("export class EmailProcessor {}\n")
    (repo / "src/config/runtime.ts").write_text("export const runtime = {}\n")
    (repo / "prisma/schema.prisma").write_text("model User { id Int @id }\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("unable to open database file")),
    )

    payload = json.loads(server.codebase_map())

    assert payload["bootstrap_mode"] == "filesystem_only"
    assert payload["framework_hints"]["app_type"] == "fullstack"
    assert {"nuxt", "nest"} <= set(payload["framework_hints"]["signals"])
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
    start_paths = [item["path"] for item in payload["start_here"][:4]]
    assert "src/main.ts" in start_paths
    assert "server/api/health.get.ts" in start_paths


def test_codebase_map_uses_indexed_symbol_metadata_for_backend_orientation(tmp_path, monkeypatch):
    repo = tmp_path / "indexed-metadata-app"
    (repo / ".srclight").mkdir(parents=True)
    (repo / "app/pages").mkdir(parents=True)
    (repo / "server/api").mkdir(parents=True)
    (repo / "src/http").mkdir(parents=True)
    (repo / "src/bootstrap").mkdir(parents=True)
    (repo / "src/persistence").mkdir(parents=True)
    (repo / "src/messaging").mkdir(parents=True)

    (repo / "package.json").write_text(json.dumps({
        "dependencies": {
            "nuxt": "^4.0.0",
            "@nestjs/core": "^11.0.0",
        },
    }))
    (repo / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
    (repo / "app/pages/index.vue").write_text("<template>Home</template>\n")
    (repo / "server/api/health.get.ts").write_text("export default defineEventHandler(() => 'ok')\n")
    (repo / "src/main.ts").write_text("async function bootstrap() {}\n")
    (repo / "src/http/orders.controller.ts").write_text("export class OrdersController {}\n")
    (repo / "src/bootstrap/runtime.module.ts").write_text("export class RuntimeModule {}\n")
    (repo / "src/persistence/drizzle.client.ts").write_text("export const db = {}\n")
    (repo / "src/messaging/orders.consumer.ts").write_text("export class OrdersConsumer {}\n")
    (repo / "src/bootstrap/cache.module.ts").write_text("export class CacheModule {}\n")

    db_path = repo / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_ids: dict[str, int] = {}

    def _file_id_for(path: str) -> int:
        if path not in file_ids:
            file_ids[path] = db.upsert_file(
                FileRecord(
                    path=path,
                    content_hash=f"indexed:{path}",
                    mtime=1000.0,
                    language="typescript" if path.endswith(".ts") else "vue",
                    size=200,
                    line_count=20,
                )
            )
        return file_ids[path]

    seeded_symbols = [
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
            "name": "DrizzleClient",
            "kind": "function",
            "path": "src/persistence/drizzle.client.ts",
            "signature": "function createDrizzleClient()",
            "content": "drizzle database client",
            "metadata": {"framework": "drizzle", "resource": "database"},
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
    ]

    for i, symbol in enumerate(seeded_symbols):
        db.insert_symbol(
            SymbolRecord(
                file_id=_file_id_for(symbol["path"]),
                kind=symbol["kind"],
                name=symbol["name"],
                qualified_name=f"indexed.{symbol['name']}",
                signature=symbol["signature"],
                start_line=i * 10 + 1,
                end_line=i * 10 + 8,
                content=symbol["content"],
                line_count=8,
                metadata=symbol["metadata"],
            ),
            symbol["path"],
        )

    db.commit()
    db.close()

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(server, "_read_index_signal", lambda root: {"timestamp": "2026-04-15T08:31:00Z"})

    payload = json.loads(server.codebase_map())

    assert payload["index"]["status"] == "ready"
    assert payload["topology"]["routes"]["systems"] == ["nest_controllers", "nitro_file_routes"]
    assert payload["topology"]["routes"]["files"] == [
        "server/api/health.get.ts",
        "src/http/orders.controller.ts",
    ]
    assert payload["topology"]["data"]["systems"] == ["drizzle"]
    assert payload["topology"]["data"]["files"] == ["src/persistence/drizzle.client.ts"]
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


def test_codebase_map_indexed_repo_does_not_walk_full_filesystem_for_large_subsystems(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "indexed-no-subsystem-walk"
    (repo / ".git").mkdir(parents=True)
    (repo / ".srclight").mkdir(parents=True)
    (repo / "src").mkdir(parents=True)
    (repo / "package.json").write_text(json.dumps({"dependencies": {}}))
    (repo / "src/main.ts").write_text("async function bootstrap() {}\n")

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    db.initialize()
    try:
        file_id = db.upsert_file(FileRecord(
            path="src/main.ts",
            content_hash="src/main.ts",
            mtime=1.0,
            language="typescript",
            size=64,
            line_count=1,
        ))
        db.insert_symbol(SymbolRecord(
            file_id=file_id,
            kind="function",
            name="bootstrap",
            qualified_name="bootstrap",
            signature="async function bootstrap()",
            start_line=1,
            end_line=1,
            content="bootstrap runtime",
            line_count=1,
            metadata={"resource": "bootstrap"},
        ), "src/main.ts")
        db.commit()

        monkeypatch.chdir(repo)
        _reset_single_repo_server_state(monkeypatch)
        monkeypatch.setattr(server, "_read_index_signal", lambda root: None)
        monkeypatch.setattr(
            server,
            "_large_subsystem_summaries",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("indexed codebase_map should not scan the full filesystem")
            ),
        )

        payload = json.loads(server.codebase_map())
    finally:
        db.close()

    assert payload["index"]["status"] == "ready"


def test_codebase_map_treats_rmq_transport_metadata_as_rabbitmq_async_system(
    tmp_path, monkeypatch
):
    repo = tmp_path / "rmq-indexed-app"
    (repo / ".srclight").mkdir(parents=True)
    (repo / "src/messaging").mkdir(parents=True)

    db_path = repo / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(
        FileRecord(
            path="src/messaging/orders.consumer.ts",
            content_hash="indexed:src/messaging/orders.consumer.ts",
            mtime=1000.0,
            language="typescript",
            size=200,
            line_count=20,
        )
    )
    db.insert_symbol(
        SymbolRecord(
            file_id=file_id,
            kind="microservice_handler",
            name="OrdersConsumer",
            qualified_name="indexed.OrdersConsumer",
            signature="function handleOrders()",
            start_line=1,
            end_line=8,
            content="rmq consumer orders.created",
            line_count=8,
            metadata={"framework": "nest", "resource": "consumer", "transport": "rmq"},
        ),
        "src/messaging/orders.consumer.ts",
    )
    db.commit()
    db.close()

    (repo / "src/messaging/orders.consumer.ts").write_text("export class OrdersConsumer {}\n")

    monkeypatch.chdir(repo)
    _reset_single_repo_server_state(monkeypatch, repo_root=repo)
    monkeypatch.setattr(server, "_read_index_signal", lambda root: {"timestamp": "2026-04-15T08:31:00Z"})

    payload = json.loads(server.codebase_map())

    assert payload["representative_files"]["async"] == ["src/messaging/orders.consumer.ts"]
    assert payload["topology"]["async"]["systems"] == ["rabbitmq"]
    assert payload["topology"]["async"]["files"] == ["src/messaging/orders.consumer.ts"]


def test_indexed_orientation_hints_keep_generic_config_out_of_async_topology():
    hints = server._indexed_orientation_hints([
        {
            "kind": "class",
            "name": "AppConfig",
            "signature": "class AppConfig",
            "file_path": "src/config/app.config.ts",
            "metadata": {"resource": "config"},
        },
    ])

    assert hints["runtime_files"] == ["src/config/app.config.ts"]
    assert hints["representative_files"]["config"] == ["src/config/app.config.ts"]
    assert hints["representative_files"]["async"] == []
    assert hints["async_systems"] == []


def test_indexed_orientation_hints_do_not_label_non_nest_routes_as_nest_controllers():
    hints = server._indexed_orientation_hints([
        {
            "kind": "route",
            "name": "healthRoute",
            "signature": "GET /health",
            "file_path": "src/http/health.route.ts",
            "metadata": {"framework": "fastify", "route_path": "/health", "http_method": "GET"},
        },
    ])

    assert hints["route_files"] == ["src/http/health.route.ts"]
    assert "nest_controllers" not in hints["route_systems"]
    assert "nest" not in hints["signals"]


def test_indexed_orientation_hints_non_nest_framework_overrides_nest_like_path_heuristics():
    hints = server._indexed_orientation_hints([
        {
            "kind": "controller",
            "name": "HealthController",
            "signature": "class HealthController",
            "file_path": "src/controllers/health.controller.ts",
            "metadata": {"framework": "fastify", "route_path": "/health", "http_method": "GET"},
        },
    ])

    assert hints["route_files"] == ["src/controllers/health.controller.ts"]
    assert "nest_controllers" not in hints["route_systems"]
    assert "nest" not in hints["signals"]


def test_orientation_symbols_collects_representative_rows_beyond_first_200_paths(tmp_path):
    db_path = tmp_path / "orientation-symbols.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_ids: dict[str, int] = {}

    def _file_id_for(path: str) -> int:
        if path not in file_ids:
            file_ids[path] = db.upsert_file(
                FileRecord(
                    path=path,
                    content_hash=f"orientation:{path}",
                    mtime=1000.0,
                    language="typescript",
                    size=200,
                    line_count=20,
                )
            )
        return file_ids[path]

    for i in range(220):
        path = f"aa/generated/{i:03d}.ts"
        db.insert_symbol(
            SymbolRecord(
                file_id=_file_id_for(path),
                kind="class",
                name=f"Generated{i}",
                qualified_name=f"generated.Generated{i}",
                signature=f"class Generated{i}",
                start_line=1,
                end_line=5,
                content="generated filler",
                line_count=5,
                metadata={"framework": "noise", "resource": "helper"},
            ),
            path,
        )

    important_symbols = [
        (
            "src/main.ts",
            "function",
            "bootstrap",
            {"framework": "nest", "resource": "bootstrap"},
        ),
        (
            "src/http/orders.controller.ts",
            "controller",
            "OrdersController",
            {"framework": "nest", "resource": "controller", "route_prefix": "/orders"},
        ),
        (
            "src/persistence/mongo.model.ts",
            "class",
            "UserModel",
            {"framework": "mongoose", "resource": "model"},
        ),
        (
            "src/messaging/orders.consumer.ts",
            "microservice_handler",
            "OrdersConsumer",
            {"framework": "rabbitmq", "resource": "consumer", "transport": "rabbitmq"},
        ),
        (
            "src/config/app.config.ts",
            "class",
            "AppConfig",
            {"resource": "config"},
        ),
    ]

    for i, (path, kind, name, metadata) in enumerate(important_symbols, start=1000):
        db.insert_symbol(
            SymbolRecord(
                file_id=_file_id_for(path),
                kind=kind,
                name=name,
                qualified_name=f"important.{name}",
                signature=f"{kind} {name}",
                start_line=i,
                end_line=i + 4,
                content="important orientation signal",
                line_count=5,
                metadata=metadata,
            ),
            path,
        )

    db.commit()

    rows = db.orientation_symbols()
    paths = {row["file_path"] for row in rows}

    assert "src/main.ts" in paths
    assert "src/http/orders.controller.ts" in paths
    assert "src/persistence/mongo.model.ts" in paths
    assert "src/messaging/orders.consumer.ts" in paths
    assert "src/config/app.config.ts" in paths

    db.close()


def test_indexed_orientation_hints_ignore_sparse_metadata_without_route_or_async_leaks():
    hints = server._indexed_orientation_hints([
        {
            "kind": "class",
            "name": "PlainService",
            "signature": "class PlainService",
            "file_path": "src/services/plain.service.ts",
            "metadata": {},
        },
    ])

    assert hints["route_systems"] == []
    assert hints["route_files"] == []
    assert hints["async_systems"] == []
    assert hints["runtime_files"] == []
