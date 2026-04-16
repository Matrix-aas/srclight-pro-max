"""Tests for new analysis tools: find_dead_code, find_pattern, find_imports."""


import json

import pytest

import srclight.server as server
import srclight.task_context as task_context
from srclight.db import Database, EdgeRecord, FileRecord, SymbolRecord
from srclight.indexer import IndexConfig, Indexer
from srclight.server import _extract_imports


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


def _insert_file(db, path="src/main.py", language="python", **kwargs):
    """Helper to insert a file and return its id."""
    defaults = dict(
        content_hash=path, mtime=1.0, size=100, line_count=10,
    )
    defaults.update(kwargs)
    return db.upsert_file(FileRecord(path=path, language=language, **defaults))


def _insert_symbol(db, file_id, name, file_path="src/main.py", kind="function",
                   start_line=1, end_line=5, content=None, **kwargs):
    """Helper to insert a symbol and return its id."""
    if content is None:
        content = f"def {name}(): pass"
    return db.insert_symbol(SymbolRecord(
        file_id=file_id, kind=kind, name=name,
        start_line=start_line, end_line=end_line,
        content=content, line_count=end_line - start_line + 1,
        **kwargs,
    ), file_path)


# --- find_dead_code tests ---


class TestFindDeadCode:
    def test_unreferenced_symbols_returned(self, db):
        """Symbols with no incoming edges are returned as dead code."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "used_fn", start_line=1, end_line=3)
        _insert_symbol(db, fid, "unused_fn", start_line=5, end_line=8)
        caller_id = _insert_symbol(db, fid, "caller", start_line=10, end_line=15,
                                   content="def caller(): used_fn()")

        # used_fn has an edge from caller
        used = db.get_symbol_by_name("used_fn")
        db.insert_edge(EdgeRecord(
            source_id=caller_id, target_id=used.id, edge_type="calls",
        ))
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        # unused_fn has no callers, caller also has no callers
        assert "unused_fn" in dead_names
        assert "caller" in dead_names
        # used_fn has an incoming edge, so should NOT be in dead code
        assert "used_fn" not in dead_names

    def test_entry_points_excluded(self, db):
        """main, __init__, test_* are excluded from dead code."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "main", start_line=1, end_line=3)
        _insert_symbol(db, fid, "__init__", start_line=5, end_line=7, kind="method")
        _insert_symbol(db, fid, "test_something", start_line=9, end_line=12)
        _insert_symbol(db, fid, "TestCase", start_line=14, end_line=20, kind="class")
        _insert_symbol(db, fid, "real_function", start_line=22, end_line=25)
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "main" not in dead_names
        assert "__init__" not in dead_names
        assert "test_something" not in dead_names
        assert "TestCase" not in dead_names
        # real_function has no callers and is not an entry point
        assert "real_function" in dead_names

    def test_public_visibility_excluded(self, db):
        """Symbols with 'public' or 'export' visibility are excluded."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "public_fn", start_line=1, end_line=3,
                       visibility="public")
        _insert_symbol(db, fid, "exported_fn", start_line=5, end_line=7,
                       visibility="export")
        _insert_symbol(db, fid, "private_fn", start_line=9, end_line=12,
                       visibility="private")
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "public_fn" not in dead_names
        assert "exported_fn" not in dead_names
        assert "private_fn" in dead_names

    def test_only_relevant_kinds(self, db):
        """Only function, method, class kinds are checked for dead code."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "MyEnum", kind="enum", start_line=1, end_line=5,
                       content="enum MyEnum { A, B }")
        _insert_symbol(db, fid, "my_func", kind="function", start_line=7, end_line=10)
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "MyEnum" not in dead_names  # enum not in checked kinds
        assert "my_func" in dead_names

    def test_limit_respected(self, db):
        """get_dead_symbols respects the limit parameter."""
        fid = _insert_file(db)
        for i in range(10):
            _insert_symbol(db, fid, f"fn_{i}", start_line=i * 5, end_line=i * 5 + 3)
        db.commit()

        dead = db.get_dead_symbols(limit=3)
        assert len(dead) == 3

    def test_typescript_constructors_and_main_bootstrap_are_not_reported_dead(self, db):
        file_id = _insert_file(db, path="apps/backend/src/main.ts", language="typescript")
        _insert_symbol(
            db,
            file_id,
            "bootstrap",
            file_path="apps/backend/src/main.ts",
            kind="function",
            start_line=1,
            end_line=5,
            content="async function bootstrap() { return NestFactory.create(AppModule); }",
        )
        _insert_symbol(
            db,
            file_id,
            "constructor",
            file_path="apps/backend/src/main.ts",
            kind="method",
            start_line=7,
            end_line=8,
            content="constructor(private readonly app: AppService) {}",
        )
        db.commit()

        dead = db.get_dead_symbols()
        dead_names = [d.name for d in dead]
        assert "bootstrap" not in dead_names
        assert "constructor" not in dead_names

    def test_nuxt_auto_imported_exported_composable_not_reported_dead(self, db, tmp_path):
        project = tmp_path / "nuxt-auto-import"
        (project / "composables").mkdir(parents=True)

        (project / "nuxt.config.ts").write_text("export default defineNuxtConfig({})\n")
        (project / "composables" / "use-pagination.ts").write_text("""\
export function usePagination() {
  return { page: ref(1) };
}
""")
        (project / "app.vue").write_text("""\
<script setup lang="ts">
const { page } = usePagination()
</script>
""")

        Indexer(db, IndexConfig(root=project)).index(project)

        dead_names = {symbol.name for symbol in db.get_dead_symbols(kind="function")}
        use_pagination = db.get_symbol_by_name("usePagination")

        assert use_pagination is not None
        assert use_pagination.visibility == "export"
        assert "usePagination" not in dead_names

    def test_nest_use_factory_provider_function_not_reported_dead(self, db, tmp_path):
        project = tmp_path / "nest-use-factory"
        src = project / "src"
        src.mkdir(parents=True)

        (src / "auth.module.ts").write_text("""\
import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

function buildAuthOptions(config: ConfigService) {
  return { secret: config.get('AUTH_SECRET') };
}

@Module({
  providers: [
    {
      provide: 'AUTH_OPTIONS',
      useFactory: buildAuthOptions,
      inject: [ConfigService],
    },
  ],
})
export class AuthModule {}
""")

        Indexer(db, IndexConfig(root=project)).index(project)

        dead_names = {symbol.name for symbol in db.get_dead_symbols(kind="function")}
        assert "buildAuthOptions" not in dead_names


# --- api_surface tests ---


class TestApiSurface:
    def test_api_surface_lists_nest_and_elysia_endpoints(self, db, monkeypatch):
        controller_file = _insert_file(db, path="apps/backend/src/controllers/coding.controller.ts", language="typescript")
        router_file = _insert_file(db, path="apps/backend/src/http/routes.ts", language="typescript")

        _insert_symbol(
            db,
            controller_file,
            "getCoding",
            file_path="apps/backend/src/controllers/coding.controller.ts",
            kind="route_handler",
            signature="GET /coding/:id",
            content="getCoding()",
            metadata={
                "framework": "nest",
                "resource": "route_handler",
                "http_method": "GET",
                "route_path": "/coding/:id",
                "route_prefix": "/coding",
            },
        )
        _insert_symbol(
            db,
            router_file,
            "codingRoutes",
            file_path="apps/backend/src/http/routes.ts",
            kind="router",
            signature="Elysia router | /api | GET /api/health | POST /api/coding",
            content="coding routes",
            metadata={
                "framework": "elysia",
                "resource": "router",
                "prefix": "/api",
                "routes": [
                    {"method": "GET", "path": "/api/health"},
                    {"method": "POST", "path": "/api/coding"},
                ],
            },
        )
        db.commit()

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)

        payload = json.loads(server.api_surface())

        assert payload["endpoint_count"] == 3
        assert payload["endpoints"][0]["method"] == "GET"
        assert {item["path"] for item in payload["endpoints"]} == {
            "/coding/:id",
            "/api/health",
            "/api/coding",
        }


# --- find_pattern tests ---


class TestFindPattern:
    def test_pattern_matches_content(self, db):
        """Regex pattern matching finds symbols with matching content."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn_with_todo", start_line=1, end_line=5,
                       content="def fn_with_todo():\n    # TODO: fix this\n    pass")
        _insert_symbol(db, fid, "clean_fn", start_line=7, end_line=10,
                       content="def clean_fn():\n    return 42")
        db.commit()

        results = db.find_pattern_in_symbols("TODO")
        assert len(results) == 1
        assert results[0]["name"] == "fn_with_todo"
        assert results[0]["match_count"] == 1

    def test_pattern_with_kind_filter(self, db):
        """Kind filter narrows results to specific symbol types."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "my_func", kind="function", start_line=1, end_line=5,
                       content="def my_func():\n    sleep(1)")
        _insert_symbol(db, fid, "MyClass", kind="class", start_line=7, end_line=12,
                       content="class MyClass:\n    def run(self):\n        sleep(2)")
        db.commit()

        # Both have sleep, but filter to function only
        results = db.find_pattern_in_symbols(r"sleep\(", kind="function")
        assert len(results) == 1
        assert results[0]["name"] == "my_func"

    def test_multiple_matches_in_symbol(self, db):
        """Multiple matches within one symbol are counted."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "many_fixmes", start_line=1, end_line=10,
                       content="def many_fixmes():\n    # FIXME: one\n    x = 1\n    # FIXME: two\n    # FIXME: three")
        db.commit()

        results = db.find_pattern_in_symbols("FIXME")
        assert len(results) == 1
        assert results[0]["match_count"] == 3

    def test_matched_lines_returned(self, db):
        """Matched lines include line offset and content."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=4,
                       content="def fn():\n    x = dangerous_call()\n    return x")
        db.commit()

        results = db.find_pattern_in_symbols("dangerous_call")
        assert len(results) == 1
        matched = results[0]["matched_lines"]
        assert len(matched) == 1
        assert matched[0]["line_offset"] == 2
        assert "dangerous_call" in matched[0]["line"]
        assert matched[0]["match"] == "dangerous_call"

    def test_no_matches_returns_empty(self, db):
        """No matches returns empty list."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=3,
                       content="def fn(): return 1")
        db.commit()

        results = db.find_pattern_in_symbols("NONEXISTENT_PATTERN")
        assert results == []

    def test_limit_respected(self, db):
        """find_pattern_in_symbols respects the limit parameter."""
        fid = _insert_file(db)
        for i in range(10):
            _insert_symbol(db, fid, f"fn_{i}", start_line=i * 5, end_line=i * 5 + 3,
                           content=f"def fn_{i}():\n    # TODO: item {i}")
        db.commit()

        results = db.find_pattern_in_symbols("TODO", limit=3)
        assert len(results) == 3

    def test_regex_special_characters(self, db):
        """Regex with special characters works correctly."""
        fid = _insert_file(db)
        _insert_symbol(db, fid, "fn", start_line=1, end_line=3,
                       content="def fn():\n    except Exception as e:")
        db.commit()

        results = db.find_pattern_in_symbols(r"except\s+\w+\s+as\s+\w+")
        assert len(results) == 1


# --- find_imports tests ---


class TestExtractImports:
    """Test the _extract_imports helper function directly."""

    def test_python_import(self):
        content = "import os\nimport sys"
        imports = _extract_imports(content, "python")
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "sys" in modules

    def test_python_from_import(self):
        content = "from pathlib import Path\nfrom os.path import join, exists"
        imports = _extract_imports(content, "python")
        assert len(imports) == 2

        path_imp = next(i for i in imports if i["module"] == "pathlib")
        assert "Path" in path_imp["names"]

        os_imp = next(i for i in imports if i["module"] == "os.path")
        assert "join" in os_imp["names"]
        assert "exists" in os_imp["names"]

    def test_python_relative_import(self):
        content = "from .db import Database\nfrom ..utils import helper"
        imports = _extract_imports(content, "python")
        modules = [i["module"] for i in imports]
        assert ".db" in modules
        assert "..utils" in modules

    def test_javascript_import(self):
        content = "import React from 'react';\nimport { useState } from 'react';"
        imports = _extract_imports(content, "javascript")
        modules = [i["module"] for i in imports]
        assert "react" in modules

    def test_javascript_require(self):
        content = "const fs = require('fs');\nconst path = require('path');"
        imports = _extract_imports(content, "javascript")
        modules = [i["module"] for i in imports]
        assert "fs" in modules
        assert "path" in modules

    def test_typescript_import(self):
        content = "import { Component } from '@angular/core';\nimport type { Config } from './config';"
        imports = _extract_imports(content, "typescript")
        modules = [i["module"] for i in imports]
        assert "@angular/core" in modules
        assert "./config" in modules

    def test_c_include(self):
        content = '#include <stdio.h>\n#include "myheader.h"'
        imports = _extract_imports(content, "c")
        modules = [i["module"] for i in imports]
        assert "stdio.h" in modules
        assert "myheader.h" in modules

    def test_cpp_include(self):
        content = '#include <vector>\n#include <string>\n#include "utils.h"'
        imports = _extract_imports(content, "cpp")
        assert len(imports) == 3

    def test_java_import(self):
        content = "import java.util.List;\nimport com.example.MyClass;"
        imports = _extract_imports(content, "java")
        modules = [i["module"] for i in imports]
        assert "java.util.List" in modules
        assert "com.example.MyClass" in modules

    def test_go_import(self):
        content = 'import (\n    "fmt"\n    "os"\n)'
        imports = _extract_imports(content, "go")
        modules = [i["module"] for i in imports]
        assert "fmt" in modules
        assert "os" in modules

    def test_csharp_using(self):
        content = "using System;\nusing System.Collections.Generic;"
        imports = _extract_imports(content, "csharp")
        modules = [i["module"] for i in imports]
        assert "System" in modules
        assert "System.Collections.Generic" in modules

    def test_dart_import(self):
        content = "import 'dart:async';\nimport 'package:flutter/material.dart';"
        imports = _extract_imports(content, "dart")
        modules = [i["module"] for i in imports]
        assert "dart:async" in modules
        assert "package:flutter/material.dart" in modules

    def test_swift_import(self):
        content = "import Foundation\nimport UIKit"
        imports = _extract_imports(content, "swift")
        modules = [i["module"] for i in imports]
        assert "Foundation" in modules
        assert "UIKit" in modules

    def test_kotlin_import(self):
        content = "import kotlin.collections.List\nimport com.example.Utils"
        imports = _extract_imports(content, "kotlin")
        modules = [i["module"] for i in imports]
        assert "kotlin.collections.List" in modules
        assert "com.example.Utils" in modules

    def test_unsupported_language_returns_empty(self):
        imports = _extract_imports("some content", "markdown")
        assert imports == []

    def test_empty_content_returns_empty(self):
        imports = _extract_imports("", "python")
        assert imports == []

    def test_no_duplicate_statements(self):
        """Same import statement on multiple lines shouldn't duplicate."""
        content = "import os\nimport os"
        imports = _extract_imports(content, "python")
        # Both lines are "import os", so deduped
        os_imports = [i for i in imports if i["module"] == "os"]
        assert len(os_imports) == 1


class TestResolveImport:
    """Test the db.resolve_import method."""

    def test_resolve_by_symbol_name(self, db):
        """Import name matching a symbol name resolves to that symbol."""
        fid = _insert_file(db, "src/db.py")
        _insert_symbol(db, fid, "Database", file_path="src/db.py", kind="class",
                       start_line=10, end_line=50, content="class Database: pass")
        db.commit()

        result = db.resolve_import("Database")
        assert result is not None
        assert result["name"] == "Database"
        assert result["file"] == "src/db.py"
        assert result["kind"] == "class"
        assert result["match_type"] == "symbol"

    def test_resolve_by_qualified_name(self, db):
        """Import matching a qualified_name resolves."""
        fid = _insert_file(db, "src/utils.py")
        _insert_symbol(db, fid, "helper", file_path="src/utils.py",
                       qualified_name="utils.helper",
                       start_line=1, end_line=5, content="def helper(): pass")
        db.commit()

        result = db.resolve_import("utils.helper")
        assert result is not None
        assert result["name"] == "helper"
        assert result["match_type"] == "qualified_name"

    def test_resolve_by_file_path(self, db):
        """Import matching a file path resolves."""
        _insert_file(db, "src/utils.py")
        db.commit()

        result = db.resolve_import("src.utils")
        assert result is not None
        assert result["file"] == "src/utils.py"
        assert result["match_type"] == "file_path"

    def test_unresolved_returns_none(self, db):
        """Unknown import returns None."""
        db.commit()
        result = db.resolve_import("nonexistent_module")
        assert result is None

    def test_resolve_js_path(self, db):
        """JS-style import path resolves to .js file."""
        _insert_file(db, "src/utils.js", language="javascript")
        db.commit()

        result = db.resolve_import("src/utils")
        assert result is not None
        assert result["file"] == "src/utils.js"
        assert result["match_type"] == "file_path"

    def test_resolve_typescript_alias_and_workspace_package(self, db, tmp_path):
        repo = tmp_path / "repo"
        (repo / "apps/backend/src/entities/coding").mkdir(parents=True)
        (repo / "packages/shared/src").mkdir(parents=True)
        (repo / "package.json").write_text(
            """\
{
  "name": "monorepo-root",
  "private": true,
  "workspaces": ["apps/*", "packages/*"]
}
"""
        )
        (repo / "apps/backend/tsconfig.json").write_text(
            """\
{
  "compilerOptions": {
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"]
    }
    }
}
"""
        )
        (repo / "apps/backend/src/entities/coding/coding.schema.ts").write_text("export class CodingSchema {}")
        (repo / "packages/shared/package.json").write_text('{"name":"@repo/shared"}')
        (repo / "packages/shared/src/index.ts").write_text("export interface SharedDto { id: string }")
        _insert_file(db, "apps/backend/src/entities/coding/coding.schema.ts", language="typescript")
        _insert_file(db, "packages/shared/src/index.ts", language="typescript")
        db.commit()

        alias_result = db.resolve_import(
            "@/entities/coding/coding.schema",
            hint_path="apps/backend/src/modules/coding/coding.service.ts",
            root_path=repo,
        )
        assert alias_result is not None
        assert alias_result["file"] == "apps/backend/src/entities/coding/coding.schema.ts"
        assert alias_result["match_type"] == "file_path"

        workspace_result = db.resolve_import(
            "@repo/shared",
            hint_path="apps/backend/src/modules/coding/coding.service.ts",
            root_path=repo,
        )
        assert workspace_result is not None
        assert workspace_result["file"] == "packages/shared/src/index.ts"
        assert workspace_result["match_type"] == "file_path"


class TestFindImportsIntegration:
    """Integration tests for the find_imports tool using the DB + file system."""

    def test_python_imports_resolved(self, db, tmp_path):
        """Python imports are extracted and resolved against the index."""
        # Set up a project directory with files
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "main.py").write_text(
            "from db import Database\nimport json\nimport os\n"
        )
        (proj / "db.py").write_text("class Database: pass\n")

        # Index: register files and symbols
        _insert_file(db, "main.py", language="python")
        fid_db = _insert_file(db, "db.py", language="python")
        _insert_symbol(db, fid_db, "Database", file_path="db.py", kind="class",
                       start_line=1, end_line=1, content="class Database: pass")
        db.commit()

        # Extract imports from the file content
        content = (proj / "main.py").read_text()
        raw_imports = _extract_imports(content, "python")

        assert len(raw_imports) == 3

        # Resolve each import
        resolved = []
        for imp in raw_imports:
            names_to_try = imp["names"] if imp["names"] else [imp["module"].split(".")[-1]]
            for name in names_to_try:
                result = db.resolve_import(name)
                if result:
                    resolved.append(result)
                    break

        # "Database" should be resolved
        resolved_names = [r["name"] for r in resolved]
        assert "Database" in resolved_names

    def test_javascript_imports_resolved(self, db, tmp_path):
        """JavaScript imports are extracted and resolved."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.js").write_text(
            "import { render } from './renderer';\n"
            "const utils = require('./utils');\n"
        )

        _insert_file(db, "app.js", language="javascript")
        fid_renderer = _insert_file(db, "renderer.js", language="javascript")
        _insert_symbol(db, fid_renderer, "render", file_path="renderer.js",
                       kind="function", start_line=1, end_line=5,
                       content="function render() {}")
        db.commit()

        content = (proj / "app.js").read_text()
        raw_imports = _extract_imports(content, "javascript")

        assert len(raw_imports) == 2
        modules = [i["module"] for i in raw_imports]
        assert "./renderer" in modules
        assert "./utils" in modules

        # "render" symbol should resolve
        result = db.resolve_import("render")
        assert result is not None
        assert result["name"] == "render"

    def test_c_includes_extracted(self, db, tmp_path):
        """C #include directives are extracted."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "main.c").write_text(
            '#include <stdio.h>\n'
            '#include "mylib.h"\n'
            '\nint main() { return 0; }\n'
        )

        content = (proj / "main.c").read_text()
        raw_imports = _extract_imports(content, "c")

        assert len(raw_imports) == 2
        modules = [i["module"] for i in raw_imports]
        assert "stdio.h" in modules
        assert "mylib.h" in modules

    def test_server_find_imports_resolves_typescript_aliases_and_workspace_packages(self, db, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        service_dir = repo / "apps/backend/src/modules/coding"
        entity_dir = repo / "apps/backend/src/entities/coding"
        shared_dir = repo / "packages/shared/src"
        service_dir.mkdir(parents=True)
        entity_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)

        (repo / "package.json").write_text(
            """\
{
  "name": "monorepo-root",
  "private": true,
  "workspaces": ["apps/*", "packages/*"]
}
"""
        )
        (repo / "apps/backend/tsconfig.json").write_text(
            """\
{
  "compilerOptions": {
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"]
    }
  }
}
"""
        )
        (service_dir / "coding.service.ts").write_text(
            """\
import { CodingSchema } from '@/entities/coding/coding.schema';
import { SharedDto } from '@repo/shared';
"""
        )
        (entity_dir / "coding.schema.ts").write_text("export class CodingSchema {}")
        (repo / "packages/shared/package.json").write_text('{"name":"@repo/shared"}')
        (shared_dir / "index.ts").write_text("export interface SharedDto { id: string }")

        _insert_file(db, "apps/backend/src/modules/coding/coding.service.ts", language="typescript")
        _insert_file(db, "apps/backend/src/entities/coding/coding.schema.ts", language="typescript")
        _insert_file(db, "packages/shared/src/index.ts", language="typescript")
        db.commit()

        monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_repo_root", repo)

        payload = json.loads(server.find_imports("apps/backend/src/modules/coding/coding.service.ts"))

        assert payload["resolved_count"] == 2
        files = {entry["module"]: entry["resolved_to"]["file"] for entry in payload["imports"]}
        assert files["@/entities/coding/coding.schema"] == "apps/backend/src/entities/coding/coding.schema.ts"
        assert files["@repo/shared"] == "packages/shared/src/index.ts"


class TestFileNavigationTools:
    def test_list_files_returns_db_backed_results(self, db, monkeypatch):
        file_paths = [
            "shared/src/domain/model.ts",
            "shared/src/domain/level/nested.ts",
            "shared/src/domain/level/deeper/more.ts",
            "shared/src/infra/http.ts",
        ]
        for path in file_paths:
            _insert_file(db, path=path, language="typescript")
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        payload = json.loads(server.list_files(path_prefix="shared/src/domain", recursive=False))
        assert payload["path_prefix"] == "shared/src/domain"
        assert payload["recursive"] is False
        assert payload["files"][0]["path"].startswith("shared/src/domain/")
        assert payload["files"] == [
            {
                "path": "shared/src/domain/model.ts",
                "language": "typescript",
                "size": 100,
                "line_count": 10,
                "summary": "Indexed typescript file. No top-level symbols indexed.",
            },
        ]

        deep_payload = json.loads(server.list_files(
            path_prefix="shared/src/domain",
            recursive=True,
            limit=2,
        ))
        assert deep_payload["recursive"] is True
        assert len(deep_payload["files"]) == 2

    def test_list_files_derives_summary_from_top_level_symbols_when_file_summary_missing(self, db, monkeypatch):
        file_path = "shared/src/domain/level/LayoutEngine.ts"
        file_id = _insert_file(db, path=file_path, language="typescript")
        _insert_symbol(
            db,
            file_id,
            "LayoutEngine",
            file_path=file_path,
            kind="class",
            start_line=1,
            end_line=40,
            signature="class LayoutEngine",
            content="export class LayoutEngine {}",
        )
        _insert_symbol(
            db,
            file_id,
            "measureNode",
            file_path=file_path,
            kind="function",
            start_line=42,
            end_line=55,
            signature="function measureNode(node)",
            content="export function measureNode(node) {}",
        )
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        payload = json.loads(server.list_files(path_prefix="shared/src/domain/level"))
        assert payload["files"]
        assert payload["files"][0]["summary"] == (
            "Top-level symbols: LayoutEngine (class), measureNode (function)."
        )

    def test_list_files_compacts_many_top_level_symbols_into_one_line_summary(self, db, monkeypatch):
        file_path = "shared/src/domain/level/LayoutEngine.ts"
        file_id = _insert_file(db, path=file_path, language="typescript")
        for index, name in enumerate(("forwardSimulate", "assignMahjongLayout", "smartAssign", "seedLayout")):
            _insert_symbol(
                db,
                file_id,
                name,
                file_path=file_path,
                kind="function",
                start_line=index * 10 + 1,
                end_line=index * 10 + 5,
                signature=f"function {name}()",
                content=f"export function {name}() {{}}",
            )
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        payload = json.loads(server.list_files(path_prefix="shared/src/domain/level"))
        assert payload["files"][0]["summary"] == (
            "Top-level symbols: forwardSimulate (function), assignMahjongLayout (function), smartAssign (function) +1 more."
        )

    def test_list_files_marks_index_barrels_when_no_symbols_are_indexed(self, db, monkeypatch):
        file_path = "shared/src/domain/level/index.ts"
        _insert_file(db, path=file_path, language="typescript")
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        payload = json.loads(server.list_files(path_prefix="shared/src/domain/level"))
        assert payload["files"]
        assert payload["files"][0]["summary"] == (
            "Index barrel file (typescript). No top-level symbols indexed."
        )

    def test_get_file_summary_returns_summary_and_top_level_symbols(self, db, monkeypatch):
        file_path = "client/src/components/ProfileCard.vue"
        file_id = _insert_file(db, path=file_path, language="vue", size=240, line_count=20)
        _insert_symbol(
            db,
            file_id,
            "ProfileCard",
            file_path=file_path,
            kind="component",
            start_line=1,
            end_line=20,
            content="<template><div /></template>",
            signature="<script setup>",
        )
        db.update_file_summary(
            file_path,
            summary="Profile card component shell.",
            metadata={"framework": "vue"},
        )
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        payload = json.loads(server.get_file_summary(file_path))
        assert payload["file"] == file_path
        assert payload["summary"] == "Profile card component shell."
        assert payload["metadata"] == {"framework": "vue"}
        assert payload["top_level_symbols"][0]["name"] == "ProfileCard"

    def test_symbols_in_file_distinguishes_missing_file_from_empty_indexed_file(self, db, monkeypatch):
        _insert_file(db, path="src/empty.ts", language="typescript")
        db.commit()

        monkeypatch.setattr("srclight.server._is_workspace_mode", lambda: False)
        monkeypatch.setattr("srclight.server._get_db", lambda: db)

        missing = json.loads(server.symbols_in_file("src/missing.ts"))
        assert missing["error"] == "File 'src/missing.ts' not found"

        empty = json.loads(server.symbols_in_file("src/empty.ts"))
        assert empty["file"] == "src/empty.ts"
        assert empty["symbol_count"] == 0
        assert empty["symbols"] == []


def test_list_projects_single_mode_returns_guidance(monkeypatch):
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_repo_root", None)

    payload = json.loads(server.list_projects())

    assert payload["error"] == "Not in workspace mode"
    assert payload["mode"] == "single"
    assert "srclight serve --workspace" in payload["hint"]


def test_get_callees_reports_ambiguous_same_name_symbols(monkeypatch, db):
    left_file = _insert_file(db, path="apps/backend/src/block.service.ts", language="typescript")
    right_file = _insert_file(db, path="apps/backend/src/coding.service.ts", language="typescript")
    _insert_symbol(
        db,
        left_file,
        "findOne",
        file_path="apps/backend/src/block.service.ts",
        kind="method",
        start_line=11,
        end_line=20,
        content="findOne() { return this.repo.findOne(); }",
    )
    _insert_symbol(
        db,
        right_file,
        "findOne",
        file_path="apps/backend/src/coding.service.ts",
        kind="method",
        start_line=16,
        end_line=25,
        content="findOne() { return this.repo.findOne(); }",
    )
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    payload = json.loads(server.get_callees("findOne"))

    assert payload["error"] == "Ambiguous symbol name 'findOne'"
    assert payload["match_count"] == 2
    assert {candidate["file"] for candidate in payload["candidates"]} == {
        "apps/backend/src/block.service.ts",
        "apps/backend/src/coding.service.ts",
    }


def test_detect_changes_compact_returns_summary_entries(monkeypatch, db, tmp_path):
    file_id = _insert_file(
        db,
        path="apps/backend/src/coding.service.ts",
        language="typescript",
        line_count=80,
    )
    _insert_symbol(
        db,
        file_id,
        "updateDescription",
        file_path="apps/backend/src/coding.service.ts",
        kind="method",
        start_line=20,
        end_line=36,
        content="updateDescription() { return this.repo.save(); }",
        qualified_name="CodingService.updateDescription",
    )
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_repo_root", lambda project=None: tmp_path)

    import srclight.community as community_mod
    import srclight.git as git_mod

    monkeypatch.setattr(
        git_mod,
        "detect_changes",
        lambda repo_root, ref=None: [
            {
                "file": "apps/backend/src/coding.service.ts",
                "hunks": [{"new_start": 22, "new_count": 4}],
            }
        ],
    )
    monkeypatch.setattr(
        community_mod,
        "compute_impact",
        lambda db_obj, symbol_id, sym_to_comm, flows: {
            "risk": "HIGH",
            "direct_dependents": 3,
            "transitive_dependents": 6,
            "affected_communities": [1, 2],
            "affected_flows": ["controller -> service -> repository"],
        },
    )

    payload = json.loads(server.detect_changes(compact=True))

    assert payload["compact"] is True
    assert payload["changed_symbol_count"] == 1
    assert payload["changed_symbols"] == [
        {
            "name": "updateDescription",
            "qualified_name": "CodingService.updateDescription",
            "kind": "method",
            "file": "apps/backend/src/coding.service.ts",
            "lines": "20-36",
            "risk": "HIGH",
            "direct_dependents": 3,
            "transitive_dependents": 6,
            "flows_affected": 1,
        }
    ]


def test_context_for_task_returns_compact_actionable_context(monkeypatch, db, tmp_path):
    service_file = _insert_file(
        db,
        path="apps/backend/src/coding.service.ts",
        language="typescript",
        line_count=120,
    )
    controller_file = _insert_file(
        db,
        path="apps/backend/src/coding.controller.ts",
        language="typescript",
        line_count=80,
    )
    types_file = _insert_file(
        db,
        path="apps/backend/src/coding.types.ts",
        language="typescript",
        line_count=40,
    )
    test_file = _insert_file(
        db,
        path="apps/backend/test/coding.service.spec.ts",
        language="typescript",
        line_count=60,
    )

    service_id = _insert_symbol(
        db,
        service_file,
        "updateDescription",
        file_path="apps/backend/src/coding.service.ts",
        kind="method",
        start_line=24,
        end_line=46,
        qualified_name="CodingService.updateDescription",
        content="updateDescription(dto: CodingDocument) { return this.repo.save(dto); }",
    )
    controller_id = _insert_symbol(
        db,
        controller_file,
        "patchCodingDescription",
        file_path="apps/backend/src/coding.controller.ts",
        kind="route_handler",
        start_line=10,
        end_line=24,
        qualified_name="CodingController.patchCodingDescription",
        signature="PATCH /coding/:id/description",
        content="patchCodingDescription() { return this.coding.updateDescription(dto); }",
        metadata={
            "framework": "nestjs",
            "resource": "route_handler",
            "http_method": "PATCH",
            "route_path": "/coding/:id/description",
            "route_prefix": "/coding",
        },
    )
    type_id = _insert_symbol(
        db,
        types_file,
        "CodingDocument",
        file_path="apps/backend/src/coding.types.ts",
        kind="interface",
        start_line=1,
        end_line=8,
        qualified_name="CodingDocument",
        content="interface CodingDocument { description: string }",
    )
    _insert_symbol(
        db,
        test_file,
        "test_updateDescription",
        file_path="apps/backend/test/coding.service.spec.ts",
        kind="function",
        start_line=5,
        end_line=18,
        qualified_name="test_updateDescription",
        content="test('updateDescription validates payload', () => {})",
    )
    db.insert_edge(EdgeRecord(source_id=controller_id, target_id=service_id, edge_type="calls", confidence=0.9))
    db.insert_edge(EdgeRecord(source_id=service_id, target_id=type_id, edge_type="uses_type", confidence=0.95))
    db.commit()

    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_repo_root", tmp_path)

    payload = json.loads(server.context_for_task("add validation to CodingService.updateDescription", budget="small"))

    assert payload["task"] == "add validation to CodingService.updateDescription"
    assert payload["budget"] == "small"
    assert payload["seeds"]
    assert payload["primary_symbols"]
    assert payload["primary_symbols"][0]["qualified_name"] == "CodingService.updateDescription"
    assert payload["primary_files"]
    assert payload["primary_files"][0]["file"] == "apps/backend/src/coding.service.ts"
    assert payload["related_api"]
    assert payload["related_api"][0]["path"] == "/coding/:id/description"
    assert payload["related_tests"]
    assert payload["related_tests"][0]["file"] == "apps/backend/test/coding.service.spec.ts"
    assert payload["data_types"]
    assert payload["data_types"][0]["name"] == "CodingDocument"
    assert payload["call_chain"]
    assert payload["next_steps"]
    assert payload["why_these_results"]


def test_context_for_task_uses_hybrid_seed_fallback_for_natural_language(monkeypatch, db, tmp_path):
    file_id = _insert_file(
        db,
        path="apps/backend/src/auth.service.ts",
        language="typescript",
        line_count=80,
    )
    _insert_symbol(
        db,
        file_id,
        "refreshSession",
        file_path="apps/backend/src/auth.service.ts",
        kind="method",
        start_line=15,
        end_line=34,
        qualified_name="AuthService.refreshSession",
        signature="refreshSession(sessionToken: string)",
        content="refreshSession() { return this.tokens.rotate(); }",
    )
    db.commit()

    hybrid_calls: list[tuple[str, int]] = []

    def _fake_hybrid_seed_candidates(db_obj, task, seed_limit):
        hybrid_calls.append((task, seed_limit))
        sym = db_obj.get_symbol_by_name("refreshSession")
        return [(35, sym, "matched hybrid search")]

    monkeypatch.setattr(db, "search_symbols", lambda *args, **kwargs: [])
    monkeypatch.setattr(task_context, "_hybrid_seed_candidates", _fake_hybrid_seed_candidates)
    monkeypatch.setattr(server, "_is_workspace_mode", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_repo_root", tmp_path)

    payload = json.loads(server.context_for_task("fix the login bug", budget="small"))

    assert hybrid_calls == [("fix the login bug", 4)]
    assert payload["seeds"]
    assert payload["seeds"][0]["qualified_name"] == "AuthService.refreshSession"
    assert payload["seeds"][0]["reason"] == "matched hybrid search"
