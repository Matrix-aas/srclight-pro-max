"""Tests for the tree-sitter indexer."""

import pytest

from srclight.db import Database
from srclight.indexer import (
    INDEXER_BUILD_ID,
    IndexConfig,
    Indexer,
    _imported_microservice_decorator_wrappers,
    _imported_name_map_from_module,
    _resolve_typescript_import_path,
    _tsconfig_alias_rules,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal sample project."""
    src = tmp_path / "project"
    src.mkdir()

    # Python file
    (src / "main.py").write_text('''\
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
''')

    # Another Python file
    (src / "utils.py").write_text('''\
import os

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path) as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
''')

    return src


@pytest.fixture
def c_project(tmp_path):
    """Create a minimal C project."""
    src = tmp_path / "cproject"
    src.mkdir()

    (src / "main.c").write_text('''\
#include <stdio.h>

/* Print a greeting message. */
void greet(const char* name) {
    printf("Hello, %s!\\n", name);
}

int main(int argc, char** argv) {
    greet("World");
    return 0;
}
''')

    (src / "utils.h").write_text('''\
#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int x;
    int y;
} Point;

int distance(Point a, Point b);

#endif
''')

    return src


def test_index_python(db, sample_project):
    """Indexes Python files and extracts symbols."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)
    stats = indexer.index(sample_project)

    assert stats.files_scanned == 2
    assert stats.files_indexed == 2
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check symbols were created
    db_stats = db.stats()
    assert db_stats["files"] == 2
    assert db_stats["symbols"] > 0
    assert "python" in db_stats["languages"]

    # Check specific symbols
    syms = db.symbols_in_file("main.py")
    names = [s.name for s in syms]
    assert "hello" in names
    assert "Calculator" in names

    syms = db.symbols_in_file("utils.py")
    names = [s.name for s in syms]
    assert "read_file" in names
    assert "write_file" in names


def test_index_c(db, c_project):
    """Indexes C files and extracts symbols."""
    config = IndexConfig(root=c_project)
    indexer = Indexer(db, config)
    stats = indexer.index(c_project)

    assert stats.files_indexed == 2
    assert stats.symbols_extracted > 0

    syms = db.symbols_in_file("main.c")
    names = [s.name for s in syms]
    assert "greet" in names
    assert "main" in names


def test_incremental_index(db, sample_project):
    """Incremental indexing skips unchanged files."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)

    # First index
    stats1 = indexer.index(sample_project)
    assert stats1.files_indexed == 2

    # Second index — nothing changed
    stats2 = indexer.index(sample_project)
    assert stats2.files_indexed == 0
    assert stats2.files_unchanged == 2

    # Modify a file
    (sample_project / "main.py").write_text("def new_function(): pass\n")

    # Third index — only modified file re-indexed
    stats3 = indexer.index(sample_project)
    assert stats3.files_indexed == 1
    assert stats3.files_unchanged == 1


def test_indexer_build_change_forces_full_reindex(db, sample_project, monkeypatch):
    """Extractor build changes should invalidate unchanged-file skipping."""
    from srclight import indexer as indexer_module

    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)

    stats1 = indexer.index(sample_project)
    assert stats1.files_indexed == 2

    monkeypatch.setattr(
        indexer_module,
        "INDEXER_BUILD_ID",
        indexer_module.INDEXER_BUILD_ID + "+test-bump",
    )

    stats2 = indexer.index(sample_project)
    assert stats2.files_indexed == 2
    assert stats2.files_unchanged == 0


def test_indexer_build_id_marks_wave2_async_extractor_version():
    """Vue metadata changes should produce a distinct build id suffix."""
    assert INDEXER_BUILD_ID.endswith("vue-normalized-metadata-v6")


def test_index_csharp_doc_comments_do_not_leak_to_methods(db, tmp_path):
    """Non-JS extraction should keep class comments off child methods."""
    project = tmp_path / "csharp-docs"
    project.mkdir()

    (project / "Example.cs").write_text(
        '''\
/// Class docs should stay on the class.
public class Example
{
    public int Method()
    {
        return 1;
    }
}
'''
    )

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    indexer.index(project)

    syms = {sym.name: sym for sym in db.symbols_in_file("Example.cs")}

    assert "Class docs should stay on the class." in (syms["Example"].doc_comment or "")
    assert syms["Method"].doc_comment is None


def test_index_typescript_class_method_does_not_inherit_class_doc(db, tmp_path):
    """JS/TS doc lookup should not leak class docs onto child methods."""
    project = tmp_path / "ts-class-docs"
    project.mkdir()

    (project / "example.ts").write_text(
        '''\
/** Class docs should stay on the class. */
class Example {
  method() {
    return 1;
  }
}
'''
    )

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    indexer.index(project)

    syms = {sym.name: sym for sym in db.symbols_in_file("example.ts")}

    assert "Class docs should stay on the class." in (syms["Example"].doc_comment or "")
    assert syms["method"].doc_comment is None


def test_index_typescript_exported_wrapper_doc_comments_are_preserved(db, tmp_path):
    """Exported JS/TS declarations should keep wrapper-level doc comments."""
    project = tmp_path / "ts-export-docs"
    project.mkdir()

    (project / "example.ts").write_text(
        '''\
/**
 * Wrapper-level export docs should remain attached.
 */
export function wrappedValue() {
  return 1;
}
'''
    )

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    indexer.index(project)

    syms = {sym.name: sym for sym in db.symbols_in_file("example.ts")}

    assert "Wrapper-level export docs should remain attached." in (
        syms["wrappedValue"].doc_comment or ""
    )


def test_index_typescript_non_latin_doc_comments_are_preserved(db, tmp_path):
    """Non-Latin JS/TS doc comments should survive indexing."""
    project = tmp_path / "ts-non-latin-docs"
    project.mkdir()

    (project / "example.ts").write_text(
        '''\
/**
 * Привет мир
 */
export function greet() {
  return 1;
}
'''
    )

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    indexer.index(project)

    syms = {sym.name: sym for sym in db.symbols_in_file("example.ts")}

    assert "Привет мир" in (syms["greet"].doc_comment or "")


def test_local_microservice_decorator_wrappers_reuse_file_scope_scan(monkeypatch):
    """Wrapper discovery should not reparse the same TS file source on repeated lookups."""
    from srclight import indexer as indexer_module

    source_text = """\
import { MessagePattern } from '@nestjs/microservices';

export function CachedRpcRequest(pattern: string): MethodDecorator {
  return MessagePattern(pattern);
}
"""
    nest_microservice_imports = _imported_name_map_from_module(
        source_text,
        "@nestjs/microservices",
    )
    call_count = 0
    original = indexer_module._typescript_function_declaration_nodes

    def counting_nodes(text):
        nonlocal call_count
        call_count += 1
        return original(text)

    monkeypatch.setattr(indexer_module, "_typescript_function_declaration_nodes", counting_nodes)

    first = indexer_module._local_microservice_decorator_wrappers(
        source_text,
        nest_microservice_imports,
    )
    second = indexer_module._local_microservice_decorator_wrappers(
        source_text,
        nest_microservice_imports,
    )

    assert first == second
    assert call_count == 1


def test_imported_microservice_decorator_wrappers_reuse_file_scope_scan(monkeypatch, tmp_path):
    """Imported wrapper discovery should not re-resolve the same file on repeated lookups."""
    from srclight import indexer as indexer_module

    root = tmp_path / "project"
    (root / "server/decorators").mkdir(parents=True)
    (root / "server/messaging").mkdir(parents=True)
    (root / "server/decorators/rpc-request.decorator.ts").write_text(
        """import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return MessagePattern(path);
}
"""
    )
    source_text = """import { RpcRequest } from '../decorators/rpc-request.decorator';

export class ImportedController {
  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {}
}
"""

    call_count = 0
    original = indexer_module._resolve_typescript_import_path

    def counting_resolve(root_path, source_file_path, module_specifier):
        nonlocal call_count
        call_count += 1
        return original(root_path, source_file_path, module_specifier)

    monkeypatch.setattr(indexer_module, "_resolve_typescript_import_path", counting_resolve)

    first = indexer_module._imported_microservice_decorator_wrappers(
        root,
        "server/messaging/imported.controller.ts",
        source_text,
    )
    second = indexer_module._imported_microservice_decorator_wrappers(
        root,
        "server/messaging/imported.controller.ts",
        source_text,
    )

    assert first == second
    assert call_count == 1


def test_imported_microservice_decorator_wrappers_only_resolve_used_decorator_imports(
    monkeypatch, tmp_path
):
    """Imported wrapper discovery should ignore unrelated imports instead of resolving all of them."""
    from srclight import indexer as indexer_module

    root = tmp_path / "project"
    root.mkdir()
    source_text = """import { RpcRequest } from '@/decorators/rpc-request.decorator';
import { createFoo } from '@/services/foo';
import { StorageKeys } from '@/config/storage';

export class ImportedController {
  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {}
}
"""

    resolved_specifiers: list[str] = []

    def fake_resolve(root_path, source_file_path, module_specifier):
        resolved_specifiers.append(module_specifier)
        if module_specifier == "@/decorators/rpc-request.decorator":
            return "server/decorators/rpc-request.decorator.ts"
        return None

    monkeypatch.setattr(indexer_module, "_resolve_typescript_import_path", fake_resolve)
    monkeypatch.setattr(
        indexer_module,
        "_read_project_text",
        lambda root_path_str, rel_path: """import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return MessagePattern(path);
}
""",
    )

    resolved = indexer_module._imported_microservice_decorator_wrappers(
        root,
        "server/messaging/imported.controller.ts",
        source_text,
    )

    assert resolved_specifiers == ["@/decorators/rpc-request.decorator"]
    assert resolved["RpcRequest"]["canonical_decorator"] == "MessagePattern"


def test_search_after_index(db, sample_project):
    """Search works after indexing."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)
    indexer.index(sample_project)

    # Search by function name
    results = db.search_symbols("hello")
    assert len(results) > 0
    assert any(r["name"] == "hello" for r in results)

    # Search by class name
    results = db.search_symbols("Calculator")
    assert len(results) > 0

    # Search by doc content
    results = db.search_symbols("greet someone")
    assert len(results) > 0


def test_search_prefers_vue_component_over_docs_for_code_like_query(db, vue_search_project):
    """Code-like Vue/Nuxt queries should rank component anchors above docs noise."""
    config = IndexConfig(root=vue_search_project)
    indexer = Indexer(db, config)
    indexer.index(vue_search_project)

    results = db.search_symbols("define model i18n locale path css module template ref query mutation")

    assert results
    assert results[0]["name"] == "ObservedNuxtUses"
    assert results[0]["kind"] == "component"


@pytest.fixture
def markdown_project(tmp_path):
    """Create a minimal Markdown project."""
    src = tmp_path / "mdproject"
    src.mkdir()

    (src / "notes.md").write_text('''\
---
title: Architecture Notes
tags: [design, architecture]
---

# Architecture

Overall system design.

## Components

The main components are listed here.

### Database Layer

SQLite with FTS5 indexes.

## Deployment

Run on any Linux server.
''')

    (src / "plain.md").write_text('''\
Just a file with no headings.

Some plain text content.
''')

    (src / "single-heading.md").write_text('''\
# Quick Note

A brief note with only one heading.
''')

    return src


def test_index_markdown(db, markdown_project):
    """Indexes Markdown files and extracts heading sections as symbols."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    stats = indexer.index(markdown_project)

    assert stats.files_scanned == 3
    assert stats.files_indexed == 3
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check notes.md — should have 4 section symbols
    syms = db.symbols_in_file("notes.md")
    names = [s.name for s in syms]
    assert "Architecture" in names
    assert "Components" in names
    assert "Database Layer" in names
    assert "Deployment" in names

    # Check kinds are all "section"
    assert all(s.kind == "section" for s in syms)

    # Check qualified names use ">" ancestry
    arch = [s for s in syms if s.name == "Architecture"][0]
    assert arch.qualified_name == "notes > Architecture"
    db_layer = [s for s in syms if s.name == "Database Layer"][0]
    assert db_layer.qualified_name == "notes > Architecture > Components > Database Layer"

    # Check own-content: "Components" section shouldn't include "Database Layer" content
    components = [s for s in syms if s.name == "Components"][0]
    assert "The main components" in components.content
    assert "SQLite" not in components.content


def test_index_markdown_no_headings(db, markdown_project):
    """Markdown file without headings produces a single document symbol."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    indexer.index(markdown_project)

    syms = db.symbols_in_file("plain.md")
    assert len(syms) == 1
    assert syms[0].kind == "document"
    assert syms[0].name == "plain"
    assert "no headings" in syms[0].content


def test_index_markdown_frontmatter(db, markdown_project):
    """YAML frontmatter is extracted as doc_comment on the first symbol."""
    config = IndexConfig(root=markdown_project)
    indexer = Indexer(db, config)
    indexer.index(markdown_project)

    syms = db.symbols_in_file("notes.md")
    # First symbol should have frontmatter in doc_comment
    first = sorted(syms, key=lambda s: s.start_line)[0]
    assert first.doc_comment is not None
    assert "title: Architecture Notes" in first.doc_comment
    assert "tags:" in first.doc_comment


@pytest.fixture
def vue_project(tmp_path):
    """Create a minimal Vue SFC project."""
    src = tmp_path / "vueproject"
    src.mkdir()

    (src / "SetupTs.vue").write_text('''\
<template>
  <div>{{ msg }}</div>
</template>

<script setup lang="ts">
export interface Props {
  msg: string;
}

function greet(name: string) {
  return name;
}
</script>
''')

    (src / "PlainJs.vue").write_text('''\
<template>
  <button type="button">Click</button>
</template>

<script>
function clickHandler() {
  return true;
}
</script>
''')

    (src / "Priority.vue").write_text('''\
<template><div /></template>

<script>
function lowPriority() {
  return false;
}
</script>

<script type="application/ld+json">
{"@context":"https://schema.org","name":"Ignored"}
</script>

<script setup lang="ts">
export function highPriority() {
  return true;
}
</script>
''')

    (src / "PriorityTypeScriptAlias.vue").write_text('''\
<template><div /></template>

<script>
function jsFallback() {
  return false;
}
</script>

<script lang="typescript">
export interface AliasWins {
  msg: string;
}
</script>
''')

    (src / "EmptySetupFallback.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
</script>

<script>
function fallbackFn() {
  return 7;
}
</script>
''')

    (src / "CommentOnlySetupFallback.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
// comments only should not suppress fallback
</script>

<script>
function recoveredFallback() {
  return 9;
}
</script>
''')

    (src / "NoScript.vue").write_text('''\
<template>
  <div>No script</div>
</template>

<style scoped>
div {
  color: red;
}
</style>
''')

    (src / "JsonOnly.vue").write_text('''\
<template><div /></template>

<script type="application/ld+json">
{"@context":"https://schema.org","name":"Ignored"}
</script>
''')

    (src / "TemplateTypeOnly.vue").write_text('''\
<template><div /></template>

<script type="text/template">
function shouldStayIgnored() {
  return false;
}
</script>
''')

    (src / "JsonLangOnly.vue").write_text('''\
<template><div /></template>

<script lang="json">
function shouldAlsoStayIgnored() {
  return true;
}
</script>
''')

    (src / "Tsx.vue").write_text('''\
<template><div /></template>

<script lang="tsx">
export function renderBox() {
  return <div className="box">tsx</div>;
}
</script>
''')

    (src / "TemplateStyleSignals.vue").write_text('''\
<template>
  <NuxtLink class="nav-link hero" @click="track" v-if="ready" :to="target">
    <BaseCard v-for="item in items" :class="$style.card">
      <slot name="header" />
    </BaseCard>
  </NuxtLink>
</template>

<style lang="postcss" module>
.card {
  color: var(--accent-color);
}

.hero {
  @apply px-4 py-2;
}

:global(.page-shell) {
  display: block;
}
</style>
''')

    (src / "CommentedSignals.vue").write_text('''\
<template>
  <!-- <NuxtLink class="ghost" v-if="nope"><BaseCard /></NuxtLink> -->
  <div />
</template>

<style lang="postcss" module>
/* .card { color: var(--ignored-color); } */
</style>
''')

    (src / "ClassOnlyTemplate.vue").write_text('''\
<template>
  <div class="foo" />
</template>
''')

    (src / "ScriptHints.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth',
  layout: 'dashboard',
})

const props = defineProps<{ id: string }>()
const emit = defineEmits<{ refresh: [] }>()
const route = useRoute()
const { data } = await useFetch('/api/items')
useHead({ title: 'Inventory' })
</script>
''')

    (src / "GraphqlHints.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const authStore = useAuthStore()

const catalogQuery = gql`
  # query FakeCatalog
  query GetCatalog {
    catalog {
      id
    }
  }
`

const saveCartMutation = gql`
  # mutation FakeSaveCart
  mutation SaveCart {
    saveCart {
      id
    }
  }
`

async function checkout() {
  await navigateTo('/checkout')
}
</script>
''')

    (src / "ObservedNuxtUses.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
defineOptions({ name: 'ObservedNuxtUses' })
const model = defineModel<string>()
const { t } = useI18n()
const localePath = useLocalePath()
const styles = useCssModule()
const button = useTemplateRef('button')
const client = useNuxtApp()
const result = useQuery()
const mutation = useMutation()
const feed = useSubscription()
</script>
''')

    (src / "FalsePositiveStrings.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const label = "useRoute definePageMeta useAuthStore useQuery"
const details = `navigateTo('/fake') gql query Fake`
const count = 1 // useFetch('/ignored')
</script>
''')

    (src / "ProtocolPaths.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const { data } = await useFetch('https://api.example.com/items')
await navigateTo('https://example.com/checkout')
</script>
''')

    (src / "GeneratedHookNames.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const me = useCurrentUserQuery()
const update = useUpdateMeMutation()
</script>
''')

    (src / "NormalizedSignals.vue").write_text('''\
<template>
  <BaseCard :class="$style.card">
    <template #header>
      {{ msg }}
    </template>
  </BaseCard>
</template>

<script setup lang="ts">
const props = defineProps<{ msg: string }>()
const emit = defineEmits<{ save: [] }>()
const route = useRoute()
const authStore = useAuthStore()

const catalogQuery = gql`
  query GetCatalog {
    catalog {
      id
    }
  }
`

const saveCartMutation = gql`
  mutation SaveCart {
    saveCart {
      id
    }
  }
`

await navigateTo('/checkout')
</script>

<style lang="postcss" module scoped>
.card {
  color: var(--accent-color);
}
</style>
''')

    (src / "NestedMacroSignals.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const props = defineProps<{
  id: string
  nested: {
    child: string
    details: {
      label: string
    }
  }
}>()

const emit = defineEmits(['save', 'cancel'])

const text = "defineProps<{ fake: true }>() defineEmits(['oops'])"
</script>
''')

    (src / "StyleOnlyCssModule.vue").write_text('''\
<template>
  <div class="hero" />
</template>

<style lang="postcss" module>
.card {
  color: var(--accent-color);
}

.badge {
  padding: 4px;
}
</style>
''')

    (src / "MacroEdgeCases.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const props = defineProps<{
  cb: (x: Foo<Bar>) => Baz
  'data-id': string
  other: string
}>()

const emit = defineEmits({
  save: null,
  cancel: payload => true,
})

const text = "defineProps<{ fake: true }>() defineEmits({ nope: null })"
</script>
''')

    (src / "RuntimeDefineProps.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
const props = defineProps({
  'data-id': String,
  other: Number,
})
</script>
''')

    (src / "DocCommentCleanup.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
/**
 * Real Vue docs should remain attached.
 */
export function meaningfulDoc() {
  return true;
}

// =====
export function separatorDoc() {
  return false;
}

/**
 * Mixed Vue docs should survive even when a separator sits closest.
 */
// -------
export function mixedNoiseDoc() {
  return true;
}

// FIXME: temporary Vue helper
export function todoDoc() {
  return true;
}
</script>
''')

    return src


def test_index_vue_dual_script_blocks_keep_both_symbol_surfaces(db, tmp_path):
    project = tmp_path / "dual-vue"
    project.mkdir()

    (project / "DualScript.vue").write_text('''\
<template>
  <div />
</template>

<script>
export function legacyHelper() {
  return 1;
}
</script>

<script setup lang="ts">
export function setupHelper() {
  return 2;
}
</script>
''')

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    stats = indexer.index(project)

    assert stats.files_indexed == 1

    syms = db.symbols_in_file("DualScript.vue")
    names = {sym.name: sym for sym in syms}

    assert "legacyHelper" in names
    assert "setupHelper" in names
    assert names["legacyHelper"].start_line == 6
    assert names["setupHelper"].start_line == 12


def test_index_vue_script_setup_ts_offsets_lines(db, vue_project):
    """Indexes Vue script setup TS content with original line numbers."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    stats = indexer.index(vue_project)

    assert stats.files_scanned == 26
    assert stats.files_indexed == 26
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    syms = db.symbols_in_file("SetupTs.vue")
    names = {s.name: s for s in syms}

    assert "Props" in names
    assert names["Props"].kind == "interface"
    assert names["Props"].start_line == 6

    assert "greet" in names
    assert names["greet"].kind == "function"
    assert names["greet"].start_line == 10


def test_index_vue_script_doc_comments_filter_noise(db, vue_project):
    """Vue script docs should keep meaningful comments and drop separator noise."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = {sym.name: sym for sym in db.symbols_in_file("DocCommentCleanup.vue")}

    assert "Real Vue docs should remain attached." in (syms["meaningfulDoc"].doc_comment or "")
    assert syms["separatorDoc"].doc_comment is None
    assert "Mixed Vue docs should survive" in (syms["mixedNoiseDoc"].doc_comment or "")
    assert "-------" not in (syms["mixedNoiseDoc"].doc_comment or "")
    assert syms["todoDoc"].doc_comment is None


def test_index_vue_plain_script_js(db, vue_project):
    """Indexes Vue plain script blocks as JavaScript."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("PlainJs.vue")
    names = [s.name for s in syms]

    assert "clickHandler" in names
    assert any(s.kind == "function" and s.start_line == 6 for s in syms)


def test_index_vue_script_priority_keeps_both_usable_blocks(db, vue_project):
    """Indexes both usable Vue script blocks when they each contain symbols."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("Priority.vue")
    names = [s.name for s in syms]

    assert "highPriority" in names
    assert "lowPriority" in names


def test_index_vue_script_priority_keeps_typescript_alias_and_js_blocks(db, vue_project):
    """Indexes both usable scripts when one uses the typescript lang alias."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("PriorityTypeScriptAlias.vue")
    names = [s.name for s in syms]

    assert "AliasWins" in names
    assert "jsFallback" in names


def test_index_vue_empty_setup_falls_back_to_usable_script(db, vue_project):
    """An empty setup block should not suppress a lower-priority usable script."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("EmptySetupFallback.vue")
    names = [s.name for s in syms]

    assert "fallbackFn" in names


def test_index_vue_symbol_less_setup_falls_back_to_usable_script(db, vue_project):
    """A symbol-less setup block should not suppress a lower-priority usable script."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("CommentOnlySetupFallback.vue")
    names = [s.name for s in syms]

    assert "recoveredFallback" in names


def test_index_vue_without_usable_script_yields_no_symbols(db, vue_project):
    """Vue files without a usable script block do not produce symbols."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("NoScript.vue")
    assert syms == []


def test_index_vue_ignores_non_js_script_types(db, vue_project):
    """Vue script tags with non-JS MIME types are ignored."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("JsonOnly.vue")
    assert syms == []

    syms = db.symbols_in_file("TemplateTypeOnly.vue")
    assert syms == []

    syms = db.symbols_in_file("JsonLangOnly.vue")
    assert syms == []


def test_index_vue_treats_tsx_lang_as_tsx_parser(db, vue_project):
    """Vue TSX blocks should be parsed with the TSX grammar, not plain TS."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("Tsx.vue")
    names = {s.name: s for s in syms}

    assert "renderBox" in names
    assert names["renderBox"].signature == "function renderBox()"


def test_index_vue_template_and_postcss_module_emit_component_signal(db, vue_project):
    """Signal-rich template/style blocks should emit a Vue component anchor symbol."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("TemplateStyleSignals.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.name == "TemplateStyleSignals"
    assert "NuxtLink" in (component.signature or "")
    assert "BaseCard" in (component.signature or "")
    assert "postcss" in (component.signature or "")
    assert "module" in (component.signature or "")
    assert ".card" in (component.signature or "")
    assert "--accent-color" in (component.signature or "")
    assert "v-if" in (component.doc_comment or "")
    assert "click" in (component.doc_comment or "")
    assert component.metadata is not None
    assert component.metadata["template"]["components"] == ["BaseCard", "NuxtLink", "slot"]
    assert component.metadata["style"]["classes"] == ["card", "hero", "page-shell"]
    assert component.metadata["style"]["vars"] == ["--accent-color"]
    assert component.metadata["style"]["langs"] == ["postcss"]
    assert component.metadata["style"]["flags"] == ["module"]


def test_search_finds_vue_template_and_postcss_component_signal(db, vue_project):
    """Vue template/style summaries should be searchable after indexing."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    results = db.search_symbols("base card accent color postcss")

    assert results
    assert results[0]["name"] == "TemplateStyleSignals"
    assert results[0]["kind"] == "component"


def test_index_vue_ignores_commented_template_and_style_signals(db, vue_project):
    """Commented-out Vue template/style content should not emit component anchors."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("CommentedSignals.vue")

    assert all(s.kind != "component" for s in syms)


def test_index_vue_ignores_class_only_templates_for_component_anchor(db, vue_project):
    """A plain static class alone is not enough to count as a signal-rich component."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("ClassOnlyTemplate.vue")

    assert all(s.kind != "component" for s in syms)


def test_index_vue_script_hints_emit_component_anchor(db, vue_project):
    """Nuxt/Vue frontend script hints should surface on the component anchor."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("ScriptHints.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.name == "ScriptHints"
    assert "definePageMeta" in (component.signature or "")
    assert "useFetch" in (component.signature or "")
    assert "auth" in (component.doc_comment or "")
    assert "/api/items" in (component.doc_comment or "")
    assert component.metadata is not None
    assert component.metadata["script"]["macros"] == [
        "defineEmits", "definePageMeta", "defineProps",
    ]
    assert component.metadata["script"]["composables"] == [
        "useFetch", "useHead", "useRoute",
    ]
    assert component.metadata["script"]["fetch_paths"] == ["/api/items"]
    assert component.metadata["script"]["page_meta_keys"] == ["layout", "middleware"]
    assert component.metadata["script"]["page_meta_values"] == ["auth", "dashboard"]


def test_search_finds_vue_script_hint_component_anchor(db, vue_project):
    """Vue script hint summaries should be searchable after indexing."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    results = db.search_symbols("page meta use fetch auth dashboard")

    assert results
    assert results[0]["name"] == "ScriptHints"
    assert results[0]["kind"] == "component"


def test_index_vue_graphql_and_store_hints_emit_component_anchor(db, vue_project):
    """GraphQL/store/navigation hints should surface on the component anchor."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("GraphqlHints.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.name == "GraphqlHints"
    assert "useAuthStore" in (component.doc_comment or "")
    assert "GetCatalog" in (component.doc_comment or "")
    assert "SaveCart" in (component.doc_comment or "")
    assert "/checkout" in (component.doc_comment or "")
    assert component.metadata is not None
    assert component.metadata["script"]["stores"] == ["useAuthStore"]
    assert component.metadata["script"]["graphql_ops"] == [
        "mutation SaveCart", "query GetCatalog",
    ]
    assert component.metadata["script"]["navigate_paths"] == ["/checkout"]


def test_search_finds_vue_graphql_store_component_anchor(db, vue_project):
    """GraphQL/store/navigation summaries should be searchable after indexing."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    results = db.search_symbols("get catalog save cart checkout auth store")

    assert results
    assert results[0]["name"] == "GraphqlHints"
    assert results[0]["kind"] == "component"


def test_index_vue_observed_nuxt_uses_emit_component_anchor(db, vue_project):
    """Frequently used real-world Nuxt/Vue hooks should surface on the component anchor."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("ObservedNuxtUses.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.name == "ObservedNuxtUses"
    assert "defineOptions" in (component.doc_comment or "")
    assert "defineModel" in (component.doc_comment or "")
    assert "useI18n" in (component.doc_comment or "")
    assert "useLocalePath" in (component.doc_comment or "")
    assert "useCssModule" in (component.doc_comment or "")
    assert "useTemplateRef" in (component.doc_comment or "")
    assert "useNuxtApp" in (component.doc_comment or "")
    assert component.metadata is not None
    assert component.metadata["script"]["macros"] == [
        "defineModel", "defineOptions",
    ]
    assert component.metadata["script"]["composables"] == [
        "useCssModule", "useI18n", "useLocalePath", "useMutation",
        "useNuxtApp", "useQuery", "useSubscription", "useTemplateRef",
    ]
    assert component.metadata["script"]["graphql_hooks"] == [
        "useMutation", "useQuery", "useSubscription",
    ]


def test_search_finds_vue_observed_nuxt_use_component_anchor(db, vue_project):
    """Observed real-world Nuxt/Vue hook summaries should be searchable."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    results = db.search_symbols("define model i18n locale path css module template ref query mutation")

    assert results
    assert results[0]["name"] == "ObservedNuxtUses"
    assert results[0]["kind"] == "component"


def test_index_vue_normalized_metadata_and_file_summary_are_persisted(db, vue_project):
    """Vue component anchors should expose normalized metadata and file summaries."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("NormalizedSignals.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["framework"] == "vue"
    assert component.metadata["resource"] == "component"
    assert component.metadata["props"] == ["msg"]
    assert component.metadata["emits"] == ["save"]
    assert component.metadata["slots"] == ["header"]
    assert component.metadata["composables_used"] == ["useRoute"]
    assert component.metadata["stores_used"] == ["useAuthStore"]
    assert component.metadata["graphql_ops_used"] == ["mutation SaveCart", "query GetCatalog"]
    assert component.metadata["routes_used"] == ["/checkout"]
    assert component.metadata["css_modules"] == ["card"]
    assert component.metadata["scoped_styles"] == ["scoped"]
    assert component.metadata["template"]["components"] == ["BaseCard"]
    assert component.metadata["style"]["flags"] == ["module", "scoped"]
    assert component.signature is not None
    assert component.signature.startswith("component NormalizedSignals")
    assert "props: msg" in component.signature
    assert "emits: save" in component.signature
    assert "slots: header" in component.signature
    assert "--accent-color" not in component.signature

    file_summary = db.get_file_summary("NormalizedSignals.vue")
    assert file_summary is not None
    assert file_summary["summary"]
    assert file_summary["metadata"] is not None
    assert file_summary["metadata"]["framework"] == "vue"
    assert file_summary["metadata"]["resource"] == "component"
    assert file_summary["metadata"]["props"] == ["msg"]
    assert file_summary["metadata"]["css_modules"] == ["card"]

    search_results = db.search_symbols("NormalizedSignals")
    assert search_results
    assert search_results[0]["signature"].startswith("component NormalizedSignals")
    assert "props: msg" in search_results[0]["signature"]
    assert "emits: save" in search_results[0]["signature"]
    assert "--accent-color" not in (search_results[0]["signature"] or "")


def test_index_vue_nested_macro_keys_ignore_string_noise(db, vue_project):
    """Vue macro extraction should only keep real top-level prop and emit keys."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("NestedMacroSignals.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["props"] == ["id", "nested"]
    assert component.metadata["emits"] == ["cancel", "save"]


def test_index_vue_style_only_css_module_uses_stylesheet_classes(db, vue_project):
    """Vue css_modules should come from module-aware stylesheet classes, not plain template classes."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("StyleOnlyCssModule.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["css_modules"] == ["badge", "card"]


def test_index_vue_macro_edge_cases_preserve_nested_generics_and_object_emits(db, vue_project):
    """Vue macro extraction should survive nested generics, quoted prop keys, and object emits."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("MacroEdgeCases.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["props"] == ["cb", "data-id", "other"]
    assert component.metadata["emits"] == ["cancel", "save"]


def test_index_vue_runtime_define_props_object_form_supports_quoted_keys(db, vue_project):
    """Vue runtime defineProps object form should recover quoted and plain keys."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("RuntimeDefineProps.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["props"] == ["data-id", "other"]


def test_index_vue_reindex_clears_stale_file_summary_when_signals_disappear(db, tmp_path):
    """Vue file summaries should be removed once a file no longer has Vue signals."""
    project = tmp_path / "vue-stale-summary"
    project.mkdir()
    file_path = project / "SignalCleanup.vue"
    file_path.write_text('''\
<template>
  <BaseCard :class="$style.card">
    <template #header>
      {{ msg }}
    </template>
  </BaseCard>
</template>

<script setup lang="ts">
const props = defineProps<{ msg: string }>()
const emit = defineEmits<{ save: [] }>()
</script>

<style lang="postcss" module scoped>
.card {
  color: var(--accent-color);
}
</style>
''')

    config = IndexConfig(root=project)
    indexer = Indexer(db, config)
    indexer.index(project)

    initial_summary = db.get_file_summary("SignalCleanup.vue")
    assert initial_summary is not None
    assert initial_summary["summary"] is not None
    assert initial_summary["metadata"]["framework"] == "vue"

    file_path.write_text('''\
<template>
  <div />
</template>
''')

    indexer.index(project)

    cleared_summary = db.get_file_summary("SignalCleanup.vue")
    assert cleared_summary is not None
    assert cleared_summary["summary"] == "Indexed vue file. No top-level symbols indexed."
    assert cleared_summary["metadata"] in (None, {})
    assert all(sym.kind != "component" for sym in db.symbols_in_file("SignalCleanup.vue"))


def test_index_vue_ignores_string_and_inline_comment_false_positives(db, vue_project):
    """String literals and inline comments should not create frontend script hints."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("FalsePositiveStrings.vue")

    assert all(s.kind != "component" for s in syms)


def test_index_vue_preserves_protocol_paths_in_fetch_and_navigation(db, vue_project):
    """Comment stripping should not truncate real protocol-based fetch/navigation paths."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("ProtocolPaths.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata is not None
    assert component.metadata["script"]["fetch_paths"] == ["https://api.example.com/items"]
    assert component.metadata["script"]["navigate_paths"] == ["https://example.com/checkout"]


def test_index_vue_generated_query_hooks_stay_searchable_without_graphql_bucket(db, vue_project):
    """Generated *Query/*Mutation hooks should stay searchable without forced GraphQL labeling."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    syms = db.symbols_in_file("GeneratedHookNames.vue")
    component = next(s for s in syms if s.kind == "component")

    assert component.metadata["script"]["composables"] == [
        "useCurrentUserQuery", "useUpdateMeMutation",
    ]
    assert component.metadata["script"]["graphql_hooks"] == []


def test_search_finds_vue_generated_hooks_component_anchor(db, vue_project):
    """Generated hook names should still be searchable from the component summary."""
    config = IndexConfig(root=vue_project)
    indexer = Indexer(db, config)
    indexer.index(vue_project)

    results = db.search_symbols("current user update me mutation")

    assert results
    assert results[0]["name"] == "GeneratedHookNames"
    assert results[0]["kind"] == "component"


def test_file_removal_detection(db, sample_project):
    """Detects and removes deleted files from index."""
    config = IndexConfig(root=sample_project)
    indexer = Indexer(db, config)

    # Index everything
    indexer.index(sample_project)
    assert db.stats()["files"] == 2

    # Delete a file
    (sample_project / "utils.py").unlink()

    # Re-index
    stats = indexer.index(sample_project)
    assert stats.files_removed == 1
    assert db.stats()["files"] == 1


@pytest.fixture
def dart_project(tmp_path):
    """Create a minimal Dart project."""
    src = tmp_path / "dartproject"
    src.mkdir()

    # Main Dart file with various constructs
    (src / "main.dart").write_text('''\
// A sample Dart file for testing.

int add(int a, int b) {
  return a + b;
}

class UserService {
  final String _name;

  UserService(this._name);

  String get name => _name;

  /// Fetches a user by ID.
  Future<User?> fetchUser(int id) async {
    return null;
  }
}

class User {
  final int id;
  final String email;

  const User({
    required this.id,
    required this.email,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] as int,
      email: json['email'] as String,
    );
  }
}

enum UserStatus {
  active,
  inactive,
}

mixin Logger {
  void log(String message) {
    print('[LOG] $message');
  }
}

class DataManager with Logger {
  Future<void> load() async {
    log('Loading data...');
  }
}

extension StringExtensions on String {
  String get capitalized {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }
}
''')

    return src


@pytest.fixture
def vue_search_project(tmp_path):
    """Create a project to test code-vs-doc ranking for Vue frontend queries."""
    src = tmp_path / "vue-search-project"
    src.mkdir()
    docs = src / "docs"
    docs.mkdir()
    (src / "ObservedNuxtUses.vue").write_text('''\
<template><div /></template>

<script setup lang="ts">
defineOptions({ name: 'ObservedNuxtUses' })
const model = defineModel<string>()
const { t } = useI18n()
const localePath = useLocalePath()
const styles = useCssModule()
const button = useTemplateRef('button')
const result = useQuery()
const mutation = useMutation()
</script>
''')
    (src / "CLAUDE.md").write_text('''\
# Vue Frontend Notes

## Authentication
define model i18n locale path css module template ref query mutation

## Search
define model i18n locale path css module template ref query mutation
''')
    (docs / "spec.md").write_text('''\
# Spec

## Implementation sketch
define model i18n locale path css module template ref query mutation

## Acceptance criteria
define model i18n locale path css module template ref query mutation
''')
    return src


def test_index_dart(db, dart_project):
    """Indexes Dart files and extracts symbols."""
    config = IndexConfig(root=dart_project)
    indexer = Indexer(db, config)
    stats = indexer.index(dart_project)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    # Check symbols were created
    db_stats = db.stats()
    assert db_stats["files"] == 1
    assert db_stats["symbols"] > 0
    assert "dart" in db_stats["languages"]

    # Check specific symbols
    syms = db.symbols_in_file("main.dart")
    names = [s.name for s in syms]

    # Top-level function
    assert "add" in names

    # Classes
    assert "UserService" in names
    assert "User" in names

    # Method in class - there may be duplicates due to how Dart AST works
    # Check that we have at least one fetchUser with kind=method
    fetch_methods = [s for s in syms if s.name == "fetchUser" and s.kind == "method"]
    assert len(fetch_methods) >= 1

    # Enum
    assert "UserStatus" in names

    # Mixin
    assert "Logger" in names
    # Verify mixin kind
    logger_syms = [s for s in syms if s.name == "Logger"]
    assert len(logger_syms) >= 1
    assert logger_syms[0].kind == "mixin"

    # Extension
    assert "StringExtensions" in names
    ext_syms = [s for s in syms if s.name == "StringExtensions"]
    assert len(ext_syms) >= 1
    assert ext_syms[0].kind == "extension"


@pytest.fixture
def php_project(tmp_path):
    """Create a minimal PHP project."""
    src = tmp_path / "phpproject"
    src.mkdir()

    (src / "app.php").write_text('''\
<?php
function greet($name) {
    echo "Hello, $name!";
}

class UserController {
    public function index(): void {
        echo "list users";
    }

    private function validate($input): bool {
        return true;
    }
}

interface Cacheable {
    public function cache(): void;
}

trait Loggable {
    public function log($msg): void {}
}

enum Status {
    case Active;
    case Inactive;
}
?>
''')
    return src


def test_index_php(db, php_project):
    """Indexes PHP files and extracts symbols."""
    config = IndexConfig(root=php_project)
    indexer = Indexer(db, config)
    stats = indexer.index(php_project)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.symbols_extracted > 0
    assert stats.errors == 0

    db_stats = db.stats()
    assert db_stats["files"] == 1
    assert db_stats["symbols"] > 0
    assert "php" in db_stats["languages"]

    syms = db.symbols_in_file("app.php")
    names = [s.name for s in syms]

    # Top-level function
    assert "greet" in names

    # Class
    assert "UserController" in names

    # Methods
    assert "index" in names
    assert "validate" in names

    # Interface
    assert "Cacheable" in names

    # Trait
    assert "Loggable" in names

    # Enum
    assert "Status" in names


@pytest.fixture
def typescript_backend_project(tmp_path):
    """Create a TS backend project with Elysia, Drizzle, and Nest patterns."""
    src = tmp_path / "backend"
    (src / "server/src/routes").mkdir(parents=True)
    (src / "server/src/db").mkdir(parents=True)
    (src / "server/src/transports").mkdir(parents=True)
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/config").mkdir(parents=True)

    (src / "server/src/routes/auth.ts").write_text(
        """\
import { Elysia } from 'elysia';

export async function issueAnonymous() {
  return { ok: true };
}

export async function issueRefresh() {
  return { ok: true };
}

export const authRouteNames = ['/anonymous', '/refresh'];

export const authRoutes = new Elysia({ prefix: '/api/auth' })
  .post('/anonymous', issueAnonymous)
  .post('/refresh', issueRefresh);
"""
    )

    (src / "server/src/routes/health.ts").write_text(
        """\
import { Elysia } from 'elysia';

export function healthHandler() {
  return { ok: true };
}

export const healthRoutes = new Elysia().get('/api/health', healthHandler);
"""
    )

    (src / "server/src/routes/doc-comments.ts").write_text(
        """\
/**
 * Real docs should remain attached.
 */
export function meaningfulDoc() {
  return true;
}

// -----
export function separatorDoc() {
  return false;
}

/**
 * Mixed backend docs should survive even when a separator sits closest.
 */
// ======
export function mixedNoiseDoc() {
  return true;
}

// TODO: remove this placeholder after cleanup
export function todoDoc() {
  return true;
}
"""
    )

    (src / "server/src/middleware.ts").write_text(
        """\
import { Elysia } from 'elysia';

export const authMiddleware = new Elysia({ name: 'auth' })
  .derive(() => ({ userId: '1' }))
  .onBeforeHandle(() => {});
"""
    )

    (src / "server/src/db/schema.ts").write_text(
        """\
import { integer, pgTable, text } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: integer('id').primaryKey(),
  username: text('username').notNull(),
});
"""
    )

    (src / "server/src/db/index.ts").write_text(
        """\
import { drizzle } from 'drizzle-orm/postgres-js';
import * as schema from './schema.js';

declare const client: unknown;

export const db = drizzle(client, { schema });
"""
    )

    (src / "server/src/transports/unknown.ts").write_text(
        """\
declare const client: unknown;

export const unknownTransport = createUnknownTransport(client);
"""
    )

    (src / "server/src/modules/auth/auth.controller.ts").write_text(
        """\
import { Controller, Get, Post } from '@nestjs/common';

@Controller('auth')
export class AuthController {
  @Get('me')
  getMe() {
    return { ok: true };
  }

  @Post('refresh')
  refresh() {
    return { ok: true };
  }
}
"""
    )

    (src / "server/src/modules/auth/auth.infra.ts").write_text(
        """\
import {
  ArgumentsHost,
  CallHandler,
  Catch,
  ExceptionFilter,
  ExecutionContext,
  Injectable,
  NestInterceptor,
  NestMiddleware,
  PipeTransform,
  CanActivate,
} from '@nestjs/common';

@Injectable()
export class AuthService {
  validate() {
    return true;
  }
}

@Injectable()
export class JwtAuthGuard implements CanActivate {
  canActivate(context: ExecutionContext) {
    return true;
  }
}

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    return { exception, host };
  }
}

@Injectable()
export class RequestContextPipe implements PipeTransform<string, string> {
  transform(value: string) {
    return value.trim();
  }
}

@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler) {
    return next.handle();
  }
}

export class AuthMiddleware implements NestMiddleware {
  use(req: unknown, res: unknown, next: () => void) {
    next();
  }
}
"""
    )

    (src / "server/src/modules/auth/auth.module.ts").write_text(
        """\
import { MiddlewareConsumer, Module, NestModule } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthController } from './auth.controller';
import {
  AllExceptionsFilter,
  AuthMiddleware,
  AuthService,
  JwtAuthGuard,
  LoggingInterceptor,
  RequestContextPipe,
} from './auth.infra';
import { ViewerResolver } from './viewer.resolver';

@Module({
  imports: [
    ConfigModule.forRoot(),
    TypeOrmModule.forFeature([User]),
    forwardRef(() => AuthModule),
  ],
  controllers: [AuthController],
  providers: [
    AuthService,
    JwtAuthGuard,
    AllExceptionsFilter,
    RequestContextPipe,
    {
      provide: 'AUTH_OPTIONS',
      useFactory: (config: ConfigService) => ({
        secret: config.get('AUTH_SECRET'),
      }),
      inject: [ConfigService],
    },
    LoggingInterceptor,
    ViewerResolver,
  ],
  exports: [AuthService],
})
export class AuthModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(AuthMiddleware).forRoutes(AuthController);
  }
}
"""
    )

    (src / "server/src/modules/auth/viewer.resolver.ts").write_text(
        """\
import { Mutation, Query, Resolver, Subscription } from '@nestjs/graphql';

@Resolver('Viewer')
export class ViewerResolver {
  @Query(() => String, { name: 'viewer' })
  viewer() {
    return 'viewer';
  }

  @Mutation(() => Boolean)
  refreshViewer() {
    return true;
  }

  @Subscription(() => String, { name: 'viewerUpdated' })
  viewerUpdated() {
    return 'viewer-updated';
  }
}
"""
    )

    (src / "server/src/config/auth.config.ts").write_text(
        """\
import { registerAs } from '@nestjs/config';

export const AuthConfig = registerAs('auth', () => ({
  jwtSecret: 'secret',
}));

export const authConfig = registerAs('auth.secondary', () => ({
  jwtAudience: 'users',
}));
"""
    )

    (src / "server/src/main.ts").write_text(
        """\
import { NestFactory } from '@nestjs/core';
import { AuthModule } from './modules/auth/auth.module';

export async function bootstrap() {
  const app = await NestFactory.create(AuthModule);
  await app.listen(3000);
}
"""
    )

    (src / "server/src/modules/auth/local-query.ts").write_text(
        """\
function Query() {
  return () => {};
}

export class LocalQueryExample {
  @Query()
  localQuery() {
    return 'local';
  }
}
"""
    )

    (src / "server/src/modules/auth/local-resolver-query.ts").write_text(
        """\
function Resolver() {
  return () => {};
}

function Query() {
  return () => {};
}

@Resolver()
export class LocalResolverExample {
  @Query()
  localQuery() {
    return 'local';
  }
}
"""
    )

    (src / "server/src/modules/auth/aliased.resolver.ts").write_text(
        """\
import {
  Query as GqlQuery,
  Resolver as GqlResolver,
} from '@nestjs/graphql';

@GqlResolver('Viewer')
export class AliasedViewerResolver {
  @GqlQuery(() => String, { name: 'viewerAlias' })
  viewerAlias() {
    return 'viewer';
  }
}
"""
    )

    return src


def test_framework_symbol_precedence_prefers_route_handler_over_method(db, typescript_backend_project):
    """Route handlers should outrank generic method classification on the same symbol."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.controller.ts")}

    assert symbols["getMe"].kind == "route_handler"


@pytest.mark.parametrize("_fallback_selector", [pytest.param(None, id="fallback")])
def test_framework_override_failure_keeps_generic_constant(
    db, typescript_backend_project, _fallback_selector
):
    """Unrecognized backend transport factories should remain generic constants."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/transports/unknown.ts")}

    assert symbols["unknownTransport"].kind == "constant"


def test_index_typescript_backend_framework_exports(db, typescript_backend_project):
    """Indexes exported Elysia and Drizzle symbols as first-class backend navigation points."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    stats = indexer.index(typescript_backend_project)

    assert stats.files_scanned == 16
    assert stats.files_indexed == 16
    assert stats.errors == 0

    auth_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/routes/auth.ts")}
    assert "authRoutes" in auth_symbols
    assert auth_symbols["authRouteNames"].kind == "constant"
    assert auth_symbols["authRoutes"].kind == "router"
    assert "/api/auth" in (auth_symbols["authRoutes"].signature or "")
    assert "POST /api/auth/anonymous" in (auth_symbols["authRoutes"].signature or "")
    assert "POST /api/auth/refresh" in (auth_symbols["authRoutes"].doc_comment or "")

    health_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/routes/health.ts")}
    assert health_symbols["healthRoutes"].kind == "router"

    auth_router = db.get_symbol_by_name("authRoutes")
    health_router = db.get_symbol_by_name("healthRoutes")
    assert auth_router is not None
    assert health_router is not None

    auth_callee_names = [item["symbol"].name for item in db.get_callees(auth_router.id)]
    health_callee_names = [item["symbol"].name for item in db.get_callees(health_router.id)]

    assert "issueAnonymous" in auth_callee_names
    assert "issueRefresh" in auth_callee_names
    assert "healthHandler" in health_callee_names
    assert auth_symbols["authRoutes"].metadata is not None
    assert auth_symbols["authRoutes"].metadata["framework"] == "elysia"
    assert auth_symbols["authRoutes"].metadata["prefix"] == "/api/auth"

    doc_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/routes/doc-comments.ts")
    }
    assert "Real docs should remain attached." in (doc_symbols["meaningfulDoc"].doc_comment or "")
    assert doc_symbols["separatorDoc"].doc_comment is None
    assert "Mixed backend docs should survive" in (doc_symbols["mixedNoiseDoc"].doc_comment or "")
    assert "======" not in (doc_symbols["mixedNoiseDoc"].doc_comment or "")
    assert doc_symbols["todoDoc"].doc_comment is None

    health_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/routes/health.ts")}
    assert "healthRoutes" in health_symbols
    assert health_symbols["healthRoutes"].kind == "router"
    assert "GET /api/health" in (health_symbols["healthRoutes"].signature or "")

    middleware_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/middleware.ts")
    }
    assert "authMiddleware" in middleware_symbols
    assert middleware_symbols["authMiddleware"].kind == "plugin"
    assert "derive" in (middleware_symbols["authMiddleware"].signature or "")
    assert "onBeforeHandle" in (middleware_symbols["authMiddleware"].doc_comment or "")

    schema_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/db/schema.ts")}
    assert "users" in schema_symbols
    assert schema_symbols["users"].kind == "table"
    assert "users" in (schema_symbols["users"].signature or "")
    assert schema_symbols["users"].metadata is not None
    assert schema_symbols["users"].metadata["framework"] == "drizzle"
    assert schema_symbols["users"].metadata["table_name"] == "users"

    db_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/db/index.ts")}
    assert "db" in db_symbols
    assert db_symbols["db"].kind == "database"
    assert "drizzle" in (db_symbols["db"].signature or "")
    assert db_symbols["db"].metadata is not None
    assert db_symbols["db"].metadata["framework"] == "drizzle"


def test_index_typescript_nest_controller_route_metadata(db, typescript_backend_project):
    """Nest controllers should expose route metadata in searchable symbol text."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.controller.ts")}

    assert "AuthController" in symbols
    assert symbols["AuthController"].kind == "controller"
    assert "Controller /auth" in (symbols["AuthController"].signature or "")

    assert "getMe" in symbols
    assert "GET /auth/me" in (symbols["getMe"].signature or "")

    assert "refresh" in symbols
    assert "POST /auth/refresh" in (symbols["refresh"].signature or "")


def test_index_typescript_nest_module_guard_filter_config_bootstrap_metadata(
    db, typescript_backend_project
):
    """Nest infra symbols should be promoted beyond generic class/function kinds."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    module_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.module.ts")
    }
    assert module_symbols["AuthModule"].kind == "module"
    assert module_symbols["AuthModule"].metadata is not None
    assert module_symbols["AuthModule"].metadata["framework"] == "nestjs"
    assert module_symbols["AuthModule"].metadata["imports"] == [
        "AuthModule",
        "ConfigModule",
        "TypeOrmModule",
    ]
    assert module_symbols["AuthModule"].metadata["controllers"] == ["AuthController"]
    assert module_symbols["AuthModule"].metadata["providers"] == [
        "AllExceptionsFilter",
        "AuthService",
        "JwtAuthGuard",
        "LoggingInterceptor",
        "RequestContextPipe",
        "ViewerResolver",
    ]
    assert module_symbols["AuthModule"].metadata["exports"] == ["AuthService"]

    infra_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.infra.ts")
    }
    assert infra_symbols["AuthService"].kind == "service"
    assert infra_symbols["JwtAuthGuard"].kind == "guard"
    assert infra_symbols["AllExceptionsFilter"].kind == "filter"
    assert infra_symbols["RequestContextPipe"].kind == "pipe"
    assert infra_symbols["LoggingInterceptor"].kind == "interceptor"
    assert infra_symbols["AuthMiddleware"].kind == "middleware"

    config_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/config/auth.config.ts")
    }
    assert config_symbols["AuthConfig"].kind == "config"
    assert config_symbols["AuthConfig"].metadata is not None
    assert config_symbols["AuthConfig"].metadata["framework"] == "nestjs"
    assert config_symbols["AuthConfig"].metadata["config_namespace"] == "auth"

    bootstrap_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/main.ts")}
    assert bootstrap_symbols["bootstrap"].kind == "bootstrap"
    assert bootstrap_symbols["bootstrap"].metadata is not None
    assert bootstrap_symbols["bootstrap"].metadata["framework"] == "nestjs"
    assert bootstrap_symbols["bootstrap"].metadata["root_module"] == "AuthModule"


def test_index_typescript_nest_module_graphql_resolver_metadata(db, typescript_backend_project):
    """Nest GraphQL resolvers should expose operation names and resolver metadata."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/viewer.resolver.ts")
    }

    assert symbols["ViewerResolver"].kind == "resolver"
    assert symbols["ViewerResolver"].metadata is not None
    assert symbols["ViewerResolver"].metadata["framework"] == "nestjs"
    assert symbols["ViewerResolver"].metadata["resolver_type"] == "Viewer"

    assert symbols["viewer"].kind == "query"
    assert "Query viewer" in (symbols["viewer"].signature or "")
    assert symbols["viewer"].metadata is not None
    assert symbols["viewer"].metadata["graphql_field"] == "viewer"

    assert symbols["refreshViewer"].kind == "mutation"
    assert "Mutation refreshViewer" in (symbols["refreshViewer"].signature or "")

    assert symbols["viewerUpdated"].kind == "subscription"
    assert "Subscription viewerUpdated" in (symbols["viewerUpdated"].signature or "")


def test_index_typescript_nest_module_metadata_handles_nested_provider_arrays(
    db, typescript_backend_project
):
    """Nested inject arrays inside @Module providers must not truncate later providers."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    module_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.module.ts")
    }

    providers = module_symbols["AuthModule"].metadata["providers"]
    assert "AuthService" in providers
    assert "LoggingInterceptor" in providers
    assert "ViewerResolver" in providers


def test_index_typescript_non_resolver_query_decorator_stays_generic(
    db, typescript_backend_project
):
    """Local decorators named Query must not be promoted without an enclosing Nest resolver."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/local-query.ts")
    }

    assert symbols["LocalQueryExample"].kind == "class"
    assert symbols["localQuery"].kind == "method"
    assert "Query localQuery" not in (symbols["localQuery"].signature or "")


def test_index_typescript_local_resolver_and_query_decorators_stay_generic(
    db, typescript_backend_project
):
    """Local decorators named Resolver and Query must not create Nest GraphQL symbols."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/modules/auth/local-resolver-query.ts")
    }

    assert symbols["LocalResolverExample"].kind == "class"
    assert symbols["localQuery"].kind == "method"


def test_index_typescript_nest_module_import_metadata_keeps_module_names(
    db, typescript_backend_project
):
    """Nest imports metadata should keep top-level module names from common call forms."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    module_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/auth.module.ts")
    }

    assert module_symbols["AuthModule"].metadata["imports"] == [
        "AuthModule",
        "ConfigModule",
        "TypeOrmModule",
    ]


def test_index_typescript_nest_graphql_aliased_imports_promote_symbols(
    db, typescript_backend_project
):
    """Aliased @nestjs/graphql imports should still enable resolver/query promotion."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/modules/auth/aliased.resolver.ts")
    }

    assert symbols["AliasedViewerResolver"].kind == "resolver"
    assert symbols["AliasedViewerResolver"].metadata is not None
    assert symbols["AliasedViewerResolver"].metadata["resolver_type"] == "Viewer"

    assert symbols["viewerAlias"].kind == "query"
    assert "Query viewerAlias" in (symbols["viewerAlias"].signature or "")
    assert symbols["viewerAlias"].metadata is not None
    assert symbols["viewerAlias"].metadata["graphql_field"] == "viewerAlias"


def test_search_typescript_backend_framework_symbols_by_export_name(db, typescript_backend_project):
    """Exported backend constants must be searchable by their real names."""
    config = IndexConfig(root=typescript_backend_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_project)

    route_results = db.search_symbols("authRoutes")
    assert route_results
    assert route_results[0]["name"] == "authRoutes"

    table_results = db.search_symbols("users")
    assert any(result["kind"] == "table" and result["name"] == "users" for result in table_results)


@pytest.fixture
def typescript_async_system_project(tmp_path):
    """Create a TS backend project with async transport, queue, and scheduler patterns."""
    src = tmp_path / "async-backend"
    (src / "server/src/messaging").mkdir(parents=True)
    (src / "server/src/decorators").mkdir(parents=True)
    (src / "server/src/scheduling").mkdir(parents=True)
    (src / "server/src/queues").mkdir(parents=True)
    (src / "server/src/transports").mkdir(parents=True)
    (src / "server/src/cache").mkdir(parents=True)

    (src / "tsconfig.json").write_text(
        """\
{
  "compilerOptions": {
    "baseUrl": "./server/src",
    "paths": {
      "@/*": ["*"]
    }
  }
}
"""
    )

    (src / "server/src/messaging/predictions.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { EventPattern, MessagePattern, Payload, Transport } from '@nestjs/microservices';

@Controller()
export class PredictionsController {
  @MessagePattern('prediction.run')
  async handlePrediction(@Payload() payload: { jobId: string }) {
    return { ok: true, payload };
  }

  @EventPattern('AchievementUnlocked')
  async onAchievementUnlocked(@Payload() payload: { userId: string }) {
    return payload.userId;
  }
}

export const achievementTransport = Transport.RMQ;
"""
    )

    (src / "server/src/scheduling/sync.scheduler.ts").write_text(
        """\
import { Cron, Interval, Timeout } from '@nestjs/schedule';

export class SyncScheduler {
  @Cron('0 2 * * *')
  dailySync() {
    return 'daily';
  }

  @Interval('heartbeat', 5000)
  emitHeartbeat() {
    return 'heartbeat';
  }

  @Timeout('startupWarmup', 1500)
  startupWarmup() {
    return 'warm';
  }
}
"""
    )

    (src / "server/src/transports/notifications.client.ts").write_text(
        """\
import { ClientProxyFactory, Transport } from '@nestjs/microservices';

export const notificationClient = ClientProxyFactory.create({
  transport: Transport.RMQ,
  options: {
    urls: ['amqp://localhost:5672'],
    queue: 'notifications',
  },
});
"""
    )

    (src / "server/src/queues/notification.processor.ts").write_text(
        """\
import { Processor, Process } from '@nestjs/bull';
import { Job } from 'bull';

@Processor('notifications')
export class NotificationProcessor {
  @Process('deliver-email')
  async deliverEmail(job: Job<{ userId: string }>) {
    return job.data.userId;
  }
}
"""
    )

    (src / "server/src/queues/notification.worker.ts").write_text(
        """\
import { Worker } from 'bullmq';

export const notificationWorker = new Worker('notifications', async (job) => {
  return job.data;
});
"""
    )

    (src / "server/src/transports/rabbitmq.consumer.ts").write_text(
        """\
import amqp from 'amqplib';

export async function startAchievementConsumer() {
  const connection = await amqp.connect('amqp://localhost:5672');
  const channel = await connection.createChannel();
  await channel.consume('achievement-events', (message) => {
    return message?.content.toString();
  });
}
"""
    )

    (src / "server/src/cache/redis.ts").write_text(
        """\
import { createClient } from 'redis';

export const redisClient = createClient({
  url: process.env.REDIS_URL,
  socket: {
    host: 'localhost',
    port: 6379,
  },
});
"""
    )

    (src / "server/src/messaging/object-patterns.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { EventPattern, MessagePattern, Transport } from '@nestjs/microservices';

@Controller()
export class ObjectPatternController {
  private readonly serverTransport = Transport.RMQ;

  @MessagePattern({ role: 'admin', cmd: 'prediction.run' })
  handleAdminPrediction() {
    return true;
  }

  @EventPattern({ event: 'AchievementUnlocked', source: 'gamification' })
  onAchievementUnlockedEvent() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/mixed-transports.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { ClientProxyFactory, EventPattern, MessagePattern, Transport } from '@nestjs/microservices';

export const analyticsClient = ClientProxyFactory.create({
  transport: Transport.KAFKA,
  options: {
    client: {
      clientId: 'analytics',
      brokers: ['localhost:9092'],
    },
  },
});

@Controller()
export class MixedTransportController {
  private readonly serverTransport = Transport.RMQ;

  @MessagePattern('prediction.mixed')
  handleMixedPrediction() {
    return true;
  }

  @EventPattern('AchievementMixed')
  onMixedAchievement() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/in-class-mixed-transports.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { ClientProxyFactory, MessagePattern, Transport } from '@nestjs/microservices';

@Controller()
export class InClassMixedTransportController {
  private readonly analyticsClient = ClientProxyFactory.create({
    transport: Transport.KAFKA,
    options: {
      client: {
        clientId: 'analytics-in-class',
        brokers: ['localhost:9092'],
      },
    },
  });

  private readonly serverTransport = Transport.RMQ;

  @MessagePattern('prediction.in-class')
  handleInClassPrediction() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/wrapped-patterns.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { MessagePattern, Transport } from '@nestjs/microservices';

export function RpcRequest(pattern: string): MethodDecorator {
  return MessagePattern(pattern);
}

@Controller()
export class WrappedPatternsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcRequest('diary.note.push')
  handleDiaryPush() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/wrapped-patterns-transitive.controller.ts").write_text(
        """\
import { Controller, Injectable } from '@nestjs/common';
import { MessagePattern, Transport } from '@nestjs/microservices';

function RpcMessage(pattern: string): MethodDecorator {
  return MessagePattern(pattern);
}

export function RpcRequest(pattern: string): MethodDecorator {
  return RpcMessage(pattern);
}

@Injectable()
@Controller()
export class WrappedPatternsTransitiveController {
  private readonly serverTransport = Transport.RMQ;

  @RpcRequest('diary.note.chain')
  handleDiaryChain() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/wrapped-events.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { EventPattern, Transport } from '@nestjs/microservices';

export function RpcEvent(pattern: string): MethodDecorator {
  return EventPattern(pattern);
}

@Controller()
export class WrappedEventsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcEvent('diary.note.archived')
  onDiaryArchived() {
    return true;
  }
}
"""
    )

    (src / "server/src/decorators/rpc-request.decorator.ts").write_text(
        """\
import { applyDecorators } from '@nestjs/common';
import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return applyDecorators(MessagePattern({ cmd: path }));
}
"""
    )

    (src / "server/src/decorators/rpc-event.decorator.ts").write_text(
        """\
import { applyDecorators } from '@nestjs/common';
import { EventPattern } from '@nestjs/microservices';

export function RpcEvent(path: string): MethodDecorator {
  return applyDecorators(EventPattern({ event: path }));
}
"""
    )

    (src / "server/src/messaging/imported-wrapped-patterns.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { Transport } from '@nestjs/microservices';

import { RpcRequest } from '@/decorators/rpc-request.decorator';

@Controller()
export class ImportedWrappedPatternsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/imported-relative-wrapped-patterns.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { Transport } from '@nestjs/microservices';

import { RpcRequest } from '../decorators/rpc-request.decorator';

@Controller()
export class ImportedRelativeWrappedPatternsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcRequest('diary.note.relative')
  handleImportedRelativeDiaryPush() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/imported-wrapped-events.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { Transport } from '@nestjs/microservices';

import { RpcEvent } from '@/decorators/rpc-event.decorator';

@Controller()
export class ImportedWrappedEventsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcEvent('diary.note.imported-archived')
  onImportedDiaryArchived() {
    return true;
  }
}
"""
    )

    (src / "server/src/messaging/invalid-default-imported-wrapped-patterns.controller.ts").write_text(
        """\
import { Controller } from '@nestjs/common';
import { Transport } from '@nestjs/microservices';

import RpcRequest from '../decorators/rpc-request.decorator';

@Controller()
export class InvalidDefaultImportedWrappedPatternsController {
  private readonly serverTransport = Transport.RMQ;

  @RpcRequest('diary.note.invalid-default')
  handleInvalidDefaultImportedDiaryPush() {
    return true;
  }
}
"""
    )

    (src / "server/src/scheduling/digit.scheduler.ts").write_text(
        """\
import { Interval, Timeout } from '@nestjs/schedule';

export class DigitScheduler {
  @Interval('heartbeat-5', 5000)
  heartbeat() {
    return 'heartbeat';
  }

  @Timeout('warmup-10', 1500)
  warmup() {
    return 'warmup';
  }
}
"""
    )

    (src / "server/src/queues/notification.bullmq.processor.ts").write_text(
        """\
import { Processor, Process } from '@nestjs/bullmq';
import { Job } from 'bullmq';

@Processor('notifications-bullmq')
export class NotificationBullmqProcessor {
  @Process('deliver-push')
  async deliverPush(job: Job<{ userId: string }>) {
    return job.data.userId;
  }
}
"""
    )

    return src


def test_index_typescript_async_messagepattern_eventpattern_cron_bullmq_redis_rabbitmq_metadata(
    db, typescript_async_system_project
):
    """Wave 2 async TS symbols should surface microservice, queue, transport, and scheduler metadata."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    stats = indexer.index(typescript_async_system_project)

    assert stats.errors == 0

    messaging_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/predictions.controller.ts")
    }
    assert messaging_symbols["handlePrediction"].kind == "microservice_handler"
    assert messaging_symbols["handlePrediction"].metadata is not None
    assert messaging_symbols["handlePrediction"].metadata["message_pattern"] == "prediction.run"
    assert messaging_symbols["handlePrediction"].metadata["pattern"] == "prediction.run"
    assert messaging_symbols["handlePrediction"].metadata["transport"] == "rmq"
    assert messaging_symbols["handlePrediction"].metadata["role"] == "consumer"

    assert messaging_symbols["onAchievementUnlocked"].kind == "microservice_handler"
    assert messaging_symbols["onAchievementUnlocked"].metadata["event_pattern"] == "AchievementUnlocked"
    assert messaging_symbols["onAchievementUnlocked"].metadata["pattern"] == "AchievementUnlocked"
    assert messaging_symbols["onAchievementUnlocked"].metadata["transport"] == "rmq"

    assert messaging_symbols["achievementTransport"].kind == "transport"
    assert messaging_symbols["achievementTransport"].metadata is not None
    assert messaging_symbols["achievementTransport"].metadata["transport"] == "rmq"

    scheduling_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/scheduling/sync.scheduler.ts")
    }
    assert scheduling_symbols["dailySync"].kind == "scheduled_job"
    assert scheduling_symbols["dailySync"].metadata["schedule_type"] == "cron"
    assert scheduling_symbols["dailySync"].metadata["cron"] == "0 2 * * *"

    assert scheduling_symbols["emitHeartbeat"].kind == "scheduled_job"
    assert scheduling_symbols["emitHeartbeat"].metadata["schedule_type"] == "interval"
    assert scheduling_symbols["emitHeartbeat"].metadata["interval_name"] == "heartbeat"
    assert scheduling_symbols["emitHeartbeat"].metadata["every_ms"] == 5000

    assert scheduling_symbols["startupWarmup"].kind == "scheduled_job"
    assert scheduling_symbols["startupWarmup"].metadata["schedule_type"] == "timeout"
    assert scheduling_symbols["startupWarmup"].metadata["timeout_name"] == "startupWarmup"
    assert scheduling_symbols["startupWarmup"].metadata["delay_ms"] == 1500

    transport_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/transports/notifications.client.ts")
    }
    assert transport_symbols["notificationClient"].kind == "transport"
    assert transport_symbols["notificationClient"].metadata["transport"] == "rmq"
    assert transport_symbols["notificationClient"].metadata["queue_name"] == "notifications"
    assert transport_symbols["notificationClient"].metadata["role"] == "producer"

    processor_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/queues/notification.processor.ts")
    }
    assert processor_symbols["NotificationProcessor"].kind == "queue_processor"
    assert processor_symbols["NotificationProcessor"].metadata["queue_name"] == "notifications"
    assert processor_symbols["NotificationProcessor"].metadata["role"] == "consumer"
    assert processor_symbols["deliverEmail"].kind == "queue_processor"
    assert processor_symbols["deliverEmail"].metadata["queue_name"] == "notifications"
    assert processor_symbols["deliverEmail"].metadata["job_name"] == "deliver-email"

    worker_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/queues/notification.worker.ts")
    }
    assert worker_symbols["notificationWorker"].kind in {"queue_processor", "worker"}
    assert worker_symbols["notificationWorker"].metadata["queue_name"] == "notifications"
    assert worker_symbols["notificationWorker"].metadata["role"] == "consumer"

    rabbitmq_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/transports/rabbitmq.consumer.ts")
    }
    assert rabbitmq_symbols["startAchievementConsumer"].kind == "microservice_handler"
    assert rabbitmq_symbols["startAchievementConsumer"].metadata["queue_name"] == "achievement-events"
    assert rabbitmq_symbols["startAchievementConsumer"].metadata["transport"] == "rmq"
    assert rabbitmq_symbols["startAchievementConsumer"].metadata["role"] == "consumer"

    redis_symbols = {sym.name: sym for sym in db.symbols_in_file("server/src/cache/redis.ts")}
    assert redis_symbols["redisClient"].kind == "transport"
    assert redis_symbols["redisClient"].metadata["transport"] == "redis"
    assert redis_symbols["redisClient"].metadata["role"] == "client"
    assert redis_symbols["redisClient"].metadata["connection_url"] == "process.env.REDIS_URL"


def test_index_typescript_async_object_form_microservice_patterns(
    db, typescript_async_system_project
):
    """Object-form Nest microservice decorators should prefer the semantic pattern keys."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/object-patterns.controller.ts")
    }

    assert symbols["handleAdminPrediction"].kind == "microservice_handler"
    assert symbols["handleAdminPrediction"].metadata["pattern"] == "prediction.run"
    assert symbols["handleAdminPrediction"].metadata["message_pattern"] == "prediction.run"
    assert symbols["handleAdminPrediction"].metadata["transport"] == "rmq"

    assert symbols["onAchievementUnlockedEvent"].kind == "microservice_handler"
    assert symbols["onAchievementUnlockedEvent"].metadata["pattern"] == "AchievementUnlocked"
    assert symbols["onAchievementUnlockedEvent"].metadata["event_pattern"] == "AchievementUnlocked"
    assert symbols["onAchievementUnlockedEvent"].metadata["transport"] == "rmq"


def test_index_typescript_async_mixed_transport_handlers_use_local_rmq_context(
    db, typescript_async_system_project
):
    """Mixed-transport files should not let Kafka client setup poison RMQ handler metadata."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/mixed-transports.controller.ts")
    }

    assert symbols["analyticsClient"].kind == "transport"
    assert symbols["analyticsClient"].metadata["transport"] == "kafka"

    assert symbols["handleMixedPrediction"].kind == "microservice_handler"
    assert symbols["handleMixedPrediction"].metadata["transport"] == "rmq"
    assert symbols["handleMixedPrediction"].metadata["message_pattern"] == "prediction.mixed"

    assert symbols["onMixedAchievement"].kind == "microservice_handler"
    assert symbols["onMixedAchievement"].metadata["transport"] == "rmq"
    assert symbols["onMixedAchievement"].metadata["event_pattern"] == "AchievementMixed"


def test_index_typescript_async_in_class_mixed_transport_prefers_server_assignment(
    db, typescript_async_system_project
):
    """Handlers should not inherit Kafka from a client factory elsewhere in the same class."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/in-class-mixed-transports.controller.ts")
    }

    assert symbols["handleInClassPrediction"].kind == "microservice_handler"
    assert symbols["handleInClassPrediction"].metadata["message_pattern"] == "prediction.in-class"
    assert symbols["handleInClassPrediction"].metadata["transport"] == "rmq"


def test_index_typescript_async_wrapped_messagepattern_handlers_resolve_direct_and_transitive_wrappers(
    db, typescript_async_system_project
):
    """Project-local wrapper factories around MessagePattern should resolve to microservice handlers."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/wrapped-patterns.controller.ts")
    }

    assert symbols["handleDiaryPush"].kind == "microservice_handler"
    assert symbols["handleDiaryPush"].metadata["framework"] == "nestjs"
    assert symbols["handleDiaryPush"].metadata["pattern"] == "diary.note.push"
    assert symbols["handleDiaryPush"].metadata["message_pattern"] == "diary.note.push"
    assert symbols["handleDiaryPush"].metadata["transport"] == "rmq"

    assert symbols["RpcRequest"].kind == "function"
    assert symbols["RpcRequest"].metadata["framework"] == "nestjs"
    assert symbols["RpcRequest"].metadata["resource"] == "decorator_wrapper"
    assert symbols["RpcRequest"].metadata["canonical_decorator"] == "MessagePattern"
    assert symbols["RpcRequest"].metadata["async_kind"] == "microservice_handler"
    assert symbols["RpcRequest"].metadata["pattern_metadata_key"] == "message_pattern"

    transitive_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/wrapped-patterns-transitive.controller.ts")
    }

    assert transitive_symbols["handleDiaryChain"].kind == "microservice_handler"
    assert transitive_symbols["handleDiaryChain"].metadata["framework"] == "nestjs"
    assert transitive_symbols["handleDiaryChain"].metadata["pattern"] == "diary.note.chain"
    assert transitive_symbols["handleDiaryChain"].metadata["message_pattern"] == "diary.note.chain"
    assert transitive_symbols["handleDiaryChain"].metadata["transport"] == "rmq"

    assert transitive_symbols["RpcRequest"].kind == "function"
    assert transitive_symbols["RpcRequest"].metadata["framework"] == "nestjs"
    assert transitive_symbols["RpcRequest"].metadata["resource"] == "decorator_wrapper"
    assert transitive_symbols["RpcRequest"].metadata["canonical_decorator"] == "MessagePattern"
    assert transitive_symbols["RpcRequest"].metadata["async_kind"] == "microservice_handler"
    assert transitive_symbols["RpcRequest"].metadata["pattern_metadata_key"] == "message_pattern"

    assert transitive_symbols["RpcMessage"].kind == "function"
    assert transitive_symbols["RpcMessage"].metadata["framework"] == "nestjs"
    assert transitive_symbols["RpcMessage"].metadata["resource"] == "decorator_wrapper"
    assert transitive_symbols["RpcMessage"].metadata["canonical_decorator"] == "MessagePattern"
    assert transitive_symbols["RpcMessage"].metadata["async_kind"] == "microservice_handler"
    assert transitive_symbols["RpcMessage"].metadata["pattern_metadata_key"] == "message_pattern"


def test_index_typescript_async_wrapped_eventpattern_handlers_resolve_local_wrappers(
    db, typescript_async_system_project
):
    """Project-local wrapper factories around EventPattern should resolve to microservice handlers."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/wrapped-events.controller.ts")
    }

    assert symbols["onDiaryArchived"].kind == "microservice_handler"
    assert symbols["onDiaryArchived"].metadata["framework"] == "nestjs"
    assert symbols["onDiaryArchived"].metadata["pattern"] == "diary.note.archived"
    assert symbols["onDiaryArchived"].metadata["event_pattern"] == "diary.note.archived"
    assert symbols["onDiaryArchived"].metadata["transport"] == "rmq"

    assert symbols["RpcEvent"].kind == "function"
    assert symbols["RpcEvent"].metadata["framework"] == "nestjs"
    assert symbols["RpcEvent"].metadata["resource"] == "decorator_wrapper"
    assert symbols["RpcEvent"].metadata["canonical_decorator"] == "EventPattern"
    assert symbols["RpcEvent"].metadata["async_kind"] == "microservice_handler"
    assert symbols["RpcEvent"].metadata["pattern_metadata_key"] == "event_pattern"


def test_index_typescript_async_imported_wrapped_messagepattern_handlers_resolve_local_alias_wrappers(
    db, typescript_async_system_project
):
    """Imported local applyDecorators(MessagePattern(...)) wrappers should resolve to handlers."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/imported-wrapped-patterns.controller.ts")
    }

    assert symbols["handleImportedDiaryPush"].kind == "microservice_handler"
    assert symbols["handleImportedDiaryPush"].metadata["framework"] == "nestjs"
    assert symbols["handleImportedDiaryPush"].metadata["pattern"] == "diary.note.push"
    assert symbols["handleImportedDiaryPush"].metadata["message_pattern"] == "diary.note.push"
    assert symbols["handleImportedDiaryPush"].metadata["transport"] == "rmq"

    wrapper_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/decorators/rpc-request.decorator.ts")
    }

    assert wrapper_symbols["RpcRequest"].kind == "function"
    assert wrapper_symbols["RpcRequest"].metadata["framework"] == "nestjs"
    assert wrapper_symbols["RpcRequest"].metadata["resource"] == "decorator_wrapper"
    assert wrapper_symbols["RpcRequest"].metadata["canonical_decorator"] == "MessagePattern"
    assert wrapper_symbols["RpcRequest"].metadata["async_kind"] == "microservice_handler"
    assert wrapper_symbols["RpcRequest"].metadata["pattern_metadata_key"] == "message_pattern"


def test_index_typescript_async_imported_wrapped_messagepattern_handlers_resolve_local_relative_wrappers(
    db, typescript_async_system_project
):
    """Imported local MessagePattern wrappers via relative paths should resolve to handlers."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/imported-relative-wrapped-patterns.controller.ts")
    }

    assert symbols["handleImportedRelativeDiaryPush"].kind == "microservice_handler"
    assert symbols["handleImportedRelativeDiaryPush"].metadata["framework"] == "nestjs"
    assert symbols["handleImportedRelativeDiaryPush"].metadata["pattern"] == "diary.note.relative"
    assert symbols["handleImportedRelativeDiaryPush"].metadata["message_pattern"] == "diary.note.relative"
    assert symbols["handleImportedRelativeDiaryPush"].metadata["transport"] == "rmq"


def test_index_typescript_async_imported_wrapped_eventpattern_handlers_resolve_local_alias_wrappers(
    db, typescript_async_system_project
):
    """Imported local applyDecorators(EventPattern(...)) wrappers should resolve to handlers."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/imported-wrapped-events.controller.ts")
    }

    assert symbols["onImportedDiaryArchived"].kind == "microservice_handler"
    assert symbols["onImportedDiaryArchived"].metadata["framework"] == "nestjs"
    assert symbols["onImportedDiaryArchived"].metadata["pattern"] == "diary.note.imported-archived"
    assert symbols["onImportedDiaryArchived"].metadata["event_pattern"] == "diary.note.imported-archived"
    assert symbols["onImportedDiaryArchived"].metadata["transport"] == "rmq"

    wrapper_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/decorators/rpc-event.decorator.ts")
    }

    assert wrapper_symbols["RpcEvent"].kind == "function"
    assert wrapper_symbols["RpcEvent"].metadata["framework"] == "nestjs"
    assert wrapper_symbols["RpcEvent"].metadata["resource"] == "decorator_wrapper"
    assert wrapper_symbols["RpcEvent"].metadata["canonical_decorator"] == "EventPattern"
    assert wrapper_symbols["RpcEvent"].metadata["async_kind"] == "microservice_handler"
    assert wrapper_symbols["RpcEvent"].metadata["pattern_metadata_key"] == "event_pattern"


def test_index_typescript_async_imported_named_wrapper_via_invalid_default_import_stays_generic(
    db, typescript_async_system_project
):
    """Default imports must not resolve to named-only local wrapper exports."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/messaging/invalid-default-imported-wrapped-patterns.controller.ts")
    }

    assert symbols["handleInvalidDefaultImportedDiaryPush"].kind == "method"
    assert symbols["handleInvalidDefaultImportedDiaryPush"].metadata is None


def test_resolve_typescript_import_path_prefers_more_specific_alias_rule(tmp_path):
    root = tmp_path / "project"
    (root / "server").mkdir(parents=True)
    (root / "server/foo.ts").write_text("export const foo = 1;\n")
    (root / "tsconfig.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@/server/*": ["server/*"]
    }
  }
}
"""
    )

    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(root, "src/main.ts", "@/server/foo") == "server/foo.ts"


def test_tsconfig_extends_contributes_inherited_alias_rules(tmp_path):
    root = tmp_path / "project"
    (root / "server/decorators").mkdir(parents=True)
    (root / "server/messaging").mkdir(parents=True)
    (root / "server/decorators/rpc-request.decorator.ts").write_text(
        """import { applyDecorators } from '@nestjs/common';
import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return applyDecorators(MessagePattern({ cmd: path }));
}
"""
    )
    (root / "server/messaging/imported.controller.ts").write_text(
        """import { RpcRequest } from '@test/decorators/rpc-request.decorator';

export class ImportedController {
  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {}
}
"""
    )
    (root / "tsconfig.base.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@test/*": ["server/*"]
    }
  }
}
"""
    )
    (root / "tsconfig.json").write_text(
        """{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {}
}
"""
    )

    _tsconfig_alias_rules.cache_clear()

    rules = _tsconfig_alias_rules(root.as_posix())
    assert ("@test/", "", True, ("server/*",), ".") in rules
    assert _resolve_typescript_import_path(
        root, "server/messaging/imported.controller.ts", "@test/decorators/rpc-request.decorator"
    ) == "server/decorators/rpc-request.decorator.ts"

    resolved = _imported_microservice_decorator_wrappers(
        root,
        "server/messaging/imported.controller.ts",
        (root / "server/messaging/imported.controller.ts").read_text(),
    )
    assert "RpcRequest" in resolved
    assert resolved["RpcRequest"]["canonical_decorator"] == "MessagePattern"


def test_tsconfig_child_paths_override_parent_paths_for_same_alias_key(tmp_path):
    root = tmp_path / "project"
    (root / "base").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "base/foo.ts").write_text("export const baseFoo = 1;\n")
    (root / "src/foo.ts").write_text("export const srcFoo = 1;\n")
    (root / "tsconfig.base.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@test/*": ["base/*"]
    }
  }
}
"""
    )
    (root / "tsconfig.json").write_text(
        """{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "paths": {
      "@test/*": ["src/*"]
    }
  }
}
"""
    )

    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(root, "src/main.ts", "@test/foo") == "src/foo.ts"


def test_tsconfig_extends_inherits_parent_baseurl_for_child_path_override(tmp_path):
    root = tmp_path / "project"
    (root / "src/app/decorators").mkdir(parents=True)
    (root / "src/app/messaging").mkdir(parents=True)
    (root / "src/app/decorators/rpc-request.decorator.ts").write_text(
        """import { applyDecorators } from '@nestjs/common';
import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return applyDecorators(MessagePattern({ cmd: path }));
}
"""
    )
    (root / "src/app/messaging/imported.controller.ts").write_text(
        """import { RpcRequest } from '@test/decorators/rpc-request.decorator';

export class ImportedController {
  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {}
}
"""
    )
    (root / "tsconfig.base.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": "./src",
    "paths": {
      "@test/*": ["lib/*"]
    }
  }
}
"""
    )
    (root / "tsconfig.json").write_text(
        """{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "paths": {
      "@test/*": ["app/*"]
    }
  }
}
"""
    )

    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(root, "src/app/messaging/imported.controller.ts", "@test/decorators/rpc-request.decorator") == "src/app/decorators/rpc-request.decorator.ts"

    resolved = _imported_microservice_decorator_wrappers(
        root,
        "src/app/messaging/imported.controller.ts",
        (root / "src/app/messaging/imported.controller.ts").read_text(),
    )
    assert "RpcRequest" in resolved
    assert resolved["RpcRequest"]["canonical_decorator"] == "MessagePattern"


def test_tsconfig_extends_without_parent_baseurl_uses_child_config_dir_for_paths(tmp_path):
    root = tmp_path / "project"
    (root / "packages/app/src/decorators").mkdir(parents=True)
    (root / "packages/app/src/messaging").mkdir(parents=True)
    (root / "packages/tsconfig.base.json").write_text(
        """{
  "compilerOptions": {
    "paths": {
      "@test/*": ["lib/*"]
    }
  }
}
"""
    )
    (root / "packages/app/src/decorators/rpc-request.decorator.ts").write_text(
        """import { applyDecorators } from '@nestjs/common';
import { MessagePattern } from '@nestjs/microservices';

export function RpcRequest(path: string): MethodDecorator {
  return applyDecorators(MessagePattern({ cmd: path }));
}
"""
    )
    (root / "packages/app/src/messaging/imported.controller.ts").write_text(
        """import { RpcRequest } from '@test/decorators/rpc-request.decorator';

export class ImportedController {
  @RpcRequest('diary.note.push')
  handleImportedDiaryPush() {}
}
"""
    )
    (root / "packages/app/tsconfig.json").write_text(
        """{
  "extends": "../tsconfig.base.json",
  "compilerOptions": {
    "paths": {
      "@test/*": ["src/*"]
    }
  }
}
"""
    )

    app_root = root / "packages/app"
    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(
        app_root,
        "src/messaging/imported.controller.ts",
        "@test/decorators/rpc-request.decorator",
    ) == "src/decorators/rpc-request.decorator.ts"

    resolved = _imported_microservice_decorator_wrappers(
        app_root,
        "src/messaging/imported.controller.ts",
        (app_root / "src/messaging/imported.controller.ts").read_text(),
    )
    assert "RpcRequest" in resolved
    assert resolved["RpcRequest"]["canonical_decorator"] == "MessagePattern"


@pytest.mark.parametrize(
    "extends_value, package_config_path",
    [
        ("@tsconfig/node20", "node_modules/@tsconfig/node20/tsconfig.json"),
        ("@scope/pkg/tsconfig.json", "node_modules/@scope/pkg/tsconfig.json"),
    ],
)
def test_tsconfig_package_extends_loads_inherited_alias_rules(tmp_path, extends_value, package_config_path):
    root = tmp_path / "project"
    package_dir = root / package_config_path.rsplit("/", 1)[0]
    package_dir.mkdir(parents=True)
    (package_dir / "src").mkdir(parents=True)
    (package_dir / "src/foo.ts").write_text("export const foo = 1;\n")
    (package_dir / package_config_path.rsplit("/", 1)[1]).write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@test/*": ["src/*"]
    }
  }
}
"""
    )
    (root / "tsconfig.json").write_text(
        """{
  "extends": "__EXTENDS__",
  "compilerOptions": {}
}
""".replace("__EXTENDS__", extends_value)
    )

    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(root, "src/main.ts", "@test/foo") == package_config_path.rsplit("/", 1)[0] + "/src/foo.ts"


def test_tsconfig_relative_extends_outside_root_path_loads_without_crash(tmp_path):
    repo = tmp_path / "repo"
    (repo / "packages/app").mkdir(parents=True)
    (repo / "packages/shared").mkdir(parents=True)
    (repo / "packages/shared/foo.ts").write_text("export const foo = 1;\n")
    (repo / "packages/tsconfig.base.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@test/*": ["shared/*"]
    }
  }
}
"""
    )
    (repo / "packages/app/tsconfig.json").write_text(
        """{
  "extends": "../tsconfig.base.json",
  "compilerOptions": {}
}
"""
    )

    _tsconfig_alias_rules.cache_clear()

    assert _resolve_typescript_import_path(
        repo / "packages/app", "src/main.ts", "@test/foo"
    ) == "../shared/foo.ts"


def test_index_typescript_async_scheduler_digits_use_numeric_timing_args(
    db, typescript_async_system_project
):
    """Scheduler names with digits must not override the actual numeric timing values."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/scheduling/digit.scheduler.ts")
    }

    assert symbols["heartbeat"].kind == "scheduled_job"
    assert symbols["heartbeat"].metadata["interval_name"] == "heartbeat-5"
    assert symbols["heartbeat"].metadata["every_ms"] == 5000

    assert symbols["warmup"].kind == "scheduled_job"
    assert symbols["warmup"].metadata["timeout_name"] == "warmup-10"
    assert symbols["warmup"].metadata["delay_ms"] == 1500


def test_index_typescript_async_bullmq_framework_metadata(
    db, typescript_async_system_project
):
    """@nestjs/bullmq processors should carry bullmq framework metadata."""
    config = IndexConfig(root=typescript_async_system_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_async_system_project)

    symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/queues/notification.bullmq.processor.ts")
    }

    assert symbols["NotificationBullmqProcessor"].kind == "queue_processor"
    assert symbols["NotificationBullmqProcessor"].metadata["framework"] == "bullmq"
    assert symbols["NotificationBullmqProcessor"].metadata["queue_name"] == "notifications-bullmq"

    assert symbols["deliverPush"].kind == "queue_processor"
    assert symbols["deliverPush"].metadata["framework"] == "bullmq"
    assert symbols["deliverPush"].metadata["queue_name"] == "notifications-bullmq"
    assert symbols["deliverPush"].metadata["job_name"] == "deliver-push"

@pytest.fixture
def typescript_backend_ownership_project(tmp_path):
    """Create a Nest backend where traversal depends on ownership/module metadata."""
    src = tmp_path / "backend-ownership"
    (src / "server/src/modules/auth").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)
    (src / "server/src/config").mkdir(parents=True)

    (src / "server/src/modules/auth/auth.controller.ts").write_text(
        """\
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
"""
    )

    (src / "server/src/modules/auth/auth.infra.ts").write_text(
        """\
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
"""
    )

    (src / "server/src/modules/persistence/persistence.infra.ts").write_text(
        """\
import { EntityRepository } from '@mikro-orm/core';

export class User {}

export class UserRepository extends EntityRepository<User> {
  findById() {
    return { id: 1 };
  }
}
"""
    )

    (src / "server/src/modules/persistence/persistence.module.ts").write_text(
        """\
import { Module } from '@nestjs/common';
import { UserRepository } from './persistence.infra';

@Module({
  providers: [UserRepository],
  exports: [UserRepository],
})
export class PersistenceModule {}
"""
    )

    (src / "server/src/config/auth.config.ts").write_text(
        """\
import { registerAs } from '@nestjs/config';

export const AuthConfig = registerAs('auth', () => ({
  jwtSecret: 'secret',
}));

export const authConfig = registerAs('auth.secondary', () => ({
  jwtAudience: 'users',
}));
"""
    )

    (src / "server/src/modules/auth/auth.module.ts").write_text(
        """\
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
"""
    )

    return src


def test_index_typescript_backend_ownership_edges_link_routes_and_module_exports(
    db, typescript_backend_ownership_project
):
    """Ownership edges should bridge route, controller, service, and imported exports."""
    config = IndexConfig(root=typescript_backend_ownership_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_backend_ownership_project)

    route_handler = db.get_symbol_by_name("getMe")
    controller = db.get_symbol_by_name("AuthController")
    service = db.get_symbol_by_name("AuthService")
    user_repository = db.get_symbol_by_name("UserRepository")
    auth_module = db.get_symbol_by_name("AuthModule")
    persistence_module = db.get_symbol_by_name("PersistenceModule")
    auth_config = db.get_symbol_by_name("AuthConfig")
    config_symbol_names = {
        sym.name for sym in db.symbols_in_file("server/src/config/auth.config.ts")
    }

    assert route_handler is not None
    assert controller is not None
    assert service is not None
    assert user_repository is not None
    assert auth_module is not None
    assert persistence_module is not None
    assert auth_config is not None
    assert "authConfig" in config_symbol_names

    route_callee_names = [item["symbol"].name for item in db.get_callees(route_handler.id)]
    assert "AuthController" in route_callee_names

    controller_callee_names = [item["symbol"].name for item in db.get_callees(controller.id)]
    assert "AuthService" in controller_callee_names
    assert "JwtAuthGuard" not in controller_callee_names
    assert "UserRepository" not in controller_callee_names

    service_callee_names = [item["symbol"].name for item in db.get_callees(service.id)]
    assert "UserRepository" in service_callee_names

    module_caller_names = [item["symbol"].name for item in db.get_callers(auth_module.id)]
    assert "AuthConfig" in module_caller_names
    assert "authConfig" in module_caller_names
    config_callee_names = [item["symbol"].name for item in db.get_callees(auth_config.id)]
    assert "AuthService" in config_callee_names
    module_callee_names = [item["symbol"].name for item in db.get_callees(auth_module.id)]
    assert "PersistenceModule" in module_callee_names
    assert "UserRepository" not in module_callee_names


@pytest.fixture
def typescript_persistence_project(tmp_path):
    """Create a TypeScript backend with Mongoose and MikroORM persistence surfaces."""
    src = tmp_path / "persistence-app"
    (src / "server/src/db").mkdir(parents=True)
    (src / "server/src/modules/persistence").mkdir(parents=True)

    (src / "server/src/db/mongoose.ts").write_text(
        """\
import { MongooseModule, Prop, Schema, SchemaFactory } from '@nestjs/mongoose';

@Schema({ collection: 'users' })
export class User {
  @Prop({ required: true })
  email!: string;
}

export const UserSchema = SchemaFactory.createForClass(User);
export const userModels = MongooseModule.forFeature([
  { name: User.name, schema: UserSchema, collection: 'users' },
]);
"""
    )

    (src / "server/src/db/mikroorm.ts").write_text(
        """\
import {
  Entity,
  EntitySchema,
  MikroORM,
  PrimaryKey,
  Property,
} from '@mikro-orm/core';
import { EntityRepository } from '@mikro-orm/postgresql';
import { defineEntity } from '@mikro-orm/postgresql';

@Entity({ tableName: 'users' })
export class User {
  @PrimaryKey()
  id!: number;

  @Property({ fieldName: 'email_address' })
  email!: string;
}

export const Book = defineEntity({
  name: 'Book',
  tableName: 'books',
  properties: {
    id: { primary: true, type: 'number' },
  },
});

export const AuditLogSchema = new EntitySchema({
  name: 'AuditLog',
  tableName: 'audit_logs',
  properties: {
    id: { primary: true, type: 'number' },
  },
});

export class UserRepository extends EntityRepository<User> {}

export const orm = MikroORM.init({
  entities: [User, Book, AuditLogSchema],
});
"""
    )

    (src / "server/src/modules/persistence/persistence.module.ts").write_text(
        """\
import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { MikroOrmModule } from '@mikro-orm/nestjs';
import {
  AuditLogSchema,
  Book,
  User,
  UserRepository,
} from '../../db/mikroorm';
import { User as MongooseUser, UserSchema as MongooseUserSchema } from '../../db/mongoose';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: MongooseUser.name, schema: MongooseUserSchema, collection: 'users' },
    ]),
    MikroOrmModule.forRoot({
      entities: [User, Book, AuditLogSchema],
    }),
    MikroOrmModule.forFeature([User]),
  ],
  providers: [UserRepository],
})
export class PersistenceModule {}
"""
    )

    (src / "server/src/db/mongoose-aliased.ts").write_text(
        """\
import {
  MongooseModule as MongoFeatureModule,
  Schema as MongoSchema,
  SchemaFactory as MongoSchemaFactory,
} from '@nestjs/mongoose';

@MongoSchema({ collection: 'accounts' })
export class Account {
  handle!: string;
}

export const AccountSchema = MongoSchemaFactory.createForClass(Account);
export const accountModels = MongoFeatureModule.forFeature([
  { name: Account.name, schema: AccountSchema, collection: 'accounts' },
]);
"""
    )

    (src / "server/src/db/mikroorm-aliased.ts").write_text(
        """\
import {
  MikroORM as MikroCore,
} from '@mikro-orm/core';
import { defineEntity as definePgEntity } from '@mikro-orm/postgresql';

export const Invoice = definePgEntity({
  name: 'Invoice',
  tableName: 'invoices',
  properties: {
    id: { primary: true, type: 'number' },
  },
});

export const aliasedOrm = MikroCore.init({
  entities: [Invoice],
});
"""
    )

    (src / "server/src/modules/persistence/persistence-aliased.module.ts").write_text(
        """\
import { Module } from '@nestjs/common';
import { MikroOrmModule as MikroNestModule } from '@mikro-orm/nestjs';
import { Invoice } from '../../db/mikroorm-aliased';

@Module({
  imports: [
    MikroNestModule.forRoot({
      entities: [Invoice],
    }),
    MikroNestModule.forFeature([Invoice]),
  ],
})
export class PersistenceAliasedModule {}
"""
    )

    (src / "server/src/db/mongoose-semicolonless.ts").write_text(
        """\
import { Schema as MongoSchema } from '@nestjs/mongoose'
import {
  SchemaFactory as MongoSchemaFactory,
  MongooseModule as MongoFeatureModule,
} from '@nestjs/mongoose'

@MongoSchema({ collection: 'profiles' })
export class Profile {
  handle!: string
}

export const ProfileSchema = MongoSchemaFactory.createForClass(Profile)
export const profileModels = MongoFeatureModule.forFeature([
  { name: Profile.name, schema: ProfileSchema, collection: 'profiles' },
])
"""
    )

    return src


def test_index_typescript_mongoose_and_mikroorm_exports(db, typescript_persistence_project):
    """Indexes Mongoose and MikroORM persistence exports as first-class backend symbols."""
    config = IndexConfig(root=typescript_persistence_project)
    indexer = Indexer(db, config)
    stats = indexer.index(typescript_persistence_project)

    assert stats.files_scanned == 7
    assert stats.files_indexed == 7
    assert stats.errors == 0

    mongoose_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/db/mongoose.ts")
    }
    assert mongoose_symbols["User"].kind == "entity"
    assert mongoose_symbols["User"].metadata is not None
    assert mongoose_symbols["User"].metadata["framework"] == "mongoose"
    assert mongoose_symbols["User"].metadata["collection_name"] == "users"

    assert mongoose_symbols["UserSchema"].kind == "schema"
    assert mongoose_symbols["UserSchema"].metadata is not None
    assert mongoose_symbols["UserSchema"].metadata["framework"] == "mongoose"
    assert mongoose_symbols["UserSchema"].metadata["entity_name"] == "User"
    assert mongoose_symbols["UserSchema"].metadata["collection_name"] == "users"

    mikro_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/db/mikroorm.ts")
    }
    assert mikro_symbols["User"].kind == "entity"
    assert mikro_symbols["User"].metadata is not None
    assert mikro_symbols["User"].metadata["framework"] == "mikroorm"
    assert mikro_symbols["User"].metadata["table_name"] == "users"
    assert mikro_symbols["User"].metadata["fields"] == [
        {"name": "id", "field_name": "id", "kind": "primary_key"},
        {"name": "email", "field_name": "email_address", "kind": "property"},
    ]

    assert mikro_symbols["Book"].kind == "entity"
    assert mikro_symbols["Book"].metadata is not None
    assert mikro_symbols["Book"].metadata["framework"] == "mikroorm"
    assert mikro_symbols["Book"].metadata["table_name"] == "books"

    assert mikro_symbols["AuditLogSchema"].kind == "entity"
    assert mikro_symbols["AuditLogSchema"].metadata is not None
    assert mikro_symbols["AuditLogSchema"].metadata["framework"] == "mikroorm"
    assert mikro_symbols["AuditLogSchema"].metadata["entity_name"] == "AuditLog"
    assert mikro_symbols["AuditLogSchema"].metadata["table_name"] == "audit_logs"

    assert mikro_symbols["UserRepository"].kind == "repository"
    assert mikro_symbols["UserRepository"].metadata is not None
    assert mikro_symbols["UserRepository"].metadata["framework"] == "mikroorm"
    assert mikro_symbols["UserRepository"].metadata["entity_name"] == "User"

    assert mikro_symbols["orm"].kind == "database"
    assert mikro_symbols["orm"].metadata is not None
    assert mikro_symbols["orm"].metadata["framework"] == "mikroorm"
    assert mikro_symbols["orm"].metadata["resource"] == "database"


def test_index_typescript_entityschema_and_module_import_metadata(
    db, typescript_persistence_project
):
    """Persistence modules should expose MikroORM-specific persistence metadata."""
    config = IndexConfig(root=typescript_persistence_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_persistence_project)

    module_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/modules/persistence/persistence.module.ts")
    }

    assert module_symbols["PersistenceModule"].kind == "module"
    assert module_symbols["PersistenceModule"].metadata["imports"] == [
        "MikroOrmModule",
        "MongooseModule",
    ]
    assert module_symbols["PersistenceModule"].metadata["mikroorm_root_entities"] == [
        "AuditLogSchema",
        "Book",
        "User",
    ]
    assert module_symbols["PersistenceModule"].metadata["mikroorm_feature_entities"] == ["User"]
    assert module_symbols["PersistenceModule"].metadata["providers"] == ["UserRepository"]


def test_index_typescript_aliased_persistence_imports(db, typescript_persistence_project):
    """Aliased Mongoose and MikroORM imports should still produce persistence symbols."""
    config = IndexConfig(root=typescript_persistence_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_persistence_project)

    mongoose_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/db/mongoose-aliased.ts")
    }
    assert mongoose_symbols["Account"].kind == "entity"
    assert mongoose_symbols["AccountSchema"].kind == "schema"
    assert mongoose_symbols["accountModels"].kind == "database"
    assert mongoose_symbols["accountModels"].metadata["collection_names"] == ["accounts"]

    mikro_symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/db/mikroorm-aliased.ts")
    }
    assert mikro_symbols["Invoice"].kind == "entity"
    assert mikro_symbols["Invoice"].metadata["table_name"] == "invoices"
    assert mikro_symbols["aliasedOrm"].kind == "database"
    assert mikro_symbols["aliasedOrm"].metadata["entity_names"] == ["Invoice"]

    module_symbols = {
        sym.name: sym
        for sym in db.symbols_in_file("server/src/modules/persistence/persistence-aliased.module.ts")
    }
    assert module_symbols["PersistenceAliasedModule"].kind == "module"
    assert module_symbols["PersistenceAliasedModule"].metadata["mikroorm_root_entities"] == [
        "Invoice"
    ]
    assert module_symbols["PersistenceAliasedModule"].metadata["mikroorm_feature_entities"] == [
        "Invoice"
    ]


def test_imported_name_map_handles_semicolonless_multiple_imports():
    """Semicolonless named imports from the same module should all resolve."""
    source_text = """\
import { Resolver as GqlResolver } from '@nestjs/graphql'
import { Query as GqlQuery, Mutation as GqlMutation } from '@nestjs/graphql'
"""

    imported = _imported_name_map_from_module(source_text, "@nestjs/graphql")

    assert imported == {
        "GqlResolver": "Resolver",
        "GqlQuery": "Query",
        "GqlMutation": "Mutation",
    }


def test_index_typescript_semicolonless_aliased_persistence_imports(
    db, typescript_persistence_project
):
    """Semicolonless aliased persistence imports should still produce persistence symbols."""
    config = IndexConfig(root=typescript_persistence_project)
    indexer = Indexer(db, config)
    indexer.index(typescript_persistence_project)

    symbols = {
        sym.name: sym for sym in db.symbols_in_file("server/src/db/mongoose-semicolonless.ts")
    }

    assert symbols["Profile"].kind == "entity"
    assert symbols["ProfileSchema"].kind == "schema"
    assert symbols["profileModels"].kind == "database"
    assert symbols["profileModels"].metadata["collection_names"] == ["profiles"]


@pytest.fixture
def nitro_nuxt_project(tmp_path):
    """Create a Nuxt/Nitro app with file-derived server entrypoints."""
    src = tmp_path / "nuxt-app"
    (src / "server/api/foo").mkdir(parents=True)
    (src / "server/api/users").mkdir(parents=True)
    (src / "server/routes/api").mkdir(parents=True)
    (src / "server/plugins").mkdir(parents=True)
    (src / "app/plugins").mkdir(parents=True)
    (src / "app/middleware").mkdir(parents=True)

    (src / "server/api/health.get.ts").write_text(
        "export default defineEventHandler(() => ({ ok: true }))\n"
    )
    (src / "server/api/users/[id].patch.ts").write_text(
        "export default defineCachedEventHandler(() => ({ ok: true }))\n"
    )
    (src / "server/api/foo/[...].ts").write_text(
        "export default defineEventHandler(() => ({ ok: true }))\n"
    )
    (src / "server/routes/api/[...slug].ts").write_text(
        "export default defineEventHandler(() => ({ ok: true }))\n"
    )
    (src / "server/routes/foo.get.ts").write_text(
        "export default defineEventHandler(() => ({ ok: true }))\n"
    )
    (src / "server/plugins/auth.ts").write_text(
        "export default defineNitroPlugin(() => {})\n"
    )
    (src / "app/plugins/apollo.client.ts").write_text(
        "export default defineNuxtPlugin(() => {})\n"
    )
    (src / "app/middleware/auth.global.ts").write_text(
        "export default defineNuxtRouteMiddleware((to) => {})\n"
    )

    return src


def test_index_nitro_file_derived_symbols(db, nitro_nuxt_project):
    """Nitro/Nuxt default exports should index as first-class file-derived symbols."""
    config = IndexConfig(root=nitro_nuxt_project)
    indexer = Indexer(db, config)
    stats = indexer.index(nitro_nuxt_project)

    assert stats.files_scanned == 8
    assert stats.files_indexed == 8
    assert stats.errors == 0

    health_symbols = db.symbols_in_file("server/api/health.get.ts")
    assert health_symbols
    assert health_symbols[0].kind == "route"
    assert health_symbols[0].metadata is not None
    assert health_symbols[0].metadata["framework"] == "nitro"
    assert health_symbols[0].metadata["http_method"] == "GET"
    assert health_symbols[0].metadata["route_path"] == "/api/health"

    user_symbols = db.symbols_in_file("server/api/users/[id].patch.ts")
    assert user_symbols
    assert user_symbols[0].kind == "route"
    assert user_symbols[0].metadata is not None
    assert user_symbols[0].metadata["http_method"] == "PATCH"
    assert user_symbols[0].metadata["route_path"] == "/api/users/:id"

    unnamed_catch_all_symbols = db.symbols_in_file("server/api/foo/[...].ts")
    assert unnamed_catch_all_symbols
    assert unnamed_catch_all_symbols[0].kind == "route"
    assert unnamed_catch_all_symbols[0].metadata is not None
    assert unnamed_catch_all_symbols[0].metadata["http_method"] == "ALL"
    assert unnamed_catch_all_symbols[0].metadata["route_path"] == "/api/foo/:pathMatch(.*)*"

    catch_all_symbols = db.symbols_in_file("server/routes/api/[...slug].ts")
    assert catch_all_symbols
    assert catch_all_symbols[0].kind == "route"
    assert catch_all_symbols[0].metadata is not None
    assert catch_all_symbols[0].metadata["http_method"] == "ALL"
    assert catch_all_symbols[0].metadata["route_path"] == "/api/:slug(.*)*"

    route_method_symbols = db.symbols_in_file("server/routes/foo.get.ts")
    assert route_method_symbols
    assert route_method_symbols[0].kind == "route"
    assert route_method_symbols[0].metadata is not None
    assert route_method_symbols[0].metadata["http_method"] == "GET"
    assert route_method_symbols[0].metadata["route_path"] == "/foo"

    server_plugin_symbols = db.symbols_in_file("server/plugins/auth.ts")
    assert server_plugin_symbols
    assert server_plugin_symbols[0].kind == "plugin"
    assert server_plugin_symbols[0].metadata is not None
    assert server_plugin_symbols[0].metadata["framework"] == "nitro"
    assert server_plugin_symbols[0].metadata["resource"] == "plugin"

    client_plugin_symbols = db.symbols_in_file("app/plugins/apollo.client.ts")
    assert client_plugin_symbols
    assert client_plugin_symbols[0].kind == "plugin"
    assert client_plugin_symbols[0].metadata is not None
    assert client_plugin_symbols[0].metadata["framework"] == "nuxt"
    assert client_plugin_symbols[0].metadata["resource"] == "plugin"

    middleware_symbols = db.symbols_in_file("app/middleware/auth.global.ts")
    assert middleware_symbols
    assert middleware_symbols[0].kind == "middleware"
    assert middleware_symbols[0].metadata is not None
    assert middleware_symbols[0].metadata["framework"] == "nuxt"
    assert middleware_symbols[0].metadata["resource"] == "middleware"
