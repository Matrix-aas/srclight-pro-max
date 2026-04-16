"""Tree-sitter based code indexer.

Walks a directory, parses files, extracts symbols, populates the database.
Incremental: only re-indexes files whose content hash has changed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import logging
import posixpath
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

from tree_sitter import Node, Parser, Query, QueryCursor

from . import __version__
from .db import Database, EdgeRecord, FileRecord, SymbolRecord, content_hash
from .extractors import DOCUMENT_EXTENSIONS, detect_document_language, get_registry
from .languages import (
    LANGUAGES,
    detect_language,
    get_language,
    get_tsx_language,
)

logger = logging.getLogger("srclight.indexer")

IndexEvent = dict[str, object]
IndexEventCallback = Callable[[IndexEvent], None]

# Bump when extraction/query behavior changes such that unchanged files must be re-indexed.
INDEXER_BUILD_ID = f"{__version__}+extractor-2026-04-16-jsdoc-cleanup-v1"


# Default ignore patterns
DEFAULT_IGNORE = [
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.a",
    "*.lib",
    "*.exe",
    "*.bin",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.rar",
    "*.7z",
    "*.pdf",
    "*.sqlite",
    "*.db",
    "*.sqlite3",
    # Build artifacts
    "CMakeFiles",
    "__cmake_systeminformation",
    "*.cmake",
    "CMakeCache.txt",
    # C# / .NET artifacts
    "bin",
    "obj",
    "packages",
    ".vs",
    "*.Designer.cs",
    "*.g.cs",
    "*.g.i.cs",
    "*.AssemblyInfo.cs",
    # Vendored / third-party
    "vendor",
    "third_party",
    "third-party",
    "ext",
    "depends",
    # Srclight index
    ".srclight",
    ".codelight",
    # Obsidian
    ".obsidian",
    ".trash",
]

# Max file size to index (1 MB)
MAX_FILE_SIZE = 1_000_000


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    symbols_extracted: int = 0
    edges_created: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IndexConfig:
    root: Path = field(default_factory=Path)
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    max_file_size: int = MAX_FILE_SIZE
    max_doc_file_size: int = 50_000_000  # 50 MB for documents (PDF, DOCX, etc.)
    languages: list[str] | None = None  # None = all supported
    embed_model: str | None = None  # e.g. "qwen3-embedding", "voyage-code-3"


def _should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    rel = str(path.relative_to(root))

    # Check each component of the relative path against directory patterns
    parts = path.relative_to(root).parts
    for part in parts:
        for pattern in patterns:
            if fnmatch.fnmatch(part, pattern):
                return True

    # Check full relative path
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True

    return False


def _git_tracked_files(root: Path) -> set[str] | None:
    """Get the set of git-tracked files (respects .gitignore).

    Returns None if not a git repo or git is unavailable.
    Returns relative paths as strings.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return {line for line in result.stdout.splitlines() if line}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _get_git_head(root: Path) -> str | None:
    """Get current git HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _extract_doc_comment(source_bytes: bytes, node: Node) -> str | None:
    """Extract doc comment preceding a symbol node."""
    # Look at the previous sibling for a comment
    prev = node.prev_named_sibling
    if prev is None:
        # Check for comment as first child or preceding line
        # Look at previous unnamed siblings too
        prev_sib = node.prev_sibling
        if prev_sib and prev_sib.type == "comment":
            return prev_sib.text.decode("utf-8", errors="replace").strip()
        return None

    if prev.type == "comment":
        return prev.text.decode("utf-8", errors="replace").strip()

    # Python: check for docstring (first child expression_statement with string)
    if node.type in ("function_definition", "class_definition"):
        body = node.child_by_field_name("body")
        if body and body.named_child_count > 0:
            first_stmt = body.named_children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.named_children[0] if first_stmt.named_child_count > 0 else None
                if expr and expr.type == "string":
                    return expr.text.decode("utf-8", errors="replace").strip().strip('"""').strip("'''").strip()

    return None


def _is_meaningful_js_ts_doc_comment(comment: str | None) -> bool:
    """Keep real JS/TS doc comments while rejecting separator and TODO noise."""
    if not comment:
        return False

    cleaned_lines: list[str] = []
    for raw_line in comment.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^/\*\*?", "", line)
        line = re.sub(r"\*/$", "", line)
        line = re.sub(r"^//+", "", line)
        line = re.sub(r"^\*+", "", line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return False

    for line in cleaned_lines:
        if re.fullmatch(r"[-=/*_# ]+", line):
            continue
        if re.match(r"^(?:todo|fixme|xxx|hack|note)\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"[A-Za-z0-9]", line):
            return True

    return False


def _extract_js_ts_doc_comment(source_bytes: bytes, node: Node) -> str | None:
    """Extract meaningful doc comments for JS/TS/Vue symbols."""
    current: Node | None = node
    for _ in range(3):
        if current is None:
            break
        doc = _extract_doc_comment(source_bytes, current)
        if _is_meaningful_js_ts_doc_comment(doc):
            return doc
        current = current.parent

    return None


def _extract_signature(source_bytes: bytes, node: Node, lang: str) -> str | None:
    """Extract function/method signature (without body)."""
    if lang == "python":
        # Everything up to the colon before the body
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        name_node = node.child_by_field_name("name")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            return source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace").strip()

    elif lang in ("c", "cpp"):
        # For function definitions, get the declarator
        declarator = node.child_by_field_name("declarator")
        ret_type = node.child_by_field_name("type")
        if declarator:
            parts = []
            if ret_type:
                parts.append(ret_type.text.decode("utf-8", errors="replace"))
            parts.append(declarator.text.decode("utf-8", errors="replace"))
            return " ".join(parts)

    elif lang in ("javascript", "typescript"):
        name_node = node.child_by_field_name("name")
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            prefix = source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace")
            # Trim off decorators/export
            lines = prefix.split("\n")
            for i, line in enumerate(lines):
                if "function " in line or "class " in line or "(" in line:
                    return "\n".join(lines[i:]).strip()
            return prefix.strip()

    elif lang == "rust":
        name_node = node.child_by_field_name("name")
        params = node.child_by_field_name("parameters")
        ret = node.child_by_field_name("return_type")
        if name_node:
            sig_end = (ret.end_byte if ret else
                       params.end_byte if params else
                       name_node.end_byte)
            return source_bytes[node.start_byte:sig_end].decode("utf-8", errors="replace").strip()

    return None


_ELYSIA_ROUTE_METHODS = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
    "options": "OPTIONS",
    "head": "HEAD",
    "all": "ALL",
    "ws": "WS",
}

_ELYSIA_PLUGIN_HOOKS = {
    "derive",
    "decorate",
    "guard",
    "macro",
    "mapDerive",
    "model",
    "onAfterHandle",
    "onBeforeHandle",
    "onError",
    "resolve",
    "state",
    "use",
}

_DRIZZLE_TABLE_BUILDERS = {
    "pgTable": "table",
    "mysqlTable": "table",
    "sqliteTable": "table",
    "singlestoreTable": "table",
    "pgView": "view",
    "mysqlView": "view",
    "sqliteView": "view",
}

_DRIZZLE_ENUM_BUILDERS = {"pgEnum", "mysqlEnum", "sqliteEnum"}

_NEST_HTTP_DECORATORS = {
    "Get": "GET",
    "Post": "POST",
    "Put": "PUT",
    "Patch": "PATCH",
    "Delete": "DELETE",
    "Options": "OPTIONS",
    "Head": "HEAD",
    "All": "ALL",
}

_NEST_GRAPHQL_DECORATORS = {
    "Query": "query",
    "Mutation": "mutation",
    "Subscription": "subscription",
}

_NEST_MICROSERVICE_DECORATORS = {
    "MessagePattern": "message_pattern",
    "EventPattern": "event_pattern",
}

_NEST_SCHEDULE_DECORATORS = {
    "Cron": "cron",
    "Interval": "interval",
    "Timeout": "timeout",
}

_NITRO_HTTP_METHODS = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
    "options": "OPTIONS",
    "head": "HEAD",
    "connect": "CONNECT",
    "trace": "TRACE",
}

FRAMEWORK_KIND_PRECEDENCE = {
    "route_handler": 100,
    "microservice_handler": 95,
    "queue_processor": 90,
    "scheduled_job": 85,
    "controller": 80,
    "transport": 70,
}


def _normalize_route_path(prefix: str | None, path: str | None) -> str:
    """Join a route prefix and child path into a normalized absolute path."""
    prefix = (prefix or "").strip()
    path = (path or "").strip()

    if not prefix:
        if not path:
            return "/"
        return path if path.startswith("/") else f"/{path}"

    if not path:
        return prefix if prefix.startswith("/") else f"/{prefix}"

    left = prefix.rstrip("/")
    right = path.lstrip("/")
    if not left:
        return f"/{right}"
    return f"{left}/{right}" if right else left


def _quoted_strings(text: str) -> list[str]:
    """Return quoted string literals from a source snippet."""
    matches = re.findall(r"(['\"`])((?:\\.|(?!\1).)*)\1", text, flags=re.DOTALL)
    return [value for _quote, value in matches]


def _decorator_parts(node: Node) -> tuple[str | None, list[str]]:
    """Extract the decorator name and string arguments from a decorator node."""
    call_node = next((child for child in node.named_children if child.type == "call_expression"), None)
    if call_node is None:
        ident = next((child for child in node.named_children if child.type == "identifier"), None)
        if ident is None:
            return None, []
        return ident.text.decode("utf-8", errors="replace"), []

    function_node = call_node.child_by_field_name("function")
    if function_node is None:
        function_node = next((child for child in call_node.named_children if child.type != "arguments"), None)
    if function_node is None:
        return None, []

    name = function_node.text.decode("utf-8", errors="replace")
    args_node = call_node.child_by_field_name("arguments")
    args = _quoted_strings(args_node.text.decode("utf-8", errors="replace")) if args_node else []
    return name, args


def _leading_decorators(node: Node) -> list[Node]:
    """Collect decorators immediately preceding a class or method node."""
    parent = node.parent
    if parent is None:
        return []

    siblings = parent.children
    try:
        idx = siblings.index(node)
    except ValueError:
        return []

    decorators: list[Node] = []
    i = idx - 1
    skip_types = {"export", "default", "abstract", "declare", "public", "private", "protected"}
    while i >= 0:
        sibling = siblings[i]
        if sibling.type == "decorator":
            decorators.append(sibling)
            i -= 1
            continue
        if sibling.type in skip_types:
            i -= 1
            continue
        if sibling.is_named:
            break
        i -= 1

    decorators.reverse()
    return decorators


def _controller_prefix_for(node: Node) -> str | None:
    """Return the enclosing Nest controller prefix for a method definition."""
    current = node.parent
    while current is not None and current.type != "class_declaration":
        current = current.parent
    if current is None:
        return None

    for decorator in _leading_decorators(current):
        name, args = _decorator_parts(decorator)
        if name == "Controller":
            return _normalize_route_path(None, args[0] if args else None)
    return None


def _enclosing_class_node(node: Node) -> Node | None:
    """Return the enclosing class declaration for a node, if any."""
    current = node.parent
    while current is not None and current.type != "class_declaration":
        current = current.parent
    return current


def _extract_identifiers_from_array(text: str, key: str) -> list[str]:
    """Recover top-level identifier entries from an object literal array property."""
    entries = _extract_object_array_entries(text, key)
    if not entries:
        return []
    identifiers = []
    for entry in entries:
        entry = entry.strip()
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", entry):
            identifiers.append(entry)
    return sorted(set(identifiers))


def _extract_module_names_from_array(text: str, key: str) -> list[str]:
    """Recover useful top-level module names from a Nest module imports array."""
    entries = _extract_object_array_entries(text, key)
    if not entries:
        return []

    names: list[str] = []
    for entry in entries:
        entry = entry.strip()

        forward_ref_match = re.search(
            r"\bforwardRef\s*\(\s*\(\s*\)\s*=>\s*([A-Z][A-Za-z0-9_]*)",
            entry,
        )
        if forward_ref_match:
            names.append(forward_ref_match.group(1))
            continue

        call_match = re.match(
            r"([A-Z][A-Za-z0-9_]*)\s*(?:\.\s*[A-Za-z_$][\w$]*)?\s*\(",
            entry,
        )
        if call_match:
            names.append(call_match.group(1))
            continue

        ident_match = re.fullmatch(r"([A-Z][A-Za-z0-9_]*)", entry)
        if ident_match:
            names.append(ident_match.group(1))

    return sorted(set(names))


def _extract_config_refs_from_module_imports(text: str, local_module_names: set[str]) -> list[str]:
    """Recover config factory identifiers passed through Nest ConfigModule imports."""
    entries = _extract_object_array_entries(text, "imports")
    if not entries or not local_module_names:
        return []

    module_pattern = "|".join(re.escape(name) for name in sorted(local_module_names))
    identifier_pattern = r"[A-Za-z_$][\w$]*"
    config_refs: list[str] = []
    for entry in entries:
        if not re.search(rf"\b(?:{module_pattern})\s*\.", entry):
            continue

        feature_match = re.search(
            rf"\b(?:{module_pattern})\.forFeature\s*\(\s*({identifier_pattern})\s*\)",
            entry,
        )
        if feature_match:
            config_refs.append(feature_match.group(1))

        load_match = re.search(r"\bload\s*:\s*\[([^\]]*)\]", entry, flags=re.DOTALL)
        if load_match:
            load_entries = _extract_object_array_entries(
                "{ load: [" + load_match.group(1) + "] }",
                "load",
            )
            for load_entry in load_entries:
                ident_match = re.fullmatch(rf"({identifier_pattern})", load_entry.strip())
                if ident_match:
                    config_refs.append(ident_match.group(1))

    return sorted(set(config_refs))


def _extract_object_array_entries(text: str, key: str) -> list[str]:
    """Return top-level array entries from an object literal property."""
    match = re.search(rf"\b{re.escape(key)}\s*:", text)
    if not match:
        return []

    i = match.end()
    length = len(text)
    while i < length and text[i].isspace():
        i += 1
    if i >= length or text[i] != "[":
        return []

    depth_bracket = 0
    depth_brace = 0
    depth_paren = 0
    in_string: str | None = None
    escape = False
    entry_start = i + 1
    entries: list[str] = []

    for j in range(i, length):
        ch = text[j]
        if in_string is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue

        if ch in {"'", '"', "`"}:
            in_string = ch
            continue

        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket -= 1
            if depth_bracket == 0:
                entry = text[entry_start:j].strip()
                if entry:
                    entries.append(entry)
                return entries
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            depth_brace -= 1
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren -= 1
            continue
        if ch == "," and depth_bracket == 1 and depth_brace == 0 and depth_paren == 0:
            entry = text[entry_start:j].strip()
            if entry:
                entries.append(entry)
            entry_start = j + 1

    return []


def _extract_graphql_field_name(decorator_text: str, fallback_name: str) -> str:
    """Recover a Nest GraphQL field name from decorator options when possible."""
    match = re.search(r"\bname\s*:\s*(['\"`])([^'\"`]+)\1", decorator_text)
    if match:
        return match.group(2)
    return fallback_name


def _imported_name_map_from_module(source_text: str, module_name: str) -> dict[str, str]:
    """Return local-name to canonical-name mappings imported from a module specifier."""
    imported: dict[str, str] = {}
    pattern = re.compile(
        r"^\s*import\s+(\{[^}]*\})\s+from\s*(['\"])"
        + re.escape(module_name)
        + r"\2\s*;?\s*$",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(source_text):
        clause = match.group(1).strip()
        inner = clause[1:-1]
        for part in inner.split(","):
            item = part.strip()
            if not item:
                continue
            alias_match = re.match(r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$", item)
            if alias_match:
                imported[alias_match.group(2)] = alias_match.group(1)
                continue
            ident_match = re.match(r"([A-Za-z_$][\w$]*)$", item)
            if ident_match:
                imported[ident_match.group(1)] = ident_match.group(1)
    return imported


def _imported_name_map_from_modules(source_text: str, module_names: list[str]) -> dict[str, str]:
    """Return local-name to canonical-name mappings imported from any listed module."""
    imported: dict[str, str] = {}
    for module_name in module_names:
        imported.update(_imported_name_map_from_module(source_text, module_name))
    return imported


def _typescript_import_bindings(source_text: str) -> dict[str, tuple[str | None, str, str]]:
    """Return TS/JS import bindings keyed by local symbol name."""
    imported: dict[str, tuple[str | None, str, str]] = {}
    pattern = re.compile(
        r"^\s*import\s+(.+?)\s+from\s*(['\"])([^'\"]+)\2\s*;?\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(source_text):
        clause = match.group(1).strip()
        module_specifier = match.group(3)
        named_clause = ""
        if "{" in clause and "}" in clause:
            named_clause = clause[clause.index("{") + 1:clause.rindex("}")]
            clause = clause[:clause.index("{")].rstrip(", ").strip()
        if clause and not clause.startswith("*"):
            ident_match = re.match(r"([A-Za-z_$][\w$]*)$", clause)
            if ident_match:
                imported[ident_match.group(1)] = (None, module_specifier, "default")
        if named_clause:
            for part in named_clause.split(","):
                item = part.strip()
                if not item:
                    continue
                alias_match = re.match(
                    r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$",
                    item,
                )
                if alias_match:
                    imported[alias_match.group(2)] = (
                        alias_match.group(1),
                        module_specifier,
                        "named",
                    )
                    continue
                ident_match = re.match(r"([A-Za-z_$][\w$]*)$", item)
                if ident_match:
                    imported[ident_match.group(1)] = (
                        ident_match.group(1),
                        module_specifier,
                        "named",
                    )
    return imported


def _local_names_for_import(import_map: dict[str, str], canonical_name: str) -> set[str]:
    """Return local identifiers that resolve to a canonical imported name."""
    return {local for local, canonical in import_map.items() if canonical == canonical_name}


def _default_import_name_from_module(source_text: str, module_name: str) -> str | None:
    """Return the default-import local name for a module specifier."""
    match = re.search(
        r"^\s*import\s+([A-Za-z_$][\w$]*)\s+from\s*(['\"])"
        + re.escape(module_name)
        + r"\2\s*;?\s*$",
        source_text,
        flags=re.MULTILINE,
    )
    if match:
        return match.group(1)
    return None


def _namespace_import_name_from_module(source_text: str, module_name: str) -> str | None:
    """Return the namespace-import local name for a module specifier."""
    match = re.search(
        r"^\s*import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s*(['\"])"
        + re.escape(module_name)
        + r"\2\s*;?\s*$",
        source_text,
        flags=re.MULTILINE,
    )
    if match:
        return match.group(1)
    return None


def _parse_jsonc(text: str) -> dict[str, object] | None:
    """Parse a JSON-with-comments document conservatively."""
    stripped: list[str] = []
    in_string = False
    quote = ""
    escape = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
                stripped.append(char)
            i += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            stripped.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            i += 1
            continue
        if char in ("'", '"'):
            in_string = True
            quote = char
            stripped.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        stripped.append(char)
        i += 1
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", "".join(stripped)))
    except json.JSONDecodeError:
        return None


def _ordered_import_candidate_paths(base_path: str, specifier: str) -> list[str]:
    """Return likely TS/JS file paths for an import specifier base path."""
    if any(specifier.endswith(ext) for ext in (".ts", ".tsx", ".js", ".jsx")):
        return [base_path]

    return [
        f"{base_path}.ts",
        f"{base_path}.tsx",
        f"{base_path}.js",
        f"{base_path}.jsx",
        posixpath.join(base_path, "index.ts"),
        posixpath.join(base_path, "index.tsx"),
        posixpath.join(base_path, "index.js"),
        posixpath.join(base_path, "index.jsx"),
    ]


def _resolve_existing_import_path(root_path: Path | None, candidate_paths: list[str]) -> str | None:
    """Return the first candidate import path that exists under root_path."""
    if root_path is None:
        return candidate_paths[0] if candidate_paths else None

    for candidate_path in candidate_paths:
        try:
            if (root_path / candidate_path).is_file():
                return candidate_path
        except OSError:
            continue
    return None


@lru_cache(maxsize=64)
def _tsconfig_alias_rules(root_path_str: str) -> tuple[tuple[str, str, bool, tuple[str, ...], str], ...]:
    """Load tsconfig/jsconfig alias rules relative to a repository root."""
    root_path = Path(root_path_str)

    def _find_config_path(start_path: Path) -> Path | None:
        for candidate_name in ("tsconfig.json", "jsconfig.json"):
            candidate_path = start_path / candidate_name
            if candidate_path.is_file():
                return candidate_path
        return None

    def _candidate_extends_paths(config_path: Path, extends_value: str) -> list[Path]:
        extends_value = extends_value.strip()
        if not extends_value:
            return []

        def _candidate_variants(base_path: Path) -> list[Path]:
            variants = [base_path]
            if base_path.suffix == "":
                variants.extend([
                    base_path.with_suffix(".json"),
                    base_path.with_suffix(".jsonc"),
                ])
                variants.extend([
                    base_path / "tsconfig.json",
                    base_path / "tsconfig.jsonc",
                    base_path / "tsconfig.base.json",
                    base_path / "tsconfig.base.jsonc",
                    base_path / "index.json",
                    base_path / "index.jsonc",
                ])
            return variants

        candidate_roots: list[Path] = []
        if extends_value.startswith(".") or extends_value.startswith("/"):
            candidate_roots.append(config_path.parent / extends_value)
        else:
            for ancestor in (config_path.parent, *config_path.parent.parents):
                candidate_roots.append(ancestor / "node_modules" / extends_value)

        candidates: list[Path] = []
        for candidate_root in candidate_roots:
            for candidate in _candidate_variants(candidate_root):
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _load_alias_rules(
        config_path: Path,
        seen: set[Path],
    ) -> tuple[
        dict[tuple[str, str, bool], tuple[str, str, bool, tuple[str, ...], str]],
        str,
        bool,
    ]:
        resolved_config_path = config_path.as_posix()
        if resolved_config_path in seen:
            return {}, posixpath.relpath(config_path.parent.as_posix(), root_path.as_posix()), False
        seen.add(resolved_config_path)

        parsed = _parse_jsonc(config_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(parsed, dict):
            return {}, posixpath.relpath(config_path.parent.as_posix(), root_path.as_posix()), False

        rules: dict[tuple[str, str, bool], tuple[str, str, bool, tuple[str, ...], str]] = {}
        config_dir = posixpath.relpath(config_path.parent.as_posix(), root_path.as_posix())
        effective_base_root = config_dir
        has_effective_base_url = False
        extends_value = parsed.get("extends")
        if isinstance(extends_value, str):
            for parent_path in _candidate_extends_paths(config_path, extends_value):
                if parent_path.is_file():
                    parent_rules, parent_base_root, parent_has_effective_base_url = _load_alias_rules(parent_path, seen)
                    rules.update(parent_rules)
                    if parent_has_effective_base_url:
                        effective_base_root = parent_base_root
                        has_effective_base_url = True
                    break

        compiler_options = parsed.get("compilerOptions")
        if not isinstance(compiler_options, dict):
            return rules, effective_base_root, has_effective_base_url

        raw_paths = compiler_options.get("paths")
        if not isinstance(raw_paths, dict):
            base_url = compiler_options.get("baseUrl")
            if isinstance(base_url, str) and base_url.strip():
                effective_base_root = posixpath.normpath(posixpath.join(config_dir, base_url))
                has_effective_base_url = True
            return rules, effective_base_root, has_effective_base_url

        base_url = compiler_options.get("baseUrl")
        if isinstance(base_url, str) and base_url.strip():
            effective_base_root = posixpath.normpath(posixpath.join(config_dir, base_url))
            has_effective_base_url = True

        base_root = effective_base_root if has_effective_base_url else config_dir

        for pattern, targets in raw_paths.items():
            if not isinstance(pattern, str) or pattern.count("*") > 1:
                continue
            target_list = tuple(item for item in targets if isinstance(item, str)) if isinstance(targets, list) else ()
            if not target_list:
                continue
            has_wildcard = "*" in pattern
            if has_wildcard:
                prefix, suffix = pattern.split("*", 1)
            else:
                prefix, suffix = pattern, ""
            rules[(prefix, suffix, has_wildcard)] = (prefix, suffix, has_wildcard, target_list, base_root)

        return rules, effective_base_root, has_effective_base_url

    config_path = _find_config_path(root_path)
    if config_path is None:
        return ()

    raw_rules, _, _ = _load_alias_rules(config_path, set())
    return tuple(
        sorted(
            raw_rules.values(),
            key=lambda item: (
                -len(item[0]),
                item[2],
                -len(item[1]),
                item[4],
                item[3],
            ),
        )
    )


def _resolve_typescript_alias_rule(
    root_path: Path | None,
    module_specifier: str,
    resolve_existing_import_path,
) -> tuple[str | None, bool]:
    """Resolve a TS/JS alias specifier and report whether a tsconfig alias matched."""
    if root_path is None:
        return None, False

    matched_rule = False
    for prefix, suffix, has_wildcard, target_patterns, base_root in _tsconfig_alias_rules(root_path.as_posix()):
        if has_wildcard:
            if not module_specifier.startswith(prefix) or not module_specifier.endswith(suffix):
                continue
            end_index = len(module_specifier) - len(suffix) if suffix else len(module_specifier)
            if end_index < len(prefix):
                continue
            wildcard = module_specifier[len(prefix):end_index]
        else:
            if module_specifier != prefix:
                continue
            wildcard = ""

        matched_rule = True
        for target_pattern in target_patterns:
            if target_pattern.count("*") > 1:
                continue
            if "*" in target_pattern:
                expanded = target_pattern.replace("*", wildcard, 1)
            elif wildcard:
                continue
            else:
                expanded = target_pattern
            expanded = expanded.strip()
            if not expanded or expanded.startswith("/"):
                continue
            base_path = posixpath.normpath(posixpath.join(base_root, expanded))
            resolved = resolve_existing_import_path(
                _ordered_import_candidate_paths(base_path, expanded),
            )
            if resolved:
                return resolved, True

    return None, matched_rule


def _resolve_typescript_import_path(
    root_path: Path | None,
    source_file_path: str,
    module_specifier: str,
) -> str | None:
    """Resolve a project-local TS/JS import path using relative, tsconfig, or workspace aliases."""
    if root_path is None:
        return None

    if module_specifier.startswith("."):
        base_path = posixpath.normpath((Path(source_file_path).parent / module_specifier).as_posix())
        return _resolve_existing_import_path(
            root_path,
            _ordered_import_candidate_paths(base_path, module_specifier),
        )

    alias_path, alias_handled = _resolve_typescript_alias_rule(
        root_path,
        module_specifier,
        lambda candidate_paths: _resolve_existing_import_path(root_path, candidate_paths),
    )
    if alias_handled:
        return alias_path

    for prefix in ("@app/", "~/", "@/"):
        if not module_specifier.startswith(prefix):
            continue
        suffix = module_specifier[len(prefix):]
        direct = _resolve_existing_import_path(
            root_path,
            _ordered_import_candidate_paths(suffix, suffix),
        )
        if direct:
            return direct
        matches = sorted({
            path.relative_to(root_path).as_posix()
            for candidate in _ordered_import_candidate_paths(suffix, suffix)
            for path in root_path.rglob(Path(candidate).name)
            if path.is_file() and path.relative_to(root_path).as_posix().endswith(candidate)
        })
        if len(matches) == 1:
            return matches[0]
        return None

    return None


@lru_cache(maxsize=512)
def _read_project_text(root_path_str: str, rel_path: str) -> str:
    """Read a project-relative file as UTF-8 with replacement."""
    try:
        return (Path(root_path_str) / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _default_export_names_from_source_text(source_text: str) -> set[str]:
    """Return named symbols that are exported as default from a TS/JS module."""
    names = set(re.findall(
        r"\bexport\s+default\s+(?:abstract\s+)?(?:class|function)\s+([A-Za-z_$][\w$]*)",
        source_text,
    ))
    names.update(re.findall(
        r"\bexport\s+default\s+([A-Za-z_$][\w$]*)\s*(?:;|$)",
        source_text,
        flags=re.MULTILINE,
    ))
    names.update(re.findall(
        r"\bexport\s*\{\s*([A-Za-z_$][\w$]*)\s+as\s+default\s*\}",
        source_text,
    ))
    return names


@lru_cache(maxsize=512)
def _default_export_names_for_project_file(root_path_str: str, rel_path: str) -> tuple[str, ...]:
    """Return default-exported symbol names for a project-relative file."""
    return tuple(sorted(_default_export_names_from_source_text(
        _read_project_text(root_path_str, rel_path),
    )))


def _object_string_property(text: str, key: str) -> str | None:
    """Extract a string literal object property from a source snippet."""
    match = re.search(rf"\b{re.escape(key)}\s*:\s*(['\"`])([^'\"`]+)\1", text)
    if match:
        return match.group(2)
    return None


def _decorated_class_block(
    source_text: str,
    decorator_names: str | set[str],
    class_name: str,
) -> str | None:
    """Return a decorator/class block for a named TS class when present."""
    if isinstance(decorator_names, str):
        names = {decorator_names}
    else:
        names = decorator_names
    if not names:
        return None
    decorator_pattern = "|".join(re.escape(name) for name in sorted(names))
    match = re.search(
        rf"@(?:{decorator_pattern})(?:\s*\([\s\S]*?\))?\s*"
        rf"(?:export\s+)?class\s+{re.escape(class_name)}\b[\s\S]*?(?=(?:\n@|\n(?:export\s+)?class\s+\w|\Z))",
        source_text,
    )
    if match:
        return match.group(0)
    return None


def _mongoose_collection_for_entity(
    source_text: str, entity_name: str, schema_names: set[str] | None = None
) -> str | None:
    """Recover a Mongoose collection name from a @Schema-decorated class."""
    block = _decorated_class_block(source_text, schema_names or {"Schema"}, entity_name)
    if not block:
        return None
    return _object_string_property(block, "collection")


def _mikroorm_table_for_entity(
    source_text: str, entity_name: str, entity_names: set[str] | None = None
) -> str | None:
    """Recover a MikroORM table name from an @Entity-decorated class."""
    block = _decorated_class_block(source_text, entity_names or {"Entity"}, entity_name)
    if not block:
        return None
    return _object_string_property(block, "tableName")


def _mikroorm_fields_for_entity(
    source_text: str,
    entity_name: str,
    entity_decorator_names: set[str] | None = None,
) -> list[dict[str, str]]:
    """Recover MikroORM property metadata from an @Entity-decorated class body."""
    block = _decorated_class_block(
        source_text,
        entity_decorator_names or {"Entity"},
        entity_name,
    )
    if not block:
        return []

    fields: list[dict[str, str]] = []
    pattern = re.compile(
        r"@(?P<decorator>PrimaryKey|Property)(?:\s*\((?P<args>[\s\S]*?)\))?\s*"
        r"(?:(?:public|private|protected|readonly|static)\s+)*"
        r"(?P<name>[A-Za-z_$][\w$]*)\s*[!?]?\s*(?::[^;]+)?;",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(block):
        decorator = match.group("decorator")
        args = match.group("args") or ""
        field_name = _object_string_property(args, "fieldName") or match.group("name")
        fields.append({
            "name": match.group("name"),
            "field_name": field_name,
            "kind": "primary_key" if decorator == "PrimaryKey" else "property",
        })
    return fields


def _extract_module_call_entities(
    text: str,
    module_names: str | set[str],
    method_name: str,
    nested_key: str | None = None,
) -> list[str]:
    """Recover entity identifiers from `Module.method(...)` imports entries."""
    entries = _extract_object_array_entries(text, "imports")
    if not entries:
        return []

    if isinstance(module_names, str):
        local_module_names = {module_names}
    else:
        local_module_names = module_names

    names: list[str] = []
    for entry in entries:
        entry = entry.strip()
        matched_module_name = next(
            (module_name for module_name in local_module_names if entry.startswith(f"{module_name}.{method_name}")),
            None,
        )
        if matched_module_name is None:
            continue

        target = entry
        if nested_key:
            nested_values = _extract_object_array_entries(entry, nested_key)
            if not nested_values:
                continue
            target = "[" + ", ".join(nested_values) + "]"

        for ident in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", target):
            if ident != matched_module_name:
                names.append(ident)

    return sorted(set(names))


def _extract_mikroorm_module_entities(
    text: str,
    local_module_names: set[str],
) -> tuple[list[str], list[str]]:
    """Recover MikroORM root/feature entity lists from Nest module imports."""
    if not local_module_names:
        return [], []

    module_pattern = "|".join(re.escape(name) for name in sorted(local_module_names))
    root_entities: set[str] = set()
    feature_entities: set[str] = set()

    for match in re.finditer(
        rf"\b(?:{module_pattern})\.forRoot\s*\(\s*\{{([\s\S]*?)\}}\s*\)",
        text,
    ):
        body = match.group(1)
        entity_arrays = _extract_object_array_entries("{" + body + "}", "entities")
        for entry in entity_arrays:
            for ident in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", entry):
                root_entities.add(ident)

    for match in re.finditer(
        rf"\b(?:{module_pattern})\.forFeature\s*\(\s*\[([^\]]*)\]",
        text,
    ):
        for ident in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", match.group(1)):
            feature_entities.add(ident)

    return sorted(root_entities), sorted(feature_entities)


def _extract_resolver_type(decorator_text: str) -> str | None:
    """Recover a Nest GraphQL resolver root type from @Resolver(...) text."""
    quoted = _quoted_strings(decorator_text)
    if quoted:
        return quoted[0]

    array_arrow_match = re.search(r"=>\s*\[\s*([A-Za-z_$][\w$]*)", decorator_text)
    if array_arrow_match:
        return array_arrow_match.group(1)

    arrow_match = re.search(r"=>\s*([A-Za-z_$][\w$]*)", decorator_text)
    if arrow_match:
        return arrow_match.group(1)
    return None


def _resolver_context_for_method(
    node: Node,
    nest_graphql_imports: dict[str, str],
) -> tuple[bool, str | None]:
    """Return whether a method is inside a Nest resolver class and its resolver type."""
    class_node = _enclosing_class_node(node)
    if class_node is None:
        return False, None

    for class_decorator in _leading_decorators(class_node):
        local_name = _decorator_parts(class_decorator)[0]
        if local_name and nest_graphql_imports.get(local_name) == "Resolver":
            return True, _extract_resolver_type(
                class_decorator.text.decode("utf-8", errors="replace")
            )
    return False, None


def _build_config_factory_overrides(symbol_name: str, value_text: str) -> dict[str, object] | None:
    """Build Nest config metadata for registerAs(...) factories."""
    match = re.search(r"\bregisterAs\s*\(\s*(['\"`])([^'\"`]+)\1", value_text)
    if not match:
        return None

    namespace = match.group(2)
    return {
        "kind": "config",
        "signature": f"Config {namespace}",
        "doc_comment": f"Nest config factory for {namespace}.",
        "metadata": {
            "framework": "nestjs",
            "resource": "config",
            "config_namespace": namespace,
            "factory_name": symbol_name,
        },
    }


def _normalize_transport_name(value: str | None) -> str | None:
    """Normalize transport identifiers to search-friendly lowercase labels."""
    if not value:
        return None
    normalized = value.strip().split(".")[-1].strip().lower()
    aliases = {
        "rabbitmq": "rmq",
    }
    return aliases.get(normalized, normalized)


def _transport_name_from_member_access(text: str, local_names: set[str]) -> str | None:
    """Extract `Transport.X` style member access from a source snippet."""
    for local_name in sorted(local_names):
        match = re.search(
            rf"\b{re.escape(local_name)}\s*\.\s*([A-Za-z_$][\w$]*)",
            text,
        )
        if match:
            return _normalize_transport_name(match.group(1))
    return None


def _all_transport_names_from_member_access(text: str, local_names: set[str]) -> list[str]:
    """Extract all distinct `Transport.X` style names from a source snippet."""
    names: list[str] = []
    for local_name in sorted(local_names):
        for match in re.finditer(
            rf"\b{re.escape(local_name)}\s*\.\s*([A-Za-z_$][\w$]*)",
            text,
        ):
            transport_name = _normalize_transport_name(match.group(1))
            if transport_name and transport_name not in names:
                names.append(transport_name)
    return names


def _transport_name_from_server_assignment(text: str, local_names: set[str]) -> str | None:
    """Extract direct server-oriented transport assignments from a class body."""
    for local_name in sorted(local_names):
        match = re.search(
            rf"\b(?:server|consumer|listener|handler|microservice)[A-Za-z_$]*Transport\b"
            rf"\s*=\s*{re.escape(local_name)}\s*\.\s*([A-Za-z_$][\w$]*)",
            text,
        )
        if match:
            return _normalize_transport_name(match.group(1))
    return None


def _exported_transport_constant_matches(
    source_text: str,
) -> list[tuple[str, str, int, int]]:
    """Find exported TS/JS transport constants like `export const foo = Transport.RMQ;`."""
    nest_microservice_imports = _imported_name_map_from_module(source_text, "@nestjs/microservices")
    transport_local_names = _local_names_for_import(nest_microservice_imports, "Transport")
    if not transport_local_names:
        return []

    local_name_pattern = "|".join(re.escape(name) for name in sorted(transport_local_names))
    matches: list[tuple[str, str, int, int]] = []
    for match in re.finditer(
        rf"^\s*export\s+const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:{local_name_pattern})\s*\.\s*([A-Za-z_$][\w$]*)\s*;?\s*$",
        source_text,
        flags=re.MULTILINE,
    ):
        transport_name = _normalize_transport_name(match.group(2))
        if transport_name is None:
            continue
        matches.append((match.group(1), transport_name, match.start(), match.end()))
    return matches


def _decorator_argument_snippets(decorator_text: str) -> list[str]:
    """Split a decorator call into top-level argument snippets."""
    open_paren = decorator_text.find("(")
    close_paren = decorator_text.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren <= open_paren:
        return []

    text = decorator_text[open_paren + 1:close_paren]
    if not text.strip():
        return []

    args: list[str] = []
    depth_brace = 0
    depth_bracket = 0
    depth_paren = 0
    in_string: str | None = None
    escape = False
    start = 0

    for i, ch in enumerate(text):
        if in_string is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue

        if ch in {"'", '"', "`"}:
            in_string = ch
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            depth_brace -= 1
            continue
        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket -= 1
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren -= 1
            continue
        if ch == "," and depth_brace == 0 and depth_bracket == 0 and depth_paren == 0:
            arg = text[start:i].strip()
            if arg:
                args.append(arg)
            start = i + 1

    last_arg = text[start:].strip()
    if last_arg:
        args.append(last_arg)
    return args


def _string_literal_value(text: str) -> str | None:
    """Extract a plain string literal value from a snippet."""
    match = re.fullmatch(r"\s*(['\"`])((?:\\.|(?!\1).)*)\1\s*", text, flags=re.DOTALL)
    if match:
        return match.group(2)
    return None


def _integer_literal_value(text: str) -> int | None:
    """Extract an integer literal value from a snippet."""
    match = re.fullmatch(r"\s*(\d+)\s*", text)
    if match:
        return int(match.group(1))
    return None


def _microservice_pattern_from_decorator(
    canonical_name: str,
    decorator_text: str,
) -> str | None:
    """Extract semantic microservice pattern names from decorator arguments."""
    args = _decorator_argument_snippets(decorator_text)
    if not args:
        return None

    first_arg = args[0]
    key_order = {
        "MessagePattern": ("cmd", "pattern", "message", "topic", "event"),
        "EventPattern": ("event", "pattern", "topic", "cmd", "message"),
    }.get(canonical_name, ("pattern", "cmd", "event", "topic", "message"))

    for key in key_order:
        value = _object_string_property(first_arg, key)
        if value:
            return value

    return _string_literal_value(first_arg)


def _typescript_function_declaration_nodes(source_text: str) -> list[Node]:
    """Return top-level TS function declarations, including exported ones."""
    parser = Parser(get_language("typescript"))
    tree = parser.parse(source_text.encode("utf-8"))
    root = tree.root_node
    functions: list[Node] = []

    for child in root.named_children:
        function_node = child
        if child.type == "export_statement":
            function_node = next(
                (named for named in child.named_children if named.type == "function_declaration"),
                child,
            )
        if function_node.type == "function_declaration":
            functions.append(function_node)

    return functions


def _microservice_decorator_target_from_call(
    call_node: Node,
    parameter_name: str,
    apply_decorators_local_names: set[str],
) -> str | None:
    """Resolve a narrow decorator-wrapper call target from a return call expression."""
    target_node = call_node.child_by_field_name("function")
    arguments_node = call_node.child_by_field_name("arguments")
    if target_node is None or target_node.type != "identifier" or arguments_node is None:
        return None

    args = [child for child in arguments_node.named_children if child.type != "type_arguments"]
    target_name = target_node.text.decode("utf-8", errors="replace")
    if len(args) == 1 and args[0].type == "identifier":
        argument_name = args[0].text.decode("utf-8", errors="replace")
        if argument_name == parameter_name:
            return target_name

    if target_name not in apply_decorators_local_names or len(args) != 1 or args[0].type != "call_expression":
        return None

    inner_call = args[0]
    inner_target = inner_call.child_by_field_name("function")
    inner_arguments = inner_call.child_by_field_name("arguments")
    if inner_target is None or inner_target.type != "identifier" or inner_arguments is None:
        return None

    inner_args = [child for child in inner_arguments.named_children if child.type != "type_arguments"]
    if len(inner_args) != 1 or inner_args[0].type != "object":
        return None

    if not re.search(
        rf"\b(?:cmd|pattern|message|topic|event)\s*:\s*{re.escape(parameter_name)}\b",
        inner_args[0].text.decode("utf-8", errors="replace"),
    ):
        return None

    return inner_target.text.decode("utf-8", errors="replace")


def _transparent_microservice_wrapper_targets(
    function_node: Node,
    apply_decorators_local_names: set[str] | None = None,
) -> tuple[str, str] | None:
    """Return `(target_name, parameter_name)` for narrow transparent wrapper factories."""
    name_node = function_node.child_by_field_name("name")
    parameters_node = function_node.child_by_field_name("parameters")
    body_node = function_node.child_by_field_name("body")
    if name_node is None or parameters_node is None or body_node is None:
        return None

    params = [child for child in parameters_node.named_children if child.type.endswith("parameter")]
    if len(params) != 1:
        return None

    parameter_name_node = params[0].child_by_field_name("pattern") or params[0].child_by_field_name("name")
    if parameter_name_node is None or parameter_name_node.type != "identifier":
        return None
    parameter_name = parameter_name_node.text.decode("utf-8", errors="replace")

    body_statements = body_node.named_children
    if len(body_statements) != 1 or body_statements[0].type != "return_statement":
        return None

    call_node = body_statements[0].child_by_field_name("argument")
    if call_node is None:
        call_node = next(
            (child for child in body_statements[0].named_children if child.type == "call_expression"),
            None,
        )
    if call_node is None or call_node.type != "call_expression":
        return None

    target_name = _microservice_decorator_target_from_call(
        call_node,
        parameter_name,
        apply_decorators_local_names or set(),
    )
    if target_name is None:
        return None

    return (
        target_name,
        parameter_name,
    )


def _local_microservice_decorator_wrappers(
    source_text: str,
    nest_microservice_imports: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Resolve narrow same-file wrapper factories around Nest microservice decorators."""
    cached = _cached_local_microservice_decorator_wrappers(
        source_text,
        tuple(sorted(nest_microservice_imports.items())),
        tuple(sorted(_local_names_for_import(
            _imported_name_map_from_module(source_text, "@nestjs/common"),
            "applyDecorators",
        ))),
    )
    return {
        wrapper_name: dict(metadata_items)
        for wrapper_name, metadata_items in cached
    }


@lru_cache(maxsize=256)
def _cached_local_microservice_decorator_wrappers(
    source_text: str,
    nest_microservice_import_items: tuple[tuple[str, str], ...],
    apply_decorators_import_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Cache wrapper discovery per file source to avoid repeated reparsing per symbol."""
    nest_microservice_imports = dict(nest_microservice_import_items)
    apply_decorators_local_names = set(apply_decorators_import_names)
    resolved: dict[str, dict[str, str]] = {}
    pending: dict[str, str] = {}

    for function_node in _typescript_function_declaration_nodes(source_text):
        name_node = function_node.child_by_field_name("name")
        wrapper_name = (
            name_node.text.decode("utf-8", errors="replace")
            if name_node is not None else
            None
        )
        if not wrapper_name:
            continue
        target = _transparent_microservice_wrapper_targets(
            function_node,
            apply_decorators_local_names,
        )
        if target is None:
            continue
        target_name, _parameter_name = target
        canonical_name = nest_microservice_imports.get(target_name)
        if canonical_name in _NEST_MICROSERVICE_DECORATORS:
            resolved[wrapper_name] = {
                "canonical_decorator": canonical_name,
                "async_kind": "microservice_handler",
                "pattern_metadata_key": _NEST_MICROSERVICE_DECORATORS[canonical_name],
            }
            continue
        pending[wrapper_name] = target_name

    changed = True
    while changed and pending:
        changed = False
        for wrapper_name, target_name in list(pending.items()):
            target_wrapper = resolved.get(target_name)
            canonical_name = (
                target_wrapper.get("canonical_decorator")
                if isinstance(target_wrapper, dict) else
                None
            )
            if canonical_name in _NEST_MICROSERVICE_DECORATORS:
                resolved[wrapper_name] = dict(target_wrapper)
                pending.pop(wrapper_name)
                changed = True

    return tuple(
        (wrapper_name, tuple(sorted(metadata.items())))
        for wrapper_name, metadata in sorted(resolved.items())
    )


def _imported_microservice_decorator_wrappers(
    root_path: Path | None,
    source_file_path: str,
    source_text: str,
) -> dict[str, dict[str, str]]:
    """Resolve directly imported project-local microservice wrapper decorators."""
    if root_path is None:
        return {}

    cached = _cached_imported_microservice_decorator_wrappers(
        root_path.as_posix(),
        source_file_path,
        source_text,
    )
    return {
        wrapper_name: dict(metadata_items)
        for wrapper_name, metadata_items in cached
    }


@lru_cache(maxsize=512)
def _cached_imported_microservice_decorator_wrappers(
    root_path_str: str,
    source_file_path: str,
    source_text: str,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Cache imported wrapper discovery per file and skip unrelated imports."""
    root_path = Path(root_path_str)
    used_decorator_names = set(re.findall(r"@([A-Za-z_$][\w$]*)", source_text))
    if not used_decorator_names:
        return ()

    resolved: dict[str, dict[str, str]] = {}
    for local_name, (imported_name, module_specifier, import_kind) in _typescript_import_bindings(source_text).items():
        if local_name not in used_decorator_names:
            continue
        import_path = _resolve_typescript_import_path(root_path, source_file_path, module_specifier)
        if import_path is None:
            continue
        imported_text = _read_project_text(root_path.as_posix(), import_path)
        if not imported_text:
            continue
        imported_wrappers = _local_microservice_decorator_wrappers(
            imported_text,
            _imported_name_map_from_module(imported_text, "@nestjs/microservices"),
        )
        if import_kind == "named":
            canonical_import_name = imported_name or local_name
            wrapper_metadata = imported_wrappers.get(canonical_import_name)
            if isinstance(wrapper_metadata, dict):
                resolved[local_name] = dict(wrapper_metadata)
            continue
        if import_kind == "default":
            default_export_names = set(_default_export_names_for_project_file(
                root_path.as_posix(),
                import_path,
            ))
            if len(default_export_names) != 1:
                continue
            default_export_name = next(iter(default_export_names))
            wrapper_metadata = imported_wrappers.get(default_export_name)
            if isinstance(wrapper_metadata, dict):
                resolved[local_name] = dict(wrapper_metadata)
    return tuple(
        (wrapper_name, tuple(sorted(metadata.items())))
        for wrapper_name, metadata in sorted(resolved.items())
    )


def _scheduler_metadata_from_decorator(
    canonical_name: str,
    decorator_text: str,
) -> dict[str, int | str]:
    """Extract scheduler name/timing metadata from decorator arguments."""
    args = _decorator_argument_snippets(decorator_text)
    if canonical_name == "Cron":
        cron = _string_literal_value(args[0]) if args else None
        return {"cron": cron} if cron is not None else {}

    if canonical_name == "Interval":
        interval_name = _string_literal_value(args[0]) if args else None
        every_ms = _integer_literal_value(args[1]) if len(args) > 1 else None
        metadata: dict[str, int | str] = {}
        if interval_name is not None:
            metadata["interval_name"] = interval_name
        if every_ms is not None:
            metadata["every_ms"] = every_ms
        return metadata

    timeout_name = _string_literal_value(args[0]) if args else None
    delay_ms = _integer_literal_value(args[1]) if len(args) > 1 else None
    metadata = {}
    if timeout_name is not None:
        metadata["timeout_name"] = timeout_name
    if delay_ms is not None:
        metadata["delay_ms"] = delay_ms
    return metadata


def _first_number_literal(text: str) -> int | None:
    """Extract the first integer literal from a snippet."""
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _transport_name_for_async_handler(
    source_text: str,
    node: Node,
    transport_local_names: set[str],
) -> str | None:
    """Infer transport for a handler from local context before falling back to file scope."""
    if not transport_local_names:
        return None

    method_text = node.text.decode("utf-8", errors="replace")
    transport = _transport_name_from_member_access(method_text, transport_local_names)
    if transport is not None:
        return transport

    class_node = _enclosing_class_node(node)
    if class_node is not None:
        class_text = class_node.text.decode("utf-8", errors="replace")
        transport = _transport_name_from_server_assignment(class_text, transport_local_names)
        if transport is not None:
            return transport

    file_transports = _all_transport_names_from_member_access(source_text, transport_local_names)
    if len(file_transports) == 1:
        return file_transports[0]
    return None


def _queue_processor_context_for_method(
    node: Node,
    processor_local_names: set[str],
) -> tuple[str | None, str | None]:
    """Return queue metadata for a method inside a Bull/BullMQ @Processor class."""
    class_node = _enclosing_class_node(node)
    if class_node is None:
        return None, None

    for decorator in _leading_decorators(class_node):
        local_name, args = _decorator_parts(decorator)
        if local_name in processor_local_names:
            return args[0] if args else None, local_name
    return None, None


def _build_typescript_async_symbol_overrides(
    rel_path: str,
    source_text: str,
    def_node: Node,
    kind: str,
    symbol_name: str | None,
    root_path: Path | None = None,
) -> dict[str, object] | None:
    """Build async-system overrides for TS microservices, queues, and transports."""
    if symbol_name is None:
        return None

    nest_microservice_imports = _imported_name_map_from_module(source_text, "@nestjs/microservices")
    nest_schedule_imports = _imported_name_map_from_module(source_text, "@nestjs/schedule")
    nest_bull_imports = _imported_name_map_from_module(source_text, "@nestjs/bull")
    nest_bullmq_imports = _imported_name_map_from_module(source_text, "@nestjs/bullmq")
    bullmq_imports = _imported_name_map_from_module(source_text, "bullmq")
    redis_imports = _imported_name_map_from_module(source_text, "redis")
    local_microservice_wrappers = _local_microservice_decorator_wrappers(
        source_text,
        nest_microservice_imports,
    )
    imported_microservice_wrappers = _imported_microservice_decorator_wrappers(
        root_path,
        rel_path,
        source_text,
    )
    transport_local_names = _local_names_for_import(nest_microservice_imports, "Transport")
    processor_local_names = (
        _local_names_for_import(nest_bull_imports, "Processor")
        | _local_names_for_import(nest_bullmq_imports, "Processor")
    )
    process_local_names = (
        _local_names_for_import(nest_bull_imports, "Process")
        | _local_names_for_import(nest_bullmq_imports, "Process")
    )
    bullmq_processor_local_names = _local_names_for_import(nest_bullmq_imports, "Processor")
    worker_local_names = _local_names_for_import(bullmq_imports, "Worker")
    create_client_local_names = _local_names_for_import(redis_imports, "createClient")

    if kind == "class":
        for decorator in _leading_decorators(def_node):
            local_name, args = _decorator_parts(decorator)
            if local_name in processor_local_names:
                queue_name = args[0] if args else None
                framework = "bullmq" if local_name in bullmq_processor_local_names else "bull"
                signature = f"Queue processor {queue_name}" if queue_name else "Queue processor"
                return {
                    "kind": "queue_processor",
                    "signature": signature,
                    "doc_comment": (
                        f"Queue processor for {queue_name}."
                        if queue_name else
                        f"Queue processor {symbol_name}."
                    ),
                    "metadata": {
                        "framework": framework,
                        "resource": "queue_processor",
                        "queue": queue_name,
                        "queue_name": queue_name,
                        "role": "consumer",
                    },
                }

    if kind == "method":
        for decorator in _leading_decorators(def_node):
            local_name, args = _decorator_parts(decorator)
            canonical_microservice_name = (
                nest_microservice_imports.get(local_name)
                if local_name is not None else
                None
            )
            wrapper_metadata = (
                local_microservice_wrappers.get(local_name)
                if local_name is not None else
                None
            )
            if not isinstance(wrapper_metadata, dict) and local_name is not None:
                wrapper_metadata = imported_microservice_wrappers.get(local_name)
            if canonical_microservice_name is None and local_name is not None:
                canonical_microservice_name = (
                    wrapper_metadata.get("canonical_decorator")
                    if isinstance(wrapper_metadata, dict) else
                    None
                )
            if canonical_microservice_name in _NEST_MICROSERVICE_DECORATORS:
                decorator_text = decorator.text.decode("utf-8", errors="replace")
                pattern = _microservice_pattern_from_decorator(
                    canonical_microservice_name,
                    decorator_text,
                )
                metadata_key = _NEST_MICROSERVICE_DECORATORS[canonical_microservice_name]
                signature = f"{canonical_microservice_name} {pattern}".strip()
                metadata = {
                    "framework": "nestjs",
                    "resource": "microservice_handler",
                    "pattern": pattern,
                    metadata_key: pattern,
                    "transport": _transport_name_for_async_handler(
                        source_text,
                        def_node,
                        transport_local_names,
                    ),
                    "role": "consumer",
                }
                return {
                    "kind": "microservice_handler",
                    "signature": signature,
                    "doc_comment": f"Nest microservice handler {pattern}.",
                    "metadata": {key: value for key, value in metadata.items() if value is not None},
                }

            canonical_schedule_name = (
                nest_schedule_imports.get(local_name)
                if local_name is not None else
                None
            )
            if canonical_schedule_name in _NEST_SCHEDULE_DECORATORS:
                decorator_text = decorator.text.decode("utf-8", errors="replace")
                if canonical_schedule_name == "Cron":
                    cron = _scheduler_metadata_from_decorator(
                        canonical_schedule_name,
                        decorator_text,
                    ).get("cron")
                    return {
                        "kind": "scheduled_job",
                        "signature": f"Cron {cron}".strip(),
                        "doc_comment": f"Scheduled cron job {cron}.",
                        "metadata": {
                            "framework": "nestjs",
                            "resource": "scheduled_job",
                            "schedule_type": "cron",
                            "cron": cron,
                        },
                    }
                if canonical_schedule_name == "Interval":
                    schedule_metadata = _scheduler_metadata_from_decorator(
                        canonical_schedule_name,
                        decorator_text,
                    )
                    interval_name = schedule_metadata.get("interval_name")
                    return {
                        "kind": "scheduled_job",
                        "signature": f"Interval {interval_name or symbol_name}".strip(),
                        "doc_comment": f"Scheduled interval job {interval_name or symbol_name}.",
                        "metadata": {
                            "framework": "nestjs",
                            "resource": "scheduled_job",
                            "schedule_type": "interval",
                            **schedule_metadata,
                        },
                    }
                schedule_metadata = _scheduler_metadata_from_decorator(
                    canonical_schedule_name,
                    decorator_text,
                )
                timeout_name = schedule_metadata.get("timeout_name")
                return {
                    "kind": "scheduled_job",
                    "signature": f"Timeout {timeout_name or symbol_name}".strip(),
                    "doc_comment": f"Scheduled timeout job {timeout_name or symbol_name}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "scheduled_job",
                        "schedule_type": "timeout",
                        **schedule_metadata,
                    },
                }

            if local_name in process_local_names:
                queue_name, _processor_local_name = _queue_processor_context_for_method(
                    def_node,
                    processor_local_names,
                )
                job_name = args[0] if args else None
                framework = (
                    "bullmq"
                    if _processor_local_name in bullmq_processor_local_names else
                    "bull"
                )
                metadata = {
                    "framework": framework,
                    "resource": "queue_processor",
                    "queue": queue_name,
                    "queue_name": queue_name,
                    "job_name": job_name,
                    "role": "consumer",
                }
                return {
                    "kind": "queue_processor",
                    "signature": f"Queue processor {queue_name or ''} {job_name or ''}".strip(),
                    "doc_comment": (
                        f"Queue processor for {queue_name} {job_name}."
                        if queue_name or job_name else
                        f"Queue processor {symbol_name}."
                    ),
                    "metadata": {key: value for key, value in metadata.items() if value is not None},
                }

    if kind == "function":
        wrapper_metadata = local_microservice_wrappers.get(symbol_name)
        if isinstance(wrapper_metadata, dict):
            return {
                "kind": "function",
                "metadata": {
                    "framework": "nestjs",
                    "resource": "decorator_wrapper",
                    **wrapper_metadata,
                },
            }

        amqplib_default = _default_import_name_from_module(source_text, "amqplib")
        amqplib_namespace = _namespace_import_name_from_module(source_text, "amqplib")
        function_text = def_node.text.decode("utf-8", errors="replace")
        amqplib_names = {name for name in (amqplib_default, amqplib_namespace) if name}
        uses_amqplib = any(
            re.search(rf"\b{re.escape(local_name)}\s*\.\s*connect\s*\(", function_text)
            for local_name in amqplib_names
        )
        if uses_amqplib and re.search(r"\.\s*consume\s*\(", function_text):
            queue_name = None
            consume_match = re.search(
                r"\.\s*consume\s*\(\s*(['\"`])([^'\"`]+)\1",
                function_text,
            )
            if consume_match:
                queue_name = consume_match.group(2)
            metadata = {
                "framework": "amqplib",
                "resource": "microservice_handler",
                "queue": queue_name,
                "queue_name": queue_name,
                "transport": "rmq",
                "role": "consumer",
            }
            return {
                "kind": "microservice_handler",
                "signature": f"RabbitMQ consumer {queue_name or symbol_name}".strip(),
                "doc_comment": f"RabbitMQ consumer for {queue_name or symbol_name}.",
                "metadata": {key: value for key, value in metadata.items() if value is not None},
            }

    if def_node.type != "variable_declarator":
        return None

    value_node = def_node.child_by_field_name("value")
    if value_node is None:
        return None
    value_text = value_node.text.decode("utf-8", errors="replace")

    client_factory_local_names = _local_names_for_import(nest_microservice_imports, "ClientProxyFactory")
    if any(
        re.search(rf"\b{re.escape(local_name)}\s*\.\s*create\s*\(", value_text)
        for local_name in client_factory_local_names
    ):
        transport = _transport_name_from_member_access(value_text, transport_local_names)
        queue_name = _object_string_property(value_text, "queue")
        metadata = {
            "framework": "nestjs",
            "resource": "transport",
            "transport": transport,
            "queue": queue_name,
            "queue_name": queue_name,
            "role": "producer",
        }
        return {
            "kind": "transport",
            "signature": f"Client transport {transport or symbol_name}".strip(),
            "doc_comment": f"Client transport for {queue_name or transport or symbol_name}.",
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    if any(re.search(rf"\bnew\s+{re.escape(local_name)}\s*\(", value_text) for local_name in worker_local_names):
        queue_names = _quoted_strings(value_text)
        queue_name = queue_names[0] if queue_names else None
        metadata = {
            "framework": "bullmq",
            "resource": "queue_processor",
            "queue": queue_name,
            "queue_name": queue_name,
            "role": "consumer",
        }
        return {
            "kind": "queue_processor",
            "signature": f"BullMQ worker {queue_name or symbol_name}".strip(),
            "doc_comment": f"BullMQ worker for {queue_name or symbol_name}.",
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    if any(re.search(rf"\b{re.escape(local_name)}\s*\(", value_text) for local_name in create_client_local_names):
        connection_url = None
        url_match = re.search(r"\burl\s*:\s*([^,}\n]+)", value_text)
        if url_match:
            connection_url = url_match.group(1).strip()
        metadata = {
            "framework": "redis",
            "resource": "transport",
            "transport": "redis",
            "connection_url": connection_url,
            "role": "client",
        }
        return {
            "kind": "transport",
            "signature": "Redis client",
            "doc_comment": f"Redis client {symbol_name}.",
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    transport_name = _transport_name_from_member_access(value_text, transport_local_names)
    if transport_name is not None:
        return {
            "kind": "transport",
            "signature": f"Transport {transport_name}",
            "doc_comment": f"Transport configuration for {transport_name}.",
            "metadata": {
                "framework": "nestjs",
                "resource": "transport",
                "transport": transport_name,
            },
        }

    return None


def _extract_service_config_refs(
    class_text: str,
    inject_local_names: set[str],
    config_type_local_names: set[str],
) -> list[str]:
    """Recover config factory identifiers from structured Nest service injections."""
    refs: list[str] = []
    for inject_name in inject_local_names:
        refs.extend(re.findall(
            rf"@\s*{re.escape(inject_name)}\s*\(\s*([A-Za-z_$][\w$]*)\.KEY\s*\)",
            class_text,
        ))
    for config_type_name in config_type_local_names:
        refs.extend(re.findall(
            rf"\b{re.escape(config_type_name)}\s*<\s*typeof\s+([A-Za-z_$][\w$]*)\s*>",
            class_text,
        ))
    return sorted(set(refs))


def _build_nest_symbol_overrides(
    source_text: str,
    def_node: Node,
    kind: str,
    symbol_name: str | None,
) -> dict[str, object] | None:
    """Build NestJS-specific overrides for classes, methods, and bootstrap functions."""
    decorators = _leading_decorators(def_node)
    decorator_parts = [_decorator_parts(node) for node in decorators]
    decorator_names = {name for name, _args in decorator_parts if name}
    nest_common_imports = _imported_name_map_from_module(source_text, "@nestjs/common")
    nest_config_imports = _imported_name_map_from_module(source_text, "@nestjs/config")
    nest_graphql_imports = _imported_name_map_from_module(source_text, "@nestjs/graphql")
    mikroorm_nest_imports = _imported_name_map_from_module(source_text, "@mikro-orm/nestjs")

    if kind == "class":
        for decorator_name, args in decorator_parts:
            if decorator_name == "Controller":
                controller_path = _normalize_route_path(None, args[0] if args else None)
                return {
                    "kind": "controller",
                    "signature": f"Controller {controller_path}",
                    "doc_comment": f"Nest controller for {controller_path}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "controller",
                        "controller_path": controller_path,
                    },
                }
            if decorator_name == "Module":
                decorator_text = next(
                    (
                        node.text.decode("utf-8", errors="replace")
                        for node in decorators
                        if _decorator_parts(node)[0] == "Module"
                    ),
                    "",
                )
                imports = _extract_module_names_from_array(decorator_text, "imports")
                providers = _extract_identifiers_from_array(decorator_text, "providers")
                controllers = _extract_identifiers_from_array(decorator_text, "controllers")
                exports = _extract_identifiers_from_array(decorator_text, "exports")
                config_refs = _extract_config_refs_from_module_imports(
                    decorator_text,
                    _local_names_for_import(nest_config_imports, "ConfigModule"),
                )
                mikroorm_root_entities, mikroorm_feature_entities = _extract_mikroorm_module_entities(
                    decorator_text,
                    _local_names_for_import(mikroorm_nest_imports, "MikroOrmModule"),
                )
                return {
                    "kind": "module",
                    "signature": f"Nest module {symbol_name or ''}".strip(),
                    "doc_comment": f"Nest module {symbol_name or ''}.".strip(),
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "module",
                        "imports": imports,
                        "providers": providers,
                        "controllers": controllers,
                        "exports": exports,
                        "config_refs": config_refs,
                        "mikroorm_root_entities": mikroorm_root_entities,
                        "mikroorm_feature_entities": mikroorm_feature_entities,
                    },
                }
            if decorator_name and nest_graphql_imports.get(decorator_name) == "Resolver":
                decorator_text = next(
                    (
                        node.text.decode("utf-8", errors="replace")
                        for node in decorators
                        if _decorator_parts(node)[0] == decorator_name
                    ),
                    "",
                )
                resolver_type = _extract_resolver_type(decorator_text)
                signature = f"Resolver {resolver_type}" if resolver_type else "Resolver"
                return {
                    "kind": "resolver",
                    "signature": signature,
                    "doc_comment": (
                        f"Nest GraphQL resolver for {resolver_type}."
                        if resolver_type else
                        "Nest GraphQL resolver."
                    ),
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "resolver",
                        "resolver_type": resolver_type,
                    },
                }
            if decorator_name == "Catch":
                return {
                    "kind": "filter",
                    "signature": f"Filter {symbol_name}",
                    "doc_comment": f"Nest exception filter {symbol_name}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "filter",
                    },
                }

        class_text = def_node.text.decode("utf-8", errors="replace")
        if symbol_name:
            if symbol_name.endswith("Guard") or "implements CanActivate" in class_text:
                return {
                    "kind": "guard",
                    "signature": f"Guard {symbol_name}",
                    "doc_comment": f"Nest guard {symbol_name}.",
                    "metadata": {"framework": "nestjs", "resource": "guard"},
                }
            if symbol_name.endswith("Pipe") or "implements PipeTransform" in class_text:
                return {
                    "kind": "pipe",
                    "signature": f"Pipe {symbol_name}",
                    "doc_comment": f"Nest pipe {symbol_name}.",
                    "metadata": {"framework": "nestjs", "resource": "pipe"},
                }
            if symbol_name.endswith("Interceptor") or "implements NestInterceptor" in class_text:
                return {
                    "kind": "interceptor",
                    "signature": f"Interceptor {symbol_name}",
                    "doc_comment": f"Nest interceptor {symbol_name}.",
                    "metadata": {"framework": "nestjs", "resource": "interceptor"},
                }
            if symbol_name.endswith("Middleware") or "implements NestMiddleware" in class_text:
                return {
                    "kind": "middleware",
                    "signature": f"Middleware {symbol_name}",
                    "doc_comment": f"Nest middleware {symbol_name}.",
                    "metadata": {"framework": "nestjs", "resource": "middleware"},
                }
            if "Injectable" in decorator_names:
                config_refs = _extract_service_config_refs(
                    class_text,
                    _local_names_for_import(nest_common_imports, "Inject"),
                    _local_names_for_import(nest_config_imports, "ConfigType"),
                )
                return {
                    "kind": "service",
                    "signature": f"Service {symbol_name}",
                    "doc_comment": f"Nest service {symbol_name}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "service",
                        "config_refs": config_refs,
                    },
                }

    if kind == "method":
        for decorator in decorators:
            decorator_name, args = _decorator_parts(decorator)
            if decorator_name in _NEST_HTTP_DECORATORS:
                controller_path = _controller_prefix_for(def_node)
                route_path = _normalize_route_path(controller_path, args[0] if args else None)
                method = _NEST_HTTP_DECORATORS[decorator_name]
                return {
                    "kind": "route_handler",
                    "signature": f"{method} {route_path}",
                    "doc_comment": f"Nest route handler {method} {route_path}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": "route_handler",
                        "controller_path": controller_path,
                        "http_method": method,
                        "route_path": route_path,
                    },
                }
            canonical_graphql_name = (
                nest_graphql_imports.get(decorator_name)
                if decorator_name is not None else
                None
            )
            if canonical_graphql_name in _NEST_GRAPHQL_DECORATORS:
                decorator_text = decorator.text.decode("utf-8", errors="replace")
                graphql_kind = _NEST_GRAPHQL_DECORATORS[canonical_graphql_name]
                has_resolver_context, resolver_type = _resolver_context_for_method(
                    def_node,
                    nest_graphql_imports,
                )
                if not has_resolver_context:
                    continue
                field_name = _extract_graphql_field_name(decorator_text, symbol_name or decorator_name)
                signature = f"{canonical_graphql_name} {field_name}"
                return {
                    "kind": graphql_kind,
                    "signature": signature,
                    "doc_comment": f"Nest GraphQL {graphql_kind} {field_name}.",
                    "metadata": {
                        "framework": "nestjs",
                        "resource": graphql_kind,
                        "graphql_field": field_name,
                        "resolver_type": resolver_type,
                    },
                }

    if kind == "function":
        body_text = def_node.text.decode("utf-8", errors="replace")
        match = re.search(r"\bNestFactory\.create\s*\(\s*([A-Za-z_$][\w$]*)", body_text)
        if match:
            root_module = match.group(1)
            return {
                "kind": "bootstrap",
                "signature": f"Nest bootstrap {root_module}",
                "doc_comment": f"Nest bootstrap function for {root_module}.",
                "metadata": {
                    "framework": "nestjs",
                    "resource": "bootstrap",
                    "root_module": root_module,
                },
            }

    return None


def _build_mongoose_symbol_overrides(
    source_text: str,
    def_node: Node,
    kind: str,
    symbol_name: str | None,
) -> dict[str, object] | None:
    """Build Mongoose/Nest Mongoose overrides for classes and exported constants."""
    if symbol_name is None:
        return None

    mongoose_imports = _imported_name_map_from_module(source_text, "@nestjs/mongoose")
    schema_local_names = _local_names_for_import(mongoose_imports, "Schema")
    schema_factory_local_names = _local_names_for_import(mongoose_imports, "SchemaFactory")
    mongoose_module_local_names = _local_names_for_import(mongoose_imports, "MongooseModule")

    if kind == "class":
        for decorator in _leading_decorators(def_node):
            decorator_name, _args = _decorator_parts(decorator)
            if decorator_name and mongoose_imports.get(decorator_name) == "Schema":
                collection_name = _mongoose_collection_for_entity(
                    source_text, symbol_name, schema_local_names
                )
                signature = f"mongoose entity | {symbol_name}"
                if collection_name:
                    signature += f" | {collection_name}"
                doc_comment = f"Mongoose entity {symbol_name}."
                if collection_name:
                    doc_comment += f" Collection: {collection_name}."
                return {
                    "kind": "entity",
                    "signature": signature,
                    "doc_comment": doc_comment,
                    "metadata": {
                        "framework": "mongoose",
                        "resource": "entity",
                        "entity_name": symbol_name,
                        "collection_name": collection_name,
                    },
                }

    if def_node.type != "variable_declarator":
        return None

    value_node = def_node.child_by_field_name("value")
    if value_node is None:
        return None
    value_text = value_node.text.decode("utf-8", errors="replace")

    schema_factory_pattern = "|".join(re.escape(name) for name in sorted(schema_factory_local_names))
    schema_factory_match = (
        re.search(
            rf"\b(?:{schema_factory_pattern})\.createForClass\s*\(\s*([A-Za-z_$][\w$]*)\s*\)",
            value_text,
        )
        if schema_factory_pattern else
        None
    )
    if schema_factory_match:
        entity_name = schema_factory_match.group(1)
        collection_name = _mongoose_collection_for_entity(source_text, entity_name, schema_local_names)
        signature = f"mongoose schema | {entity_name}"
        if collection_name:
            signature += f" | {collection_name}"
        doc_comment = f"Mongoose schema for {entity_name}."
        if collection_name:
            doc_comment += f" Collection: {collection_name}."
        return {
            "kind": "schema",
            "signature": signature,
            "doc_comment": doc_comment,
            "metadata": {
                "framework": "mongoose",
                "resource": "schema",
                "entity_name": entity_name,
                "collection_name": collection_name,
            },
        }

    module_pattern = "|".join(re.escape(name) for name in sorted(mongoose_module_local_names))
    if module_pattern and re.search(rf"\b(?:{module_pattern})\.forFeature\s*\(", value_text):
        model_names = re.findall(r"\bname\s*:\s*([A-Za-z_$][\w$]*)\.name\b", value_text)
        collection_names = re.findall(r"\bcollection\s*:\s*(['\"`])([^'\"`]+)\1", value_text)
        unique_collections = sorted({name for _quote, name in collection_names})
        signature_parts = ["mongoose model registry"]
        if model_names:
            signature_parts.extend(model_names[:2])
        if unique_collections:
            signature_parts.extend(unique_collections[:2])
        return {
            "kind": "database",
            "signature": " | ".join(signature_parts),
            "doc_comment": "MongooseModule.forFeature registration.",
            "metadata": {
                "framework": "mongoose",
                "resource": "database",
                "entity_names": sorted(set(model_names)),
                "collection_names": unique_collections,
            },
        }

    return None


def _build_mikroorm_symbol_overrides(
    source_text: str,
    def_node: Node,
    kind: str,
    symbol_name: str | None,
) -> dict[str, object] | None:
    """Build MikroORM overrides for entities, repositories, and database clients."""
    mikro_imports = _imported_name_map_from_modules(
        source_text,
        ["@mikro-orm/core", "@mikro-orm/postgresql", "@mikro-orm/mysql", "@mikro-orm/sqlite"],
    )
    entity_local_names = _local_names_for_import(mikro_imports, "Entity")
    mikroorm_local_names = _local_names_for_import(mikro_imports, "MikroORM")
    define_entity_local_names = _local_names_for_import(mikro_imports, "defineEntity")

    if kind == "class" and symbol_name is not None:
        decorators = _leading_decorators(def_node)
        for decorator in decorators:
            decorator_name, _args = _decorator_parts(decorator)
            if decorator_name and mikro_imports.get(decorator_name) == "Entity":
                table_name = _mikroorm_table_for_entity(source_text, symbol_name, entity_local_names)
                fields = _mikroorm_fields_for_entity(source_text, symbol_name, entity_local_names)
                signature = f"mikroorm entity | {symbol_name}"
                if table_name:
                    signature += f" | {table_name}"
                doc_comment = f"MikroORM entity {symbol_name}."
                if table_name:
                    doc_comment += f" Table: {table_name}."
                if fields:
                    doc_comment += " Fields: " + ", ".join(
                        f"{field['name']} ({field['kind']})" for field in fields
                    ) + "."
                return {
                    "kind": "entity",
                    "signature": signature,
                    "doc_comment": doc_comment,
                    "metadata": {
                        "framework": "mikroorm",
                        "resource": "entity",
                        "entity_name": symbol_name,
                        "table_name": table_name,
                        "fields": fields,
                    },
                }

        class_text = def_node.text.decode("utf-8", errors="replace")
        repo_match = re.search(
            r"extends\s+EntityRepository\s*<\s*([A-Za-z_$][\w$]*)\s*>",
            class_text,
        )
        if repo_match:
            entity_name = repo_match.group(1)
            table_name = _mikroorm_table_for_entity(source_text, entity_name, entity_local_names)
            signature = f"mikroorm repository | {symbol_name}"
            if entity_name:
                signature += f" | {entity_name}"
            doc_comment = f"MikroORM repository {symbol_name} for {entity_name}."
            if table_name:
                doc_comment += f" Table: {table_name}."
            return {
                "kind": "repository",
                "signature": signature,
                "doc_comment": doc_comment,
                "metadata": {
                    "framework": "mikroorm",
                    "resource": "repository",
                    "entity_name": entity_name,
                    "repository_owner": entity_name,
                    "table_name": table_name,
                },
            }

    if def_node.type != "variable_declarator":
        return None

    value_node = def_node.child_by_field_name("value")
    if value_node is None:
        return None
    value_text = value_node.text.decode("utf-8", errors="replace")

    mikroorm_pattern = "|".join(re.escape(name) for name in sorted(mikroorm_local_names))
    if mikroorm_pattern and re.match(rf"\s*(?:await\s+)?(?:{mikroorm_pattern})\.init\s*\(", value_text):
        entity_names = re.findall(r"\bentities\s*:\s*\[([^\]]*)\]", value_text)
        entities = []
        if entity_names:
            entities = sorted(
                set(re.findall(r"[A-Za-z_$][\w$]*", entity_names[0]))
            )
        signature = "mikroorm database"
        if entities:
            signature += " | " + ", ".join(entities[:3])
        return {
            "kind": "database",
            "signature": signature,
            "doc_comment": "MikroORM database initialization.",
            "metadata": {
                "framework": "mikroorm",
                "resource": "database",
                "entity_names": entities,
            },
        }

    define_entity_pattern = "|".join(re.escape(name) for name in sorted(define_entity_local_names))
    if define_entity_pattern and re.search(rf"\b(?:{define_entity_pattern})\s*\(", value_text):
        entity_name = _object_string_property(value_text, "name") or symbol_name
        table_name = _object_string_property(value_text, "tableName")
        signature = f"mikroorm entity | {entity_name}"
        if table_name:
            signature += f" | {table_name}"
        return {
            "kind": "entity",
            "signature": signature,
            "doc_comment": (
                f"MikroORM entity {entity_name} defined via defineEntity."
                + (f" Table: {table_name}." if table_name else "")
            ),
            "metadata": {
                "framework": "mikroorm",
                "resource": "entity",
                "entity_name": entity_name,
                "table_name": table_name,
            },
        }

    if re.search(r"\bnew\s+EntitySchema(?:<[\s\S]*?>)?\s*\(", value_text):
        entity_name = _object_string_property(value_text, "name") or symbol_name
        table_name = _object_string_property(value_text, "tableName")
        signature = f"mikroorm entity schema | {entity_name}"
        if table_name:
            signature += f" | {table_name}"
        return {
            "kind": "entity",
            "signature": signature,
            "doc_comment": (
                f"MikroORM EntitySchema for {entity_name}."
                + (f" Table: {table_name}." if table_name else "")
            ),
            "metadata": {
                "framework": "mikroorm",
                "resource": "entity",
                "entity_name": entity_name,
                "table_name": table_name,
                "schema_name": symbol_name,
            },
        }

    return None


def _build_elysia_symbol_overrides(symbol_name: str, value_text: str) -> dict[str, object] | None:
    """Build router/plugin metadata for Elysia exported constants."""
    if "Elysia" not in value_text:
        return None

    prefix_match = re.search(
        r"\bnew\s+Elysia\s*\(\s*\{[\s\S]*?\bprefix\s*:\s*(['\"`])([^'\"`]+)\1",
        value_text,
    )
    prefix = _normalize_route_path(None, prefix_match.group(2)) if prefix_match else None

    name_match = re.search(
        r"\bnew\s+Elysia\s*\(\s*\{[\s\S]*?\bname\s*:\s*(['\"`])([^'\"`]+)\1",
        value_text,
    )
    plugin_name = name_match.group(2) if name_match else None

    routes: list[dict[str, str]] = []
    for method, _quote, path in re.findall(
        r"\.(get|post|put|patch|delete|options|head|all|ws)\s*\(\s*(['\"`])([^'\"`]+)\2",
        value_text,
        flags=re.IGNORECASE,
    ):
        method_upper = _ELYSIA_ROUTE_METHODS[method.lower()]
        routes.append({
            "method": method_upper,
            "path": _normalize_route_path(prefix, path),
        })

    plugins = sorted(set(re.findall(r"\.use\s*\(\s*([A-Za-z_$][\w$]*)", value_text)))
    hooks = sorted(
        {
            method
            for method in re.findall(r"\.([A-Za-z_$][\w$]*)\s*\(", value_text)
            if method in _ELYSIA_PLUGIN_HOOKS
        }
    )

    if routes:
        route_summary = [f"{item['method']} {item['path']}" for item in routes[:4]]
        signature_parts = ["Elysia router"]
        if prefix:
            signature_parts.append(prefix)
        signature_parts.extend(route_summary[:3])

        doc_parts = []
        if prefix:
            doc_parts.append(f"Elysia router with prefix {prefix}.")
        else:
            doc_parts.append("Elysia router.")
        doc_parts.append("Routes: " + "; ".join(route_summary) + ".")
        if len(routes) > 4:
            doc_parts.append(f"+{len(routes) - 4} more routes.")
        if plugins:
            doc_parts.append("Uses: " + ", ".join(plugins) + ".")

        return {
            "kind": "router",
            "signature": " | ".join(signature_parts),
            "doc_comment": " ".join(doc_parts),
            "metadata": {
                "framework": "elysia",
                "name": symbol_name,
                "prefix": prefix,
                "routes": routes,
                "plugins": plugins,
                "hooks": hooks,
            },
        }

    if hooks or plugins or plugin_name:
        signature_parts = ["Elysia plugin"]
        if plugin_name:
            signature_parts.append(plugin_name)
        signature_parts.extend(hooks[:3])

        doc_parts = ["Elysia plugin."]
        if plugin_name:
            doc_parts.append(f"Name: {plugin_name}.")
        if hooks:
            doc_parts.append("Hooks: " + ", ".join(hooks) + ".")
        if plugins:
            doc_parts.append("Uses: " + ", ".join(plugins) + ".")

        return {
            "kind": "plugin",
            "signature": " | ".join(signature_parts),
            "doc_comment": " ".join(doc_parts),
            "metadata": {
                "framework": "elysia",
                "name": symbol_name,
                "plugin_name": plugin_name,
                "plugins": plugins,
                "hooks": hooks,
            },
        }

    return None


def _build_drizzle_symbol_overrides(value_text: str) -> dict[str, object] | None:
    """Build schema/database metadata for Drizzle exported constants."""
    builder_match = re.match(r"\s*([A-Za-z_$][\w$]*)\s*\(", value_text)
    if not builder_match:
        return None

    builder = builder_match.group(1)

    if builder == "drizzle":
        has_schema = bool(re.search(r"\{\s*schema\b", value_text))
        signature = "drizzle database"
        if has_schema:
            signature += " | schema"
        return {
            "kind": "database",
            "signature": signature,
            "doc_comment": "Drizzle database client." + (" Uses schema exports." if has_schema else ""),
            "metadata": {
                "framework": "drizzle",
                "resource": "database",
                "builder": builder,
                "has_schema": has_schema,
            },
        }

    if builder in _DRIZZLE_TABLE_BUILDERS:
        args = _quoted_strings(value_text)
        table_name = args[0] if args else None
        resource = _DRIZZLE_TABLE_BUILDERS[builder]
        signature = f"drizzle {resource}"
        if table_name:
            signature += f" | {table_name}"
        signature += f" | {builder}"
        return {
            "kind": resource,
            "signature": signature,
            "doc_comment": (
                f"Drizzle {resource} defined via {builder}."
                + (f" Physical name: {table_name}." if table_name else "")
            ),
            "metadata": {
                "framework": "drizzle",
                "resource": resource,
                "builder": builder,
                "table_name": table_name,
            },
        }

    if builder in _DRIZZLE_ENUM_BUILDERS:
        args = _quoted_strings(value_text)
        enum_name = args[0] if args else None
        return {
            "kind": "enum",
            "signature": f"drizzle enum | {enum_name or builder}",
            "doc_comment": (
                f"Drizzle enum defined via {builder}."
                + (f" Physical name: {enum_name}." if enum_name else "")
            ),
            "metadata": {
                "framework": "drizzle",
                "resource": "enum",
                "builder": builder,
                "enum_name": enum_name,
            },
        }

    if builder == "relations":
        return {
            "kind": "relation",
            "signature": "drizzle relations",
            "doc_comment": "Drizzle relations mapping.",
            "metadata": {
                "framework": "drizzle",
                "resource": "relation",
                "builder": builder,
            },
        }

    return None


def _nitro_segment_to_route_token(segment: str) -> str:
    """Convert Nitro/Nuxt file path segments into route path tokens."""
    if segment == "index":
        return ""

    if segment == "[...]":
        return ":pathMatch(.*)*"

    optional_catch_all = re.fullmatch(r"\[\[\.\.\.([A-Za-z_][\w]*)\]\]", segment)
    if optional_catch_all:
        return f":{optional_catch_all.group(1)}(.*)?"

    catch_all = re.fullmatch(r"\[\.\.\.([A-Za-z_][\w]*)\]", segment)
    if catch_all:
        return f":{catch_all.group(1)}(.*)*"

    param = re.fullmatch(r"\[([A-Za-z_][\w]*)\]", segment)
    if param:
        return f":{param.group(1)}"

    return segment


def _nitro_route_from_file(rel_path: str) -> tuple[str, str] | None:
    """Derive Nitro route method and normalized route path from a file path."""
    path = Path(rel_path)
    suffixes = path.suffixes
    if len(suffixes) < 1:
        return None

    stem = path.name[:-len(path.suffix)]
    method = "ALL"
    route_parts = list(path.parts)
    method_match = re.match(r"^(?P<name>.+)\.(?P<method>[a-z]+)$", stem)

    if rel_path.startswith("server/api/"):
        route_parts = ["api", *path.parts[2:]]
    elif rel_path.startswith("server/routes/"):
        route_parts = list(path.parts[2:])
    else:
        return None

    if method_match and method_match.group("method").lower() in _NITRO_HTTP_METHODS:
        route_parts[-1] = method_match.group("name")
        method = _NITRO_HTTP_METHODS[method_match.group("method").lower()]

    if route_parts and route_parts[-1].endswith(path.suffix):
        route_parts[-1] = route_parts[-1][:-len(path.suffix)]

    tokens = [
        token
        for token in (_nitro_segment_to_route_token(part) for part in route_parts)
        if token
    ]
    route_path = "/" + "/".join(tokens) if tokens else "/"
    return method, route_path


def _build_nitro_symbol_overrides(rel_path: str, source_text: str) -> dict[str, object] | None:
    """Build file-derived Nitro/Nuxt symbol metadata for default-export server entrypoints."""
    match = re.search(
        r"\bexport\s+default\s+(?:await\s+)?"
        r"(defineEventHandler|defineCachedEventHandler|defineNitroPlugin|defineNuxtPlugin|defineNuxtRouteMiddleware)\s*\(",
        source_text,
    )
    if not match:
        return None

    factory = match.group(1)
    stem = Path(rel_path).stem

    if factory in {"defineEventHandler", "defineCachedEventHandler"}:
        derived = _nitro_route_from_file(rel_path)
        if derived is None:
            return None
        http_method, route_path = derived
        return {
            "name": route_path,
            "kind": "route",
            "signature": f"{http_method} {route_path}",
            "doc_comment": f"Nitro route {http_method} {route_path} via {factory}.",
            "metadata": {
                "framework": "nitro",
                "resource": "route",
                "handler_factory": factory,
                "http_method": http_method,
                "route_path": route_path,
            },
        }

    if factory == "defineNitroPlugin" and rel_path.startswith("server/plugins/"):
        return {
            "name": stem,
            "kind": "plugin",
            "signature": f"Nitro plugin {stem}",
            "doc_comment": f"Nitro plugin {stem}.",
            "metadata": {
                "framework": "nitro",
                "resource": "plugin",
                "plugin_name": stem,
                "handler_factory": factory,
            },
        }

    if factory == "defineNuxtPlugin" and (
        rel_path.startswith("plugins/") or rel_path.startswith("app/plugins/")
    ):
        plugin_name = re.sub(r"\.(client|server)$", "", stem)
        return {
            "name": plugin_name,
            "kind": "plugin",
            "signature": f"Nuxt plugin {plugin_name}",
            "doc_comment": f"Nuxt plugin {plugin_name}.",
            "metadata": {
                "framework": "nuxt",
                "resource": "plugin",
                "plugin_name": plugin_name,
                "handler_factory": factory,
            },
        }

    if factory == "defineNuxtRouteMiddleware" and rel_path.startswith("app/middleware/"):
        middleware_name = re.sub(r"\.(global|named)$", "", stem)
        return {
            "name": middleware_name,
            "kind": "middleware",
            "signature": f"Nuxt middleware {middleware_name}",
            "doc_comment": f"Nuxt route middleware {middleware_name}.",
            "metadata": {
                "framework": "nuxt",
                "resource": "middleware",
                "middleware_name": middleware_name,
                "handler_factory": factory,
            },
        }

    return None


def _build_typescript_symbol_overrides(
    rel_path: str,
    source: bytes,
    def_node: Node,
    kind: str,
    symbol_name: str | None,
    root_path: Path | None = None,
) -> dict[str, object] | None:
    """Build framework-aware overrides for TS/JS symbols."""
    if symbol_name is None:
        if def_node.type == "call_expression":
            return _build_nitro_symbol_overrides(
                rel_path, source.decode("utf-8", errors="replace"),
            )
        return None

    source_text = source.decode("utf-8", errors="replace")

    for builder in (_build_mongoose_symbol_overrides, _build_mikroorm_symbol_overrides):
        override = builder(source_text, def_node, kind, symbol_name)
        if override is not None:
            return override

    async_override = _build_typescript_async_symbol_overrides(
        rel_path,
        source_text,
        def_node,
        kind,
        symbol_name,
        root_path,
    )
    if async_override is not None:
        return async_override

    if def_node.type == "variable_declarator":
        value_node = def_node.child_by_field_name("value")
        if value_node is None:
            return None
        value_text = value_node.text.decode("utf-8", errors="replace")

        config_override = _build_config_factory_overrides(symbol_name, value_text)
        if config_override is not None:
            return config_override

        for builder in (_build_elysia_symbol_overrides, _build_drizzle_symbol_overrides):
            override = builder(symbol_name, value_text) if builder is _build_elysia_symbol_overrides else builder(value_text)
            if override is not None:
                return override

        first_line = value_text.splitlines()[0].strip()
        first_line = first_line[:80] + ("..." if len(first_line) > 80 else "")
        return {
            "kind": "constant",
            "signature": f"const {symbol_name} = {first_line}",
            "metadata": {
                "resource": "constant",
            },
        }

    return _build_nest_symbol_overrides(source_text, def_node, kind, symbol_name)


def _resolve_symbol_kind(base_kind: str, override_kind: str | None) -> str:
    """Prefer framework-specific kinds only when they outrank the base kind."""
    if override_kind is None:
        return base_kind

    base_precedence = FRAMEWORK_KIND_PRECEDENCE.get(base_kind, 0)
    override_precedence = FRAMEWORK_KIND_PRECEDENCE.get(override_kind, 0)
    if override_precedence >= base_precedence:
        return override_kind
    return base_kind


def _kind_from_capture(capture_name: str) -> str:
    """Map tree-sitter capture names to symbol kinds."""
    prefix = capture_name.split(".")[0]
    mapping = {
        "fn": "function",
        "dec_fn": "function",
        "export_fn": "function",
        "cls": "class",
        "dec_cls": "class",
        "export_cls": "class",
        "method": "method",
        "struct": "struct",
        "enum": "enum",
        "iface": "interface",
        "type": "type_alias",
        "typedef": "type_alias",
        "ns": "namespace",
        "mod": "module",
        "macro": "macro",
        "define": "macro",
        "proto": "prototype",
        "qproto": "prototype",
        "trait": "trait",
        "impl": "impl",
        "template": "template",
        "field_fn": "method",  # method declarations in class bodies (headers)
        "var": "function",     # arrow functions
        "var2": "function",
        "ctor": "function",    # C# constructors
        "prop": "property",    # C# properties
        # Dart-specific
        "ext": "extension",
        "mixin": "mixin",
        "getter": "method",    # Dart getters
        # Bash-specific
        "export_var": "variable",
        "const": "constant",
        "default_call": "constant",
        # SQL-specific
        "table": "table",
        "view": "view",
    }
    return mapping.get(prefix, "unknown")


def _get_enclosing_scope(node: Node) -> list[str]:
    """Walk up the AST to collect enclosing namespace/class/struct names.

    Returns a list like ["myapp", "util", "ConfigManager"] for a method
    defined inside namespace myapp { namespace util { class ConfigManager { ... } } }
    """
    scopes: list[str] = []
    current = node.parent
    while current is not None:
        if current.type in (
            "namespace_definition", "class_specifier", "struct_specifier",
            "class_definition",  # Python
            "class_declaration", "namespace_declaration",  # C#
        ):
            name_node = current.child_by_field_name("name")
            if name_node:
                scopes.append(name_node.text.decode("utf-8", errors="replace"))
        elif current.type == "template_declaration":
            # Look for named child inside the template
            for child in current.children:
                if child.type in ("class_specifier", "struct_specifier"):
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        scopes.append(name_node.text.decode("utf-8", errors="replace"))
                    break
        current = current.parent
    scopes.reverse()
    return scopes


def _build_qualified_name(symbol_name: str | None, node: Node, lang: str) -> str | None:
    """Build a proper qualified name using enclosing scope context.

    For C++: "myapp::util::ConfigManager::process"
    For Python: "Calculator.add"
    For other languages: "Module.Class.method"
    """
    if symbol_name is None:
        return None

    if lang in ("c", "cpp"):
        scopes = _get_enclosing_scope(node)
        if scopes:
            return "::".join(scopes + [symbol_name])
        # If the name already has :: (from qualified_identifier), keep it
        if "::" in symbol_name:
            return symbol_name
        return symbol_name
    elif lang == "python":
        scopes = _get_enclosing_scope(node)
        if scopes:
            return ".".join(scopes + [symbol_name])
        return symbol_name
    else:
        scopes = _get_enclosing_scope(node)
        if scopes:
            return ".".join(scopes + [symbol_name])
        return symbol_name


def _extract_template_name(node: Node) -> str | None:
    """Extract the name from a template_declaration's inner declaration.

    template<T> class Container -> "Container"
    template<T> T max_value(T a) -> "max_value"
    template<T> struct Pair -> "Pair"
    """
    for child in node.children:
        if child.type in ("class_specifier", "struct_specifier", "enum_specifier"):
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
        elif child.type == "function_definition":
            declarator = child.child_by_field_name("declarator")
            if declarator:
                # Could be function_declarator -> identifier or qualified_identifier
                inner = declarator.child_by_field_name("declarator")
                if inner:
                    return inner.text.decode("utf-8", errors="replace")
                return declarator.text.decode("utf-8", errors="replace")
        elif child.type == "declaration":
            # Template variable or forward declaration
            declarator = child.child_by_field_name("declarator")
            if declarator:
                inner = declarator.child_by_field_name("declarator")
                if inner:
                    return inner.text.decode("utf-8", errors="replace")
                return declarator.text.decode("utf-8", errors="replace")
        elif child.type == "alias_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
    return None


def _active_doc_extensions() -> set[str]:
    """Return file extensions (e.g. '*.pdf') that have active extractors."""
    return {f"*{ext}" for ext in DOCUMENT_EXTENSIONS}


def _doc_languages() -> set[str]:
    """Return the set of document language names with active extractors."""
    return set(get_registry().keys()) | {"markdown"}


def _vue_script_priority(attrs: str) -> int:
    """Rank a Vue <script> block by how likely it is to contain indexable code."""
    priority = 0

    if re.search(r"\bsetup\b", attrs):
        priority += 2

    lang_match = re.search(r"\blang\s*=\s*(['\"]?)([^'\"\s>]+)\1", attrs)
    if lang_match:
        lang = lang_match.group(2).lower()
        if lang in {"ts", "tsx", "typescript"}:
            priority += 1

    return priority


def _unique_sorted(values: list[str]) -> list[str]:
    """Return unique string values sorted for stable indexing/tests."""
    return sorted(set(values))


def _humanize_signal(value: str) -> str:
    """Convert code-ish signal text into a more searchable natural phrase."""
    cleaned = value.lstrip(".-:$")
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cleaned)
    return re.sub(r"[-_]+", " ", cleaned).strip()


def _format_search_aliases(values: list[str]) -> str:
    """Format values with humanized aliases when that improves search."""
    formatted = []
    for value in values:
        humanized = _humanize_signal(value)
        if humanized and humanized.lower() != value.lower():
            formatted.append(f"{value} ({humanized})")
        else:
            formatted.append(value)
    return ", ".join(formatted)


def _vue_script_lang(attrs: str) -> tuple[str, str] | None:
    """Return the parser/query language pair for a Vue <script> block."""
    lang_match = re.search(r"\blang\s*=\s*(['\"]?)([^'\"\s>]+)\1", attrs, re.IGNORECASE)
    if not lang_match:
        return ("javascript", "javascript")

    lang = lang_match.group(2).strip().lower()
    if lang in {"js", "javascript", "jsx"}:
        return ("javascript", "javascript")
    if lang in {"ts", "typescript"}:
        return ("typescript", "typescript")
    if lang == "tsx":
        return ("tsx", "tsx")

    return None


def _vue_script_is_usable(attrs: str) -> bool:
    """Return True for script blocks that should be parsed as JS/TS."""
    if _vue_script_lang(attrs) is None:
        return False

    type_match = re.search(r"\btype\s*=\s*(['\"]?)([^'\"\s>]+)\1", attrs, re.IGNORECASE)
    if type_match:
        script_type = type_match.group(2).split(";", 1)[0].strip().lower()
        if script_type != "module" and "javascript" not in script_type and "ecmascript" not in script_type:
            return False

    return True


def _extract_vue_template_signals(text: str) -> dict[str, list[str]] | None:
    """Extract high-value Vue template signals for search/embeddings."""
    template_re = re.compile(r"<template\b[^>]*>(.*?)</template>", re.DOTALL | re.IGNORECASE)
    special_tags = {
        "client-only", "component", "keep-alive", "nuxt-layout", "nuxt-link",
        "nuxt-page", "router-link", "slot", "suspense", "teleport",
        "transition", "transition-group",
    }

    components: list[str] = []
    directives: list[str] = []
    events: list[str] = []
    bindings: list[str] = []
    classes: list[str] = []
    module_refs: list[str] = []
    slots: list[str] = []

    for match in template_re.finditer(text):
        inner = re.sub(r"<!--.*?-->", " ", match.group(1), flags=re.DOTALL)

        for tag in re.findall(r"<([A-Za-z][\w.-]*)\b", inner):
            lower = tag.lower()
            if lower == "template":
                continue
            if any(ch.isupper() for ch in tag) or "-" in tag or lower in special_tags:
                components.append(tag)

        directives.extend(re.findall(r"\b(v-[\w:-]+)", inner))
        events.extend(
            name
            for name in re.findall(r"(?:@|v-on:)([\w:-]+)", inner)
            if name
        )
        bindings.extend(
            name
            for name in re.findall(r"(?:\:|v-bind:)([\w:-]+)", inner)
            if name
        )

        for _, class_value in re.findall(r"(?<!:)\bclass\s*=\s*(['\"])(.*?)\1", inner, re.DOTALL):
            classes.extend(token for token in re.split(r"\s+", class_value.strip()) if token)

        module_refs.extend(re.findall(r"(?:\$style|styles)\.([A-Za-z_][\w-]*)", inner))
        if re.search(r"<slot\b", inner, re.IGNORECASE):
            slots.append("slot")
        slots.extend(
            name
            for name in re.findall(r"(?:#|v-slot:)([\w-]+)", inner)
            if name
        )

    signals = {
        "components": _unique_sorted(components),
        "directives": _unique_sorted(directives),
        "events": _unique_sorted(events),
        "bindings": _unique_sorted(bindings),
        "classes": _unique_sorted(classes),
        "module_refs": _unique_sorted(module_refs),
        "slots": _unique_sorted(slots),
    }

    if any(signals.values()):
        return signals
    return None


def _has_meaningful_vue_template_signals(signals: dict[str, list[str]] | None) -> bool:
    """Return True when template signals justify creating a component anchor."""
    if not signals:
        return False

    meaningful_keys = ("components", "directives", "events", "bindings", "module_refs", "slots")
    return any(signals[key] for key in meaningful_keys)


def _extract_vue_style_signals(text: str) -> dict[str, list[str]] | None:
    """Extract high-value PostCSS module style signals for search/embeddings."""
    style_re = re.compile(r"<style\b([^>]*)>(.*?)</style>", re.DOTALL | re.IGNORECASE)

    classes: list[str] = []
    vars_: list[str] = []
    langs: list[str] = []
    flags: list[str] = []
    apply_tokens: list[str] = []

    for match in style_re.finditer(text):
        attrs = match.group(1)
        body = re.sub(r"/\*.*?\*/", " ", match.group(2), flags=re.DOTALL)

        lang_match = re.search(r"\blang\s*=\s*(['\"]?)([^'\"\s>]+)\1", attrs, re.IGNORECASE)
        if not lang_match or lang_match.group(2).strip().lower() != "postcss":
            continue
        if not re.search(r"\bmodule\b", attrs, re.IGNORECASE):
            continue

        langs.append("postcss")
        if re.search(r"\bmodule\b", attrs, re.IGNORECASE):
            flags.append("module")
        if re.search(r"\bscoped\b", attrs, re.IGNORECASE):
            flags.append("scoped")

        classes.extend(re.findall(r"(?<![\w-])\.([A-Za-z_][\w-]*)", body))
        vars_.extend(re.findall(r"(--[A-Za-z0-9_-]+)", body))

        for apply_value in re.findall(r"@apply\s+([^;]+)", body):
            apply_tokens.extend(token for token in re.split(r"\s+", apply_value.strip()) if token)

    signals = {
        "classes": _unique_sorted(classes),
        "vars": _unique_sorted(vars_),
        "langs": _unique_sorted(langs),
        "flags": _unique_sorted(flags),
        "apply": _unique_sorted(apply_tokens),
    }

    if signals["classes"] or signals["vars"] or signals["apply"]:
        return signals
    return None


def _strip_js_comments(text: str) -> str:
    """Remove obvious JS/TS comments before regex-based signal extraction."""
    out: list[str] = []
    i = 0
    in_single = False
    in_double = False
    in_template = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                out.extend((" ", " "))
                in_block_comment = False
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue

        if in_single or in_double or in_template:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            elif in_template and ch == "`":
                in_template = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            out.extend((" ", " "))
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            out.extend((" ", " "))
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue
        if ch == "`":
            in_template = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _extract_vue_script_frontend_signals(text: str) -> dict[str, list[str]] | None:
    """Extract high-value Nuxt/Vue script hints from usable Vue script blocks."""
    script_re = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
    composable_allowlist = {
        "useAppConfig", "useAsyncData", "useClipboard", "useCookie", "useCssModule",
        "useDebounceFn", "useDocumentVisibility", "useFetch", "useHead", "useI18n",
        "useId", "useLazyAsyncData", "useLazyFetch", "useLocalePath", "useLocaleRoute",
        "useLazyQuery", "useMutation", "useNuxtApp", "useQuery", "useRequestHeaders", "useResizeObserver", "useRoute",
        "useRouteBaseName", "useRouter", "useRuntimeConfig", "useSeoMeta",
        "useSlots", "useState", "useStorage", "useSwitchLocalePath",
        "useSubscription", "useTemplateRef", "useViewport", "useWindowScroll",
    }

    macros: list[str] = []
    composables: list[str] = []
    graphql_hooks: list[str] = []
    stores: list[str] = []
    graphql_ops: list[str] = []
    navigate_paths: list[str] = []
    fetch_paths: list[str] = []
    page_meta_keys: list[str] = []
    page_meta_values: list[str] = []

    for match in script_re.finditer(text):
        attrs = match.group(1)
        if not _vue_script_is_usable(attrs):
            continue

        cleaned = _strip_js_comments(match.group(2))
        for gql_body in re.findall(r"\bgql\s*`(.*?)`", cleaned, re.DOTALL):
            gql_body = re.sub(r"(?m)^\s*#.*$", " ", gql_body)
            graphql_ops.extend(
                f"{op_type} {op_name}"
                for op_type, op_name in re.findall(
                    r"\b(query|mutation|subscription)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    gql_body,
                )
            )

        code_only = re.sub(r"\bgql\s*`.*?`", " ", cleaned, flags=re.DOTALL)
        code_only = re.sub(r"`[^`]*`", " ", code_only, flags=re.DOTALL)
        code_only = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", " ", code_only)
        macros.extend(
            re.findall(
                r"\b(defineProps|defineEmits|defineExpose|defineSlots|definePageMeta|defineModel|defineOptions)\s*(?:<[^()]+>)?\s*\(",
                code_only,
            )
        )

        called_helpers = set(
            re.findall(r"\b(use[A-Z][A-Za-z0-9_]*)\s*(?:<[^()]+>)?\s*\(", code_only)
        )
        composables.extend(
            name
            for name in called_helpers
            if name in composable_allowlist
            or (
                name.endswith(("Query", "LazyQuery", "Mutation", "Subscription"))
                and name not in {"useQuery", "useLazyQuery", "useMutation", "useSubscription"}
            )
        )
        stores.extend(name for name in called_helpers if re.match(r"^use[A-Z][A-Za-z0-9_]*Store$", name))
        graphql_hooks.extend(
            name
            for name in called_helpers
            if name in {"useQuery", "useLazyQuery", "useMutation", "useSubscription"}
        )
        if re.search(r"\bnavigateTo\s*\(", code_only):
            navigate_paths.extend(
                path
                for _, path in re.findall(r"\bnavigateTo\s*\(\s*(['\"])([^'\"]+)\1", cleaned)
                if path
            )

        if {"useFetch", "useLazyFetch"} & called_helpers:
            fetch_paths.extend(
                path
                for _, path in re.findall(r"\buse(?:Lazy)?Fetch\s*\(\s*(['\"])([^'\"]+)\1", cleaned)
                if path
            )

        if "definePageMeta" in macros:
            for body in re.findall(r"definePageMeta\s*\(\s*\{(.*?)\}\s*\)", cleaned, re.DOTALL):
                page_meta_keys.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:", body))
                page_meta_values.extend(
                    value
                    for _, value in re.findall(r"(['\"])([^'\"]+)\1", body)
                    if value
                )

    signals = {
        "macros": _unique_sorted(name.rstrip("(") for name in macros),
        "composables": _unique_sorted(composables),
        "graphql_hooks": _unique_sorted(graphql_hooks),
        "stores": _unique_sorted(stores),
        "graphql_ops": _unique_sorted(graphql_ops),
        "navigate_paths": _unique_sorted(navigate_paths),
        "fetch_paths": _unique_sorted(fetch_paths),
        "page_meta_keys": _unique_sorted(page_meta_keys),
        "page_meta_values": _unique_sorted(page_meta_values),
    }

    if any(signals.values()):
        return signals
    return None


def _build_vue_component_summary(
    component_name: str,
    template_signals: dict[str, list[str]] | None,
    style_signals: dict[str, list[str]] | None,
    script_signals: dict[str, list[str]] | None,
) -> tuple[str, str]:
    """Build compact signature/doc summaries for a Vue component anchor symbol."""
    signature_parts = [f"component {component_name}"]
    doc_parts = [f"Vue component {component_name}."]

    if template_signals:
        if template_signals["components"]:
            signature_parts.append(", ".join(template_signals["components"][:3]))
            doc_parts.append(
                "Template components: "
                + _format_search_aliases(template_signals["components"])
                + "."
            )
        if template_signals["directives"]:
            signature_parts.append(", ".join(template_signals["directives"][:3]))
            doc_parts.append(
                "Template directives: " + ", ".join(template_signals["directives"]) + "."
            )
        if template_signals["events"]:
            doc_parts.append("Template events: " + ", ".join(template_signals["events"]) + ".")
        if template_signals["bindings"]:
            doc_parts.append(
                "Template bindings: " + ", ".join(template_signals["bindings"]) + "."
            )
        if template_signals["classes"]:
            doc_parts.append(
                "Template classes: " + ", ".join(template_signals["classes"]) + "."
            )
        if template_signals["module_refs"]:
            doc_parts.append(
                "Template module refs: " + ", ".join(template_signals["module_refs"]) + "."
            )
        if template_signals["slots"]:
            doc_parts.append("Template slots: " + ", ".join(template_signals["slots"]) + ".")

    if style_signals:
        style_signature_bits = []
        if style_signals["langs"]:
            style_signature_bits.extend(style_signals["langs"])
        if style_signals["flags"]:
            style_signature_bits.extend(style_signals["flags"])
        if style_signals["classes"]:
            style_signature_bits.extend(f".{name}" for name in style_signals["classes"][:3])
        if style_signals["vars"]:
            style_signature_bits.extend(style_signals["vars"][:2])
        if style_signature_bits:
            signature_parts.append(", ".join(style_signature_bits))
        if style_signals["classes"]:
            doc_parts.append("Style classes: " + ", ".join(style_signals["classes"]) + ".")
        if style_signals["vars"]:
            doc_parts.append(
                "Style vars: "
                + ", ".join(
                    f"{name} ({_humanize_signal(name)})" for name in style_signals["vars"]
                )
                + "."
            )
        if style_signals["apply"]:
            doc_parts.append("Style apply: " + ", ".join(style_signals["apply"]) + ".")
        if style_signals["langs"] or style_signals["flags"]:
            doc_parts.append(
                "Style mode: "
                + ", ".join(style_signals["langs"] + style_signals["flags"])
                + "."
            )

    if script_signals:
        script_signature_bits = []
        if script_signals["macros"]:
            script_signature_bits.extend(script_signals["macros"][:2])
        if script_signals["composables"]:
            script_signature_bits.extend(script_signals["composables"][:2])
        if script_signature_bits:
            signature_parts.append(", ".join(script_signature_bits))
        if script_signals["macros"]:
            doc_parts.append(
                "Script macros: " + _format_search_aliases(script_signals["macros"]) + "."
            )
        if script_signals["composables"]:
            doc_parts.append(
                "Script composables: "
                + _format_search_aliases(script_signals["composables"])
                + "."
            )
        if script_signals["graphql_hooks"]:
            doc_parts.append(
                "GraphQL hooks: "
                + _format_search_aliases(script_signals["graphql_hooks"])
                + "."
            )
        if script_signals["stores"]:
            doc_parts.append(
                "Stores: " + _format_search_aliases(script_signals["stores"]) + "."
            )
        if script_signals["graphql_ops"]:
            doc_parts.append(
                "GraphQL ops: "
                + _format_search_aliases(script_signals["graphql_ops"])
                + "."
            )
        if script_signals["navigate_paths"]:
            doc_parts.append(
                "Navigate paths: " + ", ".join(script_signals["navigate_paths"]) + "."
            )
        if script_signals["fetch_paths"]:
            doc_parts.append(
                "Fetch paths: " + ", ".join(script_signals["fetch_paths"]) + "."
            )
        if script_signals["page_meta_keys"]:
            doc_parts.append(
                "Page meta keys: " + ", ".join(script_signals["page_meta_keys"]) + "."
            )
        if script_signals["page_meta_values"]:
            doc_parts.append(
                "Page meta values: " + ", ".join(script_signals["page_meta_values"]) + "."
            )

    return " | ".join(signature_parts), " ".join(doc_parts)


class Indexer:
    """Indexes a codebase into a Srclight database."""

    def __init__(self, db: Database, config: IndexConfig | None = None):
        self.db = db
        self.config = config or IndexConfig()
        self._parsers: dict[str, Parser] = {}
        self._queries: dict[str, Query] = {}

        # Remove ignore patterns for extensions that have active extractors
        active_exts = _active_doc_extensions()
        self.config.ignore_patterns = [
            p for p in self.config.ignore_patterns if p not in active_exts
        ]

    def _get_parser(self, lang_name: str) -> Parser | None:
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        if lang_name == "tsx":
            language = get_tsx_language()
        else:
            language = get_language(lang_name)
        if language is None:
            return None

        parser = Parser(language)
        self._parsers[lang_name] = parser
        return parser

    def _get_query(self, lang_name: str) -> Query | None:
        if lang_name in self._queries:
            return self._queries[lang_name]

        if lang_name == "tsx":
            language = get_tsx_language()
            config = LANGUAGES.get("typescript")
        else:
            language = get_language(lang_name)
            config = LANGUAGES.get(lang_name)
        if language is None or config is None:
            return None

        try:
            query = Query(language, config.symbol_query)
            self._queries[lang_name] = query
            return query
        except Exception as e:
            logger.warning("Failed to compile query for %s: %s", lang_name, e)
            return None

    def _extract_vue_symbols(
        self, file_id: int, rel_path: str, source: bytes,
    ) -> int:
        """Extract symbols from a Vue SFC by parsing its best script block."""
        text = source.decode("utf-8", errors="replace")
        script_re = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
        component_name = Path(rel_path).stem
        template_signals = _extract_vue_template_signals(text)
        style_signals = _extract_vue_style_signals(text)
        script_signals = _extract_vue_script_frontend_signals(text)
        count = 0

        if _has_meaningful_vue_template_signals(template_signals) or style_signals or script_signals:
            signature, doc = _build_vue_component_summary(
                component_name, template_signals, style_signals, script_signals,
            )
            metadata = {
                "template": template_signals or {},
                "style": style_signals or {},
                "script": script_signals or {},
            }
            body_h = hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

            sym = SymbolRecord(
                file_id=file_id,
                kind="component",
                name=component_name,
                qualified_name=component_name,
                signature=signature,
                start_line=1,
                end_line=max(line_count, 1),
                content="",
                doc_comment=doc,
                body_hash=body_h,
                line_count=max(line_count, 1),
                metadata=metadata,
            )
            self.db.insert_symbol(sym, rel_path)
            count += 1

        candidates = []
        for match in script_re.finditer(text):
            attrs = match.group(1)
            if not _vue_script_is_usable(attrs):
                continue
            candidates.append((_vue_script_priority(attrs), match))

        if not candidates:
            return count

        ordered_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)

        for _priority, match in ordered_candidates:
            attrs = match.group(1)
            script_content = match.group(2)
            lang_pair = _vue_script_lang(attrs)
            if lang_pair is None:
                continue
            parse_lang, query_lang = lang_pair

            parser = self._get_parser(parse_lang)
            query = self._get_query(query_lang)
            if parser is None or query is None:
                continue

            script_start_offset = text[:match.start(2)].count("\n")
            script_bytes = script_content.encode("utf-8")

            tree = parser.parse(script_bytes)
            root = tree.root_node

            cursor = QueryCursor(query)
            matches = cursor.matches(root)

            raw_symbols: list[tuple[Node, str, str | None]] = []
            for _pattern_idx, match_captures in matches:
                def_node = None
                symbol_name = None
                kind = "unknown"

                for capture_name, nodes in match_captures.items():
                    if capture_name.endswith(".def") and nodes:
                        def_node = nodes[0]
                        kind = _kind_from_capture(capture_name)
                    elif capture_name.endswith(".name") and nodes:
                        symbol_name = nodes[0].text.decode("utf-8", errors="replace")

                if def_node is None:
                    continue

                if symbol_name is None and kind == "template":
                    symbol_name = _extract_template_name(def_node)

                raw_symbols.append((def_node, kind, symbol_name))

            if not raw_symbols:
                continue

            container_kinds = {
                "class",
                "struct",
                "namespace",
                "impl",
                "module",
                "controller",
                "resolver",
                "service",
                "guard",
                "filter",
                "pipe",
                "interceptor",
                "middleware",
            }
            inserted: list[tuple[int, int, int, str | None]] = []

            for def_node, kind, symbol_name in raw_symbols:
                content_text = def_node.text.decode("utf-8", errors="replace")
                doc = _extract_js_ts_doc_comment(script_bytes, def_node)
                signature_lang = "typescript" if parse_lang == "tsx" else parse_lang
                sig = _extract_signature(script_bytes, def_node, signature_lang)
                metadata = None

                if signature_lang in {"javascript", "typescript", "tsx"}:
                    override = _build_typescript_symbol_overrides(
                        rel_path,
                        script_bytes,
                        def_node,
                        kind,
                        symbol_name,
                        self.config.root if self.config and self.config.root else None,
                    )
                    if override is not None:
                        kind = _resolve_symbol_kind(kind, override.get("kind"))
                        symbol_name = override.get("name", symbol_name)
                        sig = override.get("signature", sig)
                        override_doc = override.get("doc_comment")
                        if override_doc and doc:
                            doc = f"{override_doc} {doc}"
                        elif override_doc:
                            doc = override_doc
                        metadata = override.get("metadata")
                if symbol_name is None:
                    continue

                body_bytes = def_node.text
                body_h = hashlib.sha256(body_bytes).hexdigest()[:16]

                parent_id = None
                best_span = float("inf")
                for c_start, c_end, c_id, c_kind in inserted:
                    if c_kind not in container_kinds:
                        continue
                    if c_start < def_node.start_byte and def_node.end_byte <= c_end:
                        span = c_end - c_start
                        if span < best_span:
                            best_span = span
                            parent_id = c_id

                qualified = _build_qualified_name(symbol_name, def_node, parse_lang)

                sym = SymbolRecord(
                    file_id=file_id,
                    kind=kind,
                    name=symbol_name,
                    qualified_name=qualified,
                    signature=sig,
                    start_line=def_node.start_point[0] + 1 + script_start_offset,
                    end_line=def_node.end_point[0] + 1 + script_start_offset,
                    content=content_text,
                    doc_comment=doc,
                    body_hash=body_h,
                    line_count=def_node.end_point[0] - def_node.start_point[0] + 1,
                    parent_symbol_id=parent_id,
                    metadata=metadata,
                )

                sym_id = self.db.insert_symbol(sym, rel_path)
                inserted.append((def_node.start_byte, def_node.end_byte, sym_id, kind))
                count += 1

        return count

    def index(
        self,
        root: Path | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_event: IndexEventCallback | None = None,
    ) -> IndexStats:
        """Index a codebase. Returns statistics."""
        root = root or self.config.root
        root = root.resolve()
        stats = IndexStats()
        start = time.monotonic()

        logger.info("Indexing %s", root)

        # Try to use git ls-files for .gitignore-aware file listing
        git_files = _git_tracked_files(root)
        use_git = git_files is not None
        if use_git:
            logger.info("Using git ls-files (%d tracked files)", len(git_files))

        # Collect files to process
        files_to_index: list[Path] = []
        if use_git:
            for rel in sorted(git_files):
                path = root / rel
                if not path.is_file():
                    continue

                lang = detect_language(path)
                is_doc = False
                if lang is None:
                    lang = detect_document_language(path.suffix)
                    is_doc = True
                if lang is None:
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                try:
                    if path.stat().st_size > size_limit:
                        stats.files_skipped += 1
                        continue
                except OSError:
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                files_to_index.append(path)
                stats.files_scanned += 1
        else:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if _should_ignore(path, root, self.config.ignore_patterns):
                    continue

                lang = detect_language(path)
                is_doc = False
                if lang is None:
                    lang = detect_document_language(path.suffix)
                    is_doc = True
                if lang is None:
                    continue

                size_limit = self.config.max_doc_file_size if is_doc else self.config.max_file_size
                if path.stat().st_size > size_limit:
                    stats.files_skipped += 1
                    continue

                if self.config.languages and lang not in self.config.languages:
                    continue

                files_to_index.append(path)
                stats.files_scanned += 1

        if on_event:
            on_event(
                {
                    "phase": "scan",
                    "current": 0,
                    "total": len(files_to_index),
                    "message": "Scanning indexable files",
                }
            )

        # Track existing files for removal detection
        existing_paths = self.db.all_file_paths()
        indexed_paths: set[str] = set()
        index_state = self.db.get_index_state(str(root))
        force_full_reindex = False
        if index_state and index_state.get("indexer_version") != INDEXER_BUILD_ID:
            previous_version = index_state.get("indexer_version") or "unknown"
            force_full_reindex = True
            logger.info(
                "Indexer build changed (%s -> %s); forcing full reindex",
                previous_version,
                INDEXER_BUILD_ID,
            )
            if on_event:
                on_event(
                    {
                        "phase": "scan",
                        "message": "Extractor build changed; forcing full reindex",
                        "detail": f"{previous_version} -> {INDEXER_BUILD_ID}",
                    }
                )

        # Process each file
        for i, path in enumerate(files_to_index):
            rel_path = str(path.relative_to(root))
            indexed_paths.add(rel_path)

            if on_progress:
                on_progress(rel_path, i + 1, len(files_to_index))

            try:
                raw = path.read_bytes()
                file_hash = content_hash(raw)

                # Skip if unchanged
                if not force_full_reindex and not self.db.file_needs_reindex(rel_path, file_hash):
                    stats.files_unchanged += 1
                    continue

                lang = detect_language(path)
                if lang is None:
                    lang = detect_document_language(path.suffix)
                if lang is None:
                    continue

                line_count = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)

                # Upsert file record
                file_rec = FileRecord(
                    path=rel_path,
                    content_hash=file_hash,
                    mtime=path.stat().st_mtime,
                    language=lang,
                    size=len(raw),
                    line_count=line_count,
                )
                file_id = self.db.upsert_file(file_rec)

                # Clear old symbols for this file
                self.db.delete_symbols_for_file(file_id)

                # Parse and extract symbols
                n_symbols = self._extract_symbols(file_id, rel_path, raw, lang)
                stats.symbols_extracted += n_symbols
                stats.files_indexed += 1

            except Exception as e:
                logger.error("Error indexing %s: %s", path, e)
                stats.errors += 1

        # Remove files that no longer exist
        for old_path in existing_paths - indexed_paths:
            file_rec = self.db.get_file(old_path)
            if file_rec and file_rec.id is not None:
                self.db.delete_file(file_rec.id)
                stats.files_removed += 1

        # Build call graph and inheritance edges (second pass)
        if stats.files_indexed > 0:
            if on_event:
                on_event(
                    {
                        "phase": "graph",
                        "message": "Building call graph and inheritance edges",
                    }
                )
            stats.edges_created = self._build_edges()
            stats.edges_created += self._build_inheritance_edges()

        # Community detection and execution flow tracing (post-edge phase)
        # Run if new edges were created OR if communities table is empty (first run after v5 migration)
        needs_communities = stats.edges_created > 0
        if not needs_communities:
            try:
                count = self.db.conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
                has_edges = self.db.conn.execute(
                    "SELECT 1 FROM symbol_edges WHERE edge_type = 'calls' LIMIT 1"
                ).fetchone()
                needs_communities = count == 0 and has_edges is not None
            except Exception:
                pass
        if needs_communities:
            try:
                if on_event:
                    on_event(
                        {
                            "phase": "communities",
                            "message": "Detecting communities and execution flows",
                        }
                    )
                from .community import detect_communities, trace_execution_flows
                communities = detect_communities(self.db)
                if communities:
                    sym_to_comm = {}
                    for c in communities:
                        for m in c["members"]:
                            sym_to_comm[m["id"]] = c["id"]
                    flows = trace_execution_flows(self.db, sym_to_comm)
                    self.db.store_communities(communities)
                    self.db.store_execution_flows(flows)
                    logger.info(
                        "Detected %d communities, %d execution flows",
                        len(communities), len(flows),
                    )
                    if on_event:
                        on_event(
                            {
                                "phase": "communities",
                                "message": (
                                    f"Detected {len(communities)} communities, "
                                    f"{len(flows)} execution flows"
                                ),
                            }
                        )
            except ImportError:
                logger.debug("networkx not available — skipping community detection")
            except Exception:
                logger.warning("Community detection failed", exc_info=True)

        # Build embeddings (optional, only if embed_model configured)
        if self.config.embed_model:
            n_embedded = self._build_embeddings(self.config.embed_model, on_event=on_event)
            if n_embedded > 0:
                logger.info("Embedded %d symbols with %s", n_embedded, self.config.embed_model)

        # Update index state
        git_head = _get_git_head(root)
        self.db.update_index_state(
            repo_root=str(root),
            last_commit=git_head,
            files_indexed=stats.files_scanned,
            symbols_indexed=stats.symbols_extracted,
            indexer_version=INDEXER_BUILD_ID,
        )

        self.db.commit()
        stats.elapsed_seconds = time.monotonic() - start

        logger.info(
            "Indexed %d files (%d symbols, %d edges) in %.2fs. %d unchanged, %d removed, %d errors.",
            stats.files_indexed, stats.symbols_extracted, stats.edges_created,
            stats.elapsed_seconds, stats.files_unchanged, stats.files_removed, stats.errors,
        )

        # Signal index completion via timestamp file
        try:
            signal_file = root / ".srclight" / "last-indexed"
            signal_file.parent.mkdir(parents=True, exist_ok=True)
            signal_file.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": stats.files_scanned,
                "symbols": stats.symbols_extracted,
                "commit": git_head,
                "elapsed_seconds": round(stats.elapsed_seconds, 2),
            }))
        except Exception:
            logger.debug("Failed to write index signal file", exc_info=True)

        return stats

    def _extract_symbols(
        self, file_id: int, rel_path: str, source: bytes, lang: str,
    ) -> int:
        """Parse a file and extract symbols. Returns count of symbols extracted."""
        if lang == "vue":
            return self._extract_vue_symbols(file_id, rel_path, source)

        if lang == "markdown":
            return self._extract_markdown_symbols(file_id, rel_path, source)

        # Document extractors
        doc_registry = get_registry()
        if lang in doc_registry:
            return doc_registry[lang].extract(file_id, rel_path, source, self.db)

        parser = self._get_parser(lang)
        query = self._get_query(lang)
        if parser is None or query is None:
            return 0

        tree = parser.parse(source)
        root = tree.root_node

        cursor = QueryCursor(query)
        matches = cursor.matches(root)

        # First pass: collect all symbol info
        raw_symbols: list[tuple[Node, str, str | None]] = []  # (def_node, kind, name)

        for _pattern_idx, match_captures in matches:
            def_node = None
            symbol_name = None
            kind = "unknown"

            for capture_name, nodes in match_captures.items():
                if capture_name.endswith(".def") and nodes:
                    def_node = nodes[0]
                    kind = _kind_from_capture(capture_name)
                elif capture_name.endswith(".name") and nodes:
                    symbol_name = nodes[0].text.decode("utf-8", errors="replace")

            if def_node is None:
                continue

            # For templates without a name, extract from the inner declaration
            if symbol_name is None and kind == "template":
                symbol_name = _extract_template_name(def_node)

            raw_symbols.append((def_node, kind, symbol_name))

        # Second pass: insert symbols and track parent-child relationships
        # Track container symbols (classes, structs, namespaces) by their byte ranges
        container_kinds = {
            "class",
            "struct",
            "namespace",
            "impl",
            "module",
            "controller",
            "resolver",
            "service",
            "guard",
            "filter",
            "pipe",
            "interceptor",
            "middleware",
        }
        # Map (start_byte, end_byte) -> symbol_id for containers
        inserted: list[tuple[int, int, int, str | None]] = []  # (start, end, sym_id, kind)
        inserted_names: set[str] = set()
        count = 0

        for def_node, kind, symbol_name in raw_symbols:
            content_text = def_node.text.decode("utf-8", errors="replace")
            doc = _extract_js_ts_doc_comment(source, def_node)
            sig = _extract_signature(source, def_node, lang)
            metadata = None

            if lang in {"javascript", "typescript", "tsx"}:
                override = _build_typescript_symbol_overrides(
                    rel_path,
                    source,
                    def_node,
                    kind,
                    symbol_name,
                    self.config.root if self.config and self.config.root else None,
                )
                if override is not None:
                    kind = _resolve_symbol_kind(kind, override.get("kind"))
                    symbol_name = override.get("name", symbol_name)
                    sig = override.get("signature", sig)
                    override_doc = override.get("doc_comment")
                    if override_doc and doc:
                        doc = f"{override_doc} {doc}"
                    elif override_doc:
                        doc = override_doc
                    metadata = override.get("metadata")
            if symbol_name is None:
                continue

            body_bytes = def_node.text
            body_h = hashlib.sha256(body_bytes).hexdigest()[:16]

            # Find parent: look for the tightest container that encloses this symbol
            parent_id = None
            best_span = float("inf")
            for c_start, c_end, c_id, c_kind in inserted:
                if c_kind not in container_kinds:
                    continue
                if c_start < def_node.start_byte and def_node.end_byte <= c_end:
                    span = c_end - c_start
                    if span < best_span:
                        best_span = span
                        parent_id = c_id

            qualified = _build_qualified_name(symbol_name, def_node, lang)

            sym = SymbolRecord(
                file_id=file_id,
                kind=kind,
                name=symbol_name,
                qualified_name=qualified,
                signature=sig,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                content=content_text,
                doc_comment=doc,
                body_hash=body_h,
                line_count=def_node.end_point[0] - def_node.start_point[0] + 1,
                parent_symbol_id=parent_id,
                metadata=metadata,
            )

            sym_id = self.db.insert_symbol(sym, rel_path)
            inserted.append((def_node.start_byte, def_node.end_byte, sym_id, kind))
            inserted_names.add(symbol_name)
            count += 1

        if lang in {"javascript", "typescript", "tsx"}:
            source_text = source.decode("utf-8", errors="replace")
            for symbol_name, transport_name, start_byte, end_byte in _exported_transport_constant_matches(source_text):
                if symbol_name in inserted_names:
                    continue
                content_text = source[start_byte:end_byte].decode("utf-8", errors="replace")
                line_count = content_text.count("\n") + (1 if content_text and not content_text.endswith("\n") else 0)
                start_line = source[:start_byte].count(b"\n") + 1
                end_line = start_line + max(line_count - 1, 0)
                body_h = hashlib.sha256(content_text.encode("utf-8")).hexdigest()[:16]

                sym = SymbolRecord(
                    file_id=file_id,
                    kind="transport",
                    name=symbol_name,
                    qualified_name=symbol_name,
                    signature=f"Transport {transport_name}",
                    start_line=start_line,
                    end_line=end_line,
                    content=content_text,
                    doc_comment=f"Transport configuration for {transport_name}.",
                    body_hash=body_h,
                    line_count=max(line_count, 1),
                    metadata={
                        "framework": "nestjs",
                        "resource": "transport",
                        "transport": transport_name,
                    },
                )
                self.db.insert_symbol(sym, rel_path)
                inserted_names.add(symbol_name)
                count += 1

        return count

    def _extract_markdown_symbols(
        self, file_id: int, rel_path: str, source: bytes,
    ) -> int:
        """Extract symbols from a Markdown file using heading-based sections.

        Each heading section becomes a symbol (kind='section'). A file with
        no headings becomes a single 'document' symbol. YAML frontmatter is
        stored as doc_comment on the first symbol.
        """
        parser = self._get_parser("markdown")
        if parser is None:
            return 0

        tree = parser.parse(source)
        root = tree.root_node
        file_stem = Path(rel_path).stem

        # Extract frontmatter if present
        frontmatter: str | None = None
        for child in root.children:
            if child.type == "minus_metadata":
                frontmatter = child.text.decode("utf-8", errors="replace").strip()
                break

        count = 0
        # Track inserted symbols for parent lookup: (start, end, sym_id)
        inserted: list[tuple[int, int, int]] = []

        def _get_own_content(section_node: Node) -> str:
            """Get text of all children except nested sections."""
            parts = []
            for child in section_node.children:
                if child.type != "section":
                    parts.append(source[child.start_byte:child.end_byte])
            return b"".join(parts).decode("utf-8", errors="replace").strip()

        def _get_heading_info(section_node: Node) -> tuple[str | None, str | None, int]:
            """Extract heading name, markdown signature, and level from a section.

            Returns (name, signature, level). Level is 0 if no heading found.
            """
            for child in section_node.children:
                if child.type == "atx_heading":
                    # Get inline text as name
                    inlines = [c for c in child.children if c.type == "inline"]
                    name = inlines[0].text.decode("utf-8", errors="replace").strip() if inlines else None
                    sig = child.text.decode("utf-8", errors="replace").strip()
                    # Determine level from marker (atx_h1_marker, atx_h2_marker, etc.)
                    markers = [c for c in child.children if c.type.startswith("atx_h")]
                    level = int(markers[0].type[5]) if markers else 0  # "atx_h2_marker" -> 2
                    return name, sig, level
            return None, None, 0

        def _walk_sections(
            node: Node, ancestry: list[str],
        ) -> None:
            nonlocal count

            for child in node.children:
                if child.type != "section":
                    continue

                name, sig, level = _get_heading_info(child)
                if name is None:
                    # Section without heading (rare) — skip
                    _walk_sections(child, ancestry)
                    continue

                own_content = _get_own_content(child)
                current_ancestry = ancestry + [name]
                qualified = file_stem + " > " + " > ".join(current_ancestry)

                # First paragraph (after heading) as doc_comment
                doc = None
                for sc in child.children:
                    if sc.type == "paragraph":
                        doc = sc.text.decode("utf-8", errors="replace").strip()
                        break

                # Attach frontmatter to the first symbol in the file
                is_first = count == 0
                if is_first and frontmatter:
                    doc = frontmatter + ("\n\n" + doc if doc else "")

                body_h = hashlib.sha256(own_content.encode("utf-8")).hexdigest()[:16]

                # Find parent: tightest enclosing section we've inserted
                parent_id = None
                best_span = float("inf")
                for c_start, c_end, c_id in inserted:
                    if c_start < child.start_byte and child.end_byte <= c_end:
                        span = c_end - c_start
                        if span < best_span:
                            best_span = span
                            parent_id = c_id

                sym = SymbolRecord(
                    file_id=file_id,
                    kind="section",
                    name=name,
                    qualified_name=qualified,
                    signature=sig,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    content=own_content,
                    doc_comment=doc,
                    body_hash=body_h,
                    line_count=child.end_point[0] - child.start_point[0] + 1,
                    parent_symbol_id=parent_id,
                )
                sym_id = self.db.insert_symbol(sym, rel_path)
                inserted.append((child.start_byte, child.end_byte, sym_id))
                count += 1

                # Recurse into child sections
                _walk_sections(child, current_ancestry)

        _walk_sections(root, [])

        # If no sections found, create a single document symbol for the whole file
        if count == 0:
            content_text = source.decode("utf-8", errors="replace").strip()
            body_h = hashlib.sha256(source).hexdigest()[:16]
            sym = SymbolRecord(
                file_id=file_id,
                kind="document",
                name=file_stem,
                qualified_name=file_stem,
                signature=None,
                start_line=1,
                end_line=root.end_point[0] + 1,
                content=content_text,
                doc_comment=frontmatter,
                body_hash=body_h,
                line_count=root.end_point[0] + 1,
                parent_symbol_id=None,
            )
            self.db.insert_symbol(sym, rel_path)
            count = 1

        return count

    def _build_edges(self) -> int:
        """Build call graph edges by scanning symbol content for references.

        For each symbol, scan its body for references to other known symbol names.
        Creates content-reference edges plus framework ownership/dependency edges.
        Returns the number of edges created.
        """
        assert self.db.conn is not None
        from .db import is_vendored_path

        # Clear all existing edges (full rebuild)
        self.db.conn.execute("DELETE FROM symbol_edges")

        # Build name -> [(symbol_id, file_path, kind)] lookup
        # Exclude markdown and document types — sections don't "call" anything
        # and scanning their prose would create noise with zero useful edges.
        excluded = _doc_languages()
        placeholders = ",".join("?" * len(excluded))
        rows = self.db.conn.execute(
            f"""SELECT s.id, s.name, s.kind, f.path as file_path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        name_to_symbols: dict[str, list[dict]] = {}
        symbol_info: dict[int, dict] = {}
        for row in rows:
            name = row["name"]
            info = {"id": row["id"], "file": row["file_path"], "kind": row["kind"]}
            symbol_info[row["id"]] = info
            if name not in name_to_symbols:
                name_to_symbols[name] = []
            name_to_symbols[name].append(info)

        # Filter out short/common names that would create noise
        MIN_NAME_LEN = 4
        NOISE_NAMES = {
            # Common short identifiers
            "get", "set", "run", "new", "end", "add", "put", "pop", "top",
            "map", "key", "val", "len", "str", "int", "err", "log", "max",
            "min", "abs", "all", "any", "for", "not", "and", "the",
            "def", "var", "let", "con", "ret", "gen", "ptr", "pos",
            # Common C/C++ names
            "init", "main", "next", "prev", "data", "size", "type", "name",
            "node", "list", "info", "item", "test", "self", "this", "true",
            "false", "none", "null", "void", "char", "bool", "auto",
            "file", "path", "text", "line", "args", "argv", "argc",
            "read", "open", "send", "recv", "copy", "move", "swap",
            "push", "find", "sort", "hash", "lock", "call", "bind",
            "from", "into", "with", "each", "then", "done", "fail",
            "pass", "skip", "stop", "wait", "save", "load",
            "value", "begin", "close", "clear", "reset", "write",
            "check", "parse", "print", "state", "count", "index",
            "start", "empty", "erase", "front", "apply",
            # Common variable names that create cross-file noise
            "result", "output", "input", "buffer", "config", "params",
            "status", "error", "offset", "length", "width", "height",
            "tensor", "image", "model", "layer", "batch", "channel",
            # Catch2/test framework internals
            "Clara", "Detail", "Catch", "Matchers",
        }

        # Only create edges TO meaningful symbol kinds (not prototypes/namespaces)
        EDGE_TARGET_KINDS = {"function", "method", "class", "struct", "enum", "interface", "template"}

        filtered_names = {
            name: syms for name, syms in name_to_symbols.items()
            if len(name) >= MIN_NAME_LEN and name not in NOISE_NAMES
        }

        # Skip names with too many symbols (ambiguous)
        MAX_SYMBOL_FANOUT = 10
        filtered_names = {
            name: syms for name, syms in filtered_names.items()
            if len(syms) <= MAX_SYMBOL_FANOUT
        }

        # Pre-compile regex
        sorted_names = sorted(filtered_names.keys(), key=len, reverse=True)
        if not sorted_names:
            return self._build_ownership_edges()

        import re
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b"
        )

        def _dir_of(path: str) -> str:
            """Get directory component of a path."""
            idx = path.rfind("/")
            return path[:idx] if idx >= 0 else ""

        def _compute_confidence(source_file: str, target_file: str) -> float:
            """Score edge confidence by proximity."""
            if source_file == target_file:
                return 1.0
            s_vendored = is_vendored_path(source_file)
            t_vendored = is_vendored_path(target_file)
            # Cross vendored/project boundary = low confidence
            if s_vendored != t_vendored:
                return 0.2
            # Both vendored = skip entirely
            if s_vendored and t_vendored:
                return 0.1
            # Same directory
            if _dir_of(source_file) == _dir_of(target_file):
                return 0.9
            # Same top-level module (e.g., both under src/libcapture/)
            s_parts = source_file.split("/")[:3]
            t_parts = target_file.split("/")[:3]
            if s_parts == t_parts:
                return 0.7
            return 0.5

        # Scan each symbol's content for references
        edge_count = 0
        MAX_REFS_PER_SYMBOL = 30

        content_rows = self.db.conn.execute(
            f"""SELECT s.id, s.name, s.content FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL AND f.language NOT IN ({placeholders})""",
            list(excluded),
        ).fetchall()

        for row in content_rows:
            source_id = row["id"]
            source_name = row["name"]
            content = row["content"]
            source_info = symbol_info.get(source_id)
            if not source_info:
                continue
            source_file = source_info["file"]

            referenced_names = set(pattern.findall(content))
            referenced_names.discard(source_name)

            refs_for_this = 0
            for ref_name in referenced_names:
                if refs_for_this >= MAX_REFS_PER_SYMBOL:
                    break
                targets = filtered_names.get(ref_name, [])
                for target in targets:
                    if target["id"] == source_id:
                        continue
                    # Only link to meaningful symbol kinds
                    if target["kind"] not in EDGE_TARGET_KINDS:
                        continue
                    confidence = _compute_confidence(source_file, target["file"])
                    # Skip very low confidence edges
                    if confidence < 0.2:
                        continue
                    self.db.insert_edge(EdgeRecord(
                        source_id=source_id,
                        target_id=target["id"],
                        edge_type="calls",
                        confidence=confidence,
                    ))
                    edge_count += 1
                    refs_for_this += 1

        edge_count += self._build_ownership_edges()
        return edge_count

    def _build_ownership_edges(self) -> int:
        """Build derived dependency edges from framework ownership metadata."""
        assert self.db.conn is not None

        rows = self.db.conn.execute(
            """SELECT s.id, s.name, s.kind, s.parent_symbol_id, s.content, s.metadata,
                      f.path as file_path
               FROM symbols s
               JOIN files f ON s.file_id = f.id
               WHERE s.name IS NOT NULL"""
        ).fetchall()

        if not rows:
            return 0

        symbol_by_id: dict[int, dict[str, object]] = {}
        symbols_by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
        module_symbols: list[dict[str, object]] = []
        created_edges: set[tuple[int, int, str]] = set()

        def _parse_metadata(value: object) -> dict[str, object]:
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            if isinstance(value, dict):
                return value
            return {}

        def _dir_of(path: str) -> str:
            idx = path.rfind("/")
            return path[:idx] if idx >= 0 else ""

        def _dedupe(symbols: list[dict[str, object]]) -> list[dict[str, object]]:
            seen: set[int] = set()
            unique: list[dict[str, object]] = []
            for symbol in symbols:
                symbol_id = int(symbol["id"])
                if symbol_id in seen:
                    continue
                seen.add(symbol_id)
                unique.append(symbol)
            return unique

        for row in rows:
            metadata = _parse_metadata(row["metadata"])
            info = {
                "id": row["id"],
                "name": row["name"],
                "kind": row["kind"],
                "parent_symbol_id": row["parent_symbol_id"],
                "content": row["content"] or "",
                "metadata": metadata,
                "file_path": row["file_path"],
            }
            symbol_by_id[int(row["id"])] = info
            symbols_by_name[row["name"]].append(info)
            if row["kind"] == "module":
                module_symbols.append(info)

        def _is_service_symbol(symbol: dict[str, object]) -> bool:
            return str(symbol["kind"]) == "service"

        def _is_persistence_target(symbol: dict[str, object]) -> bool:
            return str(symbol["kind"]) in {"repository", "entity"}

        def _is_config_consumer(symbol: dict[str, object]) -> bool:
            return str(symbol["kind"]) == "module" or _is_service_symbol(symbol)

        root_path = Path(self.config.root) if self.config and self.config.root else None
        file_text_cache: dict[str, str] = {}
        import_binding_cache: dict[str, dict[str, tuple[str | None, str, str]]] = {}
        default_export_name_cache: dict[str, set[str]] = {}
        tsconfig_alias_rule_cache: list[tuple[str, str, bool, list[str], str]] | None = None

        def _file_text(file_path: str) -> str:
            if file_path in file_text_cache:
                return file_text_cache[file_path]

            text = ""
            if root_path is not None:
                try:
                    text = (root_path / file_path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
            file_text_cache[file_path] = text
            return text

        def _parse_jsonc(text: str) -> dict[str, object] | None:
            stripped: list[str] = []
            in_string = False
            quote = ""
            escape = False
            line_comment = False
            block_comment = False
            i = 0
            while i < len(text):
                char = text[i]
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if line_comment:
                    if char == "\n":
                        line_comment = False
                        stripped.append(char)
                    i += 1
                    continue
                if block_comment:
                    if char == "*" and nxt == "/":
                        block_comment = False
                        i += 2
                        continue
                    i += 1
                    continue
                if in_string:
                    stripped.append(char)
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == quote:
                        in_string = False
                    i += 1
                    continue
                if char in ("'", '"'):
                    in_string = True
                    quote = char
                    stripped.append(char)
                    i += 1
                    continue
                if char == "/" and nxt == "/":
                    line_comment = True
                    i += 2
                    continue
                if char == "/" and nxt == "*":
                    block_comment = True
                    i += 2
                    continue
                stripped.append(char)
                i += 1
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", "".join(stripped)))
            except json.JSONDecodeError:
                return None

        def _import_bindings(file_path: str) -> dict[str, tuple[str | None, str, str]]:
            if file_path in import_binding_cache:
                return import_binding_cache[file_path]

            imported: dict[str, tuple[str | None, str, str]] = {}
            pattern = re.compile(
                r"^\s*import\s+(.+?)\s+from\s*(['\"])([^'\"]+)\2\s*;?\s*$",
                flags=re.MULTILINE | re.DOTALL,
            )
            for match in pattern.finditer(_file_text(file_path)):
                clause = match.group(1).strip()
                module_specifier = match.group(3)
                named_clause = ""
                if "{" in clause and "}" in clause:
                    named_clause = clause[clause.index("{") + 1:clause.rindex("}")]
                    clause = clause[:clause.index("{")].rstrip(", ").strip()
                if clause and not clause.startswith("*"):
                    ident_match = re.match(r"([A-Za-z_$][\w$]*)$", clause)
                    if ident_match:
                        imported[ident_match.group(1)] = (None, module_specifier, "default")
                if named_clause:
                    for part in named_clause.split(","):
                        item = part.strip()
                        if not item:
                            continue
                        alias_match = re.match(
                            r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$",
                            item,
                        )
                        if alias_match:
                            imported[alias_match.group(2)] = (
                                alias_match.group(1),
                                module_specifier,
                                "named",
                            )
                            continue
                        ident_match = re.match(r"([A-Za-z_$][\w$]*)$", item)
                        if ident_match:
                            imported[ident_match.group(1)] = (
                                ident_match.group(1),
                                module_specifier,
                                "named",
                            )

            import_binding_cache[file_path] = imported
            return imported

        def _ordered_import_candidate_paths(base_path: str, specifier: str) -> list[str]:
            if any(specifier.endswith(ext) for ext in (".ts", ".tsx", ".js", ".jsx")):
                return [base_path]

            return [
                f"{base_path}.ts",
                f"{base_path}.tsx",
                f"{base_path}.js",
                f"{base_path}.jsx",
                posixpath.join(base_path, "index.ts"),
                posixpath.join(base_path, "index.tsx"),
                posixpath.join(base_path, "index.js"),
                posixpath.join(base_path, "index.jsx"),
            ]

        def _resolve_existing_import_path(candidate_paths: list[str]) -> str | None:
            if root_path is None:
                return candidate_paths[0] if candidate_paths else None

            for candidate_path in candidate_paths:
                try:
                    if (root_path / candidate_path).is_file():
                        return candidate_path
                except OSError:
                    continue
            return None

        def _import_candidate_path(source_file_path: str, module_specifier: str) -> str | None:
            if not module_specifier.startswith("."):
                return None

            base_path = posixpath.normpath((Path(source_file_path).parent / module_specifier).as_posix())
            return _resolve_existing_import_path(_ordered_import_candidate_paths(base_path, module_specifier))

        def _tsconfig_alias_rules() -> list[tuple[str, str, bool, list[str], str]]:
            nonlocal tsconfig_alias_rule_cache
            if tsconfig_alias_rule_cache is not None:
                return tsconfig_alias_rule_cache

            tsconfig_alias_rule_cache = []
            if root_path is None:
                return tsconfig_alias_rule_cache

            tsconfig_alias_rule_cache = [
                (prefix, suffix, has_wildcard, list(target_patterns), base_root)
                for prefix, suffix, has_wildcard, target_patterns, base_root in _tsconfig_alias_rules(root_path.as_posix())
            ]

            return tsconfig_alias_rule_cache

        def _resolve_tsconfig_alias_path(module_specifier: str) -> tuple[str | None, bool]:
            return _resolve_typescript_alias_rule(
                root_path,
                module_specifier,
                _resolve_existing_import_path,
            )

        def _workspace_alias_candidate_suffixes(module_specifier: str) -> list[str]:
            for prefix in ("@app/", "~/", "@/"):
                if module_specifier.startswith(prefix):
                    base_path = module_specifier[len(prefix):]
                    return _ordered_import_candidate_paths(base_path, base_path)
            return []

        def _resolve_workspace_alias_path(
            module_specifier: str,
            candidates: list[dict[str, object]],
        ) -> str | None:
            for suffix in _workspace_alias_candidate_suffixes(module_specifier):
                matched_paths = sorted({
                    str(candidate["file_path"])
                    for candidate in candidates
                    if str(candidate["file_path"]).endswith(suffix)
                })
                if len(matched_paths) == 1:
                    return matched_paths[0]
                if matched_paths:
                    return None
            return None

        def _default_export_names(file_path: str) -> set[str]:
            if file_path in default_export_name_cache:
                return default_export_name_cache[file_path]

            names = _default_export_names_from_source_text(_file_text(file_path))
            default_export_name_cache[file_path] = names
            return names

        def _filter_default_import_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
            deduped = _dedupe(candidates)
            default_names = set()
            for candidate in deduped:
                default_names.update(_default_export_names(str(candidate["file_path"])))
            if default_names:
                deduped = [
                    candidate for candidate in deduped
                    if str(candidate["name"] or "") in default_names
                ]
                return _dedupe(deduped)
            return deduped if len(deduped) == 1 else []

        def _ownership_evidence_text(symbol: dict[str, object]) -> str:
            text = str(symbol["content"] or "")
            text = re.sub(r"`(?:\\.|[^`])*`", " ", text, flags=re.S)
            text = re.sub(r"'(?:\\.|[^'\\\\])*'|\"(?:\\.|[^\"\\\\])*\"", " ", text)
            text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
            text = re.sub(r"//.*", " ", text)
            return text

        def _mentioned_in_symbol(symbol: dict[str, object], target_name: str) -> bool:
            content = _ownership_evidence_text(symbol)
            return bool(target_name and re.search(rf"\b{re.escape(target_name)}\b", content))

        def _target_spellings(source: dict[str, object], target: dict[str, object]) -> list[str]:
            target_name = str(target["name"] or "")
            spellings = {target_name}
            source_file_path = str(source["file_path"])
            target_id = int(target["id"])
            preferred_kinds = {str(target["kind"])}
            for local_name in _import_bindings(source_file_path):
                if any(
                    int(resolved["id"]) == target_id
                    for resolved in _resolve_targets(
                        local_name,
                        source=source,
                        preferred_kinds=preferred_kinds,
                    )
                ):
                    spellings.add(local_name)
            return sorted(spellings)

        def _has_target_evidence(source: dict[str, object], target: dict[str, object]) -> bool:
            preferred_kinds = {str(target["kind"])}
            target_id = int(target["id"])
            for spelling in _target_spellings(source, target):
                if not _mentioned_in_symbol(source, spelling):
                    continue
                if any(
                    int(resolved["id"]) == target_id
                    for resolved in _resolve_targets(
                        spelling,
                        source=source,
                        preferred_kinds=preferred_kinds,
                    )
                ):
                    return True
            return False

        def _resolve_targets(
            name: str,
            *,
            source: dict[str, object] | None = None,
            preferred_kinds: set[str] | None = None,
        ) -> list[dict[str, object]]:
            candidates = list(symbols_by_name.get(name, []))
            if source is not None:
                source_file_path = str(source["file_path"])
                binding = _import_bindings(source_file_path).get(name)
                if binding:
                    imported_name, module_specifier, import_kind = binding
                    binding_candidates = candidates
                    if import_kind == "named" and imported_name and imported_name != name:
                        binding_candidates = list(symbols_by_name.get(imported_name, []))
                    elif import_kind == "default":
                        binding_candidates = list(symbol_by_id.values())
                    if preferred_kinds:
                        preferred = [
                            candidate for candidate in binding_candidates
                            if candidate["kind"] in preferred_kinds
                        ]
                        if preferred:
                            binding_candidates = preferred
                    import_path = _import_candidate_path(source_file_path, module_specifier)
                    if import_path:
                        deduped = _dedupe([
                            candidate
                            for candidate in binding_candidates
                            if str(candidate["file_path"]) == import_path
                        ])
                        if import_kind == "default":
                            return _filter_default_import_candidates(deduped)
                        return deduped
                    tsconfig_alias_path, tsconfig_alias_handled = _resolve_tsconfig_alias_path(module_specifier)
                    if tsconfig_alias_handled:
                        if not tsconfig_alias_path:
                            return []
                        deduped = _dedupe([
                            candidate
                            for candidate in binding_candidates
                            if str(candidate["file_path"]) == tsconfig_alias_path
                        ])
                        if import_kind == "default":
                            return _filter_default_import_candidates(deduped)
                        return deduped
                    workspace_alias_path = _resolve_workspace_alias_path(
                        module_specifier,
                        binding_candidates,
                    )
                    if workspace_alias_path:
                        deduped = _dedupe([
                            candidate
                            for candidate in binding_candidates
                            if str(candidate["file_path"]) == workspace_alias_path
                        ])
                        if import_kind == "default":
                            return _filter_default_import_candidates(deduped)
                        return deduped
                    return []
            if not candidates:
                return []
            if preferred_kinds:
                preferred = [candidate for candidate in candidates if candidate["kind"] in preferred_kinds]
                if preferred:
                    candidates = preferred
            if source is not None:
                same_file = [
                    candidate for candidate in candidates
                    if candidate["file_path"] == source["file_path"]
                ]
                if same_file:
                    candidates = same_file
                else:
                    source_dir = _dir_of(str(source["file_path"]))
                    same_dir = [
                        candidate for candidate in candidates
                        if _dir_of(str(candidate["file_path"])) == source_dir
                    ]
                    if same_dir:
                        candidates = same_dir
            return _dedupe(candidates)

        def _add_edge(
            source: dict[str, object],
            target: dict[str, object],
            reason: str,
            *,
            confidence: float = 0.98,
        ) -> None:
            source_id = int(source["id"])
            target_id = int(target["id"])
            if source_id == target_id:
                return
            key = (source_id, target_id, "ownership")
            if key in created_edges:
                return
            created_edges.add(key)
            self.db.insert_edge(EdgeRecord(
                source_id=source_id,
                target_id=target_id,
                edge_type="ownership",
                confidence=confidence,
                metadata={"reason": reason},
            ))

        def _metadata_names(metadata: dict[str, object], keys: tuple[str, ...]) -> list[str]:
            values: list[str] = []
            for key in keys:
                raw = metadata.get(key)
                if isinstance(raw, str) and raw.strip():
                    values.append(raw.strip())
                elif isinstance(raw, list):
                    values.extend(str(item).strip() for item in raw if str(item).strip())
            return sorted(set(values))

        # Child entrypoints should resolve to their owning symbol.
        for symbol in symbol_by_id.values():
            parent_id = symbol.get("parent_symbol_id")
            if not parent_id:
                continue
            if symbol["kind"] != "route_handler":
                continue
            parent = symbol_by_id.get(int(parent_id))
            if parent is not None:
                _add_edge(symbol, parent, "parent_owner")

        module_exports: dict[int, list[dict[str, object]]] = defaultdict(list)
        module_contexts: list[dict[str, object]] = []

        for module in module_symbols:
            metadata = module["metadata"]
            if not isinstance(metadata, dict):
                continue

            imports = _metadata_names(metadata, ("imports",))
            controllers = _metadata_names(metadata, ("controllers",))
            providers = _metadata_names(metadata, ("providers",))
            exports = _metadata_names(metadata, ("exports",))
            entity_names = _metadata_names(
                metadata,
                ("mikroorm_root_entities", "mikroorm_feature_entities", "entity_names"),
            )

            controller_symbols = [
                target
                for name in controllers
                for target in _resolve_targets(name, source=module, preferred_kinds={"controller"})
            ]
            provider_symbols = [
                target
                for name in providers
                for target in _resolve_targets(name, source=module)
            ]
            entity_symbols = [
                target
                for name in entity_names
                for target in _resolve_targets(name, source=module, preferred_kinds={"entity"})
                if target["kind"] == "entity"
            ]

            imported_module_symbols = [
                target
                for name in imports
                for target in _resolve_targets(name, source=module, preferred_kinds={"module"})
            ]

            exported_symbols = [
                target
                for name in exports
                for target in _resolve_targets(name, source=module)
            ]
            module_exports[int(module["id"])] = _dedupe(exported_symbols)
            module_contexts.append({
                "module": module,
                "controllers": _dedupe(controller_symbols),
                "providers": _dedupe(provider_symbols),
                "entities": _dedupe(entity_symbols),
                "imports": _dedupe(imported_module_symbols),
                "exports": _dedupe(exported_symbols),
            })

        for context in module_contexts:
            module = context["module"]
            controller_symbols = context["controllers"]
            provider_symbols = context["providers"]
            entity_symbols = context["entities"]
            imported_module_symbols = context["imports"]
            exported_symbols = context["exports"]

            for target in controller_symbols:
                _add_edge(module, target, "module_controller")
            for target in provider_symbols:
                _add_edge(module, target, "module_provider")
            for target in imported_module_symbols:
                _add_edge(module, target, "module_import")
            for target in exported_symbols:
                _add_edge(module, target, "module_export")

            imported_exports: list[dict[str, object]] = []
            for imported_module in imported_module_symbols:
                imported_exports.extend(module_exports.get(int(imported_module["id"]), []))

            imported_persistence_targets = [
                target for target in imported_exports if _is_persistence_target(target)
            ]
            provider_dependencies = _dedupe(imported_persistence_targets + entity_symbols)
            provider_data_dependencies = [
                target for target in provider_symbols if _is_persistence_target(target)
            ]
            provider_dependencies = _dedupe(provider_dependencies + provider_data_dependencies)
            controller_dependencies = [
                target
                for target in provider_symbols
                if _is_service_symbol(target)
            ]
            for provider in provider_symbols:
                if not _is_service_symbol(provider):
                    continue
                service_targets = [
                    target
                    for target in provider_dependencies
                    if _has_target_evidence(provider, target)
                ]
                for target in _dedupe(service_targets):
                    _add_edge(provider, target, "provider_module_dependency")
            for controller in controller_symbols:
                controller_targets = [
                    target
                    for target in controller_dependencies
                    if _has_target_evidence(controller, target)
                ]
                for target in _dedupe(controller_targets):
                    _add_edge(controller, target, "controller_module_dependency")

        for symbol in symbol_by_id.values():
            metadata = symbol["metadata"]
            if not isinstance(metadata, dict):
                continue

            config_ref_names = _metadata_names(metadata, ("config_refs", "config_namespaces"))
            if config_ref_names and _is_config_consumer(symbol):
                for name in config_ref_names:
                    for target in _resolve_targets(name, source=symbol, preferred_kinds={"config"}):
                        _add_edge(target, symbol, "config_metadata_consumer")

            if symbol["kind"] == "config":
                continue

            if symbol["kind"] == "microservice_handler":
                pattern_names = _metadata_names(metadata, ("pattern", "event_pattern", "message_pattern", "topic"))
                transport_names = _metadata_names(metadata, ("transport", "transport_name"))
                for name in pattern_names:
                    for target in _resolve_targets(name, source=symbol, preferred_kinds={"event_pattern", "pattern"}):
                        _add_edge(symbol, target, "microservice_pattern")
                for name in transport_names:
                    for target in _resolve_targets(name, source=symbol, preferred_kinds={"transport"}):
                        _add_edge(symbol, target, "microservice_transport")

            if symbol["kind"] == "queue_processor":
                queue_names = _metadata_names(metadata, ("queue", "queue_name"))
                for name in queue_names:
                    for target in _resolve_targets(name, source=symbol, preferred_kinds={"queue"}):
                        _add_edge(symbol, target, "queue_processor_queue")

        return len(created_edges)

    def _build_inheritance_edges(self) -> int:
        """Build 'inherits' edges by parsing base class specifiers.

        Scans class/struct symbols for base class references in their content.
        For C++: "class Foo : public Bar" → Foo inherits Bar
        For Python: "class Foo(Bar)" → Foo inherits Bar
        Returns the number of edges created.
        """
        assert self.db.conn is not None
        import re

        # Get all class/struct symbols
        class_rows = self.db.conn.execute(
            """SELECT s.id, s.name, s.kind, s.content, f.language, f.path as file_path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.kind IN ('class', 'struct') AND s.name IS NOT NULL"""
        ).fetchall()

        # Build name → symbol_id mapping for classes/structs only
        class_name_to_ids: dict[str, list[int]] = {}
        for row in class_rows:
            name = row["name"]
            if name not in class_name_to_ids:
                class_name_to_ids[name] = []
            class_name_to_ids[name].append(row["id"])

        # C++ base class pattern: "class Foo : public Bar, private Baz"
        # Also handles struct: "struct Foo : Bar"
        cpp_base_pattern = re.compile(
            r'(?:class|struct)\s+\w+\s*(?:<[^>]*>)?\s*:\s*'
            r'((?:(?:public|protected|private)\s+)?[\w:]+(?:\s*<[^>]*>)?'
            r'(?:\s*,\s*(?:(?:public|protected|private)\s+)?[\w:]+(?:\s*<[^>]*>)?)*)'
        )
        # Extract individual base class names
        cpp_base_name_pattern = re.compile(
            r'(?:public|protected|private)?\s*([\w]+)(?:::\w+)*(?:\s*<[^>]*>)?'
        )

        # Python base class pattern: "class Foo(Bar, Baz):"
        py_base_pattern = re.compile(r'class\s+\w+\s*\(([^)]+)\)')

        edge_count = 0
        for row in class_rows:
            symbol_id = row["id"]
            content = row["content"]
            lang = row["language"]

            base_names: list[str] = []

            if lang in ("cpp", "c"):
                match = cpp_base_pattern.search(content)
                if match:
                    bases_str = match.group(1)
                    for base_match in cpp_base_name_pattern.finditer(bases_str):
                        base_name = base_match.group(1)
                        if base_name and base_name not in ("public", "protected", "private"):
                            base_names.append(base_name)

            elif lang == "python":
                match = py_base_pattern.search(content)
                if match:
                    bases_str = match.group(1)
                    for base in bases_str.split(","):
                        base = base.strip()
                        # Remove keyword args like metaclass=...
                        if "=" in base:
                            continue
                        # Get just the name (strip module prefix)
                        parts = base.split(".")
                        base_names.append(parts[-1])

            # Create edges
            for base_name in base_names:
                target_ids = class_name_to_ids.get(base_name, [])
                for target_id in target_ids:
                    if target_id == symbol_id:
                        continue
                    self.db.insert_edge(EdgeRecord(
                        source_id=symbol_id,
                        target_id=target_id,
                        edge_type="inherits",
                    ))
                    edge_count += 1

        return edge_count

    def _build_embeddings(
        self,
        model_spec: str,
        on_event: IndexEventCallback | None = None,
    ) -> int:
        """Generate embeddings for symbols that need them.

        Only embeds symbols missing embeddings or with changed body_hash.
        Uses the configured embedding provider (Ollama or Voyage).

        Returns the number of symbols embedded.
        """
        from .embeddings import _index_embed_request_timeout, embed_symbols, get_provider

        try:
            provider = get_provider(model_spec, timeout=_index_embed_request_timeout())
        except (ValueError, ConnectionError) as e:
            logger.warning("Cannot initialize embedding provider '%s': %s", model_spec, e)
            return 0

        # Get symbols needing embeddings
        symbols = self.db.get_symbols_needing_embeddings(provider.name)
        if not symbols:
            logger.debug("All symbols already embedded with %s", provider.name)
            return 0

        logger.info("Embedding %d symbols with %s...", len(symbols), provider.name)
        total_batches = (len(symbols) + 32 - 1) // 32
        batch_size = 32

        def _known_dimensions() -> int | None:
            """Return dimensions only from cached/static state.

            Avoid touching provider.dimensions here because some providers only
            populate it after the first batch has been embedded.
            """
            cached = getattr(provider, "_dimensions", None)
            if isinstance(cached, int) and cached > 0:
                return cached

            try:
                static_dimensions = inspect.getattr_static(provider, "dimensions")
            except AttributeError:
                return None

            if isinstance(static_dimensions, int) and static_dimensions > 0:
                return static_dimensions
            return None

        def _embedding_detail(
            *,
            embedded_count: int = 0,
            rate_symbols: float | None = None,
            rate_batches: float | None = None,
            dimensions: int | None = None,
        ) -> str:
            parts = [f"{len(symbols)} symbols", provider.name]
            if dimensions:
                parts.append(f"{dimensions}d")
            if embedded_count > 0:
                parts.append(f"{embedded_count} embedded")
            if rate_symbols and rate_symbols > 0:
                parts.append(f"{rate_symbols:.1f} sym/s")
            if rate_batches and rate_batches > 0:
                parts.append(f"{rate_batches:.2f} batch/s")
            return " | ".join(parts)

        if on_event:
            on_event(
                {
                    "phase": "embeddings",
                    "current": 0,
                    "total": total_batches,
                    "detail": _embedding_detail(dimensions=_known_dimensions()),
                    "message": f"Embedding {len(symbols)} symbols with {provider.name}",
                }
            )

        embed_start = time.monotonic()
        previous_elapsed: float | None = None
        smoothed_batch_seconds: float | None = None

        def _on_progress(batch_num: int, total: int) -> None:
            nonlocal previous_elapsed, smoothed_batch_seconds
            elapsed = time.monotonic() - embed_start
            if previous_elapsed is not None:
                batch_seconds = max(elapsed - previous_elapsed, 0.0)
                if smoothed_batch_seconds is None:
                    smoothed_batch_seconds = batch_seconds
                else:
                    smoothed_batch_seconds = (smoothed_batch_seconds * 0.7) + (batch_seconds * 0.3)
            previous_elapsed = elapsed

            processed_symbols = min(batch_num * batch_size, len(symbols))
            rate_batches = batch_num / elapsed if elapsed > 0 else 0
            rate_symbols = processed_symbols / elapsed if elapsed > 0 else 0
            remaining = (
                (total - batch_num) * smoothed_batch_seconds
                if smoothed_batch_seconds is not None and batch_num >= 2
                else 0
            )
            logger.info("  Embedding batch %d/%d (%.0fs elapsed, ~%.0fs remaining)",
                        batch_num, total, elapsed, remaining)
            if on_event:
                on_event(
                    {
                        "phase": "embeddings",
                        "current": batch_num,
                        "total": total,
                        "detail": _embedding_detail(
                            embedded_count=processed_symbols,
                            rate_symbols=rate_symbols,
                            rate_batches=rate_batches,
                        ),
                        "elapsed_seconds": elapsed,
                        "remaining_seconds": remaining,
                    }
                )

        try:
            results = embed_symbols(provider, symbols, on_progress=_on_progress)
        except ConnectionError as e:
            logger.error("Embedding failed: %s", e)
            return 0

        # Store embeddings
        dims = _known_dimensions()
        if dims is None and results:
            dims = len(results[0][1]) // 4
        for symbol_id, emb_bytes in results:
            # Find body_hash from the symbols list
            sym = next((s for s in symbols if s["id"] == symbol_id), None)
            body_hash = sym["body_hash"] if sym else None
            self.db.upsert_embedding(symbol_id, provider.name, dims, emb_bytes, body_hash)

        self.db.commit()

        # Build .npy sidecar for GPU-resident vector cache
        if results:
            try:
                from .vector_cache import VectorCache
                srclight_dir = self.config.root / ".srclight"
                cache = VectorCache(srclight_dir)
                cache.build_from_db(self.db.conn)
                logger.info("Embedding sidecar built: %d vectors", len(results))
            except Exception as e:
                logger.warning("Failed to build embedding sidecar: %s", e)

        total_embed_elapsed = time.monotonic() - embed_start
        if on_event:
            on_event(
                {
                    "phase": "embeddings",
                    "current": total_batches,
                    "total": total_batches,
                    "detail": _embedding_detail(
                        embedded_count=len(results),
                        rate_symbols=(
                            len(results) / total_embed_elapsed
                            if total_embed_elapsed > 0
                            else None
                        ),
                        rate_batches=(
                            total_batches / total_embed_elapsed
                            if total_embed_elapsed > 0
                            else None
                        ),
                        dimensions=dims,
                    ),
                    "elapsed_seconds": total_embed_elapsed,
                }
            )

        return len(results)
