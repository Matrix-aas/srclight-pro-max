"""Compact response shapers for token-efficient MCP outputs."""

from __future__ import annotations

from typing import Any


def shape_compact_symbol_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a stable compact payload for high-frequency symbol lookups."""
    return {
        "match_count": len(matches),
        "compact": True,
        "hint": "Too many exact matches. Use get_signature(), symbols_in_file(), or search by qualified name/file.",
        "symbols": [
            {
                key: value
                for key, value in match.items()
                if key in {
                    "project",
                    "name",
                    "qualified_name",
                    "kind",
                    "signature",
                    "file",
                    "start_line",
                    "doc_comment",
                }
            }
            for match in matches
        ],
    }


def shape_compact_changed_symbols(symbol_impacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact detect_changes entries without heavy nested payloads."""
    compact_rows: list[dict[str, Any]] = []
    for item in symbol_impacts:
        compact_rows.append({
            "name": item.get("name"),
            "qualified_name": item.get("qualified_name"),
            "kind": item.get("kind"),
            "file": item.get("file"),
            "lines": item.get("lines"),
            "risk": item.get("risk"),
            "direct_dependents": item.get("direct_dependents", 0),
            "transitive_dependents": item.get("transitive_dependents", 0),
            "flows_affected": len(item.get("affected_flows") or []),
        })
    return compact_rows
