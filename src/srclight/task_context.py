"""Deterministic task-oriented context assembly for AI agents."""

from __future__ import annotations

import re
from typing import Any

from .db import Database, SymbolRecord, tokenized_query_hint

_BUDGETS: dict[str, dict[str, int]] = {
    "small": {
        "seed_limit": 4,
        "primary_symbols": 2,
        "primary_files": 2,
        "related_api": 2,
        "related_tests": 2,
        "data_types": 2,
        "call_chain": 4,
        "next_steps": 3,
    },
    "medium": {
        "seed_limit": 6,
        "primary_symbols": 4,
        "primary_files": 4,
        "related_api": 3,
        "related_tests": 3,
        "data_types": 3,
        "call_chain": 6,
        "next_steps": 4,
    },
    "large": {
        "seed_limit": 8,
        "primary_symbols": 6,
        "primary_files": 6,
        "related_api": 5,
        "related_tests": 5,
        "data_types": 5,
        "call_chain": 8,
        "next_steps": 5,
    },
}

_TYPE_KINDS = {"interface", "type_alias", "class", "struct", "enum"}


def _task_tokens(task: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_$]+", task)]


def _identifier_candidates(task: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        normalized = value.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    for value in re.findall(r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b", task):
        _add(value)
        parts = [part for part in value.split(".") if part]
        for part in parts:
            _add(part)

    for value in re.findall(
        r"\b(?:use[A-Z][A-Za-z0-9_$]*|[A-Z][A-Za-z0-9_$]*|[a-z]+[A-Z][A-Za-z0-9_$]*)\b",
        task,
    ):
        _add(value)

    hint = tokenized_query_hint(task)
    if hint:
        _add(hint)

    return candidates


def _symbol_brief(sym: SymbolRecord) -> dict[str, Any]:
    return {
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "kind": sym.kind,
        "file": sym.file_path,
        "line": sym.start_line,
        "signature": sym.signature,
        "doc_comment": sym.doc_comment,
    }


def _edge_brief(entry: dict[str, Any], direction: str) -> dict[str, Any]:
    symbol = entry["symbol"]
    edge_types = entry.get("edge_types")
    if not edge_types:
        edge_types = [entry.get("edge_type")] if entry.get("edge_type") else []
    return {
        "direction": direction,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "kind": symbol.kind,
        "file": symbol.file_path,
        "line": symbol.start_line,
        "edge_types": sorted(set(edge_types)),
        "confidence": entry.get("confidence"),
    }


def _dedup_graph_entries(entries: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        symbol = entry["symbol"]
        key = (
            symbol.id,
            symbol.qualified_name or "",
            symbol.file_path or "",
            symbol.start_line,
        )
        brief = _edge_brief(entry, direction)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = brief
            continue
        existing["edge_types"] = sorted(set(existing["edge_types"]) | set(brief["edge_types"]))
        if (brief.get("confidence") or 0) > (existing.get("confidence") or 0):
            existing["confidence"] = brief["confidence"]
    return list(deduped.values())


def _score_symbol_candidate(sym: SymbolRecord, candidate: str) -> int:
    name = (sym.name or "").lower()
    qualified = (sym.qualified_name or "").lower()
    candidate_lower = candidate.lower()
    last_segment = candidate_lower.split(".")[-1]
    score = 0
    if qualified == candidate_lower:
        score += 120
    if candidate_lower in qualified:
        score += 50
    if name == last_segment:
        score += 70
    elif last_segment in name:
        score += 20
    if sym.kind in {"method", "function", "route_handler", "service", "controller"}:
        score += 10
    return score


def _seed_symbols(
    db: Database,
    task: str,
    seed_limit: int,
) -> tuple[list[dict[str, Any]], list[SymbolRecord]]:
    candidates = _identifier_candidates(task)
    scored: dict[int, tuple[int, SymbolRecord, str]] = {}

    for candidate in candidates:
        lookup = candidate.split(".")[-1]
        for sym in db.get_symbols_by_name(lookup, limit=seed_limit * 4):
            if sym.id is None:
                continue
            score = _score_symbol_candidate(sym, candidate)
            if score <= 0:
                continue
            existing = scored.get(sym.id)
            if existing is None or score > existing[0]:
                scored[sym.id] = (score, sym, f"matched identifier '{candidate}'")

    if not scored:
        for result in db.search_symbols(task, limit=seed_limit * 3):
            symbol_id = result.get("symbol_id")
            if symbol_id is None:
                continue
            for sym in db.get_symbols_by_name(str(result.get("name") or ""), limit=seed_limit * 4):
                if sym.id != symbol_id:
                    continue
                scored[sym.id] = (25, sym, "matched keyword search")
                break

    ranked = sorted(
        scored.values(),
        key=lambda item: (-item[0], item[1].file_path or "", item[1].start_line),
    )
    selected_symbols = [sym for _score, sym, _reason in ranked[:seed_limit]]
    seeds = [
        {
            **_symbol_brief(sym),
            "reason": reason,
            "score": score,
        }
        for score, sym, reason in ranked[:seed_limit]
    ]
    return seeds, selected_symbols


def _file_briefs(db: Database, symbols: list[SymbolRecord], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    for sym in symbols:
        if not sym.file_path or sym.file_path in seen:
            continue
        seen.add(sym.file_path)
        summary = db.get_file_summary(sym.file_path)
        if summary is None:
            continue
        files.append({
            "file": summary["file"],
            "language": summary["language"],
            "summary": summary["summary"],
            "top_level_symbols": summary["top_level_symbols"][:4],
        })
        if len(files) >= limit:
            break
    return files


def _related_tests(db: Database, symbols: list[SymbolRecord], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None]] = set()
    tests: list[dict[str, Any]] = []
    for sym in symbols:
        for item in db.get_tests_for(sym.name or ""):
            test_symbol = item["symbol"]
            key = (test_symbol.file_path or "", test_symbol.start_line)
            if key in seen:
                continue
            seen.add(key)
            tests.append({
                "name": test_symbol.name,
                "kind": test_symbol.kind,
                "file": test_symbol.file_path,
                "line": test_symbol.start_line,
                "confidence": item.get("confidence"),
            })
            if len(tests) >= limit:
                return tests
    return tests


def _call_chain(db: Database, symbols: list[SymbolRecord], limit: int) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int | None]] = set()
    for sym in symbols:
        if sym.id is None:
            continue
        for entry in _dedup_graph_entries(db.get_callers(sym.id), "caller"):
            key = (
                entry["direction"],
                entry["qualified_name"] or "",
                entry["file"] or "",
                entry["line"],
            )
            if key in seen:
                continue
            seen.add(key)
            chain.append(entry)
            if len(chain) >= limit:
                return chain
        for entry in _dedup_graph_entries(db.get_callees(sym.id), "callee"):
            key = (
                entry["direction"],
                entry["qualified_name"] or "",
                entry["file"] or "",
                entry["line"],
            )
            if key in seen:
                continue
            seen.add(key)
            chain.append(entry)
            if len(chain) >= limit:
                return chain
    return chain


def _data_types(db: Database, symbols: list[SymbolRecord], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str | None, str | None]] = set()
    types: list[dict[str, Any]] = []
    for sym in symbols:
        if sym.id is None:
            continue
        for entry in db.get_callees(sym.id):
            target = entry["symbol"]
            if target.kind not in _TYPE_KINDS:
                continue
            key = (target.name or "", target.file_path, target.qualified_name)
            if key in seen:
                continue
            seen.add(key)
            types.append(_symbol_brief(target))
            if len(types) >= limit:
                return types
        if sym.file_path:
            summary = db.get_file_summary(sym.file_path)
            if not summary:
                continue
            for item in summary["top_level_symbols"]:
                if item["kind"] not in _TYPE_KINDS:
                    continue
                key = (item["name"], sym.file_path, item.get("signature"))
                if key in seen:
                    continue
                seen.add(key)
                types.append({
                    "name": item["name"],
                    "qualified_name": item["name"],
                    "kind": item["kind"],
                    "file": sym.file_path,
                    "line": item["line"],
                    "signature": item.get("signature"),
                    "doc_comment": None,
                })
                if len(types) >= limit:
                    return types
    return types


def _related_api(
    db: Database,
    primary_symbols: list[SymbolRecord],
    call_chain: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    candidate_files = {sym.file_path for sym in primary_symbols if sym.file_path}
    candidate_names = {sym.name for sym in primary_symbols if sym.name}
    for item in call_chain:
        if item.get("file"):
            candidate_files.add(item["file"])
        if item.get("name"):
            candidate_names.add(item["name"])

    endpoints: list[dict[str, Any]] = []
    for endpoint in db.api_surface(limit=limit * 8):
        if (
            endpoint.get("file") in candidate_files
            or endpoint.get("symbol") in candidate_names
        ):
            endpoints.append(endpoint)
        if len(endpoints) >= limit:
            break
    return endpoints


def _next_steps(
    primary_symbols: list[SymbolRecord],
    primary_files: list[dict[str, Any]],
    related_api: list[dict[str, Any]],
    related_tests: list[dict[str, Any]],
    data_types: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    steps: list[str] = []
    if primary_symbols:
        steps.append(
            "Inspect "
            f"{primary_symbols[0].qualified_name or primary_symbols[0].name} "
            f"in {primary_symbols[0].file_path}:{primary_symbols[0].start_line}."
        )
    if related_api:
        steps.append(
            "Check the API entrypoint "
            f"{related_api[0]['method']} {related_api[0]['path']} "
            "before changing validation behavior."
        )
    if related_tests:
        steps.append(f"Review or extend tests in {related_tests[0]['file']}.")
    else:
        steps.append("No nearby tests found; add or update coverage before shipping the change.")
    if data_types:
        steps.append(f"Validate the data contract in {data_types[0]['name']}.")
    elif primary_files:
        steps.append(
            f"Use get_file_summary('{primary_files[0]['file']}') "
            "if you need more file-level context."
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step in seen:
            continue
        seen.add(step)
        deduped.append(step)
    return deduped[:limit]


def build_task_context(
    db: Database,
    task: str,
    *,
    budget: str = "medium",
) -> dict[str, Any]:
    """Build a compact, deterministic task context packet from the current index."""
    if budget not in _BUDGETS:
        raise ValueError(f"Unsupported budget '{budget}'")

    limits = _BUDGETS[budget]
    seeds, seed_symbols = _seed_symbols(db, task, limits["seed_limit"])
    primary_symbols = seed_symbols[: limits["primary_symbols"]]
    primary_files = _file_briefs(db, primary_symbols, limits["primary_files"])
    call_chain = _call_chain(db, primary_symbols, limits["call_chain"])
    related_api = _related_api(db, primary_symbols, call_chain, limits["related_api"])
    related_tests = _related_tests(db, primary_symbols, limits["related_tests"])
    data_types = _data_types(db, primary_symbols, limits["data_types"])

    why_these_results = []
    if seeds:
        why_these_results.append(
            "Primary symbols were chosen from exact identifier matches "
            "found in the task text."
        )
    if call_chain:
        why_these_results.append(
            "Call-chain entries show nearby callers/callees that frame "
            "the change surface."
        )
    if related_api:
        why_these_results.append(
            "Related API endpoints were pulled from route metadata "
            "connected to the same files or symbols."
        )
    if related_tests:
        why_these_results.append(
            "Nearby tests were included to show existing coverage "
            "before editing code."
        )
    if data_types:
        why_these_results.append(
            "Data types capture DTOs/interfaces/classes that shape "
            "the method contract."
        )

    return {
        "task": task,
        "budget": budget,
        "seeds": seeds,
        "primary_symbols": [_symbol_brief(sym) for sym in primary_symbols],
        "primary_files": primary_files,
        "related_api": related_api,
        "related_tests": related_tests,
        "data_types": data_types,
        "call_chain": call_chain,
        "next_steps": _next_steps(
            primary_symbols,
            primary_files,
            related_api,
            related_tests,
            data_types,
            limits["next_steps"],
        ),
        "why_these_results": why_these_results,
    }
