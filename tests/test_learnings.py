"""Tests for workspace-scoped learning tools."""

import json

import pytest

import srclight.server as server
from srclight.workspace import WorkspaceConfig


@pytest.fixture
def learnings_workspace(tmp_path, monkeypatch):
    import srclight.workspace as ws_mod

    workspaces_dir = tmp_path / "workspaces"
    monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", workspaces_dir)
    monkeypatch.setattr(server, "_workspace_name", "demo")
    monkeypatch.setattr(server, "_learnings_db", None)
    monkeypatch.setattr(server, "_workspace_db", None)
    monkeypatch.setattr(server, "_workspace_config_mtime", None)
    WorkspaceConfig(name="demo", projects={}).save()
    yield workspaces_dir
    monkeypatch.setattr(server, "_workspace_name", None)
    monkeypatch.setattr(server, "_learnings_db", None)


def test_learning_tools_require_workspace_mode(monkeypatch):
    monkeypatch.setattr(server, "_workspace_name", None)
    monkeypatch.setattr(server, "_learnings_db", None)

    payload = json.loads(server.learning_stats())

    assert payload["error"] == "Learnings require workspace mode"
    assert "srclight serve --workspace" in payload["hint"]


def test_learning_tools_record_and_search_entries(learnings_workspace):
    recorded = json.loads(server.record_learning(
        kind="decision",
        content="Prefer Vue-first indexing for frontend repos.",
        reasoning="It improves project orientation for AI agents.",
        project="meanjong",
        source_type="conversation",
        source_ref="session-1",
    ))
    assert recorded["status"] == "recorded"
    assert recorded["learning_id"] >= 1

    summary = json.loads(server.conversation_summary(
        session_id="session-1",
        task_summary="Indexed meanjong and validated navigation tools.",
        project="meanjong",
        model="gpt-5.4",
    ))
    assert summary["status"] == "recorded"
    assert summary["conversation_id"] >= 1

    stats = json.loads(server.learning_stats(project="meanjong"))
    assert stats["total"] == 1
    assert stats["by_kind"]["decision"] == 1

    results = json.loads(server.relevant_learnings("Vue indexing", project="meanjong"))
    assert results["count"] == 1
    assert results["results"][0]["content"] == "Prefer Vue-first indexing for frontend repos."
