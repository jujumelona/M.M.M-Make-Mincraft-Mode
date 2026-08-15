from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.custom_generation_search_contract import (
    _ResearchEvidenceRouter,
    _width,
)


def _complex_module():
    return SimpleNamespace(
        kind="custom_java",
        config={
            "network": True,
            "runtime": True,
            "payload": "x" * 2500,
        },
        depends_on=("core", "api"),
        required_gates=("jdt", "runtime"),
    )


def test_auto_custom_search_does_not_duplicate_single_native_lane(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert _width(_complex_module()) == 1


def test_explicit_custom_search_remains_user_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert _width(_complex_module()) == 2


def test_research_router_preserves_fresh_evidence_binding(monkeypatch, tmp_path) -> None:
    class Router:
        def __init__(self) -> None:
            self.binds = []
            self.calls = []

        def bind_agent_workspace(self, root, *, require_fresh_evidence=False):
            self.binds.append((root, require_fresh_evidence))
            return self

        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, messages, dict(kwargs)))
            return "ok"

    base = Router()
    proxy = _ResearchEvidenceRouter(
        base,
        owner=SimpleNamespace(policy=None, _cached_index=None, _cached_root=None),
        project_root=tmp_path,
        module=_complex_module(),
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.1",
    )
    proxy.bind_agent_workspace(tmp_path, require_fresh_evidence=True)
    assert base.binds == [(tmp_path, True)]

    messages = ({"role": "user", "content": "plan"},)
    result = proxy.generate_text(
        "planner",
        messages,
        response_format="json",
        enable_tools=True,
    )
    assert result == "ok"
    assert base.calls[-1][1] == messages
    assert base.calls[-1][2]["enable_tools"] is True
