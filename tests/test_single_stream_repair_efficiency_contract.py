from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import minecraft_mod_ai.agentic_optimization_contract as agentic
from minecraft_mod_ai.planner_single_stream_search_contract import (
    _host_evidence_repair_router,
    _single_stream_active,
    install,
)


def _evidence() -> dict:
    return {
        "diagnostics": {
            "diagnostics": [
                {"path": "A.java", "message": "bad one", "severity": 1},
                {"path": "B.java", "message": "bad two", "severity": 1},
            ]
        },
        "build": {"status": "FAIL", "error": "x" * 100},
    }


def test_missing_active_parallel_defaults_to_single_stream(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    assert _single_stream_active() is True


def test_auto_repair_search_collapses_before_first_server_launch(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)

    class RepairEngine:
        def __init__(self, *, router=None, **_kwargs):
            self.router = router

        @staticmethod
        def _signature(evidence):
            return str(evidence)

    repair_module = SimpleNamespace(RepairEngine=RepairEngine)
    install(agentic, repair_module)
    engine = SimpleNamespace(
        _signature=lambda evidence: str(evidence),
        _mmm_signature_counts=Counter(),
    )
    assert agentic._repair_candidate_count(engine, _evidence(), []) == 1


def test_explicit_repair_search_still_allows_multiple_candidates(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")
    engine = SimpleNamespace(
        _signature=lambda evidence: str(evidence),
        _mmm_signature_counts=Counter(),
    )
    assert agentic._repair_candidate_count(engine, _evidence(), []) == 2


def test_repair_host_evidence_removes_forced_rag_but_keeps_tools(monkeypatch, tmp_path) -> None:
    class Router:
        def __init__(self):
            self.binds = []
            self.calls = []

        def bind_agent_workspace(self, root, *, require_fresh_evidence=False):
            self.binds.append((root, require_fresh_evidence))
            return self

        def generate_text(self, role, messages, **kwargs):
            self.calls.append((role, messages, dict(kwargs)))
            return "ok"

    base = Router()
    proxy = _host_evidence_repair_router(base)
    proxy.bind_agent_workspace(tmp_path, require_fresh_evidence=True)
    assert base.binds == [(tmp_path, False)]

    result = proxy.generate_text(
        "coder",
        [{"role": "user", "content": "repair"}],
        response_format="json",
        enable_tools=True,
    )
    assert result == "ok"
    assert base.calls[-1][2]["enable_tools"] is True
