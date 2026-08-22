from __future__ import annotations

import pytest

from minecraft_mod_ai.causal_state_ledger import CausalStateLedger


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _rag_ok() -> dict[str, object]:
    return {
        "role": "tool",
        "name": "search_code_rag",
        "content": {
            "ok": True,
            "result": {
                "hits": [{"path": "src/Main.java", "score": 1.0}],
                "receipt": {
                    "result_count": 1,
                    "coverage_score": 1.0,
                    "relevance_score": 1.0,
                },
            },
        },
    }


def _candidates() -> tuple[dict[str, object], ...]:
    return (
        _schema("search_code_rag"),
        _schema("apply_source_edit"),
    )


def _resolve(messages: list[dict[str, object]]) -> frozenset[str]:
    return CausalStateLedger().resolve(
        messages,
        _candidates(),
        require_fresh_evidence=True,
        query_fn=lambda _messages: "repair project",
    ).state


def test_transport_failed_source_mutation_requires_fresh_evidence() -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {"ok": False, "error": "edit rejected"},
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" not in state
    assert "code_evidence" not in state


def test_semantic_failed_source_mutation_requires_fresh_evidence() -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {
                "ok": True,
                "result": {"status": "FAILED", "reason": "stale edit"},
            },
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" not in state
    assert "code_evidence" not in state


def test_fresh_evidence_after_failure_starts_new_repair_epoch() -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {"ok": False},
        },
        _rag_ok(),
    ]

    state = _resolve(messages)

    assert "evidence_ready" in state
    assert "code_evidence" in state


@pytest.mark.parametrize(
    "marker",
    (
        "[workspace_impact=unchanged]",
        "[mmm-workspace-impact:unchanged]",
    ),
)
def test_safe_prewrite_failure_markers_preserve_evidence(marker: str) -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {
                "ok": False,
                "error": f"AgentToolRuntimeError: exact edit rejected {marker}",
            },
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" in state
    assert "code_evidence" in state


@pytest.mark.parametrize(
    "suffix",
    (
        "[workspace_impact=unchanged] [workspace_impact=uncertain]",
        "[mmm-workspace-impact:unchanged] [workspace_impact=applied]",
        "[workspace_impact=unchanged] [mmm-workspace-impact:unrecognized]",
    ),
)
def test_invalidating_or_unknown_impact_overrides_spoofed_unchanged(
    suffix: str,
) -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {
                "ok": False,
                "error": f"AgentToolRuntimeError: conflicting impact {suffix}",
            },
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" not in state
    assert "code_evidence" not in state


def test_structured_uncertain_impact_overrides_nested_unchanged() -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {
                "ok": False,
                "result": {
                    "workspace_impact": "unchanged",
                    "transaction": {"workspace_impact": "uncertain"},
                },
                "error": "AgentToolRuntimeError: rejected",
            },
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" not in state
    assert "code_evidence" not in state


def test_error_drift_overrides_structured_unchanged_impact() -> None:
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": {
                "ok": False,
                "result": {"workspace_impact": "unchanged"},
                "error": (
                    "AgentToolRuntimeError: transaction conflict "
                    "[mmm-workspace-impact:drift]"
                ),
            },
        },
    ]

    state = _resolve(messages)

    assert "evidence_ready" not in state
    assert "code_evidence" not in state
