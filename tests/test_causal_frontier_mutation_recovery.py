from __future__ import annotations

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
