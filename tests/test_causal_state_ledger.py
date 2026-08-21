from __future__ import annotations

from minecraft_mod_ai import causal_state_ledger as ledger_module
from minecraft_mod_ai.causal_state_ledger import CausalStateLedger
from minecraft_mod_ai.causal_tool_graph import verified_state_from_messages
from minecraft_mod_ai.grounding_policy import host_baseline_causal_facts


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _rag_ok() -> dict:
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


def _failed_patch() -> dict:
    return {
        "role": "tool",
        "name": "apply_source_patch",
        "content": {"ok": False, "error": "anchor mismatch"},
    }


def _query(messages) -> str:
    return "repair project" if messages else ""


def test_append_only_round_advances_suffix_without_replaying_full_transcript(monkeypatch) -> None:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    messages = [{"role": "user", "content": "repair project"}]
    calls: list[int] = []
    original = ledger_module.verified_state_from_messages

    def tracked(items, tool_schemas, *, require_fresh_evidence=False):
        calls.append(len(items))
        return original(
            items,
            tool_schemas,
            require_fresh_evidence=require_fresh_evidence,
        )

    monkeypatch.setattr(ledger_module, "verified_state_from_messages", tracked)
    ledger = CausalStateLedger()
    first = ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert first.replayed_full_transcript is True
    assert calls == [1]

    messages = [*messages, _rag_ok()]
    second = ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert second.replayed_full_transcript is False
    assert second.processed_suffix_messages == 1
    assert calls == [1]

    expected = set(
        verified_state_from_messages(
            messages,
            schemas,
            require_fresh_evidence=True,
        )
    )
    expected.update(host_baseline_causal_facts(messages))
    assert second.state == frozenset(expected)
    assert "evidence_ready" in second.state


def test_failed_mutation_invalidates_old_evidence_until_new_evidence_arrives() -> None:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    ledger = CausalStateLedger()
    messages = [
        {"role": "user", "content": "repair project"},
        _rag_ok(),
    ]
    ready = ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert "evidence_ready" in ready.state
    assert "code_evidence" in ready.state

    messages = [*messages, _failed_patch()]
    stale = ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert stale.replayed_full_transcript is False
    assert "evidence_ready" not in stale.state
    assert "code_evidence" not in stale.state
    assert "project_observed" not in stale.state

    messages = [*messages, _rag_ok()]
    refreshed = ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert refreshed.replayed_full_transcript is False
    assert "evidence_ready" in refreshed.state
    assert "code_evidence" in refreshed.state
    assert "project_observed" in refreshed.state


def test_transcript_rewrite_falls_back_to_full_replay() -> None:
    schemas = (_schema("search_code_rag"),)
    ledger = CausalStateLedger()
    messages = [{"role": "user", "content": "repair project"}, _rag_ok()]
    ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=_query,
    )

    rewritten = [{"role": "user", "content": "different task"}, _rag_ok()]
    snapshot = ledger.resolve(
        rewritten,
        schemas,
        require_fresh_evidence=True,
        query_fn=lambda _messages: "different task",
    )
    assert snapshot.replayed_full_transcript is True
    assert snapshot.query == "different task"


def test_tool_surface_change_falls_back_to_full_replay() -> None:
    ledger = CausalStateLedger()
    messages = [{"role": "user", "content": "repair project"}]
    ledger.resolve(
        messages,
        (_schema("search_code_rag"),),
        require_fresh_evidence=True,
        query_fn=_query,
    )
    snapshot = ledger.resolve(
        messages,
        (_schema("java_workspace_symbols"),),
        require_fresh_evidence=True,
        query_fn=_query,
    )
    assert snapshot.replayed_full_transcript is True
