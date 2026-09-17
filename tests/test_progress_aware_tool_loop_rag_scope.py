from types import SimpleNamespace

from minecraft_mod_ai.progress_aware_tool_loop import LoopPhase, _call_is_evidence_tool


def test_non_rag_external_mcp_call_uses_explicit_classifier() -> None:
    call = SimpleNamespace(name="external_mcp_call", arguments={"capability": "read_file"})
    seen = []

    def classify(arguments):
        seen.append(arguments)
        return False

    assert _call_is_evidence_tool(call, LoopPhase.OBSERVE, frozenset(), classify) is False
    assert seen == [{"capability": "read_file"}]


def test_external_rag_call_uses_explicit_classifier() -> None:
    call = SimpleNamespace(name="external_mcp_call", arguments={"capability": "source_search"})
    assert _call_is_evidence_tool(
        call, LoopPhase.OBSERVE, frozenset(), lambda arguments: True
    ) is True


def test_explicit_rag_tool_set_is_honored_without_classifier_call() -> None:
    call = SimpleNamespace(name="future_rag_tool", arguments={})

    def should_not_run(_arguments):
        raise AssertionError("external classifier must not run for explicit RAG tools")

    assert _call_is_evidence_tool(
        call, LoopPhase.OBSERVE, frozenset({"future_rag_tool"}), should_not_run
    ) is True
