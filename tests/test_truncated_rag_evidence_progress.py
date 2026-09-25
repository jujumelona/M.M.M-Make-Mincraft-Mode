from types import SimpleNamespace

from minecraft_mod_ai import model_router
from minecraft_mod_ai.progress_aware_tool_loop import (
    LoopPhase,
    _model_rejection_progress_key,
)


def _host_truncated_result(*, result_count: int = 18, preview: str | None = None):
    return {
        "_mmm_observation": {
            "sanitized": True,
            "truncated": True,
            "trust": "untrusted_data_only",
        },
        "truncated": True,
        "original_bytes": 20419,
        "preserved_evidence": [{"result_count": result_count}],
        "preview": preview
        if preview is not None
        else '{"structured_content":{"hits":[{"path":"src/main/java/demo/X.java","text":"class X {}"}]}}',
    }


def test_host_truncated_positive_rag_result_remains_usable() -> None:
    assert model_router._usable_rag_result(_host_truncated_result()) is True


def test_truncated_rag_evidence_requires_host_marker_positive_count_and_collection() -> None:
    no_host_marker = _host_truncated_result()
    no_host_marker.pop("_mmm_observation")
    assert model_router._usable_rag_result(no_host_marker) is False

    assert model_router._usable_rag_result(
        _host_truncated_result(result_count=0)
    ) is False

    assert model_router._usable_rag_result(
        _host_truncated_result(preview='{"metadata":{"result_count":18}}')
    ) is False


def test_forced_evidence_rejection_progress_ignores_query_text_noise() -> None:
    state = SimpleNamespace(
        phase=LoopPhase.OBSERVE,
        validation_status="PENDING",
        latest_verifier_fingerprint=None,
    )
    first = _model_rejection_progress_key(
        state,
        [
            {
                "failure_code": "TOOL_NOT_VISIBLE",
                "original_tool": "search_code_rag",
                "raw_arguments": '{"query":"first query","limit":10}',
            }
        ],
        forced_evidence_tool="search_project_rag",
    )
    second = _model_rejection_progress_key(
        state,
        [
            {
                "failure_code": "TOOL_NOT_VISIBLE",
                "original_tool": "search_code_rag",
                "raw_arguments": '{"query":"completely different query","limit":50}',
            }
        ],
        forced_evidence_tool="search_project_rag",
    )
    assert first == second


def test_unforced_rejection_progress_keeps_full_rejection_payload() -> None:
    state = SimpleNamespace(
        phase=LoopPhase.OBSERVE,
        validation_status="PENDING",
        latest_verifier_fingerprint=None,
    )
    first = _model_rejection_progress_key(
        state,
        [{"failure_code": "TOOL_NOT_VISIBLE", "raw_arguments": "one"}],
        forced_evidence_tool=None,
    )
    second = _model_rejection_progress_key(
        state,
        [{"failure_code": "TOOL_NOT_VISIBLE", "raw_arguments": "two"}],
        forced_evidence_tool=None,
    )
    assert first != second


def test_different_mcp_capability_is_not_the_same_rejected_action():
    from minecraft_mod_ai.progress_aware_tool_loop import HostRunState

    state = HostRunState()
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = ("mapping_resolution", "registry_lookup")
    state._external_mcp_completed_capabilities = set()
    state._external_mcp_schema_capability = "mapping_resolution"
    rejection = [{"failure_code": "TOOL_NOT_VISIBLE", "original_tool": "apply_source_edit"}]
    first = _model_rejection_progress_key(state, rejection, forced_evidence_tool="external_mcp_call")
    state._external_mcp_completed_capabilities = {"mapping_resolution"}
    state._external_mcp_schema_capability = "registry_lookup"
    second = _model_rejection_progress_key(state, rejection, forced_evidence_tool="external_mcp_call")
    assert first != second
    assert not state.record_no_progress_result(first)
    assert not state.record_no_progress_result(second)
    assert state.record_no_progress_result(second)
