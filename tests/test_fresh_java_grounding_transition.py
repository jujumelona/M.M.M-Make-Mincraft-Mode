from __future__ import annotations

from minecraft_mod_ai import progress_aware_tool_loop as loop


def _tool(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _fresh_context():
    return loop.TargetMutationContext(
        target_path="src/main/java/dev/mmm/debugfixture/DebugToken.java",
        target_symbol="DebugToken",
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
    )


def _code_rag_result():
    return {
        "structured_content": {
            "schema_version": "mmm/code-rag-result-v1",
            "hits": [
                {
                    "path": "src/main/java/dev/example/ExampleMod.java",
                    "text": "public final class ExampleMod { public static final String ID = \"example\"; }",
                }
            ],
            "receipt": {"result_count": 1, "status": "FOUND"},
        }
    }


def test_fresh_java_required_grounding_exposes_code_rag_only():
    selected = loop._filter_tools_for_phase(
        (
            _tool("search_project_rag"),
            _tool("java_workspace_symbols"),
            _tool("search_code_rag"),
        ),
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=_fresh_context(),
        attempted_sources=frozenset(),
        localization_active=True,
        semantic_retrieval_choice=True,
    )
    names = [schema["function"]["name"] for schema in selected]
    assert names == ["search_code_rag"]


def test_usable_code_rag_advances_fresh_java_directly_to_act():
    state = loop.HostRunState(mutation_context=_fresh_context())
    state.record_failure("java_workspace_symbols", "stale infrastructure failure")
    state.semantic_fixed_point = True
    state.no_progress_streak = 1
    state.seen_no_progress_digests.add("stale")

    assert state.record_evidence(_code_rag_result(), usable=True) is True

    assert state.has_fresh_evidence is True
    assert state.phase == loop.LoopPhase.ACT
    assert state.semantic_fixed_point is False
    assert state.no_progress_streak == 0
    assert state.seen_no_progress_digests == set()
    assert state.last_failure_reason is None
    assert state.last_failure_digest is None


def test_metadata_catalog_alone_does_not_force_act_transition():
    state = loop.HostRunState(mutation_context=_fresh_context())
    catalog = {
        "structured_content": {
            "schema_version": "mmm/rag-result-v2",
            "sources": [{"source_id": "fabric-api-maven", "title": "Fabric API"}],
        }
    }

    assert state.record_evidence(catalog, usable=True) is True
    assert state.phase == loop.LoopPhase.OBSERVE


def test_existing_target_code_rag_does_not_bypass_localization_contract():
    context = loop.TargetMutationContext(
        target_path="src/main/java/dev/example/Existing.java",
        target_symbol="Existing",
        source_body="public final class Existing {}",
        is_new_file=False,
        evidence_source="search_code_rag",
    )
    state = loop.HostRunState(mutation_context=context)

    assert state.record_evidence(_code_rag_result(), usable=True) is True
    assert state.phase == loop.LoopPhase.OBSERVE
