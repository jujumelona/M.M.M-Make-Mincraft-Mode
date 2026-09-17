from __future__ import annotations

from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai.model_router import _usable_rag_result


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
        evidence_source="host_task_authority",
        writable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        creatable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        target_pinned=True,
    )


def test_fresh_java_required_grounding_exposes_code_rag_first():
    selected = loop._filter_tools_for_phase(
        (
            _tool("search_project_rag"),
            _tool("java_workspace_symbols"),
            _tool("external_mcp_call"),
            _tool("search_code_rag"),
        ),
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=_fresh_context(),
        attempted_sources=frozenset(),
        localization_active=True,
        semantic_retrieval_choice=True,
    )
    assert [schema["function"]["name"] for schema in selected] == ["search_code_rag"]


def test_generic_fresh_evidence_does_not_unlock_fresh_java():
    state = loop.HostRunState(mutation_context=_fresh_context())
    assert state.record_evidence(
        {"schema_version": "mmm/project-convention-v1", "content": "use a final utility class"},
        usable=True,
    ) is True
    assert state.has_fresh_evidence is True
    assert state.has_authoritative_java_evidence is False
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is False


def test_concrete_code_rag_unlocks_fresh_java():
    state = loop.HostRunState(mutation_context=_fresh_context())
    evidence = {
        "schema_version": "mmm/code-rag-result-v1",
        "hits": [{
            "source_path": "src/main/java/dev/mmm/ExistingItems.java",
            "text": "import net.minecraft.world.item.Item; final class ExistingItems {}",
        }],
        "receipt": {"result_count": 1, "coverage_score": 1.0, "relevance_score": 1.0},
    }
    assert _usable_rag_result(evidence) is True
    assert state.record_evidence(evidence, usable=True) is True
    assert state.has_authoritative_java_evidence is True
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is True


def test_target_neutral_project_rag_cannot_unlock_fresh_java():
    state = loop.HostRunState(mutation_context=_fresh_context())
    evidence = {
        "schema_version": "mmm/rag-result-v2",
        "content": "Fabric item registration background documentation.",
    }
    assert state.record_evidence(evidence, usable=True) is True
    assert state.has_authoritative_java_evidence is False
