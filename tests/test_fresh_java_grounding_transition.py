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
    assert [schema["function"]["name"] for schema in selected] == ["search_code_rag"]


def test_usable_code_rag_is_progress_for_fresh_java():
    context = _fresh_context()
    assert loop._fresh_java_code_rag_progress(
        "search_code_rag",
        recorded=True,
        usable=True,
        context=context,
    ) is True
    assert loop._fresh_java_code_rag_progress(
        "search_project_rag",
        recorded=True,
        usable=True,
        context=context,
    ) is False


def test_existing_java_does_not_use_fresh_grounding_shortcut():
    context = loop.TargetMutationContext(
        target_path="src/main/java/dev/example/Existing.java",
        target_symbol="Existing",
        source_body="public final class Existing {}",
        is_new_file=False,
        evidence_source="search_code_rag",
    )
    assert loop._fresh_java_code_rag_progress(
        "search_code_rag",
        recorded=True,
        usable=True,
        context=context,
    ) is False
