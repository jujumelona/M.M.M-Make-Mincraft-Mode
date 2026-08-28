from __future__ import annotations

from minecraft_mod_ai import progress_aware_tool_loop as loop


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _names(schemas) -> tuple[str, ...]:
    return tuple(schema["function"]["name"] for schema in schemas)


def test_need_symbol_never_uses_diagnostics_or_reexposes_exhausted_source() -> None:
    tools = tuple(
        _schema(name)
        for name in (
            "java_workspace_symbols",
            "java_diagnostics",
            "search_code_rag",
            "search_project_rag",
        )
    )
    ctx = loop.TargetMutationContext(target_path="src/main/java/example/Foo.java")

    first = loop._filter_tools_for_phase(
        tools, loop.LoopPhase.OBSERVE, "coder", mutation_context=ctx
    )
    assert _names(first) == ("java_workspace_symbols",)

    second = loop._filter_tools_for_phase(
        tools,
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=ctx,
        attempted_sources={"java_workspace_symbols"},
    )
    assert _names(second) == ("search_code_rag",)
    assert "java_diagnostics" not in _names(second)

    exhausted = loop._filter_tools_for_phase(
        tools,
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=ctx,
        attempted_sources={
            "java_workspace_symbols",
            "search_code_rag",
            "search_project_rag",
        },
    )
    assert exhausted == ()


def test_localization_attempt_memory_is_stage_scoped() -> None:
    state = loop.HostRunState()
    args = {"query": "Foo service"}
    state.record_attempted_source(
        "search_code_rag",
        args,
        localization_stage=loop.LocalizationStage.NEED_FILE,
    )

    assert state.attempted_sources_for_localization_stage(
        loop.LocalizationStage.NEED_FILE
    ) == frozenset({"search_code_rag"})
    assert state.attempted_sources_for_localization_stage(
        loop.LocalizationStage.NEED_BODY
    ) == frozenset()
    assert state.next_untried_internal_tool(
        {"search_code_rag"},
        preferred=("search_code_rag",),
        localization_stage=loop.LocalizationStage.NEED_FILE,
    ) is None
    assert state.next_untried_internal_tool(
        {"search_code_rag"},
        preferred=("search_code_rag",),
        localization_stage=loop.LocalizationStage.NEED_BODY,
    ) == "search_code_rag"


def test_verify_keeps_java_diagnostics_available() -> None:
    tools = (_schema("java_diagnostics"), _schema("search_code_rag"))
    selected = loop._filter_tools_for_phase(tools, loop.LoopPhase.VERIFY, "coder")
    assert "java_diagnostics" in _names(selected)
