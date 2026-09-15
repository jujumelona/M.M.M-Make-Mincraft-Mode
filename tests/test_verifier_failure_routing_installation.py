from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai.verifier_failure_routing_installation import install


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_jdt_release_mismatch_never_becomes_source_failure() -> None:
    install()
    payload = {
        "ok": True,
        "result": {
            "status": "FAIL",
            "diagnostics": [
                {
                    "code": "JDT_RELEASE_UNAVAILABLE",
                    "message": "release 25 is not found in the system",
                    "severity": 2,
                }
            ],
        },
    }
    assert loop._verification_outcome("java_diagnostics", payload) == "UNAVAILABLE"


def test_real_source_failure_remains_source_failure() -> None:
    install()
    payload = {
        "ok": True,
        "result": {
            "status": "FAIL",
            "diagnostics": [
                {
                    "code": "JDT_COMPILE_ERROR",
                    "message": "Foo.java:12: error: cannot find symbol",
                    "severity": 2,
                }
            ],
        },
    }
    assert loop._verification_outcome("java_diagnostics", payload) == "FAIL"


def test_recover_exposes_reviewed_mutation_tool() -> None:
    install()
    exposed = (
        _tool("search_code_rag"),
        _tool("java_workspace_symbols"),
        _tool("apply_source_edit"),
        _tool("java_diagnostics"),
    )
    selected = loop._filter_tools_for_phase(
        exposed,
        loop.LoopPhase.RECOVER,
        "coder",
        mutation_context=None,
        attempted_sources=frozenset(),
        localization_active=False,
        semantic_retrieval_choice=False,
    )
    names = {loop._tool_name(schema) for schema in selected}
    assert "apply_source_edit" in names
    assert "search_code_rag" in names
    assert "java_diagnostics" not in names
