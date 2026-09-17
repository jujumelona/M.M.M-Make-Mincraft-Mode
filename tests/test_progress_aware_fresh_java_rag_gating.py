from pathlib import Path


def _policy_source() -> str:
    source = Path("minecraft_mod_ai/progress_aware_tool_loop.py").read_text(encoding="utf-8")
    start = source.index("    require_rag = bool(\n", source.index("def _generate_with_tools_impl"))
    end = source.index("\n\n    if require_rag:", start)
    return source[start:end]


def test_host_grounded_fresh_java_does_not_force_initial_rag() -> None:
    policy = _policy_source()
    assert "and not host_grounded" in policy
    assert "and (router._agent_require_fresh_evidence or fresh_java_target)" in policy
    assert "or fresh_java_target\n" not in policy


def test_fresh_java_still_requires_rag_when_host_is_not_grounded() -> None:
    policy = _policy_source()
    namespace = {
        "role": "coder",
        "all_names": {"search_code_rag", "apply_source_edit"},
        "_RAG_EVIDENCE_TOOLS": frozenset({"search_code_rag"}),
        "host_grounded": False,
        "fresh_java_target": True,
        "router": type("Router", (), {"_agent_require_fresh_evidence": False})(),
    }
    expression = policy.split("=", 1)[1].strip()
    assert eval(expression, {}, namespace) is True


def test_grounded_fresh_java_can_enter_act_without_initial_rag() -> None:
    policy = _policy_source()
    namespace = {
        "role": "coder",
        "all_names": {"search_code_rag", "apply_source_edit"},
        "_RAG_EVIDENCE_TOOLS": frozenset({"search_code_rag"}),
        "host_grounded": True,
        "fresh_java_target": True,
        "router": type("Router", (), {"_agent_require_fresh_evidence": True})(),
    }
    expression = policy.split("=", 1)[1].strip()
    assert eval(expression, {}, namespace) is False
