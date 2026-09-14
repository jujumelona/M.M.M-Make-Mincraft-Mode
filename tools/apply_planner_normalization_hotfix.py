from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: expected source text not found")
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source match")
    return text.replace(old, new, 1)


def patch_pipeline() -> None:
    path = ROOT / "minecraft_mod_ai" / "planning_state_pipeline.py"
    text = path.read_text(encoding="utf-8")

    helper_marker = "\ndef _resolve_requirements_or_wait(\n"
    helper = '''\n\ndef _research_stage_needed(state: Mapping[str, Any]) -> bool:\n    \"\"\"Enter research only when an explicit unresolved obligation needs it.\"\"\"\n\n    queue = state.get(\"research_queue\")\n    if isinstance(queue, list) and any(\n        isinstance(row, Mapping) and row.get(\"status\") == \"pending\"\n        for row in queue\n    ):\n        return True\n\n    unresolved = state.get(\"unresolved\")\n    return isinstance(unresolved, list) and any(\n        isinstance(row, Mapping)\n        and row.get(\"status\") == \"open\"\n        and row.get(\"resolution_route\") == \"default_policy\"\n        for row in unresolved\n    )\n'''
    if "def _research_stage_needed(" not in text:
        if text.count(helper_marker) != 1:
            raise RuntimeError("pipeline helper insertion marker changed")
        text = text.replace(helper_marker, helper + helper_marker, 1)

    research_start = text.index(
        '    try:\n'
        '        state = _transition(\n'
        '            "collect_prompt_research",\n'
    )
    compile_marker = (
        '\n    try:\n'
        '        state = _transition(\n'
        '            "compile_researched_requirements",\n'
    )
    research_end = text.index(compile_marker, research_start)
    research_block = text[research_start:research_end]
    if not research_block.startswith("    if _research_stage_needed(state):"):
        guarded = "    if _research_stage_needed(state):\n" + textwrap.indent(
            research_block, "    "
        )
        text = text[:research_start] + guarded + text[research_end:]

    path.write_text(text, encoding="utf-8")


def patch_stale_verifier_test() -> None:
    path = ROOT / "tests" / "test_generation_verifier_resilience.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("import queue\nimport threading\nimport time\nfrom collections import deque\n", "")
    text = text.replace("    _collect_diagnostics_progress_aware,\n", "")
    text = text.replace(
        "def test_jdt_failure_is_fail_closed_without_gradle_fallback(tmp_path):",
        "def test_jdt_failure_fails_closed_when_no_verification_backend_is_healthy(tmp_path):",
    )
    text = text.replace(
        'with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="JDT is unavailable"):',
        'with pytest.raises(\n        agent_tool_runtime.AgentToolRuntimeError,\n        match="Generation verification has no healthy backend",\n    ):',
    )
    text = text.replace(
        "def diagnostics(_rpc, *, expected_uris, timeout_seconds, quiet_seconds):\n",
        "def diagnostics(\n        _rpc, *, expected_uris, timeout_seconds, quiet_seconds, deadline=None\n    ):\n",
    )
    legacy_test = "\ndef test_jdt_progress_refreshes_idle_deadline():\n"
    if legacy_test in text:
        text = text[: text.index(legacy_test)].rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_applicability_tests() -> None:
    path = ROOT / "tests" / "test_planning_detail_applicability.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "    WORKSHEET_SECTIONS,\n",
        "    CONDITIONAL_WORKSHEET_SECTIONS,\n    WORKSHEET_SECTIONS,\n",
        1,
    )
    text = text.replace(
        "def test_missing_applicability_keeps_full_fail_safe_contract() -> None:\n"
        "    assert required_detail_sections_for_requirement(_requirement()) == WORKSHEET_SECTIONS\n",
        "def test_missing_applicability_uses_minimal_core_without_speculative_work() -> None:\n"
        "    assert required_detail_sections_for_requirement(_requirement()) == CORE_WORKSHEET_SECTIONS\n",
    )
    text = text.replace(
        "def test_unknown_applicability_keeps_conditional_sections() -> None:",
        "def test_unknown_applicability_does_not_schedule_conditional_sections() -> None:",
    )
    text = text.replace("    assert selected == WORKSHEET_SECTIONS\n", "    assert selected == CORE_WORKSHEET_SECTIONS\n", 1)
    text = text.replace(
        "def test_required_and_unknown_conditionals_are_retained() -> None:",
        "def test_only_required_conditionals_are_retained() -> None:",
    )
    text = text.replace(
        '    assert "resources_and_ui" in selected\n',
        '    assert "resources_and_ui" not in selected\n',
        1,
    )
    text = text.replace(
        "def test_prompt_wording_never_omits_a_section() -> None:",
        "def test_prompt_wording_never_expands_section_selection() -> None:",
    )
    text = text.replace(
        "    assert required_detail_sections_for_requirement(requirement) == WORKSHEET_SECTIONS\n",
        "    assert required_detail_sections_for_requirement(requirement) == CORE_WORKSHEET_SECTIONS\n",
        1,
    )
    text = text.replace(
        '        "REQ-2": WORKSHEET_SECTIONS,\n',
        '        "REQ-2": CORE_WORKSHEET_SECTIONS,\n',
    )
    old_expected = '''    assert normalized == {\n        "authority_and_network": "unknown",\n        "persistence": "not_applicable",\n        "resources_and_ui": "unknown",\n    }\n'''
    new_expected = '''    assert normalized == {\n        section: ("not_applicable" if section == "persistence" else "unknown")\n        for section in CONDITIONAL_WORKSHEET_SECTIONS\n    }\n'''
    text = text.replace(old_expected, new_expected, 1)
    path.write_text(text, encoding="utf-8")


def patch_resume_regression_test() -> None:
    path = ROOT / "tests" / "test_planning_progress_monotone.py"
    text = path.read_text(encoding="utf-8")
    name = "test_checkpointed_atomic_results_are_not_regenerated_after_scheduler_interrupt"
    if name in text:
        return

    test = r'''


def test_checkpointed_atomic_results_are_not_regenerated_after_scheduler_interrupt(monkeypatch):
    requirements = _requirements(1, acceptance_count=2)
    _patch_compile_boundaries(monkeypatch, requirements)
    monkeypatch.setattr(adaptive, "router_native_model_parallelism", lambda _router: 1)
    checkpoints: list[dict[str, object]] = []
    generated: list[int] = []

    def compile_criterion(_router, *, requirement_ref, criterion_index, **_kwargs):
        generated.append(criterion_index)
        return _real_fragment()

    monkeypatch.setattr(adaptive, "_compile_criterion", compile_criterion)

    def interrupt_after_persisting_all(
        items,
        worker,
        *,
        max_workers,
        stage,
        sort_key=None,
        on_result=None,
    ):
        del max_workers, sort_key
        pending = list(items)
        for item in pending:
            receipt = worker(item)
            assert on_result is not None
            on_result(item, receipt)
        raise adaptive.ParallelTaskError(
            stage=stage,
            item=pending[-1],
            cause=ValueError("forced interruption after durable checkpoint"),
        )
        yield  # pragma: no cover

    monkeypatch.setattr(
        adaptive,
        "iter_completed_with_deadlines",
        interrupt_after_persisting_all,
    )

    with pytest.raises(ValueError, match="forced interruption after durable checkpoint"):
        adaptive.compile_progress_monotone_detailed_plans(
            _Router(),
            "prompt",
            _base_state(),
            required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
            checkpoint=lambda state: checkpoints.append(deepcopy(state)),
        )

    assert sorted(generated) == [0, 1]
    assert checkpoints
    resume_state = deepcopy(checkpoints[-1])
    assert len(resume_state.get("detail_progress", [])) == 2

    monkeypatch.setattr(
        adaptive,
        "_compile_criterion",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpointed criterion must not be regenerated on resume"
        ),
    )

    def forbidden_scheduler(*_args, **_kwargs):
        pytest.fail("resume must assemble fully checkpointed work before scheduling")
        yield  # pragma: no cover

    monkeypatch.setattr(adaptive, "iter_completed_with_deadlines", forbidden_scheduler)

    result = adaptive.compile_progress_monotone_detailed_plans(
        _Router(),
        "prompt",
        resume_state,
        required_sections_by_requirement={"req_1": WORKSHEET_SECTIONS},
    )

    assert result["plan_ready"] is True
    assert result["detail_progress"] == []
'''
    path.write_text(text.rstrip() + test.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    patch_pipeline()
    patch_stale_verifier_test()
    patch_applicability_tests()
    patch_resume_regression_test()


if __name__ == "__main__":
    main()
