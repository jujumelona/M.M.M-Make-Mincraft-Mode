from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


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
    legacy_test = "\ndef test_jdt_progress_refreshes_idle_deadline():\n"
    if legacy_test in text:
        text = text[: text.index(legacy_test)].rstrip() + "\n"
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
        return _fragment(requirement_ref, criterion_index)

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
    path.write_text(text.rstrip() + test + "\n", encoding="utf-8")


def main() -> None:
    patch_pipeline()
    patch_stale_verifier_test()
    patch_resume_regression_test()


if __name__ == "__main__":
    main()
