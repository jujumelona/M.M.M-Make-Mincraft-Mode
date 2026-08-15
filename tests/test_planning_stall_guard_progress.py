from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.agentic_pre_design_rag as pre_design_rag
import minecraft_mod_ai.agentic_research_game_design as agentic_research
import minecraft_mod_ai.planning_stall_guard_contract as stall_guard


class _OneHeartbeat:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self, _interval: float) -> bool:
        self.calls += 1
        return self.calls > 1


def test_heartbeat_repeats_last_known_structured_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = stall_guard._PlanningProgress()
    progress.record(total=7, checkpoint="brief-ready")
    progress.begin_domain("mk:resources")
    progress.begin_page("mk:resources", page=2, page_total=5)
    progress.begin_page("mk:resources", page=2, page_total=5)
    progress.record(checkpoint="checkpoint-saved")
    monkeypatch.setattr(stall_guard, "_env_float", lambda *args, **kwargs: 0.0)

    stall_guard._heartbeat("pre-design", _OneHeartbeat(), 0.0, progress)

    output = capsys.readouterr().out
    assert "pre-design still running" in output
    assert "stage=page-retry" in output
    assert "domain=mk:resources" in output
    assert "page=2/5" in output
    assert "attempt=2" in output
    assert "checkpoint=checkpoint-saved" in output
    assert "completed=0" in output
    assert "total=7" in output
    assert "idle=" in output
    assert "elapsed=" in output


def test_page_continuation_and_retry_are_distinct_progress_states() -> None:
    progress = stall_guard._PlanningProgress()

    progress.begin_page(
        "mk:item.registration",
        page=1,
        page_total=4,
        continuation_offset=512,
    )
    first = progress.snapshot()
    assert first["stage"] == "page-continuation"
    assert first["attempt"] == 1
    assert first["checkpoint"] == "page-offset-512"

    progress.begin_page(
        "mk:item.registration",
        page=1,
        page_total=4,
        continuation_offset=512,
    )
    second = progress.snapshot()
    assert second["stage"] == "page-retry"
    assert second["attempt"] == 2


def test_progress_source_wrappers_report_domains_pages_retries_and_checkpoints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {"worker": 0, "page": 0, "write": 0}
    pre_design = SimpleNamespace()

    def normalize(_prompt: str, _seed: dict) -> dict:
        return {"domains": [{"domain_id": "one"}, {"domain_id": "two"}]}

    def materialize(domain_id: str, _evidence: dict) -> dict:
        return {"domain_id": domain_id, "page_count": 3}

    def page_messages(*_args, **_kwargs) -> list:
        calls["page"] += 1
        return []

    def write(_path: Path, _content: str) -> None:
        calls["write"] += 1

    def worker(
        _router,
        *,
        prompt: str,
        domain: dict,
        deterministic: dict,
        trace_metadata: dict | None,
    ) -> dict:
        del prompt, deterministic, trace_metadata
        calls["worker"] += 1
        document = pre_design._materialize_domain_evidence_document(
            domain["domain_id"],
            {},
        )
        page = {"page_index": 1, "page_count": 3}
        for _attempt in range(2):
            pre_design._research_page_messages(
                domain=domain,
                document=document,
                page=page,
            )
        pre_design._atomic_write_text(Path("one.checkpoint.json"), "{}")
        return {"domain_id": domain["domain_id"], "sufficient": True}

    agentic = SimpleNamespace(
        normalize_research_brief=normalize,
        _research_domain_with_agent=worker,
    )
    pre_design._materialize_domain_evidence_document = materialize
    pre_design._research_page_messages = page_messages
    pre_design._atomic_write_text = write

    stall_guard._patch_pre_design_progress_sources(agentic, pre_design)
    # A second install must not stack duplicate wrappers or duplicate completion counts.
    stall_guard._patch_pre_design_progress_sources(agentic, pre_design)

    progress = stall_guard._PlanningProgress()
    token = stall_guard._ACTIVE_PROGRESS.set(progress)
    try:
        agentic.normalize_research_brief("prompt", {})
        result = agentic._research_domain_with_agent(
            object(),
            prompt="prompt",
            domain={"domain_id": "one"},
            deterministic={},
            trace_metadata=None,
        )
    finally:
        stall_guard._ACTIVE_PROGRESS.reset(token)

    assert result["domain_id"] == "one"
    assert calls == {"worker": 1, "page": 2, "write": 1}
    snapshot = progress.snapshot()
    assert snapshot["domain"] == "one"
    assert snapshot["page"] == 2
    assert snapshot["page_total"] == 3
    assert snapshot["attempt"] == 2
    assert snapshot["completed"] == 1
    assert snapshot["total"] == 2
    assert snapshot["checkpoint"] == "domain-saved"

    output = capsys.readouterr().out
    assert "stage=research-brief" in output
    assert "stage=page-retry" in output
    assert "checkpoint=checkpoint-saved" in output
    assert "stage=domain-complete" in output
    assert "completed=1" in output
    assert "gaps=0" in output
    assert "terminal=1" in output
    assert "total=2" in output


def test_live_wrapper_composition_has_one_progress_owner_per_probe() -> None:
    live_collect = agentic_research.collect_pre_design_research
    assert live_collect.__dict__.get("_mmm_pre_design_heartbeat") is live_collect
    assert pre_design_rag._PROGRESS_HOOK is stall_guard._research_progress_hook
    owners = (
        (
            agentic_research.normalize_research_brief,
            stall_guard._PROGRESS_NORMALIZE_MARKER,
        ),
        (
            agentic_research._research_domain_with_agent,
            stall_guard._PROGRESS_WORKER_MARKER,
        ),
        (
            pre_design_rag._materialize_domain_evidence_document,
            stall_guard._PROGRESS_MATERIALIZE_MARKER,
        ),
        (
            pre_design_rag._research_page_messages,
            stall_guard._PROGRESS_PAGE_MARKER,
        ),
        (
            pre_design_rag._atomic_write_text,
            stall_guard._PROGRESS_WRITE_MARKER,
        ),
    )
    for function, marker in owners:
        assert function.__dict__.get(marker) is function


def test_structured_research_hook_maps_exact_checkpoint_events() -> None:
    progress = stall_guard._PlanningProgress()
    progress.record(total=7)
    progress_token = stall_guard._ACTIVE_PROGRESS.set(progress)
    cursor_token = stall_guard._ACTIVE_PROGRESS_CURSOR.set({})
    try:
        stall_guard._research_progress_hook(
            {"event": "domain_start", "domain_id": "mk:resources", "page_count": 5}
        )
        stall_guard._research_progress_hook(
            {
                "event": "page_start",
                "domain_id": "mk:resources",
                "page_index": 2,
                "page_count": 5,
            }
        )
        stall_guard._research_progress_hook(
            {
                "event": "page_ledgered",
                "domain_id": "mk:resources",
                "page_index": 2,
                "page_count": 5,
                "page_ref": "must-not-be-rendered",
            }
        )
        ledgered = progress.snapshot()
        assert ledgered["stage"] == "page-ledgered"
        assert ledgered["checkpoint"] == "page-ledger-saved"
        stall_guard._research_progress_hook(
            {"event": "bounded_json_repair", "attempt": 2, "error": "ignored"}
        )
        repair = progress.snapshot()
        assert repair["stage"] == "bounded-json-repair"
        assert repair["domain"] == "mk:resources"
        assert repair["page"] == 2
        assert repair["page_total"] == 5
        assert repair["attempt"] == 2
        assert repair["checkpoint"] == "repairing-bounded-json"

        stall_guard._research_progress_hook(
            {"event": "page_checkpoint_hit", "offset": 4096, "label": "ignored"}
        )
        assert progress.snapshot()["checkpoint"] == "page-offset-4096-loaded"

        stall_guard._research_progress_hook(
            {
                "event": "page_adaptive_split",
                "start_offset": 4096,
                "midpoint": 6144,
                "end_offset": 8192,
                "label": "ignored",
            }
        )
        split = progress.snapshot()
        assert split["stage"] == "page-adaptive-split"
        assert split["checkpoint"] == "split-4096-6144-8192"

        stall_guard._research_progress_hook(
            {
                "event": "synthesis_checkpoint_hit",
                "domain_id": "mk:resources",
                "level": 3,
                "group_index": 4,
            }
        )
        synthesis = progress.snapshot()
        assert synthesis["stage"] == "synthesis-checkpoint-hit"
        assert synthesis["checkpoint"] == "synthesis-3-4-loaded"

        stall_guard._research_progress_hook(
            {
                "event": "domain_gap_receipt",
                "domain_id": "mk:resources",
                "status": "failed",
                "page_count": 5,
            }
        )
    finally:
        stall_guard._ACTIVE_PROGRESS_CURSOR.reset(cursor_token)
        stall_guard._ACTIVE_PROGRESS.reset(progress_token)

    terminal = progress.snapshot()
    assert terminal["stage"] == "domain-gap-receipt"
    assert terminal["checkpoint"] == "domain-gap-saved"
    assert terminal["completed"] == 0
    assert terminal["gaps"] == 1
    assert terminal["terminal"] == 1
    assert terminal["total"] == 7


def test_inherited_heartbeat_marker_does_not_suppress_outer_wrapper() -> None:
    calls: list[str] = []

    def owner(_router, _prompt: str, *, trace_metadata=None) -> dict:
        del trace_metadata
        calls.append("owner")
        return {"ok": True}

    owner._mmm_pre_design_heartbeat = owner  # type: ignore[attr-defined]

    @wraps(owner)
    def composed(_router, _prompt: str, *, trace_metadata=None) -> dict:
        del trace_metadata
        calls.append("composed")
        return {"ok": True}

    # wraps copied the old owner's marker onto composed, but composed does not own it.
    assert composed.__dict__["_mmm_pre_design_heartbeat"] is owner
    module = SimpleNamespace(collect_pre_design_research=composed)

    stall_guard._patch_pre_design_observability(module)
    observed = module.collect_pre_design_research
    assert observed is not composed
    assert observed.__dict__["_mmm_pre_design_heartbeat"] is observed
    assert observed.__wrapped__ is composed

    # Reapplying the patch recognizes the real owner and remains idempotent.
    stall_guard._patch_pre_design_observability(module)
    assert module.collect_pre_design_research is observed
    assert observed(object(), "prompt") == {"ok": True}
    assert calls == ["composed"]


def test_copied_worker_contexts_update_one_progress_snapshot_safely() -> None:
    progress = stall_guard._PlanningProgress()
    progress.record(total=12)
    token = stall_guard._ACTIVE_PROGRESS.set(progress)
    try:
        contexts = [copy_context() for _ in range(12)]

        def update(domain_index: int) -> None:
            domain = f"domain-{domain_index}"
            assert stall_guard.report_planner_research_progress(
                stage="page-research",
                domain=domain,
                page=domain_index + 1,
                page_total=12,
                attempt=1,
                checkpoint="checkpoint-saved",
                completed_domain=domain,
                emit=False,
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(context.run, update, index)
                for index, context in enumerate(contexts)
            ]
            for future in futures:
                future.result()
    finally:
        stall_guard._ACTIVE_PROGRESS.reset(token)

    snapshot = progress.snapshot()
    assert snapshot["completed"] == 12
    assert snapshot["total"] == 12
    assert snapshot["stage"] == "page-research"
    assert snapshot["checkpoint"] == "checkpoint-saved"
    assert not stall_guard.report_planner_research_progress(emit=False)


def test_progress_report_is_request_local_and_sanitizes_console_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not stall_guard.report_planner_research_progress(
        stage="outside",
        emit=False,
    )

    progress = stall_guard._PlanningProgress()
    token = stall_guard._ACTIVE_PROGRESS.set(progress)
    try:
        assert stall_guard.report_planner_research_progress(
            stage="page\nretry",
            domain="domain\rname",
            checkpoint="saved checkpoint",
            completed_domain="domain\rname",
        )
    finally:
        stall_guard._ACTIVE_PROGRESS.reset(token)

    output = capsys.readouterr().out
    assert "stage=page_retry" in output
    assert "domain=domain_name" in output
    assert "checkpoint=saved_checkpoint" in output
    assert "\r" not in output
    assert "\nretry" not in output
    assert progress.snapshot()["completed"] == 1


def test_observed_failure_stops_request_local_progress(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def collect(_router, _prompt: str, *, trace_metadata=None):
        del trace_metadata
        assert stall_guard.report_planner_research_progress(
            stage="page-research",
            domain="broken",
            page=4,
            page_total=9,
            attempt=3,
            checkpoint="checkpoint-saved",
            total=7,
            emit=False,
        )
        raise RuntimeError("boom")

    module = SimpleNamespace(collect_pre_design_research=collect)
    stall_guard._patch_pre_design_observability(module)

    with pytest.raises(RuntimeError, match="boom"):
        module.collect_pre_design_research(object(), "prompt")

    output = capsys.readouterr().out
    assert "stage=failed" in output
    assert "domain=broken" in output
    assert "page=4/9" in output
    assert "attempt=3" in output
    assert "checkpoint=last-safe-checkpoint" in output
    assert not stall_guard.report_planner_research_progress(emit=False)
