from __future__ import annotations

import minecraft_mod_ai.planning_stall_guard_contract as stall_guard


def test_retired_stall_guard_install_is_idempotent_noop() -> None:
    assert stall_guard.install() is None
    assert stall_guard.install() is None


def test_legacy_planner_progress_monkeypatch_api_stays_retired() -> None:
    retired_symbols = {
        "_PlanningProgress",
        "_ACTIVE_PROGRESS",
        "_ACTIVE_PROGRESS_CURSOR",
        "_heartbeat",
        "_patch_pre_design_progress_sources",
        "_patch_pre_design_observability",
        "_research_progress_hook",
        "report_planner_research_progress",
    }
    assert retired_symbols.isdisjoint(vars(stall_guard))


def test_public_surface_exposes_only_retired_install_hook() -> None:
    assert stall_guard.__all__ == ["install"]
