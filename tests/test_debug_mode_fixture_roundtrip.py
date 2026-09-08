from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.colab_run_modes import (
    FULL_MODE,
    run_plan_dialog,
    write_debug_example_plan,
)
from minecraft_mod_ai.complete_spec import CompleteProposal


def _load_complete_proposal(path: Path) -> CompleteProposal:
    raw = json.loads(path.read_text(encoding="utf-8"))
    proposal = CompleteProposal.from_dict(raw)
    proposal.validate()
    return proposal


def test_debug_example_plan_round_trips_native_name_platform(tmp_path: Path) -> None:
    target = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="26.2",
        loader="fabric",
    )

    raw = json.loads(target.read_text(encoding="utf-8"))
    platform = raw["base_proposal"]["spec"]["platform"]
    assert platform["minecraft_version"] == "26.2"
    assert platform["mappings_kind"] == ""
    assert platform["mappings_version"] == ""
    assert platform["yarn_mappings"] == ""

    _load_complete_proposal(target)


class _DebugSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def plan(self, prompt: str):
        raise AssertionError(f"Debug Mode must not call planner: {prompt}")

    def load_plan(self, path: Path):
        target = Path(path)
        self.calls.append(("load", str(target)))
        proposal = _load_complete_proposal(target)
        return SimpleNamespace(
            message="Debug fixture loaded",
            complete_proposal=proposal,
        )


def test_debug_mode_skips_planner_and_loads_round_trip_valid_fixture(tmp_path: Path) -> None:
    target = tmp_path / "proposal.json"
    session = _DebugSession()

    result = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="",
        plan_path=target,
        debug_mode=True,
        minecraft_version="26.2",
        loader="fabric",
        print_fn=lambda *_, **__: None,
    )

    assert result.approved is True
    assert result.plan_path == target
    assert session.calls == [("load", str(target))]
