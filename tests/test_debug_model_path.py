"""Debug's default must exercise real authored-plan ABI, not the old host fixture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.colab_run_modes import (
    DEBUG_DEFAULT_PROMPT,
    FULL_MODE,
    run_plan_dialog,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def _plan(prompt: str) -> AuthoredPlan:
    text = "# Behavior\nA crystal item with a crafting recipe.\n"
    specification = {key: [] for key in DETAIL_RECORDS["verification"]}
    specification["inapplicable_concerns"] = []
    return AuthoredPlan(
        requested_prompt=prompt,
        text=text,
        structured_sections={
            "verification": {
                "specification": specification,
                "constraint_evidence_refs": [],
            },
        },
        typed_plan_ir={
            "schema_version": "mmm/typed-plan-ir-v1",
            "source_sha256": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "functions": [],
            "initialize": [],
            "platform_modules": [],
            "event_bindings": [],
        },
        content_design={
            "modules": [
                {
                    "module_id": "crystal_fragment",
                    "kind": "item",
                    "config": {},
                    "depends_on": [],
                    "required_gates": [],
                }
            ],
            "_implementation_facts": [
                {
                    "fact_id": "crystal_fragment.exists",
                    "fact_type": "item_exists",
                    "subject": "crystal_fragment",
                }
            ],
        },
    )


class _ModelSession:
    def __init__(self, *, corrupt_reload: bool = False) -> None:
        self.prompts: list[str] = []
        self.save_targets: list[Path | None] = []
        self.loads = 0
        self.corrupt_reload = corrupt_reload
        self.proposal: AuthoredPlan | None = None

    def plan(self, prompt: str, *, save_plan_path: Path | None = None):
        self.prompts.append(prompt)
        self.save_targets.append(save_plan_path)
        self.proposal = _plan(prompt)
        return SimpleNamespace(complete_proposal=self.proposal, message=self.proposal.text)

    def save_plan(self, target: Path):
        assert self.proposal is not None
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.proposal.to_dict()), encoding="utf-8")
        return path

    def load_plan(self, path: Path):
        self.loads += 1
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.corrupt_reload:
            data["text"] += "modified"
        plan = AuthoredPlan.from_dict(data)
        return SimpleNamespace(complete_proposal=plan, message=plan.text)


def test_model_path_uses_real_authored_output_and_roundtrip(tmp_path: Path) -> None:
    session = _ModelSession()
    target = tmp_path / "proposal.json"
    result = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="custom user requested gameplay",
        plan_path=target,
        debug_mode=True, debug_strategy="model_path",
        print_fn=lambda *args, **kwargs: None,
    )
    assert session.prompts == ["custom user requested gameplay"]
    assert session.save_targets == [target]
    assert session.loads == 1
    assert result.plan_path == target
    assert isinstance(result.reply.complete_proposal, AuthoredPlan)
    assert result.reply.complete_proposal.content_design["modules"][0]["module_id"] == "crystal_fragment"
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == "mmm/authored-plan-v2"


def test_debug_default_routes_user_prompt_into_real_model(tmp_path: Path) -> None:
    session = _ModelSession()
    result = run_plan_dialog(
        session=session, run_mode=FULL_MODE,
        prompt="grow seasonal crops and cook recipes",
        plan_path=tmp_path / "proposal.json",
        debug_mode=True,
        print_fn=lambda *args, **kwargs: None,
    )
    assert session.prompts == ["grow seasonal crops and cook recipes"]
    assert result.reply.complete_proposal.requested_prompt == session.prompts[0]


def test_model_path_rejects_plan_for_different_user_prompt(tmp_path: Path) -> None:
    session = _ModelSession()

    def wrong_plan(_prompt: str, *, save_plan_path=None):
        generated = _plan("unrelated canned crystal fixture")
        return SimpleNamespace(complete_proposal=generated, message=generated.text)

    session.plan = wrong_plan
    with pytest.raises(RuntimeError, match="DEBUG_MODEL_PROMPT_BINDING_MISMATCH"):
        run_plan_dialog(
            session=session, run_mode=FULL_MODE, prompt="seasonal farming",
            plan_path=tmp_path / "proposal.json",
            debug_mode=True, print_fn=lambda *args, **kwargs: None,
        )


def test_api_model_plan_can_save_to_debug_without_overwriting_regular_plan(
    tmp_path: Path, monkeypatch,
) -> None:
    import minecraft_mod_ai.api as api

    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    session = api.CompleteModAISession.__new__(api.CompleteModAISession)
    session.output_root = tmp_path
    session.existing_input = None
    session.brief = ""
    session.complete_proposal = None
    session.planner = SimpleNamespace(plan=lambda prompt, **_kwargs: _plan(prompt))
    ordinary = tmp_path / "proposal.json"
    ordinary.write_text("original user plan", encoding="utf-8")
    target = tmp_path / "debug" / "model_path-proposal.json"

    result = session.plan("an original user request", save_plan_path=target)

    assert target.is_file()
    assert result.complete_proposal.requested_prompt == "an original user request"
    assert ordinary.read_text(encoding="utf-8") == "original user plan"


def test_model_path_default_prompt_runs_model_not_host_fixture(tmp_path: Path) -> None:
    session = _ModelSession()
    run_plan_dialog(
        session=session, run_mode=FULL_MODE, prompt="   ",
        plan_path=tmp_path / "proposal.json", debug_mode=True, debug_strategy="model_path",
        print_fn=lambda *args, **kwargs: None,
    )
    assert session.prompts == [DEBUG_DEFAULT_PROMPT]


def test_model_path_fails_closed_when_reload_changes_authored_plan(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="DEBUG_MODEL_PLAN_ROUNDTRIP_MISMATCH"):
        run_plan_dialog(
            session=_ModelSession(corrupt_reload=True),
            run_mode=FULL_MODE, prompt="real gameplay",
            plan_path=tmp_path / "proposal.json", debug_mode=True, debug_strategy="model_path",
            print_fn=lambda *args, **kwargs: None,
        )


def test_debug_rejects_unknown_strategy_before_planner(tmp_path: Path) -> None:
    session = _ModelSession()
    with pytest.raises(ValueError, match="지원하지 않는 Debug 전략"):
        run_plan_dialog(
            session=session, run_mode=FULL_MODE, prompt="test",
            plan_path=tmp_path / "proposal.json", debug_mode=True,
            debug_strategy="unverified_stub",
            print_fn=lambda *args, **kwargs: None,
        )
    assert session.prompts == []


def test_model_replay_uses_saved_actual_authored_plan_without_planner(tmp_path: Path) -> None:
    session = _ModelSession()
    target = tmp_path / "proposal.json"
    original = _plan("recorded model choice")
    target.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    replayed = run_plan_dialog(
        session=session, run_mode=FULL_MODE, prompt="ignored on replay",
        plan_path=target, debug_mode=True, debug_strategy="model_replay",
        print_fn=lambda *args, **kwargs: None,
    )
    assert session.prompts == []
    assert session.loads == 1
    assert replayed.reply.complete_proposal.calculate_hash() == original.calculate_hash()


def test_model_replay_requires_original_recorded_plan(tmp_path: Path) -> None:
    session = _ModelSession()
    with pytest.raises(FileNotFoundError, match="DEBUG_MODEL_REPLAY_PLAN_MISSING"):
        run_plan_dialog(
            session=session, run_mode=FULL_MODE, prompt="",
            plan_path=tmp_path / "missing.json", debug_mode=True,
            debug_strategy="model_replay", print_fn=lambda *_, **__: None,
        )
    assert session.prompts == []


def test_debug_rejects_empty_non_executable_model_plan(tmp_path: Path) -> None:
    session = _ModelSession()

    def empty_plan(prompt: str, *, save_plan_path=None):
        original = _plan(prompt)
        value = original.to_dict()
        value["content_design"] = {}
        plan = AuthoredPlan.from_dict(value)
        return SimpleNamespace(complete_proposal=plan, message=plan.text)

    session.plan = empty_plan
    with pytest.raises(RuntimeError, match="DEBUG_MODEL_PLAN_EMPTY_IMPLEMENTATION"):
        run_plan_dialog(
            session=session, run_mode=FULL_MODE, prompt="empty",
            plan_path=tmp_path / "proposal.json", debug_mode=True, debug_strategy="model_path",
            print_fn=lambda *_, **__: None,
        )
