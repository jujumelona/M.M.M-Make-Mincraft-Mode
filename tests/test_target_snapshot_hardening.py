from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import api, complete_orchestrator, complete_planner
from minecraft_mod_ai.spec import SpecValidationError


def _write_existing_project(root: Path) -> None:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "settings.gradle.kts").write_text(
        'rootProject.name = "weather-existing"\n', encoding="utf-8"
    )
    (root / "gradle.properties").write_text(
        "minecraft_version=1.21.1\nloader_version=0.16.10\n",
        encoding="utf-8",
    )
    (root / "src/main/java/example/WeatherCompass.java").write_text(
        "package example; public final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "weather_existing",
                "version": "1.0.0",
                "depends": {"minecraft": "1.21.1"},
            }
        ),
        encoding="utf-8",
    )


def _archive_project(project: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as output:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                output.write(path, "project/" + path.relative_to(project).as_posix())


def test_existing_report_inventory_and_proposal_share_one_observed_archive_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    archive = tmp_path / "existing.zip"
    _write_existing_project(project)
    _archive_project(project, archive)
    owner = SimpleNamespace()
    api._attach_existing_target(owner, archive)
    inventory = owner._mmm_existing_project_inventory_future.result(timeout=30)
    expected = owner._mmm_existing_archive_sha256

    assert owner._mmm_existing_project_report["archive_sha256"] == expected
    assert inventory.source_sha256 == expected
    assert api._verified_existing_input_sha256(
        owner, archive, await_inventory=True
    ) == expected

    captured: dict[str, str] = {}

    class Planner:
        def plan(self, prompt, *, media_paths=(), existing_input_sha256=""):
            captured["sha256"] = existing_input_sha256
            return SimpleNamespace(
                requested_prompt=prompt,
                game_design={},
                modules=(),
                acceptance_tests=(),
                existing_input_sha256=existing_input_sha256,
                calculate_hash=lambda: "sha256:" + "a" * 64,
            )

    session = object.__new__(api.CompleteModAISession)
    session.router = owner
    session.existing_input = archive
    session.planner = Planner()
    session.brief = ""
    session.complete_proposal = None
    monkeypatch.setattr(api.CompleteModAISession, "save_plan", lambda self: archive)
    from minecraft_mod_ai import plan_render

    monkeypatch.setattr(plan_render, "render_complete_plan", lambda **_kwargs: "plan")
    reply = session.chat("Keep the existing project.")

    assert captured["sha256"] == expected
    assert reply.complete_proposal.existing_input_sha256 == expected

    archive.write_bytes(b"changed archive bytes")
    with pytest.raises(SpecValidationError, match="changed after the session observed"):
        api._verified_existing_input_sha256(owner, archive, await_inventory=True)


def test_fast_mode_does_not_reduce_model_context_or_completion_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = {
        role: SimpleNamespace(max_context=262_144, max_new_tokens=65_536)
        for role in ("planner", "coder", "researcher", "coder_safe", "visual_critic")
    }

    class Registry:
        def role(self, _profile: str, role: str):
            return configs[role]

    class Router:
        def __init__(self, *, profile: str):
            self.profile = profile
            self.registry = Registry()

    class Planner:
        def __init__(self, router):
            self.router = router

    class Orchestrator:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr("minecraft_mod_ai.model_router.ModelRouter", Router)
    monkeypatch.setattr(complete_planner, "CompleteGameDesignPlanner", Planner)
    monkeypatch.setattr(
        complete_orchestrator, "CompleteProductionOrchestrator", Orchestrator
    )

    session = api.CompleteModAISession(output_root=tmp_path, fast_mode=True)

    assert session.fast_mode is True
    assert {
        (config.max_context, config.max_new_tokens) for config in configs.values()
    } == {(262_144, 65_536)}


def test_retired_target_monkeypatch_owners_are_physically_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (package / "platform_central_ai_contract.py").exists()
    assert not (package / "platform_planning_contract.py").exists()
    assert not (package / "platform_optimizer.py").exists()
    assert (package / "platform_evidence_pipeline.py").exists()
    assert (package / "live_module_lowering.py").exists()
