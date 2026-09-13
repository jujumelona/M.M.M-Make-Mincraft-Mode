from __future__ import annotations

import json
from types import SimpleNamespace

import minecraft_mod_ai.complete_orchestrator as orchestrator
from minecraft_mod_ai import prepared_project_resume_integrity as resume_integrity
from minecraft_mod_ai.scalable_generator import ScalableFabricProjectGenerator
from minecraft_mod_ai.spec import ModSpec


def _spec(platform) -> ModSpec:
    return ModSpec(
        mod_id="resume_fixture",
        mod_name="Resume Fixture",
        package_name="dev.mmm.resumefixture",
        version="1.0.0",
        summary="resume integrity fixture",
        contents=(),
        platform=platform,
    )


def test_resume_rejects_stale_base_workspace(
    tmp_path, synthetic_platform_lock, monkeypatch
) -> None:
    spec = _spec(synthetic_platform_lock)
    root = tmp_path / "project"
    ScalableFabricProjectGenerator().generate(spec, root)

    assert resume_integrity.prepared_project_matches_spec(root, spec)
    assert orchestrator.CompleteProductionOrchestrator._project_matches_spec(root, spec)
    assert getattr(
        orchestrator.CompleteProductionOrchestrator._project_matches_spec,
        "_mmm_prepared_project_resume_integrity",
        False,
    )

    metadata = root / ".minecraft_ai"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "base-proposal.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        resume_integrity.Proposal,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(spec=spec)),
    )
    assert orchestrator.CompleteProductionOrchestrator._valid_project_root(root)

    pack = root / "src/main/resources/pack.mcmeta"
    pack.unlink()
    assert not resume_integrity.prepared_project_matches_spec(root, spec)
    assert not orchestrator.CompleteProductionOrchestrator._valid_project_root(root)


def test_resume_rejects_platform_lock_drift(tmp_path, synthetic_platform_lock) -> None:
    spec = _spec(synthetic_platform_lock)
    root = tmp_path / "project"
    ScalableFabricProjectGenerator().generate(spec, root)

    fabric_path = root / "src/main/resources/fabric.mod.json"
    fabric = json.loads(fabric_path.read_text(encoding="utf-8"))
    fabric["depends"]["minecraft"] = "stale-version"
    fabric_path.write_text(json.dumps(fabric, indent=2) + "\n", encoding="utf-8")

    assert not resume_integrity.prepared_project_matches_spec(root, spec)
