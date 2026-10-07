from __future__ import annotations

import zipfile
from types import SimpleNamespace

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.validator import (
    ProjectValidator,
    _complete_artifact_owner_ids,
    _complete_native_modules,
    _validate_complete_jar,
)


def _artifact_owned_gui_proposal():
    module = ProductionModule(
        module_id="spaceport_planet_target_list_gui_economy_system_001",
        kind="gui",
        config={
            "requires_custom_generation": True,
            "executable_relations": [],
        },
    )
    job = ArtifactJob(
        job_id="gui_candidate_job",
        template_id="candidate/gui",
        owner_module=module.module_id,
        target_path=(
            "src/main/java/ai/minecraft/generated/demo/client/generated/"
            "SpaceportPlanetTargetList.java"
        ),
        operation="create",
        context_id="ctx",
        canonical_leaf="gui.screen",
        implementation_id="template:gui.screen",
    )
    return SimpleNamespace(
        modules=(module,),
        game_design={"_artifact_jobs": [job.to_dict()]},
    )


def test_artifact_owned_gui_does_not_require_native_system_source(tmp_path) -> None:
    proposal = _artifact_owned_gui_proposal()

    assert _complete_artifact_owner_ids(proposal) == {
        "spaceport_planet_target_list_gui_economy_system_001"
    }
    assert _complete_native_modules(proposal) == ()

    findings = []
    validator = ProjectValidator()
    validator._validate_complete_sources(
        tmp_path,
        SimpleNamespace(
            package_name="ai.minecraft.generated.demo",
            mod_id="demo",
        ),
        proposal,
        findings,
    )

    assert not [
        finding
        for finding in findings
        if finding.code == "COMPLETE_SYSTEM_SOURCE_MISSING"
    ]


def test_artifact_owned_gui_does_not_require_native_system_class_in_jar(
    tmp_path,
) -> None:
    proposal = _artifact_owned_gui_proposal()
    archive_path = tmp_path / "empty.jar"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("fabric.mod.json", "{}")

    findings = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        _validate_complete_jar(
            archive,
            set(archive.namelist()),
            SimpleNamespace(package_name="ai.minecraft.generated.demo"),
            proposal,
            findings,
        )

    assert not [
        finding
        for finding in findings
        if finding.code == "JAR_CLASS_MISSING"
        and "GuiNetworkingSystem" in finding.path
    ]
