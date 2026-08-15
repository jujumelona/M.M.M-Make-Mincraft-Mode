from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.runner import BuildReport, CommandResult
from minecraft_mod_ai.spec import (
    ContentKind,
    ContentSpec,
    PlatformLock,
    Proposal,
    SpecValidationError,
)


def test_every_platform_component_is_exactly_pinned(
    synthetic_platform_lock: PlatformLock,
) -> None:
    platform = synthetic_platform_lock
    platform.validate()
    for field_name in (
        "yarn_mappings",
        "fabric_loader",
        "fabric_api",
        "fabric_loom",
        "gradle",
    ):
        drifted = replace(platform, **{field_name: "unreviewed-version"})
        with pytest.raises(SpecValidationError, match=r"Platform (lock|adapter)"):
            drifted.validate()


def test_recipe_string_is_not_coerced_to_boolean() -> None:
    proposal = MinecraftModPipeline().plan("Create a frost item")
    payload = proposal.to_dict()
    payload["spec"]["contents"][0]["recipe"] = "false"
    payload["approval_hash"] = ""

    with pytest.raises(SpecValidationError, match="JSON boolean"):
        Proposal.from_dict(payload)


def test_java_and_registry_identifier_collisions_fail_in_the_spec() -> None:
    base = MinecraftModPipeline().plan(
        "Create a frost boss and item"
    ).spec
    with pytest.raises(SpecValidationError, match="reserved package"):
        replace(base, package_name="com.class.generated").validate()
    with pytest.raises(
        SpecValidationError,
        match="platform package prefix",
    ):
        replace(base, package_name="java.user.generated").validate()
    with pytest.raises(
        SpecValidationError,
        match="generated Java field",
    ):
        replace(
            base,
            contents=(
                ContentSpec(
                    "logger",
                    ContentKind.ITEM,
                    "Logger",
                    "로거",
                ),
            ),
        ).validate()
    with pytest.raises(SpecValidationError, match="spawn egg ID"):
        replace(
            base,
            contents=(
                ContentSpec(
                    f"{base.boss.entity_id}_spawn_egg",
                    ContentKind.ITEM,
                    "Collision Egg",
                    "충돌 알",
                ),
            ),
        ).validate()
    with pytest.raises(SpecValidationError, match="Reserved mod_id"):
        replace(base, mod_id="minecraft").validate()


def test_boss_spawn_egg_model_is_generated_without_map_commands(tmp_path: Path) -> None:
    proposal = MinecraftModPipeline().plan(
        "Create a frost boss, 3D model, item and block"
    )
    spec = proposal.spec
    FabricProjectGenerator().generate(spec, tmp_path)

    egg_model = (
        tmp_path
        / f"src/main/resources/assets/{spec.mod_id}/models/item/"
        f"{spec.boss.entity_id}_spawn_egg.json"
    )
    assert egg_model.is_file()
    assert not list((tmp_path / "src/main/resources/data" / spec.mod_id).rglob("*.mcfunction"))


def test_gametest_gate_requires_real_passing_xml(tmp_path: Path) -> None:
    spec = MinecraftModPipeline().plan("Create a frost item").spec
    report_path = tmp_path / "gametest-report.xml"
    command = CommandResult(
        name="gametest",
        command=("gradle", "runGameTestServer"),
        exit_code=0,
        duration_seconds=1.0,
        log_path=str(tmp_path / "gametest.log"),
    )

    missing_report = BuildReport(
        status="PASS",
        gradle_version="8.5",
        commands=(command,),
        jar_path=None,
        gametest_report=str(report_path),
    )
    assert not MinecraftModPipeline._gametest_passed(
        missing_report,
        spec,
    )

    report_path.write_text(
        '<testsuite><testcase name="generatedRegistriesAreLive">'
        '<failure message="boom"/></testcase></testsuite>',
        encoding="utf-8",
    )
    assert not MinecraftModPipeline._gametest_passed(
        missing_report,
        spec,
    )

    report_path.write_text(
        '<testsuite><testcase name="generatedRegistriesAreLive">'
        '<skipped/></testcase></testsuite>',
        encoding="utf-8",
    )
    assert not MinecraftModPipeline._gametest_passed(
        missing_report,
        spec,
    )

    report_path.write_text(
        '<testsuite><testcase '
        'name="frostworksmodgametests.generatedRegistriesAreLive"/>'
        '</testsuite>',
        encoding="utf-8",
    )
    assert MinecraftModPipeline._gametest_passed(
        missing_report,
        spec,
    )


def test_external_gradle_cache_is_rejected_before_writes(
    tmp_path: Path,
) -> None:
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan("Create a frost item")
    output = tmp_path / "approved-output"

    with pytest.raises(
        SpecValidationError,
        match="inside the approved output",
    ):
        pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=output,
            gradle_cache=tmp_path / "outside-cache",
            build=False,
        )
    assert not output.exists()


def test_release_obj_material_keeps_its_texture_handoff(
    tmp_path: Path,
) -> None:
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan(
        "Create a frost boss with a 3D model and item"
    )
    result = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=tmp_path / "output",
        build=False,
    )
    art_root = Path(result.release_dir) / "art_sources"
    texture_name = f"{proposal.spec.boss.entity_id}.png"
    assert (art_root / texture_name).is_file()
    material = (
        art_root / f"{proposal.spec.boss.entity_id}.mtl"
    ).read_text(encoding="utf-8")
    assert f"map_Kd {texture_name}" in material
    assert "../../src/" not in material
