from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.broker import (
    LocalPolicyBroker,
    PolicyDenied,
    ToolAction,
    ToolRequest,
)
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.spec import ProposalStatus, SpecValidationError
from minecraft_mod_ai.validator import ProjectValidator, validate_jar


def _relative_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_plan_does_not_write_and_wrong_approval_hash_cannot_start_execution(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output-must-not-exist"
    pipeline = MinecraftModPipeline()

    proposal = pipeline.plan("Create a frost item and block")

    assert proposal.status is ProposalStatus.AWAITING_APPROVAL
    assert proposal.approval_hash == proposal.calculate_hash()
    assert not output_root.exists()

    with pytest.raises(SpecValidationError, match="Approval hash mismatch"):
        pipeline.execute(
            proposal,
            approval_hash="sha256:" + ("0" * 64),
            output_root=output_root,
            build=False,
        )

    assert not output_root.exists()


def test_tampering_with_a_hashed_proposal_is_detected() -> None:
    proposal = MinecraftModPipeline().plan("Create a storm item")
    data = proposal.to_dict()
    data["requested_prompt"] = "Create an unrelated fire item"

    with pytest.raises(
        SpecValidationError,
        match="approval_hash does not match",
    ):
        type(proposal).from_dict(data)


def test_different_prompts_produce_distinct_non_hardcoded_specs() -> None:
    pipeline = MinecraftModPipeline()

    frost = pipeline.plan("Create a frost item and block")
    ember = pipeline.plan("Create a fire item and block")

    assert frost.requested_prompt != ember.requested_prompt
    assert frost.spec.mod_id == "frost_works"
    assert ember.spec.mod_id == "ember_works"
    assert frost.spec.mod_id != ember.spec.mod_id
    assert [item.content_id for item in frost.spec.contents] != [
        item.content_id for item in ember.spec.contents
    ]
    assert frost.approval_hash != ember.approval_hash
    assert frost.spec.platform == ember.spec.platform


def test_boss_and_3d_outputs_are_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    proposal = MinecraftModPipeline().plan(
        "Create a frost boss with a 3D model, one item and one block"
    )
    spec = proposal.spec
    assert spec.boss is not None

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    generator = FabricProjectGenerator()
    generator.generate(spec, first_root)
    generator.generate(spec, second_root)

    package_path = Path(*spec.package_name.split("."))
    boss_class = "".join(part.capitalize() for part in spec.boss.entity_id.split("_"))
    boss_class += "ModEntity"
    renderer_class = "".join(
        part.capitalize() for part in spec.boss.entity_id.split("_")
    )
    renderer_class += "ModRenderer"

    expected_paths = {
        Path("src/main/java") / package_path / "entity" / f"{boss_class}.java",
        Path("src/main/java") / package_path / "client" / f"{renderer_class}.java",
        Path(
            f"src/main/resources/assets/{spec.mod_id}/textures/entity/"
            f"{spec.boss.entity_id}.png"
        ),
        Path(
            f"src/main/resources/data/{spec.mod_id}/loot_tables/entities/"
            f"{spec.boss.entity_id}.json"
        ),
        Path(f".minecraft_ai/art_sources/{spec.boss.entity_id}.bbmodel"),
        Path(f".minecraft_ai/art_sources/{spec.boss.entity_id}.obj"),
        Path(f".minecraft_ai/art_sources/{spec.boss.entity_id}.mtl"),
    }
    missing = sorted(path.as_posix() for path in expected_paths if not (first_root / path).is_file())
    assert missing == []
    assert not list((first_root / "src/main/resources/data" / spec.mod_id).rglob("*.mcfunction"))

    bbmodel = json.loads(
        (first_root / f".minecraft_ai/art_sources/{spec.boss.entity_id}.bbmodel").read_text(
            encoding="utf-8"
        )
    )
    assert bbmodel["model_identifier"] == f"{spec.mod_id}:{spec.boss.entity_id}"
    assert ProjectValidator().validate(first_root, spec).passed
    assert ProjectValidator().validate(second_root, spec).passed
    assert _relative_digests(first_root) == _relative_digests(second_root)


def test_broker_rejects_project_root_path_traversal(tmp_path: Path) -> None:
    proposal = MinecraftModPipeline().plan("Create a crafted item")
    approved = proposal.approve(proposal.approval_hash)
    workspace = tmp_path / "approved-workspace"
    escaped_project = workspace / ".." / "escaped-project"
    request = ToolRequest(
        action=ToolAction.SCAFFOLD,
        project_root=escaped_project,
        workspace_root=workspace,
        approved_hash=approved.calculate_hash(),
    )

    with pytest.raises(PolicyDenied, match="escaped"):
        LocalPolicyBroker().authorize(request, approved)


def test_fake_jars_are_rejected(tmp_path: Path) -> None:
    spec = MinecraftModPipeline().plan("Create a crafted item").spec

    non_zip = tmp_path / "not-really-a.jar"
    non_zip.write_bytes(b"this is not a ZIP archive")
    non_zip_report = validate_jar(non_zip, spec)
    assert not non_zip_report.passed
    assert {finding.code for finding in non_zip_report.findings} == {"NOT_A_JAR"}

    metadata_only = tmp_path / "metadata-only.jar"
    with zipfile.ZipFile(metadata_only, "w") as archive:
        archive.writestr("fabric.mod.json", json.dumps({"id": spec.mod_id}))
    metadata_report = validate_jar(metadata_only, spec)
    assert not metadata_report.passed
    assert "JAR_NO_CLASSES" in {
        finding.code for finding in metadata_report.findings
    }


def test_build_false_returns_source_ready_without_publishing_a_jar(
    tmp_path: Path,
) -> None:
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan("Create a crafted item and block")

    result = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=tmp_path / "run",
        build=False,
    )

    assert result.status == "SOURCE_READY"
    assert result.validation_status == "PASS"
    assert result.build_status == "NOT_RUN"
    assert result.gametest_status == "NOT_RUN"
    assert result.release_ready is False
    assert result.jar_path is None
    assert Path(result.project_root).is_dir()
    assert Path(result.release_dir).is_dir()
    assert Path(result.release_zip).is_file()
    assert list(Path(result.release_dir, "binaries").iterdir()) == []
    assert list((tmp_path / "run").rglob("*.jar")) == []
