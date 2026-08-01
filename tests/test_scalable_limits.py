from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.scalable_generator import ScalableFabricProjectGenerator
from minecraft_mod_ai.scalable_pipeline import ScalableMinecraftModPipeline
from minecraft_mod_ai.scalable_validator import ScalableProjectValidator
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec


def test_257_contents_generate_in_deterministic_shards(tmp_path: Path) -> None:
    contents = tuple(
        ContentSpec(
            content_id=f"scale_item_{index:03d}",
            kind=ContentKind.ITEM,
            display_name_en=f"Scale Item {index}",
            display_name_ko=f"규모 아이템 {index}",
            recipe=index % 2 == 0,
        )
        for index in range(257)
    )
    spec = ModSpec(
        mod_id="scale_test",
        mod_name="Scale Test",
        package_name="ai.minecraft.scale_test",
        version="1.0.0",
        summary="scale contract",
        contents=contents,
    )
    policy = ScalePolicy(java_shard_size=32)
    root = tmp_path / "project"
    result = ScalableFabricProjectGenerator(policy=policy).generate(spec, root)
    assert result.root == root.resolve()

    registrar_units = sorted(
        root.rglob("GeneratedContentUnit*.java")
    )
    gametest_units = sorted(
        root.rglob("ScalableContentGameTestUnit*.java")
    )
    assert len(registrar_units) == 257
    assert len(gametest_units) == 9
    gametest_root = next(
        root.rglob("ScalableContentGameTest.java")
    )
    assert all(
        path.stat().st_size < policy.max_single_file_bytes
        for path in registrar_units + gametest_units + [gametest_root]
    )
    root_text = gametest_root.read_text(encoding="utf-8")
    assert "Files.list(directory)" in root_text
    assert "ScalableContentGameTestUnit" in root_text
    assert "scale_item_256" not in root_text

    metadata = json.loads(
        (root / "src/main/resources/fabric.mod.json").read_text(
            encoding="utf-8"
        )
    )
    entries = metadata["entrypoints"]["fabric-gametest"]
    assert sum("ScalableContentGameTest" in str(item) for item in entries) == 1
    assert (
        "ai.minecraft.scale_test.ScalableContentGameTest"
        in entries
    )
    report = ScalableProjectValidator(policy=policy).validate(root, spec)
    assert report.passed, report.to_dict()


def test_300_module_chain_validates_without_recursion_or_count_cap() -> None:
    base = ScalableMinecraftModPipeline(
        planner=HeuristicPlanner()
    ).plan("Create one scale anchor item")
    modules = tuple(
        ProductionModule(
            module_id=f"module_{index:03d}",
            kind="integration",
            config={"sequence": index},
            depends_on=(f"module_{index - 1:03d}",) if index else (),
        )
        for index in range(300)
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Validate a 300 module dependency chain",
        base_proposal=base,
        game_design={"title": "Scale DAG"},
        modules=modules,
        acceptance_tests=("All 300 modules remain in dependency order",),
    )
    proposal.validate()
    assert len(proposal.modules) == 300
    assert proposal.modules[-1].depends_on == ("module_298",)


def test_validator_uses_explicit_file_policy_not_legacy_4_mib(
    tmp_path: Path,
) -> None:
    spec = ModSpec(
        mod_id="large_file_test",
        mod_name="Large File Test",
        package_name="ai.minecraft.large_file_test",
        version="1.0.0",
        summary="policy file size contract",
        contents=(
            ContentSpec(
                content_id="anchor_item",
                kind=ContentKind.ITEM,
                display_name_en="Anchor Item",
                display_name_ko="기준 아이템",
            ),
        ),
    )
    policy = ScalePolicy(max_single_file_bytes=8 * 1024 * 1024)
    root = tmp_path / "project"
    ScalableFabricProjectGenerator(policy=policy).generate(spec, root)
    large = root / ".minecraft_ai" / "large-policy-proof.bin"
    large.parent.mkdir(parents=True, exist_ok=True)
    large.write_bytes(b"x" * (5 * 1024 * 1024))
    report = ScalableProjectValidator(policy=policy).validate(root, spec)
    assert report.passed, report.to_dict()
    assert not any(
        finding.code == "FILE_TOO_LARGE"
        for finding in report.findings
    )
