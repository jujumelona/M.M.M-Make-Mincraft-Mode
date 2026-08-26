from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.api import _attach_existing_target
from minecraft_mod_ai.complete_orchestrator import _semantic_execution_observation
from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.project_inventory import inspect_project_inventory
from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.work_graph import build_production_work_plan


def _existing_weather_project(root: Path) -> None:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/test/java/example").mkdir(parents=True)
    (root / "src/main/resources/assets/weather_existing/models/item").mkdir(
        parents=True
    )
    (root / "settings.gradle.kts").write_text(
        'rootProject.name = "weather-existing"\n',
        encoding="utf-8",
    )
    (root / "gradle.properties").write_text(
        "minecraft_version=1.21.1\nloader_version=0.16.10\n",
        encoding="utf-8",
    )
    (root / "src/main/java/example/WeatherCompass.java").write_text(
        "package example; public final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    (root / "src/test/java/example/WeatherCompassTest.java").write_text(
        "package example; public final class WeatherCompassTest {}\n",
        encoding="utf-8",
    )
    (
        root
        / "src/main/resources/assets/weather_existing/models/item/weather_compass.json"
    ).write_text('{"parent":"minecraft:item/generated"}\n', encoding="utf-8")


def _retained_plan(root: Path, prompt: str) -> tuple[dict, dict]:
    _existing_weather_project(root)
    inventory = inspect_project_inventory(root).to_dict()
    design = {
        "pitch": "Keep the existing weather compass behavior.",
        "modules": [
            {
                "plugin_id": "weather_compass",
                "reason": "weather_compass",
            }
        ],
        "acceptance_tests": ["The existing weather compass remains available."],
        "_existing_project_inventory": inventory,
        "_existing_snapshot": inventory,
        "_platform_selection": {
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
            },
            "preserved_existing_target": True,
            "migration_requested": False,
        },
    }
    return design, compile_evidence_first_plan(prompt, design)


def test_retained_only_plan_builds_validation_graph_without_generation(
    tmp_path: Path,
) -> None:
    prompt = "Keep the weather compass."
    design, plan = _retained_plan(tmp_path / "existing", prompt)
    assert plan["gap_catalog"] == []
    assert plan["tasks"] == []

    compiled = compile_production_contract(
        requested_prompt=prompt,
        game_design={"pitch": design["pitch"]},
        modules=(),
        assets=(),
        acceptance_tests=("The existing weather compass remains available.",),
        evidence_plan=plan,
    )
    implementation_kinds = {
        item["source_kind"] for item in compiled.contract["implementation_catalog"]
    }
    assert implementation_kinds == {"evidence_plan", "retained_component"}
    atom_groups = [
        item
        for item in compiled.contract["coverage_groups"]
        if next(
            requirement
            for requirement in compiled.contract["requirement_catalog"]
            if requirement["requirement_ref"] == item["requirement_ref"]
        )["source"]
        == "evidence_plan"
    ]
    retained_components = plan["acceptance_release_bindings"][0]["component_refs"]
    assert atom_groups[0]["implementation_refs"] == [
        f"implementation:retained_component:{component}"
        for component in retained_components
    ]

    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(prompt)
    proposal = complete_proposal_from_parts(
        requested_prompt=prompt,
        base_proposal=base,
        game_design={
            **design,
            "_evidence_first_plan": plan,
            "_production_contract": compiled.contract,
        },
        modules=(),
        acceptance_tests=compiled.acceptance_tests,
    )
    proposal.validate()
    work = build_production_work_plan(proposal)
    assert work.module_count == 0
    assert not any(node.stage.startswith("generate:") for node in work.nodes)
    assert {node.node_id for node in work.nodes} >= {
        "prepare-project",
        "validate-source",
        "build-project",
        "validate-jar",
    }


def test_retained_only_proposal_rejects_stale_evidence_plan(tmp_path: Path) -> None:
    prompt = "Keep the weather compass."
    design, plan = _retained_plan(tmp_path / "existing", prompt)
    compiled = compile_production_contract(
        requested_prompt=prompt,
        game_design={"pitch": design["pitch"]},
        modules=(),
        acceptance_tests=("The existing weather compass remains available.",),
        evidence_plan=plan,
    )
    plan["verified_provides"] = []
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(prompt)
    with pytest.raises(SpecValidationError, match="evidence-first implementation plan"):
        complete_proposal_from_parts(
            requested_prompt=prompt,
            base_proposal=base,
            game_design={
                **design,
                "_evidence_first_plan": plan,
                "_production_contract": compiled.contract,
            },
            modules=(),
            acceptance_tests=compiled.acceptance_tests,
        )


def test_existing_archive_inventory_starts_before_planning_and_is_hash_bound(
    tmp_path: Path,
) -> None:
    project = tmp_path / "source"
    _existing_weather_project(project)
    metadata = project / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
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
    archive = tmp_path / "existing.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                output.write(path, "project/" + path.relative_to(project).as_posix())

    owner = SimpleNamespace()
    _attach_existing_target(owner, archive)
    inventory = owner._mmm_existing_project_inventory_future.result(timeout=30)

    inventory.validate()
    assert owner._mmm_existing_minecraft_version == "1.21.1"
    assert owner._mmm_existing_project_report["archive_sha256"].startswith("sha256:")
    assert inventory.source_kind == "archive"
    assert "capability:weather_compass" in {
        value for component in inventory.components for value in component.provides
    }


def test_existing_zip_retains_symbol_resource_and_test_then_generates_only_gap(
    tmp_path: Path,
) -> None:
    project = tmp_path / "source"
    _existing_weather_project(project)
    archive = tmp_path / "existing.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                output.write(path, "project/" + path.relative_to(project).as_posix())

    owner = SimpleNamespace()
    _attach_existing_target(owner, archive)
    inventory = owner._mmm_existing_project_inventory_future.result(timeout=30)
    payload = inventory.to_dict()
    prompt = "Keep the existing weather compass and add quests."
    design = {
        "modules": [
            {"plugin_id": "weather_compass", "reason": "weather_compass"},
            {"plugin_id": "quests", "reason": "quests"},
        ],
        "acceptance_tests": [
            "The weather compass remains unchanged.",
            "Quests work.",
        ],
        "_existing_project_inventory": payload,
        "_existing_snapshot": payload,
        "_platform_selection": {
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "source_api_family": "fabric_live_ai",
            },
            "preserved_existing_target": True,
            "migration_requested": False,
        },
    }
    plan = compile_evidence_first_plan(prompt, design)
    weather = next(
        binding
        for binding in plan["acceptance_release_bindings"]
        if binding["capability"] == "weather_compass"
    )
    retained = {
        component["kind"]
        for component in plan["component_catalog"]
        if component["component_id"] in weather["component_refs"]
    }

    assert weather["status"] == "retained"
    assert {"symbol", "resource", "test"} <= retained
    assert all(
        "weather_compass" not in task["semantic_outcome"].casefold()
        for task in plan["tasks"]
    )
    assert any("quest" in task["semantic_outcome"].casefold() for task in plan["tasks"])


def test_one_requirement_can_bind_every_semantic_slice_without_a_fixed_ref_cap() -> None:
    prompt = "Add a persistent networked machine with a GUI and generated resources."
    design = {
        "modules": [{"plugin_id": "machine", "reason": prompt}],
        "acceptance_tests": ["The complete machine vertical slice works."],
        "_platform_selection": {
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "source_api_family": "fabric_live_ai",
            },
            "preserved_existing_target": False,
            "migration_requested": False,
        },
    }
    plan = compile_evidence_first_plan(prompt, design)
    modules = tuple(
        ProductionModule(
            module_id=str(task["task_id"]),
            kind="custom_java",
            config={"evidence_task": dict(task)},
            depends_on=tuple(task["depends_on"]),
            required_gates=tuple(task["required_gates"]),
        )
        for task in plan["tasks"]
    )
    compiled = compile_production_contract(
        requested_prompt=prompt,
        game_design={"pitch": prompt},
        modules=modules,
        acceptance_tests=("The complete machine vertical slice works.",),
        evidence_plan=plan,
    )
    exact_groups = [
        group
        for group in compiled.contract["coverage_groups"]
        if len(group["implementation_refs"]) > 8
    ]
    assert exact_groups
    assert compiled.contract["catalog_stats"][
        "max_direct_implementation_refs_per_group"
    ] == max(
        len(group["implementation_refs"])
        for group in compiled.contract["coverage_groups"]
    )


def test_execution_observation_binds_patch_and_transitive_impact_to_task() -> None:
    task = {
        "task_id": "task_registry",
        "task_sha256": "sha256:" + "1" * 64,
        "requirement_refs": ["req_machine"],
        "gap_refs": ["gap_machine"],
        "impact_probes": ["changed_symbols", "affected_tests"],
    }
    module = ProductionModule(
        module_id="task_registry",
        kind="custom_java",
        config={"evidence_task": task},
    )
    observation = _semantic_execution_observation(
        module,
        {
            "operation_count": 1,
            "touched_paths": ["src\\main\\java\\example\\Registry.java"],
            "patch_receipt": {"receipt_sha256": "sha256:" + "2" * 64},
            "source_observation_receipt": {
                "observations_sha256": "sha256:" + "3" * 64
            },
        },
        dependent_ids=("task_test", "task_block"),
    )

    assert observation is not None
    assert observation["touched_paths"] == [
        "src/main/java/example/Registry.java"
    ]
    assert observation["affected_downstream_task_ids"] == [
        "task_block",
        "task_test",
    ]
    assert observation["observation_sha256"].startswith("sha256:")
