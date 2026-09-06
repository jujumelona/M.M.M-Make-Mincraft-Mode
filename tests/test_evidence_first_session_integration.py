from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.api import _attach_existing_target
from minecraft_mod_ai.complete_orchestrator import _semantic_execution_observation
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.evidence_first_planning import compile_evidence_first_plan
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.project_inventory import inspect_project_inventory
from tests.planning_authority_fixtures import request_catalog


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


def _design_with_inventory(root: Path) -> dict:
    _existing_weather_project(root)
    inventory = inspect_project_inventory(root).to_dict()
    return {
        "pitch": "Preserve existing project content while implementing the authored request.",
        "acceptance_tests": ["The requested behavior is observable in Minecraft."],
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


def _missing_semantic_provides(plan: dict) -> set[str]:
    return {
        str(provided)
        for gap in plan["gap_catalog"]
        for provided in gap.get("missing_provides", [])
    }


def test_existing_inventory_name_alias_cannot_silently_close_semantic_gap(
    tmp_path: Path,
) -> None:
    prompt = "Keep the weather compass."
    design = _design_with_inventory(tmp_path / "existing")
    design["_evidence_request_catalog"] = request_catalog(
        prompt,
        [
            {
                "requirement_id": "req_keep_weather_behavior",
                "capability": "weather_compass.preserve_behavior",
                "statement": "Preserve the authored weather-compass behavior, not merely its name.",
                "implementation_capabilities": ["weather_compass.preserve_behavior"],
            }
        ],
    )
    plan = compile_evidence_first_plan(prompt, design)

    assert "capability:weather_compass" in {
        provided
        for component in plan["component_catalog"]
        for provided in component.get("provides", [])
    }
    assert "capability:weather_compass" in plan["verified_provides"]
    assert _missing_semantic_provides(plan).isdisjoint(plan["verified_provides"])
    assert plan["gap_catalog"]
    assert plan["tasks"]
    assert plan["acceptance_release_bindings"][0]["status"] == "planned_gap"


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


def test_existing_zip_preserves_inventory_but_only_verified_semantics_can_retain(
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
        "acceptance_tests": [
            "The existing behavior is preserved and the requested quest behavior works."
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
    design["_evidence_request_catalog"] = request_catalog(
        prompt,
        [
            {
                "requirement_id": "req_preserve_weather",
                "capability": "weather_compass.preserve_behavior",
                "statement": "Preserve the existing weather-compass behavior.",
                "source_text": "Keep the existing weather compass",
                "implementation_capabilities": ["weather_compass.preserve_behavior"],
            },
            {
                "requirement_id": "req_quests",
                "capability": "quest.progression",
                "statement": "Add quests.",
                "source_text": "add quests",
                "implementation_capabilities": ["quest.state", "quest.progression", "quest.reward"],
            },
        ],
    )
    plan = compile_evidence_first_plan(prompt, design)

    assert plan["component_catalog"]
    assert "capability:weather_compass" in plan["verified_provides"]
    assert _missing_semantic_provides(plan).isdisjoint(plan["verified_provides"])
    assert all(
        binding["status"] == "planned_gap"
        for binding in plan["acceptance_release_bindings"]
    )
    assert plan["tasks"]


def test_one_requirement_can_bind_every_semantic_slice_without_fixed_ref_cap() -> None:
    prompt = "Add a persistent networked machine with a GUI and generated resources."
    design = {
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
    design["_evidence_request_catalog"] = request_catalog(
        prompt,
        [
            {
                "requirement_id": "req_machine",
                "capability": "automation.machine",
                "statement": prompt,
                "implementation_capabilities": [
                    "automation.machine",
                    "persistence.state_store",
                    "network.action_sync",
                    "ui.container",
                ],
                "acceptance": ["The complete machine vertical slice works."],
            }
        ],
    )
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
