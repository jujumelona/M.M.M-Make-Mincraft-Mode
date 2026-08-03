from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.builder_contract_service import (
    ArchitectureCatalog,
    BuilderContractService,
)
from minecraft_mod_ai.buildspec import (
    BuildSpecValidationError,
    validate_builder_result,
    validate_buildspec,
)


def _world() -> dict:
    return {
        "origin": [0, 64, 0],
        "bbox": [0, 0, 0, 160, 80, 160],
        "context_blocks_ref": "world_context.npz",
        "terrain_ref": "terrain.npz",
        "protected_mask_ref": "protected.npz",
    }


def _spec() -> dict:
    return {
        "schema_version": "buildspec_v2",
        "world": _world(),
        "zones": [
            {
                "zone_id": "access_zone",
                "zone_type_id": "walkable_path",
                "bbox": [0, 0, 0, 20, 4, 20],
            }
        ],
        "components": [
            {
                "component_id": 1,
                "component_type_id": "structural_frame",
                "bbox": [25, 0, 25, 100, 65, 105],
                "orientation": "south",
                "completion_ratio": 0.6,
            },
            {
                "component_id": 2,
                "component_type_id": "tower_crane",
                "bbox": [105, 0, 35, 125, 78, 75],
                "orientation": "west",
            },
        ],
        "parts": [
            {
                "part_id": 10,
                "component_id": 1,
                "structural_role": "column",
                "bbox": [25, 0, 25, 32, 65, 32],
            }
        ],
        "relations": [
            {
                "subject_id": 2,
                "predicate": "adjacent_to",
                "object_id": 1,
                "clearance": 5,
            }
        ],
        "ports": [
            {
                "port_id": "entrance_1",
                "component_id": 1,
                "port_type_id": "entrance",
                "position": [62, 0, 25],
                "direction": "south",
                "width": 5,
            }
        ],
        "patterns": [],
        "operators": [
            {
                "operator_id": "stack_floors",
                "operator_type": "stack",
                "target_ids": [1],
                "count": 9,
                "interval": 5,
            }
        ],
        "task": {
            "type": "generate",
            "target_component_ids": [1, 2],
            "completed_component_ids": [],
            "open_port_ids": ["entrance_1"],
        },
        "constraints": {
            "hard": [
                {
                    "constraint_id": "support_1",
                    "type_id": "support_required",
                    "target_ids": [1, 2],
                }
            ],
            "soft": [],
        },
    }


def _result() -> dict:
    return {
        "add_blocks_ref": "add.npz",
        "remove_blocks_ref": "remove.npz",
        "replace_blocks_ref": "replace.npz",
        "resolved_ports": ["entrance_1"],
        "remaining_open_ports": [],
        "validation_predictions": {
            "supported": True,
            "connected": True,
            "constraint_violations": [],
        },
    }


def test_buildspec_v2_accepts_machine_contract() -> None:
    normalized = validate_buildspec(_spec())
    assert normalized["schema_version"] == "buildspec_v2"
    assert normalized["components"][0]["component_type_id"] == "structural_frame"


@pytest.mark.parametrize(
    "key",
    [
        "prompt",
        "brief",
        "style",
        "style_description",
        "scene_meaning",
        "image_caption",
        "natural_language",
    ],
)
def test_buildspec_rejects_natural_language_fields(key: str) -> None:
    spec = _spec()
    spec["components"][0][key] = "unfinished high rise construction site"
    with pytest.raises(BuildSpecValidationError):
        validate_buildspec(spec)


def test_buildspec_rejects_unknown_task_component() -> None:
    spec = _spec()
    spec["task"]["target_component_ids"] = [999]
    with pytest.raises(BuildSpecValidationError):
        validate_buildspec(spec)


def test_builder_result_must_partition_open_ports() -> None:
    result = _result()
    result["resolved_ports"] = []
    with pytest.raises(BuildSpecValidationError):
        validate_builder_result(_spec(), result)


def test_builder_result_accepts_exact_delta_contract() -> None:
    normalized = validate_builder_result(_spec(), _result())
    assert normalized["add_blocks_ref"] == "add.npz"
    assert normalized["remaining_open_ports"] == []


def test_architecture_catalog_is_central_only() -> None:
    result = ArchitectureCatalog().search("공사장 타워크레인 골조")
    assert result
    assert all(item["central_only"] is True for item in result)


def test_central_agent_output_is_validated(tmp_path) -> None:
    candidate = _spec()

    class FakeRouter:
        def generate_text(self, *args, **kwargs):
            return json.dumps(candidate)

    service = BuilderContractService(
        workspace_root=tmp_path,
        router_factory=lambda: FakeRouter(),
    )
    planned = service.plan_buildspec(
        "unfinished concrete construction site",
        _world(),
    )
    assert planned["builder_boundary"] == "STRUCTURED_SPEC_ONLY"
    assert planned["buildspec"] == validate_buildspec(candidate)


def test_handoff_does_not_claim_builder_execution(tmp_path) -> None:
    service = BuilderContractService(workspace_root=tmp_path)
    result = service.prepare_builder_handoff(
        _spec(),
        require_world_artifacts=False,
    )
    assert result["status"] == "READY_FOR_EXTERNAL_BUILDER"
    assert result["builder_execution"] == "NOT_EXECUTED_BY_MMM"
