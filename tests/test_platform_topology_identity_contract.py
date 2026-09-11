from __future__ import annotations

import pytest

from test_resolved_version_context import target_fixture
from minecraft_mod_ai.platform_resolver import compile_target_decision
from minecraft_mod_ai.spec import SpecValidationError


def _selection() -> dict[str, object]:
    return {
        "target": target_fixture().public_dict(),
        "resolved_version_context": target_fixture().version_context.to_dict(),
        "preserved_existing_target": True,
        "migration_requested": False,
    }


def test_target_lowering_separates_logical_module_ids_from_gradle_paths() -> None:
    inventory = {
        "target": {"loaders": ["fabric"]},
        "modules": [
            {"module_id": ":", "source_sets": ["main"]},
            {"module_id": ":client", "source_sets": ["client"]},
        ],
    }

    decision = compile_target_decision(_selection(), existing_inventory=inventory)
    topology = decision["project_topology"]

    assert topology["module_ids"] == ["root", "client"]
    assert topology["gradle_project_paths"] == [":", ":client"]
    assert topology["source_sets"] == ["client", "main"]
    assert topology["loaders"] == ["fabric"]


def test_target_lowering_rejects_logical_module_identity_collisions() -> None:
    inventory = {
        "modules": [
            {"module_id": ":foo-bar", "source_sets": ["main"]},
            {"module_id": ":foo_bar", "source_sets": ["main"]},
        ]
    }

    with pytest.raises(SpecValidationError, match="duplicate logical module identities"):
        compile_target_decision(_selection(), existing_inventory=inventory)
