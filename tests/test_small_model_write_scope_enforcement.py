from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_module_generator, host_grounding
from minecraft_mod_ai.small_model_write_scope_enforcement import (
    assert_installed,
    exact_task_writable_paths,
    validate_exact_task_operations,
)


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "evidence_task": {
                "task_id": "task_exact_scope",
                "task_sha256": "sha256:" + "a" * 64,
                "sequence": 1,
                "execution_role": "production_with_verification",
                "semantic_outcome": "Implement one exact feature.",
                "requirement_refs": ["req_exact"],
                "depends_on": [],
                "target_cell": {
                    "minecraft_version": "1.21.1",
                    "loader": "fabric",
                    "mappings": "yarn",
                    "java_version": "21",
                },
                "owned_anchors": [
                    {
                        "kind": "symbol",
                        "locator": "src/main/java/demo/Owned.java#Owned",
                        "status": "host_reserved",
                        "module_id": ":",
                        "source_set": "main",
                    },
                    {
                        "kind": "test",
                        "locator": "src/test/java/demo/OwnedTest.java#OwnedTest",
                        "status": "host_reserved",
                        "module_id": ":",
                        "source_set": "test",
                    },
                ],
                "implementation_capabilities": ["demo.exact"],
                "implementation_obligations": ["Implement only the owned feature."],
                "consumes": [],
                "provides": ["demo_exact"],
                "required_gates": ["source_static_validation", "target_compile"],
                "public_acceptance": ["The exact feature works."],
                "runtime_acceptance": ["Owned runtime state changes."],
                "artifact_obligations": [
                    {
                        "kind": "source_code",
                        "locator": "src/main/java/demo/Owned.java#Owned",
                        "purpose": "owned implementation",
                    }
                ],
            }
        }
    )


def test_exact_paths_are_derived_from_host_owned_task_contract() -> None:
    assert exact_task_writable_paths(_module()) == (
        "src/main/java/demo/Owned.java",
        "src/test/java/demo/OwnedTest.java",
    )


def test_exact_validator_rejects_other_source_file_even_inside_allowed_source_tree() -> None:
    allowed = exact_task_writable_paths(_module())
    validate_exact_task_operations(
        [{"operation": "replace", "path": "src/main/java/demo/Owned.java"}],
        allowed,
    )
    with pytest.raises(RuntimeError, match="TASK_WRITE_SCOPE_ESCAPE"):
        validate_exact_task_operations(
            [{"operation": "replace", "path": "src/main/java/demo/Other.java"}],
            allowed,
        )


def test_runtime_installs_exact_scope_and_single_coarse_path_authority() -> None:
    assert_installed(
        custom_module_generator_module=custom_module_generator,
        host_grounding_module=host_grounding,
    )
    assert custom_module_generator._agent_mutable_path is host_grounding.custom_module_path_allowed
    assert custom_module_generator._agent_mutable_path("src/main/java/demo/Owned.java") is True
    assert custom_module_generator._agent_mutable_path("build.gradle") is False
    assert custom_module_generator._agent_mutable_path(".minecraft_ai/generated/x.json") is False
