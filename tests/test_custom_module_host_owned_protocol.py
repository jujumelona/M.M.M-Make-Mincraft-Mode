from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
    _canonicalize_planned_path,
    _file_content_schema,
    _file_plan_schema,
    _validate_file_content_payload,
)


def test_fabric_metadata_path_is_host_canonicalized() -> None:
    assert (
        _canonicalize_planned_path("fabric.mod.json")
        == "src/main/resources/fabric.mod.json"
    )


def test_planned_path_cannot_escape_project() -> None:
    with pytest.raises(CustomModuleGenerationError, match="project-relative"):
        _canonicalize_planned_path("../outside.java")


def test_file_content_contract_requires_text_not_nested_json_object() -> None:
    schema = _file_content_schema()
    assert schema["properties"]["content"]["type"] == "string"

    with pytest.raises(CustomModuleGenerationError, match="UTF-8 text"):
        _validate_file_content_payload(
            {"content": {"schemaVersion": 1}, "runtime_tests": []}
        )


def test_model_plan_has_no_patch_or_progress_fields() -> None:
    schema = _file_plan_schema()
    properties = set(schema["properties"])
    assert properties == {"files", "runtime_tests"}
    assert not {
        "operation",
        "operations",
        "expected_sha256",
        "complete",
        "next_cursor",
        "context_page_complete",
    } & properties


def test_host_plan_validation_canonicalizes_before_scope_check() -> None:
    generator = object.__new__(CustomModuleGenerator)
    planned, tests = generator._validate_file_plan(
        {
            "files": [
                {
                    "path": "fabric.mod.json",
                    "purpose": "Fabric metadata",
                }
            ],
            "runtime_tests": ["metadata loads"],
        }
    )

    assert planned == [
        {
            "path": "src/main/resources/fabric.mod.json",
            "purpose": "Fabric metadata",
        }
    ]
    assert tests == ["metadata loads"]
