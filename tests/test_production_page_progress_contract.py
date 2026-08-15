from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError

from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai import production_page_durable_contract as durable


def _production_page(**overrides):
    page = {
        "modules": [],
        "assets": [],
        "audio": [],
        "acceptance_tests": [],
        "completed_deliverables": [],
        "complete": False,
        "next_cursor": "host_remaining_1",
    }
    page.update(overrides)
    return page


def _module(**overrides):
    value = {
        "module_id": "platform_lock",
        "kind": "custom_java",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }
    value.update(overrides)
    return value


def _schema():
    return runtime._schema_for_contract(_production_page())


def test_production_schema_rejects_all_empty_concrete_outputs() -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page())


@pytest.mark.parametrize(
    "field,value",
    [
        ("modules", [_module()]),
        ("assets", [{}]),
        ("audio", [{}]),
        ("acceptance_tests", ["platform lock is host-verifiable"]),
    ],
)
def test_production_schema_accepts_each_concrete_output_family(field, value) -> None:
    validator = Draft202012Validator(_schema())
    validator.validate(_production_page(**{field: value}))


@pytest.mark.parametrize("field", ["module_id", "kind"])
def test_production_schema_rejects_blank_required_module_strings(field) -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(modules=[_module(**{field: ""})]))


def test_production_schema_rejects_unsupported_module_kind() -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(modules=[_module(kind="config")]))


@pytest.mark.parametrize(
    "field",
    ["depends_on", "required_gates", "implements_deliverables"],
)
def test_production_schema_rejects_blank_module_string_array_items(field) -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(modules=[_module(**{field: [""]})]))


@pytest.mark.parametrize(
    "asset",
    [
        {"kind": "texture"},
        {"kind": "item", "width": "16"},
        {"kind": "item", "height": 16.5},
    ],
)
def test_production_schema_rejects_asset_parser_mismatches(asset) -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(assets=[asset]))


@pytest.mark.parametrize(
    "audio",
    [
        {"kind": "sfx"},
        {"kind": "effect", "duration_seconds": "1"},
        {"kind": "effect", "loop": "false"},
    ],
)
def test_production_schema_rejects_audio_parser_mismatches(audio) -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(audio=[audio]))


def test_production_schema_accepts_canonical_asset_and_audio_fields() -> None:
    validator = Draft202012Validator(_schema())
    validator.validate(
        _production_page(
            assets=[
                {
                    "asset_id": "icon",
                    "kind": "item",
                    "prompt": "icon",
                    "target_path": "assets/mod/textures/icon.png",
                    "width": 16,
                    "height": 16,
                }
            ]
        )
    )
    validator.validate(
        _production_page(
            audio=[
                {
                    "sound_id": "click",
                    "kind": "effect",
                    "duration_seconds": 1.0,
                    "frequency_hz": 440.0,
                    "volume": 0.8,
                    "loop": False,
                }
            ]
        )
    )


def test_production_schema_rejects_blank_acceptance_test() -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page(acceptance_tests=[""]))


def test_stale_loose_production_checkpoints_are_invalidated() -> None:
    assert durable._VERSION >= 4
    assert durable._ITEM_VERSION >= 3
