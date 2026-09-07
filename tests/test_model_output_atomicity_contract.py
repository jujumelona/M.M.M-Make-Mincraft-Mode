from __future__ import annotations

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


def test_large_model_authored_schema_is_rejected_before_generation() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {f"nested_{inner}": {"type": "string"} for inner in range(4)},
            }
            for index in range(20)
        },
    }
    with pytest.raises(ModelConfigurationError, match="MODEL_STRUCTURE_ATOMICITY"):
        assert_atomic_model_schema(schema, surface="regression")


def test_small_atomic_schema_remains_allowed() -> None:
    assert_atomic_model_schema(
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        surface="regression",
    )


def test_runtime_installer_wraps_bounded_argument_protocol() -> None:
    assert forced.host_selected_argument_turn.__module__ == "minecraft_mod_ai.model_output_atomicity_contract"
    assert forced.host_selected_mutation_turn.__module__ == "minecraft_mod_ai.model_output_atomicity_contract"
    assert getattr(forced.host_selected_argument_turn, "_mmm_atomic_model_output_boundary", False) is True
    assert getattr(forced.host_selected_mutation_turn, "_mmm_atomic_model_output_boundary", False) is True
