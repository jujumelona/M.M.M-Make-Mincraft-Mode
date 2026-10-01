from __future__ import annotations

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    PRODUCTION_COMPILE_REPAIR_LIMIT,
    PRODUCTION_RETRY_STRUCTURAL_REJECTIONS,
    SCHEMA_CONTRACT_PROFILE_KEY,
    SCHEMA_STRING_CLASS_KEY,
    SOURCE_REPAIR_HARD_ATTEMPTS,
    SOURCE_REPAIR_MAX_SOURCE_CHARS,
    SOURCE_REPAIR_MAX_SPAN_CHARS,
    SOURCE_REPAIR_SCHEMA_PROFILE,
    STRING_CLASS_REPAIR_SPAN,
    STRING_CLASS_SOURCE,
)
from minecraft_mod_ai.java_generation_policy import (
    PRODUCTION_COMPILE_REPAIR_LIMIT as JAVA_PRODUCTION_COMPILE_REPAIR_LIMIT,
    PRODUCTION_RETRY_STRUCTURAL_REJECTIONS as JAVA_RETRY_STRUCTURAL_REJECTIONS,
)
from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import (
    _model_transport_schema,
    assert_atomic_model_schema,
)
from minecraft_mod_ai.repair_engine import _HARD_REPAIR_ATTEMPTS
from minecraft_mod_ai.repair_response_contract import repair_response_schema
from minecraft_mod_ai.verifier_repair_window import (
    MAX_REPAIR_WINDOW_CHARS,
    MIN_REPAIR_REPLACEMENT_CHARS,
)


def test_repair_schema_and_atomicity_share_one_contract_profile() -> None:
    schema = repair_response_schema(64 * 1024)

    assert schema[SCHEMA_CONTRACT_PROFILE_KEY] == SOURCE_REPAIR_SCHEMA_PROFILE
    branches = schema["properties"]["operations"]["items"]["anyOf"]
    create_content = branches[0]["properties"]["content"]
    replace_content = branches[1]["properties"]["content"]
    replacement = branches[2]["properties"]["replacements"]["items"]["properties"]

    assert create_content["maxLength"] == SOURCE_REPAIR_MAX_SOURCE_CHARS
    assert replace_content["maxLength"] == SOURCE_REPAIR_MAX_SOURCE_CHARS
    assert create_content[SCHEMA_STRING_CLASS_KEY] == STRING_CLASS_SOURCE
    assert replacement["old"]["maxLength"] == SOURCE_REPAIR_MAX_SPAN_CHARS
    assert replacement["new"]["maxLength"] == SOURCE_REPAIR_MAX_SPAN_CHARS
    assert replacement["old"][SCHEMA_STRING_CLASS_KEY] == STRING_CLASS_REPAIR_SPAN

    # Regression for the logged failure:
    # maxLength=16384 must be legal only because the root repair contract selected it.
    assert_atomic_model_schema(schema, surface="repair regression")


def test_generic_atomic_schema_cannot_claim_repair_payload_limits() -> None:
    oversized = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "maxLength": SOURCE_REPAIR_MAX_SOURCE_CHARS,
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    with pytest.raises(ModelConfigurationError, match="MODEL_ATOMICITY_STRING_EXCEEDED"):
        assert_atomic_model_schema(oversized, surface="generic remains bounded")

    smuggled = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "maxLength": SOURCE_REPAIR_MAX_SOURCE_CHARS,
                SCHEMA_STRING_CLASS_KEY: STRING_CLASS_SOURCE,
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    with pytest.raises(ModelConfigurationError, match="MODEL_ATOMICITY_STRING_CLASS_INVALID"):
        assert_atomic_model_schema(smuggled, surface="source class requires repair profile")


def test_production_repair_paths_are_enabled_and_read_the_central_policy() -> None:
    assert PRODUCTION_COMPILE_REPAIR_LIMIT >= 1
    assert PRODUCTION_RETRY_STRUCTURAL_REJECTIONS is True
    assert JAVA_PRODUCTION_COMPILE_REPAIR_LIMIT == PRODUCTION_COMPILE_REPAIR_LIMIT
    assert JAVA_RETRY_STRUCTURAL_REJECTIONS is PRODUCTION_RETRY_STRUCTURAL_REJECTIONS
    assert _HARD_REPAIR_ATTEMPTS == SOURCE_REPAIR_HARD_ATTEMPTS


def test_verifier_repair_window_reads_central_string_limits() -> None:
    assert MAX_REPAIR_WINDOW_CHARS == SOURCE_REPAIR_MAX_SPAN_CHARS
    assert MIN_REPAIR_REPLACEMENT_CHARS == DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars


def test_host_contract_annotations_do_not_leak_to_model_tool_schema() -> None:
    schema = repair_response_schema(64 * 1024)
    transport = _model_transport_schema(schema)

    assert SCHEMA_CONTRACT_PROFILE_KEY not in transport
    rendered = repr(transport)
    assert SCHEMA_STRING_CLASS_KEY not in rendered
    assert transport["properties"]["operations"]["items"]["anyOf"][0]["properties"]["content"]["maxLength"] == SOURCE_REPAIR_MAX_SOURCE_CHARS
