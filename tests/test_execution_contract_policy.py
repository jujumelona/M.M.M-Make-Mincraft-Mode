from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES,
    JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES,
    JAVA_ATOMIC_ASSEMBLY_MAX_CALLS,
    JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS,
    JAVA_ATOMIC_INITIALIZE_PARAMETERS,
    JAVA_ATOMIC_LOGIC_MEMBERS_PARAMETERS,
    JAVA_ATOMIC_MEMBERS_PARAMETERS,
    JAVA_NESTED_TYPE_REQUIRED_VISIBILITY,
    JAVA_TYPE_OWNING_CONCERNS,
    PLANNER_RECORD_ARRAY_ITEM_MAX_CHARS,
    PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING,
    PLANNER_RECORD_FIELD_MAX_CHARS,
    PLANNER_RECORD_PAGE_MAX_FIELDS,
    PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING,
    PRODUCTION_COMPILE_REPAIR_LIMIT,
    PRODUCTION_RETRY_STRUCTURAL_REJECTIONS,
    RECOVERABLE_ATOMIC_ERROR_PREFIXES,
    SCHEMA_CONTRACT_PROFILE_KEY,
    SCHEMA_STRING_CLASS_KEY,
    SOURCE_REPAIR_HARD_ATTEMPTS,
    SOURCE_REPAIR_MAX_SOURCE_CHARS,
    SOURCE_REPAIR_MAX_SPAN_CHARS,
    SOURCE_REPAIR_SCHEMA_PROFILE,
    STRING_CLASS_REPAIR_SPAN,
    STRING_CLASS_SOURCE,
    TERMINAL_AFTER_NORMALIZATION_PREFIXES,
    assert_execution_contract_consistent,
    atomic_error_recoverable,
    atomic_error_terminal_after_normalization,
    authorized_concern_nested_type_symbols,
    java_atomic_assembly_system_prompt,
    java_atomic_parameters_for_request,
    java_generation_recipe_contract,
    java_region_system_prompt_contract,
)
from minecraft_mod_ai.java_generation_policy import (
    PRODUCTION_COMPILE_REPAIR_LIMIT as JAVA_PRODUCTION_COMPILE_REPAIR_LIMIT,
)
from minecraft_mod_ai.java_generation_policy import (
    PRODUCTION_RETRY_STRUCTURAL_REJECTIONS as JAVA_RETRY_STRUCTURAL_REJECTIONS,
)
from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import (
    _model_transport_schema,
    assert_atomic_model_schema,
)


def _source_repair_schema() -> dict[str, object]:
    return {
        SCHEMA_CONTRACT_PROFILE_KEY: SOURCE_REPAIR_SCHEMA_PROFILE,
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "maxLength": SOURCE_REPAIR_MAX_SOURCE_CHARS,
                SCHEMA_STRING_CLASS_KEY: STRING_CLASS_SOURCE,
            },
            "old": {
                "type": "string",
                "maxLength": SOURCE_REPAIR_MAX_SPAN_CHARS,
                SCHEMA_STRING_CLASS_KEY: STRING_CLASS_REPAIR_SPAN,
            },
            "new": {
                "type": "string",
                "maxLength": SOURCE_REPAIR_MAX_SPAN_CHARS,
                SCHEMA_STRING_CLASS_KEY: STRING_CLASS_REPAIR_SPAN,
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }


def test_source_repair_profile_and_transport_share_one_contract() -> None:
    schema = _source_repair_schema()

    assert_atomic_model_schema(schema, surface="source repair regression")
    transport = _model_transport_schema(schema)

    assert SCHEMA_CONTRACT_PROFILE_KEY not in transport
    assert SCHEMA_STRING_CLASS_KEY not in repr(transport)
    properties = transport["properties"]
    assert properties["content"]["maxLength"] == SOURCE_REPAIR_MAX_SOURCE_CHARS
    assert properties["old"]["maxLength"] == SOURCE_REPAIR_MAX_SPAN_CHARS
    assert properties["new"]["maxLength"] == SOURCE_REPAIR_MAX_SPAN_CHARS


def test_generic_schema_cannot_claim_source_repair_string_class() -> None:
    explicit_domain_bound = {
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
    assert_atomic_model_schema(
        explicit_domain_bound,
        surface="explicit domain bound is not a global model ceiling",
    )

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


def test_production_generation_policy_reads_the_central_contract() -> None:
    assert PRODUCTION_COMPILE_REPAIR_LIMIT >= 1
    assert PRODUCTION_RETRY_STRUCTURAL_REJECTIONS is True
    assert JAVA_PRODUCTION_COMPILE_REPAIR_LIMIT == PRODUCTION_COMPILE_REPAIR_LIMIT
    assert JAVA_RETRY_STRUCTURAL_REJECTIONS is PRODUCTION_RETRY_STRUCTURAL_REJECTIONS
    assert SOURCE_REPAIR_HARD_ATTEMPTS >= 1
    assert DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES > 0


def test_execution_contract_self_check_and_error_taxonomy_are_central() -> None:
    assert_execution_contract_consistent()
    assert set(TERMINAL_AFTER_NORMALIZATION_PREFIXES) <= set(
        RECOVERABLE_ATOMIC_ERROR_PREFIXES
    )
    for prefix in TERMINAL_AFTER_NORMALIZATION_PREFIXES:
        assert atomic_error_recoverable(prefix + " detail")
        assert atomic_error_terminal_after_normalization(prefix + " detail")


def test_nested_type_authority_recognizes_logged_error_label() -> None:
    authority = {
        "source_requirements": {
            "diagnostics": "ERROR: ShipFuelCalculationException",
        }
    }
    assert authorized_concern_nested_type_symbols(authority) == (
        "ShipFuelCalculationException",
    )
    assert JAVA_NESTED_TYPE_REQUIRED_VISIBILITY == "private"


def test_coder_prompt_and_recipe_are_derived_from_canonical_contract() -> None:
    system = java_region_system_prompt_contract(
        section="failure_and_limits",
        concern_name="diagnostics",
        response_region="members",
        platform_api_policy="forbidden",
    )
    recipe = java_generation_recipe_contract("diagnostics")

    assert "only concern-owned private nested runtime types" in system
    assert "Do not reference net.minecraft.*" in system
    assert any(
        "Nested runtime types live inside a host-owned outer class" in rule
        for rule in recipe["compiler_first_rules"]
    )


def test_host_owned_sections_never_enter_the_small_model_java_route() -> None:
    for payload, expected in (
        (
            {"concern": {"name": "stored_state"}},
            "ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN",
        ),
        (
            {"section": "state_model", "concern": {"name": "variables"}},
            "ATOMIC_HOST_ONLY_SECTION_CODER_FORBIDDEN",
        ),
        (
            {"section": "behavior_contract", "concern": {"name": "updates"}},
            "ATOMIC_HOST_ONLY_SECTION_CODER_FORBIDDEN",
        ),
    ):
        with pytest.raises(ValueError, match=expected):
            java_atomic_parameters_for_request(payload, response_region="members")


def test_atomic_contract_caps_remain_bounded_and_canonical() -> None:
    assert DEFAULT_ATOMIC_SCHEMA_LIMITS.max_fields > 0
    assert PLANNER_RECORD_PAGE_MAX_FIELDS == 3
    assert 0 < PLANNER_RECORD_ARRAY_ITEM_MAX_CHARS <= PLANNER_RECORD_FIELD_MAX_CHARS
    assert PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING < PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING
    assert JAVA_ATOMIC_ASSEMBLY_MAX_CALLS > 0
    assert JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS > 0
    assert JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES >= 0
    assert "Lock/ReentrantLock live in java.util.concurrent.locks" in (
        java_atomic_assembly_system_prompt()
    )


def test_execution_contract_has_one_definition_authority_repo_wide() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    authority = package_root / "execution_contract_policy.py"
    definition_names = (
        "SOURCE_REPAIR_MAX_SOURCE_CHARS",
        "SOURCE_REPAIR_MAX_SPAN_CHARS",
        "SOURCE_REPAIR_HARD_ATTEMPTS",
        "DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES",
        "PRODUCTION_REGION_ATTEMPT_LIMIT",
        "PRODUCTION_RETRY_STRUCTURAL_REJECTIONS",
        "PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS",
        "PRODUCTION_COMPILE_REPAIR_LIMIT",
        "PLANNER_RECORD_PAGE_MAX_FIELDS",
        "PLANNER_RECORD_FIELD_MAX_CHARS",
        "PLANNER_RECORD_ARRAY_ITEM_MAX_CHARS",
        "PLANNER_RECORD_COUNT_OUTPUT_TOKEN_CEILING",
        "PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING",
        "RECOVERABLE_ATOMIC_ERROR_PREFIXES",
        "TERMINAL_AFTER_NORMALIZATION_PREFIXES",
        "JAVA_NESTED_TYPE_REQUIRED_VISIBILITY",
        "JAVA_TYPE_OWNING_CONCERNS",
        "JAVA_ATOMIC_ASSEMBLY_MAX_CALLS",
        "JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS",
        "JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES",
    )
    definition_re = re.compile(
        r"^\s*(?:" + "|".join(map(re.escape, definition_names)) + r")\s*=",
        re.MULTILINE,
    )

    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path == authority:
            continue
        source = path.read_text(encoding="utf-8")
        for match in definition_re.finditer(source):
            violations.append(
                f"{path.relative_to(package_root)} defines {match.group(0).strip()}"
            )

    assert not violations, "execution contract drift outside canonical authority:\n" + "\n".join(
        violations
    )


def test_current_execution_contract_consumers_import_the_authority_directly() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    consumers = (
        "fixed_template_generation.py",
        "java_generation_policy.py",
        "java_region_parser.py",
        "model_output_atomicity_contract.py",
    )

    missing = []
    for filename in consumers:
        source = (package_root / filename).read_text(encoding="utf-8")
        if "from .execution_contract_policy import" not in source:
            missing.append(filename)
    assert not missing, f"execution contract consumer bypasses canonical policy: {missing!r}"


def test_authorized_nested_runtime_type_reaches_structured_schema() -> None:
    parameters, _shape = java_atomic_parameters_for_request(
        {
            "response_region": "members",
            "host_selected_class": "AuthoredFailureLimits",
            "concern": {"name": "diagnostics"},
            "authorized_nested_runtime_types": ["ShipFuelCalculationException"],
        },
        response_region="members",
    )

    for category in ("classes", "records", "enums"):
        assert category in parameters["properties"]
        name_schema = parameters["properties"][category]["items"]["properties"]["name"]
        assert "enum" not in name_schema
        assert "host maps" in name_schema["description"].lower()
        assert name_schema["not"] == {"enum": ["AuthoredFailureLimits"]}


def test_model_facing_java_schema_factory_never_returns_shared_mutable_state() -> None:
    cases = (
        (
            {"concern": {"name": "diagnostics"}},
            "members",
            JAVA_ATOMIC_LOGIC_MEMBERS_PARAMETERS,
        ),
        (
            {"concern": {"name": "variables"}},
            "members",
            JAVA_ATOMIC_MEMBERS_PARAMETERS,
        ),
        (
            {"concern": {"name": "integration"}},
            "initialize",
            JAVA_ATOMIC_INITIALIZE_PARAMETERS,
        ),
    )
    for payload, region, canonical in cases:
        first, _ = java_atomic_parameters_for_request(payload, response_region=region)
        second, _ = java_atomic_parameters_for_request(payload, response_region=region)
        assert first is not canonical
        assert second is not canonical
        assert first is not second
        first["properties"]["__mutation_probe__"] = {"type": "string"}
        assert "__mutation_probe__" not in second["properties"]
        assert "__mutation_probe__" not in canonical["properties"]


def test_repo_has_no_legacy_java_model_schema_authority() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    authority = package_root / "execution_contract_policy.py"
    forbidden_definition = re.compile(
        r"^\s*(?:"
        r"_ATOMIC_(?:PARAMETER|FIELD|METHOD|OUTER_METHOD|CONSTRUCTOR|RECORD|ENUM|CLASS)_SCHEMA"
        r"|_ATOMIC_(?:MEMBERS|LOGIC_MEMBERS|DECLARATION_MEMBERS|INITIALIZE)_PARAMETERS"
        r"|JAVA_ATOMIC_(?:PARAMETER|FIELD|METHOD|OUTER_METHOD|CONSTRUCTOR|RECORD|ENUM|CLASS)_SCHEMA"
        r"|JAVA_ATOMIC_(?:MEMBERS|LOGIC_MEMBERS|DECLARATION_MEMBERS|INITIALIZE)_PARAMETERS"
        r")\s*=",
        re.MULTILINE,
    )
    violations = []
    for path in sorted(package_root.rglob("*.py")):
        if path == authority:
            continue
        source = path.read_text(encoding="utf-8")
        for match in forbidden_definition.finditer(source):
            violations.append(
                f"{path.relative_to(package_root)} defines {match.group(0).strip()}"
            )
    assert not violations, "model-facing Java schema authority escaped canonical policy:\n" + "\n".join(
        violations
    )
