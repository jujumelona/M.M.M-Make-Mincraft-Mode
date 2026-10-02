from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    RECOVERABLE_ATOMIC_ERROR_PREFIXES,
    TERMINAL_AFTER_NORMALIZATION_PREFIXES,
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES,
    JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES,
    JAVA_ATOMIC_ASSEMBLY_MAX_CALLS,
    JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS,
    JAVA_NESTED_TYPE_REQUIRED_VISIBILITY,
    JAVA_TYPE_OWNING_CONCERNS,
    assert_execution_contract_consistent,
    atomic_error_recoverable,
    atomic_error_terminal_after_normalization,
    authorized_concern_nested_type_symbols,
    java_atomic_assembly_system_prompt,
    java_atomic_parameters_for_request,
    java_generation_recipe_contract,
    java_region_system_prompt_contract,
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


def test_consumers_do_not_redefine_canonical_contract_literals() -> None:
    import minecraft_mod_ai.atomic_concern_source as concern_source
    import minecraft_mod_ai.java_generation_policy as generation_policy
    import minecraft_mod_ai.java_region_parser as region_parser

    generation_source = inspect.getsource(generation_policy)
    concern_source_text = inspect.getsource(concern_source)
    parser_source = inspect.getsource(region_parser)

    assert "RECOVERABLE_ATOMIC_ERROR_PREFIXES = (" not in generation_source
    assert "TERMINAL_AFTER_NORMALIZATION_PREFIXES = (" not in generation_source
    assert "COMPILER_FIRST_RULES = (" not in generation_source
    assert "Return only compile-ready Java class-body source" not in concern_source_text
    assert "must be private because outer type ownership is host-owned" not in parser_source
    assert "from .java_generation_policy import" not in parser_source
    assert "str(return_type or \"\").strip() == \"void\"" not in generation_source



def test_secondary_atomic_and_repair_caps_read_the_canonical_policy() -> None:
    import minecraft_mod_ai.atomic_java_assembly as java_assembly
    import minecraft_mod_ai.central_atomic_generation_contract as central_atomic
    import minecraft_mod_ai.generation_diagnostic_repair as diagnostic_repair

    atomic_source = inspect.getsource(central_atomic)
    diagnostic_source = inspect.getsource(diagnostic_repair)
    assembly_source = inspect.getsource(java_assembly)

    assert "_MAX_ITEMS = 4" not in atomic_source
    assert "_MAX_CHARS = 256" not in atomic_source
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items" in atomic_source
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars" in atomic_source

    assert "_MAX_REPAIR_SOURCE_BYTES = 12 * 1024" not in diagnostic_source
    assert "DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES" in diagnostic_source
    assert DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES > 0

    assert "MAX_MODEL_STRING_CHARS" not in assembly_source
    assert "MAX_ASSEMBLY_CALLS = 128" not in assembly_source
    assert "MAX_PART_ITEMS = 32" not in assembly_source
    assert JAVA_ATOMIC_ASSEMBLY_MAX_CALLS > 0
    assert JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS > 0
    assert JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES >= 0
    assert "java_atomic_assembly_system_prompt()" in assembly_source
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
    canonical_phrases = (
        "Nested runtime types live inside a host-owned outer class.",
        "Return only compile-ready Java class-body source",
        "must be private because outer type ownership is host-owned",
        "The host adds static to outer fields/methods and owns nested-type visibility.",
    )

    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path == authority:
            continue
        source = path.read_text(encoding="utf-8")
        for match in definition_re.finditer(source):
            violations.append(f"{path.relative_to(package_root)} defines {match.group(0).strip()}")
        for phrase in canonical_phrases:
            if phrase in source:
                violations.append(
                    f"{path.relative_to(package_root)} duplicates canonical prompt/policy phrase {phrase!r}"
                )

    assert not violations, "execution contract drift outside canonical authority:\n" + "\n".join(
        violations
    )


def test_core_execution_contract_consumers_import_the_authority_directly() -> None:
    package_root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    consumers = (
        "atomic_concern_source.py",
        "atomic_java_assembly.py",
        "central_atomic_generation_contract.py",
        "custom_module_generator.py",
        "fixed_template_generation.py",
        "generation_diagnostic_repair.py",
        "java_region_parser.py",
        "model_output_atomicity_contract.py",
        "repair_engine.py",
        "repair_response_contract.py",
        "verifier_repair_window.py",
    )

    missing = []
    for filename in consumers:
        source = (package_root / filename).read_text(encoding="utf-8")
        if "from .execution_contract_policy import" not in source:
            missing.append(filename)
    assert not missing, f"execution contract consumer bypasses canonical policy: {missing!r}"



def test_schema_selector_and_prompt_share_type_owning_concerns() -> None:
    import minecraft_mod_ai.custom_module_generator as generator

    source = inspect.getsource(generator)
    assert "_ATOMIC_TYPE_OWNING_CONCERNS" not in source
    assert "JAVA_TYPE_OWNING_CONCERNS" in source
    assert "JAVA_DECLARATION_ONLY_CONCERNS" in source
    assert {"variables", "inputs", "outputs", "stored_state", "payloads"} == set(
        JAVA_TYPE_OWNING_CONCERNS
    )



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
        assert name_schema["enum"] == ["ShipFuelCalculationException"]
        assert name_schema["not"] == {"enum": ["AuthoredFailureLimits"]}


def test_atomic_java_assembly_reads_model_field_bound_directly_from_canonical_policy() -> None:
    import minecraft_mod_ai.atomic_java_assembly as assembly

    source = inspect.getsource(assembly)
    assert "MAX_MODEL_FIELDS" not in source
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_fields" in source



def test_model_facing_java_schema_is_owned_by_canonical_contract() -> None:
    import minecraft_mod_ai.custom_module_generator as generator

    source = inspect.getsource(generator)
    for legacy in (
        "_ATOMIC_PARAMETER_SCHEMA",
        "_ATOMIC_FIELD_SCHEMA",
        "_ATOMIC_METHOD_SCHEMA",
        "_ATOMIC_RECORD_SCHEMA",
        "_ATOMIC_ENUM_SCHEMA",
        "_ATOMIC_CLASS_SCHEMA",
        "_ATOMIC_MEMBERS_PARAMETERS",
        "_ATOMIC_LOGIC_MEMBERS_PARAMETERS",
        "_ATOMIC_DECLARATION_MEMBERS_PARAMETERS",
        "_ATOMIC_INITIALIZE_PARAMETERS",
    ):
        assert legacy not in source
    assert "java_atomic_parameters_for_request(" in source
