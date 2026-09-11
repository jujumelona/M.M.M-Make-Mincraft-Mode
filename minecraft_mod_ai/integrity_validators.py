"""Executable validators. Exceptions never become successful validation receipts."""
from __future__ import annotations

import json
from jsonschema import Draft202012Validator

from .implementation_identity import compute_content_hash, compute_json_schema_hash


def validate_json_schema(value, schema):
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(value)
    return {"validator": "json_schema", "status": "PASS"}


def validate_java_syntax(source, *, java_version, filename="IntegrityInput.java", prefix="", suffix=""):
    from .javac_bridge import analyze_java
    analyze_java(prefix + source + suffix, java_version=java_version, filename=filename, parse_only=True)
    return {"validator": "java_syntax", "status": "PASS"}


def validate_semantic_contract(value, *, contract, context_id, leaf_id):
    """Validate explicit output constraints and exact requirement/context bindings.

    This validates the declared contract only. Behavioral proof remains a separate
    compile/GameTest/runtime obligation and is never inferred from this receipt.
    """
    if contract.get("leaf_id") != leaf_id or contract.get("context_id") != context_id:
        raise ValueError("SEMANTIC_BINDING_MISMATCH")
    if not contract.get("requirement") or not contract.get("bindings"):
        raise ValueError("SEMANTIC_REQUIREMENT_MISSING")
    schema = contract.get("output_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("SEMANTIC_OUTPUT_CONSTRAINTS_MISSING")
    parsed = json.loads(value) if contract["language"] == "json" else value
    validate_json_schema(parsed, schema)
    return {"validator": "semantic_contract", "status": "PASS",
            "leaf_id": leaf_id, "context_id": context_id,
            "content_sha256": compute_content_hash(value.encode()),
            "contract_sha256": compute_json_schema_hash(contract)}


def validate_side(source, *, leaf_id, side, classpath, java_version, classpath_sides=None):
    from .java_symbol_resolver import JavaSymbolResolver
    from .side_enforcement import check_side_compatibility
    symbols = JavaSymbolResolver(classpath, java_version, classpath_sides=classpath_sides).extract_symbols_from_source(source)
    for symbol in symbols:
        valid, reason = check_side_compatibility(side, symbol.side)
        if not valid:
            raise ValueError(f"SIDE_VIOLATION: {leaf_id}: {symbol.owner}.{symbol.name}: {reason}")
    return {"validator": "client_side_only", "status": "PASS", "symbols_checked": len(symbols)}


def validate_mod_integration(*, store, evidence_id, expected):
    from .integrity_evidence import verify_execution_evidence
    verify_execution_evidence(store, evidence_id, expected=expected, required_gates=("compile", "gametest"))
    return {"validator": "mod_integration_test", "status": "PASS", "evidence_id": evidence_id}
