"""Compile canonical declaration records without a model or model repair path.

Stored-state rows describe what is persisted, its owner and scope. They do not
specify a runtime value type or a persistence algorithm. Keep those authored
values as data; never guess fields, codecs or Java types from their prose.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .authored_atomic_contract import task_concern_authority
from .authored_execution_schema import concern_contracts
from .custom_module_errors import CustomModuleGenerationError
from .execution_contract_policy import JAVA_DECLARATION_ONLY_CONCERNS


def compile_declaration_concern(
    task: Mapping[str, Any],
    *,
    section: str,
    concern: str,
    host_symbol: str,
) -> str:
    """Lower admitted records into a host-built declaration tree and render it."""
    if concern not in JAVA_DECLARATION_ONLY_CONCERNS or (section, concern) != (
        "persistence", "stored_state"
    ):
        raise CustomModuleGenerationError(
            f"HOST_DECLARATION_COMPILER_REQUIRED: {section}.{concern} has no host compiler."
        )

    # Use the host template, never a task/model-supplied schema or Java shape.
    contract = next(row for row in concern_contracts(section) if row["concern"] == concern)
    schema = contract["record_schema"]
    try:
        records = task_concern_authority(task, concern, strict_records=True)["structured_records"]
    except ValueError as exc:
        raise CustomModuleGenerationError(
            f"HOST_DECLARATION_RECORDS_INVALID: {section}.{concern}: {exc}"
        ) from exc
    if not records:
        raise CustomModuleGenerationError(
            f"HOST_DECLARATION_RECORDS_INVALID: {section}.{concern} requires canonical records."
        )
    validator = Draft202012Validator(schema)
    for index, record in enumerate(records):
        try:
            validator.validate(record)
        except ValidationError as exc:
            raise CustomModuleGenerationError(
                f"HOST_DECLARATION_RECORDS_INVALID: {section}.{concern}[{index}]: {exc.message}"
            ) from exc

    # Kind, identity, component types/order and storage are compiler policy. No
    # identifier is inferred from a row's natural-language state/owner/scope.
    fields = ("state", "owner", "scope")
    type_name = "StoredState" if host_symbol != "StoredState" else "StoredStateRecord"
    values = [
        "new " + type_name + "("
        + ", ".join(json.dumps(record[key], ensure_ascii=True) for key in fields)
        + ")"
        for record in records
    ]
    declarations = {
        "records": [{
            "name": type_name,
            "components": [{"name": key, "type": "String"} for key in fields],
        }],
        "fields": [{
            "name": "STORED_STATES",
            "type": f"java.util.List<{type_name}>",
            "modifiers": ["private", "static", "final"],
            "initializer": "java.util.List.of(\n    " + ",\n    ".join(values) + "\n)",
        }],
    }
    # Share Java syntax rendering, but never JavaStructureAssembly, a task
    # capsule, declaration paging, model completion or model-authored modifiers.
    from .custom_module_generator import _render_atomic_java_structure

    return _render_atomic_java_structure(
        declarations, response_region="members", host_symbol=host_symbol,
    )
