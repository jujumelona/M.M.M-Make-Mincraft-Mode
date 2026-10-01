from __future__ import annotations

"""Build the model-facing repair schema from the central execution contract."""

from .execution_contract_policy import (
    SCHEMA_CONTRACT_PROFILE_KEY,
    SCHEMA_STRING_CLASS_KEY,
    SOURCE_REPAIR_MAX_SOURCE_CHARS,
    SOURCE_REPAIR_MAX_SPAN_CHARS,
    SOURCE_REPAIR_SCHEMA_PROFILE,
    STRING_CLASS_REPAIR_SPAN,
    STRING_CLASS_SOURCE,
)
from .model_response_templates import response_schema


def repair_response_schema(max_patch_bytes: int) -> dict:
    schema = response_schema("repair")
    schema[SCHEMA_CONTRACT_PROFILE_KEY] = SOURCE_REPAIR_SCHEMA_PROFILE

    source_limit = max(
        1,
        min(SOURCE_REPAIR_MAX_SOURCE_CHARS, int(max_patch_bytes)),
    )
    span_limit = min(SOURCE_REPAIR_MAX_SPAN_CHARS, source_limit)

    for branch in schema["properties"]["operations"]["items"]["anyOf"]:
        properties = branch["properties"]
        if "content" in properties:
            properties["content"]["maxLength"] = source_limit
            properties["content"][SCHEMA_STRING_CLASS_KEY] = STRING_CLASS_SOURCE
        if "replacements" in properties:
            replacement = properties["replacements"]["items"]["properties"]
            replacement["old"]["maxLength"] = span_limit
            replacement["old"][SCHEMA_STRING_CLASS_KEY] = STRING_CLASS_REPAIR_SPAN
            replacement["new"]["maxLength"] = span_limit
            replacement["new"][SCHEMA_STRING_CLASS_KEY] = STRING_CLASS_REPAIR_SPAN
    return schema
