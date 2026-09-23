from __future__ import annotations

"""Bound source repair values by patch policy, not generic 256-character text slots."""

from .model_response_templates import response_schema


def repair_response_schema(max_patch_bytes: int) -> dict:
    schema = response_schema("repair")
    source_limit = max(1, min(16384, int(max_patch_bytes)))
    span_limit = min(4096, source_limit)
    for branch in schema["properties"]["operations"]["items"]["anyOf"]:
        properties = branch["properties"]
        if "content" in properties:
            properties["content"]["maxLength"] = source_limit
        if "replacements" in properties:
            replacement = properties["replacements"]["items"]["properties"]
            replacement["old"]["maxLength"] = span_limit
            replacement["new"]["maxLength"] = span_limit
    return schema
