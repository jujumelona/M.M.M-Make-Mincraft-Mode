"""Small closed JSON data shapes shared by host-owned source compilers."""
from __future__ import annotations


def closed_object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}
