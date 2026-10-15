"""Forward Java node slots must not admit transitive cyclic references."""
from __future__ import annotations

import re

from minecraft_mod_ai.construction_operation_binding import forbidden_input_references
from minecraft_mod_ai.native_string_exclusions import string_excluding


def _node(name, refs):
    return {"node_id": name, "input_refs": list(refs)}


def test_forbidden_input_refs_include_all_transitive_consumers():
    # operation_1 -> operation_1:bool -> missing boolean input.
    # Resolving the boolean from operation_1 closes a graph cycle.
    nodes = [
        _node("operation_1", ["operation_1:bool"]),
        _node("operation_2", ["operation_1"]),
        _node("operation_3", ["other"]),
    ]
    blocked = forbidden_input_references(nodes, "operation_1:bool")
    assert blocked == {"operation_1:bool", "operation_1", "operation_2"}


def test_native_reference_schema_rejects_cycles_without_banning_forward_refs():
    nodes = [_node("operation_1", ["operation_1:bool"])]
    blocked = forbidden_input_references(nodes, "operation_1:bool")
    schema = string_excluding(
        blocked, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    )
    pattern = schema["pattern"]
    assert not re.fullmatch(pattern, "operation_1:bool")
    assert not re.fullmatch(pattern, "operation_1")
    assert re.fullmatch(pattern, "operation_4:int")
    assert re.fullmatch(pattern, "other")
