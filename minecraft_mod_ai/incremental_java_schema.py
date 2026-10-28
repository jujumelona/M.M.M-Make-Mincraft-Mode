"""Small closed declaration and operation pages for host-rendered Java."""
from __future__ import annotations

from .structured_json_schema import closed_object_schema as obj

TEXT = {"type": "string", "maxLength": 240}
TYPE = {"type": "string", "minLength": 1, "maxLength": 240}
NAME = {"type": "string", "minLength": 1, "maxLength": 128,
        "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$"}
BOOL = {"type": "boolean"}


def choice(name, values):
    return obj({name: {"type": "string", "enum": list(values)}})


COUNT = obj({"count": {"type": "integer", "minimum": 0, "maximum": 255}})
SCHEMAS = {
    "local": obj({"type": TYPE}),
    "assign": obj({"operator": {"type": "string", "enum": ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>="]}}),
    "foreach": obj({"type": TYPE}),
    "for": obj({"type": TYPE}),
    "lookup": obj({"query": TYPE, "offset": {"type": "integer", "minimum": 0, "maximum": 1000000}}),
    "field": obj({"type": TYPE, "purpose": TYPE, "static": BOOL, "final": BOOL}),
    "method": obj({"return_type": TYPE, "purpose": TYPE, "static": BOOL}),
    "parameter": obj({"type": TYPE, "purpose": TEXT}),
    "type": obj({"kind": {"type": "string", "enum": ["class", "record", "enum"]},
                 "purpose": TYPE, "superclass": TEXT}),
    "constant": obj({"name": NAME}),
    "lambda": obj({"type": TYPE, "purpose": TYPE}),
    "lambda_contract": obj({"return_type": TYPE}),
    "catch": obj({"type": TYPE}),
    "constructor_call": choice("target", ["super", "this"]),
}
BLOCK_OPS = frozenset({"if", "while", "do", "foreach", "for", "block", "synchronized"})
SIMPLE_OPS = ("call", "local", "assign", "if", "while", "do", "foreach", "for", "return",
              "throw", "block", "synchronized", "switch", "try", "lambda", "register")
DECLARATION_OPS = ("field", "method", "type")

# This bounds the authoring *structure*, not the size of a requested feature.
# A feature may have arbitrarily many sequential operations or helper members,
# but uncontrolled recursive Java blocks cannot consume the Python call stack.
MAX_CONTROL_FLOW_NESTING = 24
_NESTED_OPERATIONS = frozenset({"if", "while", "do", "foreach", "for",
                                "block", "synchronized", "switch", "try", "lambda"})


def operation_options(member, scope, body, blocked_declarations, *, nesting_depth=0):
    if member["kind"] in {"class", "record", "enum"}:
        # Type layouts have no executable operations. Reusing one overload must
        # not remove the ability to declare other methods or constructors.
        return [*DECLARATION_OPS, "constructor", "override", "lookup", "complete"]
    options = [*SIMPLE_OPS, *DECLARATION_OPS, "lookup", "complete"]
    if scope["loop_depth"] or scope["switch_depth"]:
        options.append("break")
    if scope["loop_depth"]:
        options.append("continue")
    if member["kind"] == "constructor" and not body:
        options.append("constructor_call")
    # A bare block does not perform an action, and opening a new empty block
    # inside another empty block is a no-progress loop for a small model.
    if not body:
        options = [option for option in options if option != "block"]
    if nesting_depth >= MAX_CONTROL_FLOW_NESTING:
        options = [option for option in options if option not in _NESTED_OPERATIONS]
    # Model completion is not evidence of implemented behavior. A registrar or
    # callback with no observable operation must keep authoring instead of
    # claiming success and failing only at final rendering.
    if (member.get("registrar") or member.get("event")) and not _has_observable_action(body):
        options = [option for option in options if option != "complete"]
    return [option for option in options if option not in blocked_declarations]


def _has_observable_action(body):
    from .incremental_java_history import walk_nodes
    for node in walk_nodes(body):
        op = node.get("op")
        if op in {"call", "assign", "register", "constructor_call", "throw"}:
            return True
        if op == "return" and node.get("value"):
            return True
    return False
