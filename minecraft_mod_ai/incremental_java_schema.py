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


def operation_options(member, scope, body, blocked_declarations):
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
    return [option for option in options if option not in blocked_declarations]
