"""Host-owned Java scalar literal lowering for canonical state writers.

A model may select a typed JSON value, not a Java statement or expression.
This is a syntax/type boundary; the originating behavior still needs a real
trigger, meaningful state transitions, and runtime gameplay verification.
"""
from __future__ import annotations

import json
import math


_TYPES = {
    "boolean": "boolean", "java.lang.Boolean": "boolean", "Boolean": "boolean",
    "byte": "byte", "short": "short", "int": "int", "long": "long",
    "float": "float", "double": "double",
    "String": "string", "java.lang.String": "string",
}


def typed_value_schema(java_type: str) -> dict:
    """Use the JSON decoder to disallow code tokens in a scalar slot."""
    kind = _TYPES.get(java_type.strip())
    if kind is None:
        raise ValueError("CALLBACK_REQUIRED_STATE_TYPED_VALUE_UNSUPPORTED: " + java_type)
    value = {"type": "string"} if kind == "string" else (
        {"type": "boolean"} if kind == "boolean" else
        {"type": "integer"} if kind in {"byte", "short", "int", "long"} else
        {"type": "number"}
    )
    if kind in {"byte", "short", "int", "long"}:
        limits = {"byte": (-128, 127), "short": (-32768, 32767),
                  "int": (-(2 ** 31), 2 ** 31 - 1),
                  "long": (-(2 ** 63), 2 ** 63 - 1)}
        value["minimum"], value["maximum"] = limits[kind]
    if kind == "string":
        value["maxLength"] = 1024
    return {"type": "object", "properties": {"value": value},
            "required": ["value"], "additionalProperties": False}


def lower_typed_java_value(row: dict, java_type: str) -> str:
    """Compile exactly one model-selected JSON primitive to a Java literal."""
    kind = _TYPES.get(java_type.strip())
    if kind is None:
        raise ValueError("CALLBACK_REQUIRED_STATE_TYPED_VALUE_UNSUPPORTED: " + java_type)
    if not isinstance(row, dict) or set(row) != {"value"}:
        raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_SCHEMA_INVALID")
    value = row["value"]
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_TYPE_MISMATCH")
        return "true" if value else "false"
    if kind == "string":
        if not isinstance(value, str) or len(value) > 1024:
            raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_TYPE_MISMATCH")
        # JSON escapes Java string syntax too, except JSON permits \u2028 and
        # \u2029 as literal characters. Escape those for portable Java source.
        return json.dumps(value, ensure_ascii=True)
    if kind in {"byte", "short", "int", "long"}:
        if type(value) is not int:
            raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_TYPE_MISMATCH")
        low, high = {"byte": (-128, 127), "short": (-32768, 32767),
                     "int": (-(2 ** 31), 2 ** 31 - 1),
                     "long": (-(2 ** 63), 2 ** 63 - 1)}[kind]
        if not low <= value <= high:
            raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_OUT_OF_RANGE")
        return str(value) + ("L" if kind == "long" else "")
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_TYPE_MISMATCH")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_TYPE_MISMATCH") from exc
    if not math.isfinite(number):
        raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_OUT_OF_RANGE")
    if kind == "float" and abs(number) > 3.4028235e38:
        raise ValueError("CALLBACK_REQUIRED_STATE_VALUE_OUT_OF_RANGE")
    return repr(number) + ("f" if kind == "float" else "d")


def typed_state_transition_schema(java_type: str) -> dict:
    """One checked state change: no Java source, arbitrary target or method.

    "set" preserves legacy typed-literal contracts. Numeric states also
    allow additive transitions so e.g. repeated resource rewards do not
    silently reset a balance to the same constant every time.
    """
    schema = typed_value_schema(java_type)
    family = _TYPES[java_type.strip()]
    permitted = ["set", "add", "subtract"] if family in {
        "int", "long", "float", "double"} else ["set"]
    schema["properties"]["operator"] = {
        "type": "string", "enum": permitted,
    }
    return schema


def lower_typed_state_transition(row: dict, java_type: str, *,
                                 reader_expression: str) -> str:
    """Render the value argument of a verified state writer.

    For arithmetic, the host generates a read-modify-write of the *same*
    canonical state and scope. Java integer overflow fails explicitly instead
    of silently corrupting a persisted score or currency.
    """
    if not isinstance(row, dict) or not set(row) <= {"value", "operator"}:
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_INVALID")
    if "value" not in row:
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_INVALID")
    family = _TYPES.get(java_type.strip())
    operator = row.get("operator", "set")
    if operator not in ("set", "add", "subtract"):
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_OPERATOR_INVALID")
    if operator != "set" and family not in {
        "int", "long", "float", "double"}:
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_TYPE_INVALID")
    value = lower_typed_java_value({"value": row["value"]}, java_type)
    if operator == "set":
        return value
    if not isinstance(reader_expression, str) or not reader_expression.strip():
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_READER_MISSING")
    if row["value"] == 0:
        raise ValueError("CALLBACK_REQUIRED_STATE_TRANSITION_NOOP")
    if family in {"int", "long"}:
        method = "addExact" if operator == "add" else "subtractExact"
        return ("java.lang.Math." + method + "(" + reader_expression
                + ", " + value + ")")
    sign = "+" if operator == "add" else "-"
    expr = "(" + reader_expression + " " + sign + " " + value + ")"
    return expr
