"""Host-owned predicate/effect/result control flow for every Fabric callback."""
from __future__ import annotations

import json
from collections.abc import Mapping

from .fixed_template_generation import _validate_native_arguments
from .host_callback_body import parse_expression


def _result_fields(event, symbols):
    result_type = event["return_type"]
    classes = {row["owner"]: row for row in symbols.values()
               if isinstance(row, Mapping) and row.get("kind") == "CLASS" and row.get("owner")}

    def assignable(type_name):
        pending, seen = [type_name], set()
        while pending:
            current = pending.pop()
            if current == result_type:
                return True
            if not current or current in seen:
                continue
            seen.add(current)
            row = classes.get(current, {})
            pending.extend([row.get("superclass"), *row.get("interfaces", ())])
        return False

    return sorted({result_type + "." + row["name"] for row in symbols.values()
                   if isinstance(row, Mapping) and row.get("kind") == "FIELD"
                   and row.get("owner") == result_type and row.get("name")
                   and int(row.get("access") or 0) & 9 == 9
                   and assignable(row.get("return_type"))})


def select_callback_results(router, event, symbols):
    """Select semantic result values separately from gameplay implementations.

    Types and public constant fields come from the inspected classpath. No
    guessed PASS/SUCCESS names are injected into generated Java. Callbacks
    without constants can use a checked expression; javac remains the type
    authority. Void callbacks do not need any result inference.
    """
    type_name = event["return_type"]
    if type_name == "void":
        return {}
    choices = ["false", "true"] if type_name == "boolean" else _result_fields(event, symbols)
    expression_schema = ({"type": "string", "enum": choices} if choices else
                         {"type": "string", "minLength": 1, "maxLength": 2048})
    schema = {"type": "object", "properties": {"expression": expression_schema},
              "required": ["expression"], "additionalProperties": False}
    selected = {}
    for branch in ("unhandled", "handled"):
        response = router.generate_structured_data(
            "coder", [{"role": "system", "content": (
                "Choose ONLY the result VALUE for this exact inspected Fabric callback. "
                "The host owns all return statements, conditional dispatch and gameplay "
                "effects. unhandled means the original gameplay condition did not match; "
                "handled means its actual effects have executed. Preserve this callback's "
                "listener chaining semantics; use null only when its API specifies null "
                "as the pass-through result. Do not use a result as a gameplay effect. "
                "Return one expression, without return keyword or semicolon.")},
                {"role": "user", "content": json.dumps({
                    "verified_callback": event, "branch": branch,
                    "verified_result_values": choices, "selected_results": selected,
                }, ensure_ascii=False)}],
            data_name="select_callback_result_" + branch, parameters=schema,
        )
        value = _validate_native_arguments(response, schema)["expression"]
        selected[branch] = parse_expression(value)
    return selected


def render_callback_dispatch(condition, effect, arguments, results, *, observers=()):
    # Every missing world/UI source is now a concrete, separately compiled
    # predicate. No gameplay effect can run merely because the broad event
    # predicate matched; all required observers must also evaluate true.
    guards = [condition, *observers]
    if len(guards) != len(set(guards)):
        raise ValueError("CALLBACK_OBSERVER_METHOD_DUPLICATE")
    if not all(isinstance(name, str) and name.isidentifier() for name in guards):
        raise ValueError("CALLBACK_OBSERVER_METHOD_INVALID")
    predicate = " && ".join(f"this.{name}({arguments})" for name in guards)
    if not results:
        return (f"if ({predicate}) {{\n"
                f"    this.{effect}({arguments});\n}}")
    return (f"if (!({predicate})) {{\n"
            f"    return {results['unhandled']};\n}}\n"
            f"if (!this.{effect}({arguments})) {{\n"
            f"    return {results['unhandled']};\n}}\nreturn {results['handled']};")
