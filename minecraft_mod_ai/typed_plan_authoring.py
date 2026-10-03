from __future__ import annotations

"""Native-tool authoring for typed PlanIR.

The planner emits one scalar typed decision at a time. It never emits Java or an
opaque JSON blob, and a hard call bound prevents recursive authoring from running
without a host-observed termination measure.
"""

import json
from collections.abc import Mapping
from typing import Any


_TYPES = ["boolean", "int", "long", "double", "string", "object"]


class TypedOperationAuthor:
    def __init__(
        self,
        router: Any,
        source_text: str,
        structured_sections: Mapping[str, Any] | None,
        capabilities: Mapping[str, Any] | None,
        *,
        max_calls: int = 256,
    ) -> None:
        self.router = router
        self.source_text = str(source_text)
        self.structured_sections = dict(structured_sections or {})
        self.capabilities = dict(capabilities or {})
        self.max_calls = max(1, int(max_calls))
        self.call_count = 0

    def _ask(self, field: str, schema: Mapping[str, Any], *, scope: str) -> Any:
        if self.call_count >= self.max_calls:
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_LIMIT: exceeded {self.max_calls} native decisions"
            )
        self.call_count += 1
        parameters = {
            "type": "object",
            "properties": {"value": dict(schema)},
            "required": ["value"],
            "additionalProperties": False,
        }
        payload = {
            "source_text": self.source_text,
            "structured_sections": self.structured_sections,
            "available_capabilities": sorted(self.capabilities),
            "scope": scope,
            "field": field,
            "decision_index": self.call_count,
            "policy": (
                "Choose only one typed PlanIR scalar. Do not emit Java, prose, markdown, "
                "or additional fields. Preserve approved behavior exactly."
            ),
        }
        result = self.router.generate_tool_decision(
            "planner",
            (
                {
                    "role": "system",
                    "content": (
                        "Author executable behavior through the host typed PlanIR only. "
                        "Every answer is one required native scalar decision."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ),
            tool_name="emit_typed_plan_value",
            parameters=parameters,
            description="Emit exactly one host-requested typed PlanIR value.",
        )
        if not isinstance(result, Mapping) or set(result) != {"value"}:
            raise ValueError("TYPED_PLAN_AUTHORING_RESPONSE_INVALID")
        return result["value"]

    def _enum(self, field: str, values: list[str], *, scope: str) -> str:
        value = self._ask(field, {"type": "string", "enum": values}, scope=scope)
        if value not in values:
            raise ValueError(f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {field}")
        return str(value)

    def _identifier(self, field: str, *, scope: str) -> str:
        return str(self._ask(
            field,
            {"type": "string", "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$"},
            scope=scope,
        ))

    def _type(self, field: str, *, scope: str, allow_void: bool = False) -> str:
        values = [*_TYPES, *(["void"] if allow_void else [])]
        return self._enum(field, values, scope=scope)

    def _literal_value(self, kind: str, *, scope: str) -> Any:
        schema: dict[str, Any]
        if kind == "boolean":
            schema = {"type": "boolean"}
        elif kind in {"int", "long"}:
            schema = {"type": "integer"}
        elif kind == "double":
            schema = {"type": "number"}
        elif kind == "string":
            schema = {"type": "string"}
        else:
            schema = {"type": ["string", "number", "integer", "boolean", "null"]}
        return self._ask("literal_value", schema, scope=scope)

    def expression(self, scope: str) -> dict[str, Any]:
        op = self._enum(
            "expression_op",
            ["literal", "ref", "unary", "binary", "list", "map", "call", "capability", "state_get"],
            scope=scope,
        )
        if op == "literal":
            kind = self._type("literal_type", scope=scope)
            return {"op": "literal", "type": kind,
                    "value": self._literal_value(kind, scope=scope)}
        if op == "ref":
            return {"op": "ref", "name": self._identifier("reference_name", scope=scope)}
        if op == "unary":
            operator = self._enum("unary_operator", ["!", "-"], scope=scope)
            return {"op": "unary", "operator": operator,
                    "value": self.expression(scope + ".unary")}
        if op == "binary":
            operator = self._enum(
                "binary_operator",
                ["+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||"],
                scope=scope,
            )
            return {
                "op": "binary", "operator": operator,
                "left": self.expression(scope + ".left"),
                "right": self.expression(scope + ".right"),
            }
        if op == "list":
            count = int(self._ask(
                "list_item_count",
                {"type": "integer", "minimum": 0, "maximum": 64},
                scope=scope,
            ))
            return {
                "op": "list",
                "items": [self.expression(f"{scope}.item[{i}]") for i in range(count)],
            }
        if op == "map":
            count = int(self._ask(
                "map_entry_count",
                {"type": "integer", "minimum": 0, "maximum": 64},
                scope=scope,
            ))
            entries = []
            for index in range(count):
                entries.append({
                    "key": self.expression(f"{scope}.entry[{index}].key"),
                    "value": self.expression(f"{scope}.entry[{index}].value"),
                })
            return {"op": "map", "entries": entries}
        if op == "call":
            function = self._identifier("function_id", scope=scope)
            count = int(self._ask(
                "argument_count",
                {"type": "integer", "minimum": 0, "maximum": 32},
                scope=scope,
            ))
            return {
                "op": "call", "function": function,
                "args": [self.expression(f"{scope}.arg[{i}]") for i in range(count)],
            }
        if op == "capability":
            ids = sorted(self.capabilities)
            if not ids:
                raise ValueError("TYPED_PLAN_AUTHORING_CAPABILITY_UNAVAILABLE")
            cap_id = self._enum("capability_id", ids, scope=scope)
            params = self.capabilities.get(cap_id, {}).get("parameters", [])
            count = len(params) if isinstance(params, list) else 0
            return {
                "op": "capability", "id": cap_id,
                "args": [
                    self.expression(f"{scope}.capability_arg[{i}]")
                    for i in range(count)
                ],
            }
        if op == "state_get":
            return {
                "op": "state_get",
                "key": self.expression(scope + ".state_key"),
                "type": self._type("state_type", scope=scope),
                "context": self.expression(scope + ".state_context"),
            }
        raise AssertionError(op)

    def statement(self, scope: str) -> dict[str, Any] | None:
        op = self._enum(
            "statement_op",
            ["let", "set", "return", "assert", "if", "while", "foreach", "state_set", "expr", "done"],
            scope=scope,
        )
        if op == "done":
            return None
        if op == "let":
            return {
                "op": "let",
                "name": self._identifier("local_name", scope=scope),
                "type": self._type("local_type", scope=scope),
                "value": self.expression(scope + ".value"),
            }
        if op == "set":
            return {
                "op": "set",
                "name": self._identifier("local_name", scope=scope),
                "value": self.expression(scope + ".value"),
            }
        if op == "return":
            has_value = bool(self._ask(
                "return_has_value", {"type": "boolean"}, scope=scope
            ))
            return (
                {"op": "return", "value": self.expression(scope + ".value")}
                if has_value
                else {"op": "return"}
            )
        if op == "assert":
            return {
                "op": "assert",
                "condition": self.expression(scope + ".condition"),
                "message": str(self._ask(
                    "assert_message", {"type": "string"}, scope=scope
                )),
            }
        if op == "if":
            return {
                "op": "if",
                "condition": self.expression(scope + ".condition"),
                "then": self.body(scope + ".then"),
                "else": self.body(scope + ".else"),
            }
        if op == "while":
            return {
                "op": "while",
                "condition": self.expression(scope + ".condition"),
                "max_iterations": int(self._ask(
                    "max_iterations",
                    {"type": "integer", "minimum": 1, "maximum": 1000000},
                    scope=scope,
                )),
                "body": self.body(scope + ".body"),
            }
        if op == "foreach":
            return {
                "op": "foreach",
                "name": self._identifier("item_name", scope=scope),
                "type": self._type("item_type", scope=scope),
                "collection": self.expression(scope + ".collection"),
                "body": self.body(scope + ".body"),
            }
        if op == "state_set":
            return {
                "op": "state_set",
                "key": self.expression(scope + ".state_key"),
                "value": self.expression(scope + ".state_value"),
                "context": self.expression(scope + ".state_context"),
            }
        if op == "expr":
            return {"op": "expr", "value": self.expression(scope + ".value")}
        raise AssertionError(op)

    def body(self, scope: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        while True:
            statement = self.statement(f"{scope}.statement[{len(result)}]")
            if statement is None:
                return result
            result.append(statement)


__all__ = ["TypedOperationAuthor"]
