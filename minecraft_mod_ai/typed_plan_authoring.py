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


def _active_concern_refs(
    structured_sections: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    from .authored_structured_design import active_concern_records
    from .planning_detail_template import WORKSHEET_SECTIONS

    refs: list[str] = []
    for section in WORKSHEET_SECTIONS:
        for concern, rows in active_concern_records(
            structured_sections,
            section,
        ).items():
            if rows:
                refs.append(f"{section}.{concern}")
    return tuple(dict.fromkeys(refs))


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
        self.function_signatures: dict[str, tuple[str, ...]] = {}

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
            "known_functions": {
                name: list(parameters)
                for name, parameters in self.function_signatures.items()
            },
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

    def expression(
        self,
        scope: str,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        bindings = dict(env or {})
        expression_ops = [
            "literal", "unary", "binary", "list", "map", "state_get"
        ]
        if bindings:
            expression_ops.append("ref")
        if self.function_signatures:
            expression_ops.append("call")
        if self.capabilities:
            expression_ops.append("capability")
        op = self._enum(
            "expression_op",
            expression_ops,
            scope=scope,
        )
        if op == "literal":
            kind = self._type("literal_type", scope=scope)
            return {
                "op": "literal",
                "type": kind,
                "value": self._literal_value(kind, scope=scope),
            }
        if op == "ref":
            name = self._enum(
                "reference_name",
                sorted(bindings),
                scope=scope,
            )
            return {"op": "ref", "name": name}
        if op == "unary":
            operator = self._enum("unary_operator", ["!", "-"], scope=scope)
            return {
                "op": "unary",
                "operator": operator,
                "value": self.expression(scope + ".unary", bindings),
            }
        if op == "binary":
            operator = self._enum(
                "binary_operator",
                ["+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||"],
                scope=scope,
            )
            return {
                "op": "binary",
                "operator": operator,
                "left": self.expression(scope + ".left", bindings),
                "right": self.expression(scope + ".right", bindings),
            }
        if op == "list":
            count = int(self._ask(
                "list_item_count",
                {"type": "integer", "minimum": 0, "maximum": 64},
                scope=scope,
            ))
            return {
                "op": "list",
                "items": [
                    self.expression(f"{scope}.item[{i}]", bindings)
                    for i in range(count)
                ],
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
                    "key": self.expression(
                        f"{scope}.entry[{index}].key",
                        bindings,
                    ),
                    "value": self.expression(
                        f"{scope}.entry[{index}].value",
                        bindings,
                    ),
                })
            return {"op": "map", "entries": entries}
        if op == "call":
            function = self._enum(
                "function_id",
                sorted(self.function_signatures),
                scope=scope,
            )
            parameter_types = self.function_signatures[function]
            return {
                "op": "call",
                "function": function,
                "args": [
                    self.expression(f"{scope}.arg[{i}]", bindings)
                    for i in range(len(parameter_types))
                ],
            }
        if op == "capability":
            ids = sorted(self.capabilities)
            cap_id = self._enum("capability_id", ids, scope=scope)
            params = self.capabilities.get(cap_id, {}).get("parameters", [])
            count = len(params) if isinstance(params, list) else 0
            return {
                "op": "capability",
                "id": cap_id,
                "args": [
                    self.expression(
                        f"{scope}.capability_arg[{i}]",
                        bindings,
                    )
                    for i in range(count)
                ],
            }
        if op == "state_get":
            return {
                "op": "state_get",
                "key": self.expression(scope + ".state_key", bindings),
                "type": self._type("state_type", scope=scope),
                "context": self.expression(
                    scope + ".state_context",
                    bindings,
                ),
            }
        raise AssertionError(op)

    def statement(
        self,
        scope: str,
        *,
        env: Mapping[str, str] | None = None,
        return_type: str | None = None,
    ) -> dict[str, Any] | None:
        bindings = dict(env or {})
        statement_ops = [
            "let", "return", "assert", "if", "while", "foreach",
            "state_set", "expr", "done",
        ]
        if bindings:
            statement_ops.insert(1, "set")
        op = self._enum(
            "statement_op",
            statement_ops,
            scope=scope,
        )
        if op == "done":
            return None
        if op == "let":
            return {
                "op": "let",
                "name": self._identifier("local_name", scope=scope),
                "type": self._type("local_type", scope=scope),
                "value": self.expression(scope + ".value", bindings),
            }
        if op == "set":
            name = self._enum(
                "local_name",
                sorted(bindings),
                scope=scope,
            )
            return {
                "op": "set",
                "name": name,
                "value": self.expression(scope + ".value", bindings),
            }
        if op == "return":
            if return_type is None:
                has_value = bool(self._ask(
                    "return_has_value",
                    {"type": "boolean"},
                    scope=scope,
                ))
            else:
                has_value = return_type != "void"
            return (
                {
                    "op": "return",
                    "value": self.expression(scope + ".value", bindings),
                }
                if has_value
                else {"op": "return"}
            )
        if op == "assert":
            return {
                "op": "assert",
                "condition": self.expression(
                    scope + ".condition",
                    bindings,
                ),
                "message": str(self._ask(
                    "assert_message",
                    {"type": "string"},
                    scope=scope,
                )),
            }
        if op == "if":
            return {
                "op": "if",
                "condition": self.expression(
                    scope + ".condition",
                    bindings,
                ),
                "then": self.body(
                    scope + ".then",
                    env=bindings,
                    return_type=return_type,
                ),
                "else": self.body(
                    scope + ".else",
                    env=bindings,
                    return_type=return_type,
                ),
            }
        if op == "while":
            return {
                "op": "while",
                "condition": self.expression(
                    scope + ".condition",
                    bindings,
                ),
                "max_iterations": int(self._ask(
                    "max_iterations",
                    {"type": "integer", "minimum": 1, "maximum": 1000000},
                    scope=scope,
                )),
                "body": self.body(
                    scope + ".body",
                    env=bindings,
                    return_type=return_type,
                ),
            }
        if op == "foreach":
            name = self._identifier("item_name", scope=scope)
            item_type = self._type("item_type", scope=scope)
            nested = dict(bindings)
            nested[name] = item_type
            return {
                "op": "foreach",
                "name": name,
                "type": item_type,
                "collection": self.expression(
                    scope + ".collection",
                    bindings,
                ),
                "body": self.body(
                    scope + ".body",
                    env=nested,
                    return_type=return_type,
                ),
            }
        if op == "state_set":
            return {
                "op": "state_set",
                "key": self.expression(
                    scope + ".state_key",
                    bindings,
                ),
                "value": self.expression(
                    scope + ".state_value",
                    bindings,
                ),
                "context": self.expression(
                    scope + ".state_context",
                    bindings,
                ),
            }
        if op == "expr":
            return {
                "op": "expr",
                "value": self.expression(scope + ".value", bindings),
            }
        raise AssertionError(op)

    def body(
        self,
        scope: str,
        *,
        env: Mapping[str, str] | None = None,
        return_type: str | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        bindings = dict(env or {})
        while True:
            statement = self.statement(
                f"{scope}.statement[{len(result)}]",
                env=bindings,
                return_type=return_type,
            )
            if statement is None:
                return result
            result.append(statement)
            if statement.get("op") == "let":
                bindings[str(statement["name"])] = str(statement["type"])


def author_typed_plan_ir(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    *,
    max_calls: int = 2048,
) -> dict[str, Any]:
    """Author a complete typed PlanIR using only bounded scalar native decisions."""

    import hashlib

    author = TypedOperationAuthor(
        router,
        source_text,
        structured_sections,
        capabilities,
        max_calls=max_calls,
    )
    function_count = int(author._ask(
        "function_count",
        {"type": "integer", "minimum": 1, "maximum": 64},
        scope="program",
    ))

    specs: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for index in range(function_count):
        scope = f"function[{index}]"
        function_id = author._identifier("function_id", scope=scope)
        if function_id in known_ids:
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_DUPLICATE_FUNCTION: {function_id}"
            )
        known_ids.add(function_id)

        parameter_count = int(author._ask(
            "parameter_count",
            {"type": "integer", "minimum": 0, "maximum": 32},
            scope=scope,
        ))
        parameters: list[dict[str, str]] = []
        parameter_names: set[str] = set()
        for parameter_index in range(parameter_count):
            parameter_scope = f"{scope}.parameter[{parameter_index}]"
            name = author._identifier(
                "parameter_name",
                scope=parameter_scope,
            )
            if name in parameter_names:
                raise ValueError(
                    f"TYPED_PLAN_AUTHORING_DUPLICATE_PARAMETER: "
                    f"{function_id}.{name}"
                )
            parameter_names.add(name)
            parameters.append({
                "name": name,
                "type": author._type(
                    "parameter_type",
                    scope=parameter_scope,
                ),
            })

        return_type = author._type(
            "return_type",
            scope=scope,
            allow_void=True,
        )
        coverage_count = int(author._ask(
            "coverage_count",
            {"type": "integer", "minimum": 1, "maximum": 128},
            scope=scope,
        ))
        covers = [
            str(author._ask(
                "coverage_ref",
                {"type": "string", "minLength": 1},
                scope=f"{scope}.coverage[{coverage_index}]",
            ))
            for coverage_index in range(coverage_count)
        ]
        specs.append({
            "id": function_id,
            "parameters": parameters,
            "return_type": return_type,
            "covers": covers,
        })

    author.function_signatures = {
        str(spec["id"]): tuple(
            str(parameter["type"])
            for parameter in spec["parameters"]
        )
        for spec in specs
    }

    functions: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        env = {
            str(parameter["name"]): str(parameter["type"])
            for parameter in spec["parameters"]
        }
        functions.append({
            **spec,
            "body": author.body(
                f"function[{index}].body",
                env=env,
                return_type=str(spec["return_type"]),
            ),
        })

    coverage_refs = _active_concern_refs(structured_sections)
    from .typed_platform_ir import (
        PLATFORM_KINDS,
        platform_config_schema,
        validate_platform_modules,
    )

    platform_count = int(author._ask(
        "platform_module_count",
        {"type": "integer", "minimum": 0, "maximum": 64},
        scope="platform",
    ))
    platform_modules: list[dict[str, Any]] = []
    seen_platform_ids: set[str] = set()
    for index in range(platform_count):
        scope = f"platform[{index}]"
        kind = author._enum(
            "platform_kind",
            sorted(PLATFORM_KINDS),
            scope=scope,
        )
        module_id = str(author._ask(
            "platform_module_id",
            {
                "type": "string",
                "pattern": r"^[a-z][a-z0-9_]{1,63}$",
            },
            scope=scope,
        ))
        if module_id in seen_platform_ids:
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_DUPLICATE_PLATFORM_MODULE: {module_id}"
            )
        seen_platform_ids.add(module_id)

        config = author._ask(
            "platform_config",
            platform_config_schema(kind),
            scope=scope,
        )
        if not isinstance(config, Mapping):
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {scope}.platform_config"
            )
        if not coverage_refs:
            raise ValueError(
                "TYPED_PLAN_PLATFORM_COVERAGE_REQUIRED: platform modules require "
                "at least one active canonical concern."
            )
        coverage_count = int(author._ask(
            "platform_coverage_count",
            {
                "type": "integer",
                "minimum": 1,
                "maximum": min(64, len(coverage_refs)),
            },
            scope=scope,
        ))
        covers: list[str] = []
        for coverage_index in range(coverage_count):
            cover = author._enum(
                "platform_coverage_ref",
                list(coverage_refs),
                scope=f"{scope}.coverage[{coverage_index}]",
            )
            if cover not in covers:
                covers.append(cover)
        platform_modules.append({
            "module_id": module_id,
            "kind": kind,
            "config": dict(config),
            "covers": covers,
        })

    platform_modules = validate_platform_modules(platform_modules)

    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest(),
        "functions": functions,
        "initialize": author.body(
            "initialize",
            env={},
            return_type="void",
        ),
        "platform_modules": platform_modules,
    }
    from .typed_plan_ir import validate_typed_plan_ir

    return validate_typed_plan_ir(plan, capabilities=capabilities)


__all__ = ["TypedOperationAuthor", "author_typed_plan_ir"]
