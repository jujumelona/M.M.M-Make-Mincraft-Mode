from __future__ import annotations

"""Native-tool authoring for typed PlanIR.

The planner emits one bounded typed AST node or host-requested decision at a time.
It never emits Java or an opaque full-program blob.  The host owns the finite work
graph, semantic coverage, AST depth, fan-out and termination measure.
"""

import hashlib
import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any


_TYPES = ["boolean", "int", "long", "double", "string", "object"]

_HOST_UNRESOLVED = object()


def _host_resolved_schema_value(schema: Mapping[str, Any]) -> Any:
    """Return a schema-forced value without spending a model decision."""

    if "const" in schema:
        return deepcopy(schema["const"])

    enum = schema.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes, bytearray))
        and len(enum) == 1
    ):
        return deepcopy(enum[0])

    if schema.get("type") == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if (
            not isinstance(properties, Mapping)
            or not isinstance(required, Sequence)
            or isinstance(required, (str, bytes, bytearray))
            or schema.get("additionalProperties", True) is not False
        ):
            return _HOST_UNRESOLVED

        required_names = [str(name) for name in required]
        if len(required_names) != len(properties) or set(required_names) != set(properties):
            return _HOST_UNRESOLVED

        resolved: dict[str, Any] = {}
        for name in required_names:
            child = properties.get(name)
            if not isinstance(child, Mapping):
                return _HOST_UNRESOLVED
            value = _host_resolved_schema_value(child)
            if value is _HOST_UNRESOLVED:
                return _HOST_UNRESOLVED
            resolved[name] = value
        return resolved

    if schema.get("type") == "array":
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if type(minimum) is not int or minimum < 0 or maximum != minimum:
            return _HOST_UNRESOLVED
        item_schema = schema.get("items")
        if minimum == 0:
            return []
        if not isinstance(item_schema, Mapping):
            return _HOST_UNRESOLVED
        item = _host_resolved_schema_value(item_schema)
        if item is _HOST_UNRESOLVED:
            return _HOST_UNRESOLVED
        return [deepcopy(item) for _ in range(minimum)]

    return _HOST_UNRESOLVED


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


def _state_store_config_from_structured(
    structured_sections: Mapping[str, Any] | None,
    *,
    transfer_required: bool,
) -> dict[str, Any]:
    from .authored_structured_design import active_concern_records

    persistence = active_concern_records(
        structured_sections,
        "persistence",
    )
    rows = tuple(persistence.get("migration", ()))
    migrations: list[dict[str, Any]] = []
    edges: dict[str, str] = {}
    destinations: set[str] = set()
    sources: set[str] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_RECORD_INVALID: {index}"
            )
        source = str(row.get("source_version") or "").strip()
        destination = str(row.get("destination_version") or "").strip()
        operation = str(row.get("operation") or "").strip()
        if not source or not destination:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_VERSION_REQUIRED: {index}"
            )
        if operation not in {
            "preserve",
            "rename_key",
            "delete_key",
            "set_default",
        }:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_OPERATION_INVALID: {operation!r}"
            )
        previous = edges.get(source)
        if previous is not None and previous != destination:
            raise ValueError(
                "TYPED_PLAN_MIGRATION_BRANCHING_FORBIDDEN: "
                f"{source!r}"
            )
        edges[source] = destination
        sources.add(source)
        destinations.add(destination)

        source_key = row.get("source_key")
        destination_key = row.get("destination_key")
        value = row.get("value")
        if operation in {"rename_key", "delete_key"} and not str(
            source_key or ""
        ).strip():
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_SOURCE_KEY_REQUIRED: {index}"
            )
        if operation in {"rename_key", "set_default"} and not str(
            destination_key or ""
        ).strip():
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_DESTINATION_KEY_REQUIRED: {index}"
            )
        migrations.append({
            "from_version": source,
            "to_version": destination,
            "operation": operation,
            "source_key": source_key,
            "destination_key": destination_key,
            "value": value,
        })

    if rows:
        sinks = sorted(destinations - sources)
        if len(sinks) != 1:
            raise ValueError(
                "TYPED_PLAN_MIGRATION_TARGET_AMBIGUOUS: "
                f"{sinks!r}"
            )
        schema_version = sinks[0]
    else:
        schema_version = "1"

    return {
        "namespace": "authored_state",
        "schema_version": schema_version,
        "migrations": migrations,
        "malformed_policy": "backup_and_reset",
        "transfer_on_respawn": bool(transfer_required),
    }


def _integration_entry_points(
    structured_sections: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    from .authored_structured_design import active_concern_records

    active = active_concern_records(structured_sections, "integration")
    rows = active.get("entry_points", ())
    return tuple(
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
    )


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
        self.function_covers: dict[str, tuple[str, ...]] = {}
        self.scope_covers: dict[str, tuple[str, ...]] = {}

    def set_scope_covers(
        self,
        scope: str,
        covers: Sequence[str],
    ) -> None:
        self.scope_covers[str(scope)] = tuple(
            dict.fromkeys(str(item) for item in covers if str(item))
        )

    def _scope_semantic_context(self, scope: str) -> dict[str, Any]:
        from .authored_structured_design import active_concern_records

        selected_covers: tuple[str, ...] = ()
        for prefix in sorted(self.scope_covers, key=len, reverse=True):
            if scope == prefix or scope.startswith(prefix + "."):
                selected_covers = self.scope_covers[prefix]
                break

        selected: dict[str, Any] = {}
        if selected_covers:
            by_section: dict[str, set[str]] = {}
            for ref in selected_covers:
                section, dot, concern = ref.partition(".")
                if dot:
                    by_section.setdefault(section, set()).add(concern)
            for section, concerns in by_section.items():
                active = active_concern_records(
                    self.structured_sections,
                    section,
                )
                rows = {
                    concern: active[concern]
                    for concern in sorted(concerns)
                    if concern in active
                }
                if rows:
                    selected[section] = rows
        else:
            if scope.startswith("integration"):
                sections = ("integration",)
            elif scope.startswith("platform"):
                sections = (
                    "authority_and_network",
                    "persistence",
                    "resources_and_ui",
                )
            elif scope.startswith("initialize"):
                sections = ("integration", "behavior_contract", "algorithm")
            else:
                sections = (
                    "behavior_contract",
                    "algorithm",
                    "failure_and_limits",
                )
            for section in sections:
                active = active_concern_records(
                    self.structured_sections,
                    section,
                )
                if active:
                    selected[section] = active

        # State symbols are the only cross-cutting execution context needed by
        # expressions.  Include just that section rather than retransmitting the
        # complete worksheet on every scalar decision.
        state = active_concern_records(
            self.structured_sections,
            "state_model",
        )
        if state:
            selected.setdefault("state_model", state)
        return selected

    def _semantic_statement_budget(self, scope: str) -> int:
        from .authored_structured_design import active_concern_records

        selected_covers: tuple[str, ...] = ()
        for prefix in sorted(self.scope_covers, key=len, reverse=True):
            if scope == prefix or scope.startswith(prefix + "."):
                selected_covers = self.scope_covers[prefix]
                break

        units = 0
        for ref in selected_covers:
            section, dot, concern = ref.partition(".")
            if not dot:
                continue
            rows = active_concern_records(
                self.structured_sections,
                section,
            ).get(concern, ())
            if isinstance(rows, Sequence) and not isinstance(
                rows, (str, bytes, bytearray)
            ):
                units += max(1, len(rows))
        if units == 0:
            units = 1

        # Normal completion is semantic: every covered concern has been compiled
        # into this body and the model emits done/return.  This small derived bound
        # exists only to reject runaway authoring before transport/token ceilings.
        return min(12, max(2, 1 + units * 2))

    @staticmethod
    def _statement_terminates(statement: Mapping[str, Any]) -> bool:
        if statement.get("op") == "return":
            return True
        if statement.get("op") == "if":
            then = statement.get("then")
            otherwise = statement.get("else")
            return (
                isinstance(then, list)
                and isinstance(otherwise, list)
                and bool(then)
                and bool(otherwise)
                and TypedOperationAuthor._statement_terminates(then[-1])
                and TypedOperationAuthor._statement_terminates(otherwise[-1])
            )
        return False

    def _ask(self, field: str, schema: Mapping[str, Any], *, scope: str) -> Any:
        host_value = _host_resolved_schema_value(schema)
        if host_value is not _HOST_UNRESOLVED:
            return host_value

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
            "source_sha256": hashlib.sha256(
                self.source_text.encode("utf-8")
            ).hexdigest(),
            "semantic_context": self._scope_semantic_context(scope),
            "available_capabilities": sorted(self.capabilities),
            "known_functions": {
                name: list(parameters)
                for name, parameters in self.function_signatures.items()
            },
            "known_function_covers": {
                name: list(self.function_covers.get(name, ()))
                for name in self.function_signatures
            },
            "scope": scope,
            "field": field,
            "decision_index": self.call_count,
            "policy": (
                "Choose only the requested bounded typed value or AST node. Do not emit Java, prose, "
                "markdown, or additional fields. Stop at the compile-ready semantic unit; "
                "the host owns iteration and completion."
            ),
        }
        token_ceiling = {
            "statement_node": 256,
            "expression_node": 512,
            "event_config": 192,
            "function_signature": 768,
            "platform_config": 1024,
        }.get(field, 256)
        result = self.router.generate_tool_decision(
            "planner",
            (
                {
                    "role": "system",
                    "content": (
                        "Author executable behavior through the host typed PlanIR only. "
                        "Every answer is one bounded host-requested typed decision."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ),
            tool_name="emit_typed_plan_value",
            parameters=parameters,
            description="Emit exactly one host-requested typed PlanIR value.",
            output_token_ceiling=token_ceiling,
            force_non_thinking=True,
        )
        if not isinstance(result, Mapping) or set(result) != {"value"}:
            raise ValueError("TYPED_PLAN_AUTHORING_RESPONSE_INVALID")
        return result["value"]

    def _enum(self, field: str, values: list[str], *, scope: str) -> str:
        if not values:
            raise ValueError(f"TYPED_PLAN_AUTHORING_ENUM_EMPTY: {field}")
        if len(values) == 1:
            return str(values[0])
        value = self._ask(field, {"type": "string", "enum": values}, scope=scope)
        if value not in values:
            raise ValueError(f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {field}")
        return str(value)

    def expression(
        self,
        scope: str,
        env: Mapping[str, str] | None = None,
        *,
        _depth: int = 0,
        _node_budget: list[int] | None = None,
    ) -> dict[str, Any]:
        bindings = dict(env or {})
        if _node_budget is None:
            _node_budget = [16]
        if _node_budget[0] <= 0:
            raise ValueError(
                f"TYPED_PLAN_EXPRESSION_BUDGET_EXHAUSTED: {scope}"
            )
        _node_budget[0] -= 1

        def closed(
            properties: Mapping[str, Any],
            required: Sequence[str],
        ) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            }

        branches: list[dict[str, Any]] = []
        literal_values = {
            "boolean": {"type": "boolean"},
            "int": {
                "type": "integer",
                "minimum": -(2**31),
                "maximum": 2**31 - 1,
            },
            "long": {
                "type": "integer",
                "minimum": -(2**63),
                "maximum": 2**63 - 1,
            },
            "double": {"type": "number"},
            "string": {"type": "string", "maxLength": 1024},
            "object": {
                "type": [
                    "string",
                    "number",
                    "integer",
                    "boolean",
                    "null",
                ],
            },
        }
        for kind in _TYPES:
            branches.append(closed(
                {
                    "op": {"const": "literal"},
                    "type": {"const": kind},
                    "value": literal_values[kind],
                },
                ("op", "type", "value"),
            ))

        if bindings:
            branches.append(closed(
                {
                    "op": {"const": "ref"},
                    "name": {
                        "type": "string",
                        "enum": sorted(bindings),
                    },
                },
                ("op", "name"),
            ))

        if _depth < 3:
            branches.extend([
                closed(
                    {
                        "op": {"const": "unary"},
                        "operator": {
                            "type": "string",
                            "enum": ["!", "-"],
                        },
                    },
                    ("op", "operator"),
                ),
                closed(
                    {
                        "op": {"const": "binary"},
                        "operator": {
                            "type": "string",
                            "enum": [
                                "+", "-", "*", "/", "%",
                                "==", "!=", "<", "<=", ">", ">=",
                                "&&", "||",
                            ],
                        },
                    },
                    ("op", "operator"),
                ),
                closed(
                    {
                        "op": {"const": "list"},
                        "count": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                        },
                    },
                    ("op", "count"),
                ),
                closed(
                    {
                        "op": {"const": "map"},
                        "count": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                        },
                    },
                    ("op", "count"),
                ),
                closed(
                    {
                        "op": {"const": "state_get"},
                        "type": {
                            "type": "string",
                            "enum": list(_TYPES),
                        },
                    },
                    ("op", "type"),
                ),
            ])
            if self.function_signatures:
                branches.append(closed(
                    {
                        "op": {"const": "call"},
                        "function": {
                            "type": "string",
                            "enum": sorted(self.function_signatures),
                        },
                    },
                    ("op", "function"),
                ))
            if self.capabilities:
                branches.append(closed(
                    {
                        "op": {"const": "capability"},
                        "id": {
                            "type": "string",
                            "enum": sorted(self.capabilities),
                        },
                    },
                    ("op", "id"),
                ))

        head = self._ask(
            "expression_node",
            {"oneOf": branches},
            scope=scope,
        )
        if not isinstance(head, Mapping):
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {scope}.expression_node"
            )
        op = str(head.get("op") or "")

        if op == "literal":
            return {
                "op": "literal",
                "type": str(head["type"]),
                "value": head["value"],
            }
        if op == "ref":
            return {"op": "ref", "name": str(head["name"])}
        if op == "unary":
            return {
                "op": "unary",
                "operator": str(head["operator"]),
                "value": self.expression(
                    scope + ".unary",
                    bindings,
                    _depth=_depth + 1,
                    _node_budget=_node_budget,
                ),
            }
        if op == "binary":
            return {
                "op": "binary",
                "operator": str(head["operator"]),
                "left": self.expression(
                    scope + ".left",
                    bindings,
                    _depth=_depth + 1,
                    _node_budget=_node_budget,
                ),
                "right": self.expression(
                    scope + ".right",
                    bindings,
                    _depth=_depth + 1,
                    _node_budget=_node_budget,
                ),
            }
        if op == "list":
            count = int(head["count"])
            return {
                "op": "list",
                "items": [
                    self.expression(
                        f"{scope}.item[{index}]",
                        bindings,
                        _depth=_depth + 1,
                        _node_budget=_node_budget,
                    )
                    for index in range(count)
                ],
            }
        if op == "map":
            count = int(head["count"])
            return {
                "op": "map",
                "entries": [
                    {
                        "key": self.expression(
                            f"{scope}.entry[{index}].key",
                            bindings,
                            _depth=_depth + 1,
                            _node_budget=_node_budget,
                        ),
                        "value": self.expression(
                            f"{scope}.entry[{index}].value",
                            bindings,
                            _depth=_depth + 1,
                            _node_budget=_node_budget,
                        ),
                    }
                    for index in range(count)
                ],
            }
        if op == "call":
            function = str(head["function"])
            parameter_types = self.function_signatures[function]
            return {
                "op": "call",
                "function": function,
                "args": [
                    self.expression(
                        f"{scope}.arg[{index}]",
                        bindings,
                        _depth=_depth + 1,
                        _node_budget=_node_budget,
                    )
                    for index in range(len(parameter_types))
                ],
            }
        if op == "capability":
            cap_id = str(head["id"])
            params = self.capabilities.get(cap_id, {}).get(
                "parameters",
                [],
            )
            count = len(params) if isinstance(params, list) else 0
            return {
                "op": "capability",
                "id": cap_id,
                "args": [
                    self.expression(
                        f"{scope}.capability_arg[{index}]",
                        bindings,
                        _depth=_depth + 1,
                        _node_budget=_node_budget,
                    )
                    for index in range(count)
                ],
            }
        if op == "state_get":
            return {
                "op": "state_get",
                "key": self.expression(
                    scope + ".state_key",
                    bindings,
                    _depth=_depth + 1,
                    _node_budget=_node_budget,
                ),
                "type": str(head["type"]),
                "context": self.expression(
                    scope + ".state_context",
                    bindings,
                    _depth=_depth + 1,
                    _node_budget=_node_budget,
                ),
            }
        raise ValueError(
            f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: unsupported expression {op!r}"
        )

    def statement(
        self,
        scope: str,
        *,
        env: Mapping[str, str] | None = None,
        return_type: str | None = None,
        _depth: int = 0,
    ) -> dict[str, Any] | None:
        bindings = dict(env or {})

        def closed(
            properties: Mapping[str, Any],
            required: Sequence[str],
        ) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            }

        branches: list[dict[str, Any]] = [
            closed({"op": {"const": "done"}}, ("op",)),
            closed(
                {
                    "op": {"const": "let"},
                    "name": {
                        "type": "string",
                        "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$",
                    },
                    "type": {
                        "type": "string",
                        "enum": list(_TYPES),
                    },
                },
                ("op", "name", "type"),
            ),
            closed({"op": {"const": "return"}}, ("op",)),
            closed(
                {
                    "op": {"const": "assert"},
                    "message": {
                        "type": "string",
                        "maxLength": 512,
                    },
                },
                ("op", "message"),
            ),
            closed({"op": {"const": "state_set"}}, ("op",)),
            closed({"op": {"const": "expr"}}, ("op",)),
        ]
        if bindings:
            branches.append(closed(
                {
                    "op": {"const": "set"},
                    "name": {
                        "type": "string",
                        "enum": sorted(bindings),
                    },
                },
                ("op", "name"),
            ))
        if _depth < 3:
            branches.extend([
                closed({"op": {"const": "if"}}, ("op",)),
                closed(
                    {
                        "op": {"const": "while"},
                        "max_iterations": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1_000_000,
                        },
                    },
                    ("op", "max_iterations"),
                ),
                closed(
                    {
                        "op": {"const": "foreach"},
                        "name": {
                            "type": "string",
                            "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$",
                        },
                        "type": {
                            "type": "string",
                            "enum": list(_TYPES),
                        },
                    },
                    ("op", "name", "type"),
                ),
            ])

        head = self._ask(
            "statement_node",
            {"oneOf": branches},
            scope=scope,
        )
        if not isinstance(head, Mapping):
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {scope}.statement_node"
            )
        op = str(head.get("op") or "")
        expression_budget = [16]

        def expr(expr_scope: str, expr_env: Mapping[str, str]) -> dict[str, Any]:
            return self.expression(
                expr_scope,
                expr_env,
                _node_budget=expression_budget,
            )

        if op == "done":
            return None
        if op == "let":
            name = str(head["name"])
            if name in bindings:
                raise ValueError(
                    f"TYPED_PLAN_AUTHORING_DUPLICATE_LOCAL: {scope}.{name}"
                )
            return {
                "op": "let",
                "name": name,
                "type": str(head["type"]),
                "value": expr(
                    scope + ".value",
                    bindings,
                ),
            }
        if op == "set":
            return {
                "op": "set",
                "name": str(head["name"]),
                "value": expr(
                    scope + ".value",
                    bindings,
                ),
            }
        if op == "return":
            has_value = (
                bool(head.get("has_value"))
                if return_type is None
                else return_type != "void"
            )
            return (
                {
                    "op": "return",
                    "value": expr(
                        scope + ".value",
                        bindings,
                    ),
                }
                if has_value
                else {"op": "return"}
            )
        if op == "assert":
            return {
                "op": "assert",
                "condition": expr(
                    scope + ".condition",
                    bindings,
                ),
                "message": str(head["message"]),
            }
        if op == "if":
            return {
                "op": "if",
                "condition": expr(
                    scope + ".condition",
                    bindings,
                ),
                "then": self.body(
                    scope + ".then",
                    env=bindings,
                    return_type=return_type,
                    _depth=_depth + 1,
                ),
                "else": self.body(
                    scope + ".else",
                    env=bindings,
                    return_type=return_type,
                    _depth=_depth + 1,
                ),
            }
        if op == "while":
            return {
                "op": "while",
                "condition": expr(
                    scope + ".condition",
                    bindings,
                ),
                "max_iterations": int(head["max_iterations"]),
                "body": self.body(
                    scope + ".body",
                    env=bindings,
                    return_type=return_type,
                    _depth=_depth + 1,
                ),
            }
        if op == "foreach":
            name = str(head["name"])
            if name in bindings:
                raise ValueError(
                    f"TYPED_PLAN_AUTHORING_DUPLICATE_LOCAL: {scope}.{name}"
                )
            item_type = str(head["type"])
            nested = dict(bindings)
            nested[name] = item_type
            return {
                "op": "foreach",
                "name": name,
                "type": item_type,
                "collection": expr(
                    scope + ".collection",
                    bindings,
                ),
                "body": self.body(
                    scope + ".body",
                    env=nested,
                    return_type=return_type,
                    _depth=_depth + 1,
                ),
            }
        if op == "state_set":
            return {
                "op": "state_set",
                "key": expr(
                    scope + ".state_key",
                    bindings,
                ),
                "value": expr(
                    scope + ".state_value",
                    bindings,
                ),
                "context": expr(
                    scope + ".state_context",
                    bindings,
                ),
            }
        if op == "expr":
            return {
                "op": "expr",
                "value": expr(
                    scope + ".value",
                    bindings,
                ),
            }
        raise ValueError(
            f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: unsupported statement {op!r}"
        )

    def body(
        self,
        scope: str,
        *,
        env: Mapping[str, str] | None = None,
        return_type: str | None = None,
        max_statements: int | None = None,
        _depth: int = 0,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        bindings = dict(env or {})
        requested_limit = (
            self._semantic_statement_budget(scope)
            if max_statements is None
            else max(1, int(max_statements))
        )
        nesting_limit = 24 if _depth == 0 else max(2, 16 // (2 ** _depth))
        limit = min(requested_limit, nesting_limit)
        for _ in range(limit):
            statement = self.statement(
                f"{scope}.statement[{len(result)}]",
                env=bindings,
                return_type=return_type,
                _depth=_depth,
            )
            if statement is None:
                return result
            result.append(statement)
            if statement.get("op") == "let":
                bindings[str(statement["name"])] = str(statement["type"])
            if self._statement_terminates(statement):
                return result
        raise ValueError(
            "TYPED_PLAN_SEMANTIC_BUDGET_EXHAUSTED: "
            f"{scope} exceeded {limit} statements before semantic completion"
        )


def author_typed_plan_ir(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    *,
    max_calls: int | None = None,
) -> dict[str, Any]:
    """Author a complete typed PlanIR using bounded native decisions only."""

    import hashlib

    from .typed_event_ir import (
        EVENT_PARAMETERS,
        EVENT_SIGNATURES,
        event_config_schema,
        infer_event_config,
        infer_event_type,
        is_mod_initialize_trigger,
        validate_event_bindings,
    )
    from .typed_platform_ir import (
        PLATFORM_HOST_KINDS,
        PLATFORM_KINDS,
        platform_config_schema,
        platform_coverable_refs,
        validate_platform_modules,
    )

    coverage_refs = _active_concern_refs(structured_sections)
    entry_points = _integration_entry_points(structured_sections)
    semantic_units = len(coverage_refs) + len(entry_points)
    effective_max_calls = (
        max(1, int(max_calls))
        if max_calls is not None
        else max(16, min(128, 8 + semantic_units * 4))
    )
    author = TypedOperationAuthor(
        router,
        source_text,
        structured_sections,
        capabilities,
        max_calls=effective_max_calls,
    )

    specs: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    event_bindings: list[dict[str, Any]] = []

    # Event handler structure is host-owned. The planner only chooses the typed
    # event enum and, for command events, the bounded command configuration.
    for entry_point_index, row in enumerate(entry_points):
        if is_mod_initialize_trigger(row.get("trigger")):
            continue

        event_scope = f"integration.entry_points[{entry_point_index}]"
        event = infer_event_type(row.get("trigger")) or author._enum(
            "event_type",
            sorted(EVENT_SIGNATURES),
            scope=event_scope,
        )
        function_id = f"entryPoint{entry_point_index + 1}_{event}"
        if function_id in known_ids:
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_DUPLICATE_FUNCTION: {function_id}"
            )
        known_ids.add(function_id)

        parameters = [
            {"name": name, "type": type_name}
            for name, type_name in EVENT_PARAMETERS[event]
        ]
        return_type = EVENT_SIGNATURES[event][1]
        config: dict[str, Any] = {}
        if event == "command":
            inferred_config = infer_event_config(
                event,
                row.get("trigger"),
            )
            if inferred_config is not None:
                config = dict(inferred_config)
            else:
                raw_config = author._ask(
                    "event_config",
                    event_config_schema(event),
                    scope=event_scope,
                )
                if not isinstance(raw_config, Mapping):
                    raise ValueError(
                        "TYPED_PLAN_AUTHORING_RESPONSE_INVALID: event_config"
                    )
                config = dict(raw_config)

        specs.append({
            "id": function_id,
            "parameters": parameters,
            "return_type": return_type,
            "covers": ["integration.entry_points"],
        })
        event_bindings.append({
            "event": event,
            "function": function_id,
            "entry_point_index": entry_point_index,
            "config": config,
        })

    logic_refs = sorted(
        ref
        for ref in coverage_refs
        if ref.startswith((
            "behavior_contract.",
            "algorithm.",
            "failure_and_limits.",
        ))
    )
    has_mod_initialize = any(
        is_mod_initialize_trigger(row.get("trigger"))
        for row in entry_points
    )
    if logic_refs and not event_bindings and not has_mod_initialize:
        raise ValueError(
            "TYPED_PLAN_LOGIC_ENTRY_POINT_REQUIRED: executable semantics "
            "have no runtime entry point."
        )

    logic_dispatch_id = "logic_dispatch"
    logic_dispatch_spec: dict[str, Any] | None = None
    logic_dispatch_body: list[dict[str, Any]] = []
    if logic_refs:
        logic_dispatch_spec = {
            "id": logic_dispatch_id,
            "parameters": [
                {"name": "event", "type": "string"},
                {"name": "primary", "type": "object"},
                {"name": "secondary", "type": "object"},
                {"name": "flag", "type": "boolean"},
            ],
            "return_type": "int",
            "covers": list(logic_refs),
        }
        dispatch_scope = "function[logic_dispatch].body"
        body_author = TypedOperationAuthor(
            router,
            source_text,
            structured_sections,
            capabilities,
            max_calls=128,
        )
        # Runtime wrappers are host-owned. The semantic dispatcher may use only
        # typed operations, state and host capabilities; it cannot call wrappers
        # or recursively build another function graph.
        body_author.function_signatures = {}
        body_author.function_covers = {}
        body_author.set_scope_covers(dispatch_scope, logic_refs)
        statement_limit = body_author._semantic_statement_budget(
            dispatch_scope
        )
        body_author.max_calls = max(
            16,
            min(128, statement_limit * 8),
        )
        logic_dispatch_body = body_author.body(
            dispatch_scope,
            env={
                "event": "string",
                "primary": "object",
                "secondary": "object",
                "flag": "boolean",
            },
            return_type="int",
            max_statements=statement_limit,
        )

    def literal(kind: str, value: Any) -> dict[str, Any]:
        return {"op": "literal", "type": kind, "value": value}

    def ref(name: str) -> dict[str, Any]:
        return {"op": "ref", "name": name}

    def dispatch_call(
        event: str,
        *,
        primary: dict[str, Any],
        secondary: dict[str, Any] | None = None,
        flag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if logic_dispatch_spec is None:
            raise ValueError(
                "TYPED_PLAN_INTERNAL: logic dispatcher is unavailable"
            )
        return {
            "op": "call",
            "function": logic_dispatch_id,
            "args": [
                literal("string", event),
                primary,
                secondary or literal("object", None),
                flag or literal("boolean", False),
            ],
        }

    binding_by_function = {
        str(binding["function"]): binding
        for binding in event_bindings
    }
    functions: list[dict[str, Any]] = []
    for spec in specs:
        function_id = str(spec["id"])
        binding = binding_by_function.get(function_id)
        if binding is None:
            raise ValueError(
                f"TYPED_PLAN_INTERNAL: event wrapper {function_id!r} "
                "has no host binding"
            )
        event = str(binding["event"])
        if logic_dispatch_spec is None:
            body = (
                [{
                    "op": "return",
                    "value": literal("int", 1),
                }]
                if event == "command"
                else []
            )
        else:
            if event in {
                "server_started",
                "server_stopping",
                "server_tick",
            }:
                call = dispatch_call(event, primary=ref("server"))
            elif event in {"player_join", "player_disconnect"}:
                call = dispatch_call(event, primary=ref("player"))
            elif event == "player_respawn":
                call = dispatch_call(
                    event,
                    primary=ref("newPlayer"),
                    secondary=ref("oldPlayer"),
                    flag=ref("alive"),
                )
            elif event == "command":
                call = dispatch_call(event, primary=ref("source"))
            else:
                raise ValueError(
                    f"TYPED_PLAN_INTERNAL: unsupported event {event!r}"
                )
            body = (
                [{"op": "return", "value": call}]
                if event == "command"
                else [{"op": "expr", "value": call}]
            )
        functions.append({**spec, "body": body})

    if logic_dispatch_spec is not None:
        functions.append({
            **logic_dispatch_spec,
            "body": logic_dispatch_body,
        })

    signature_map = {
        str(spec["id"]): (
            tuple(
                str(parameter["type"])
                for parameter in spec["parameters"]
            ),
            str(spec["return_type"]),
        )
        for spec in functions
    }
    event_bindings = validate_event_bindings(
        event_bindings,
        signatures=signature_map,
    )

    platform_modules: list[dict[str, Any]] = []
    seen_platform_ids: set[str] = set()
    uncovered = {
        ref
        for ref in coverage_refs
        if ref.startswith((
            "authority_and_network.",
            "persistence.",
            "resources_and_ui.",
        ))
    }
    platform_index = 0
    while uncovered:
        previous_uncovered = len(uncovered)
        scope = f"platform[{platform_index}]"
        available_kinds = [
            kind
            for kind in sorted(PLATFORM_KINDS)
            if platform_coverable_refs(kind, sorted(uncovered))
        ]
        if not available_kinds:
            raise ValueError(
                "TYPED_PLAN_PLATFORM_KIND_UNAVAILABLE: active concerns have "
                "no deterministic platform backend."
            )

        # Dedicated host backends are correctness policy, not a design choice.
        # Prefer them whenever one covers an active concern so the small model
        # never spends a turn choosing between deterministic infrastructure paths.
        dedicated_host_kind: str | None = None
        host_priorities = (
            "state_store",
            "network_sync",
            "resource_policy",
        )
        for candidate in host_priorities:
            if candidate not in available_kinds:
                continue
            if platform_coverable_refs(candidate, sorted(uncovered)):
                dedicated_host_kind = candidate
                break
        kind = (
            dedicated_host_kind
            if dedicated_host_kind is not None
            else author._enum(
                "platform_kind",
                available_kinds,
                scope=scope,
            )
        )
        candidate_covers = platform_coverable_refs(
            kind,
            sorted(uncovered),
        )
        author.set_scope_covers(scope, candidate_covers)
        if kind in PLATFORM_HOST_KINDS:
            module_id = {
                "state_store": "typed_state_store",
                "network_sync": "typed_network_sync",
                "resource_policy": "typed_resource_policy",
            }[kind]
        else:
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

        config_schema = platform_config_schema(kind)
        if kind == "network_sync":
            config = {}
        elif kind == "resource_policy":
            config = {}
        elif kind == "state_store":
            config = _state_store_config_from_structured(
                structured_sections,
                transfer_required=(
                    "persistence.transfers" in uncovered
                ),
            )
        elif (
            config_schema.get("type") == "object"
            and config_schema.get("properties") == {}
            and config_schema.get("additionalProperties") is False
        ):
            config = {}
        else:
            config = author._ask(
                "platform_config",
                config_schema,
                scope=scope,
            )
            if not isinstance(config, Mapping):
                raise ValueError(
                    f"TYPED_PLAN_AUTHORING_RESPONSE_INVALID: {scope}.platform_config"
                )
        coverable_refs = candidate_covers
        if not coverable_refs:
            raise ValueError(
                f"TYPED_PLAN_PLATFORM_COVERAGE_REQUIRED: {kind} has no "
                "uncovered canonical concern to implement."
            )
        covers = list(coverable_refs)
        uncovered.difference_update(covers)
        if len(uncovered) >= previous_uncovered:
            raise ValueError(
                "TYPED_PLAN_PLATFORM_NO_PROGRESS: host coverage must strictly decrease"
            )
        platform_modules.append({
            "module_id": module_id,
            "kind": kind,
            "config": dict(config),
            "covers": covers,
        })
        platform_index += 1

    platform_modules = validate_platform_modules(platform_modules)

    if has_mod_initialize and logic_dispatch_spec is not None:
        initialize_body = [{
            "op": "expr",
            "value": dispatch_call(
                "mod_initialize",
                primary=literal("object", None),
            ),
        }]
    else:
        initialize_body = []

    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest(),
        "functions": functions,
        "initialize": initialize_body,
        "platform_modules": platform_modules,
        "event_bindings": event_bindings,
    }
    from .typed_plan_ir import validate_typed_plan_ir

    return validate_typed_plan_ir(plan, capabilities=capabilities)

__all__ = ["TypedOperationAuthor", "author_typed_plan_ir"]
