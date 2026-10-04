from __future__ import annotations

"""Host-paged schema authoring for typed PlanIR.

The planner emits one bounded typed AST node or host-requested decision at a time
through tools=0 schema-constrained JSON. It never emits Java or an opaque full-program
blob. The host owns the finite work graph, semantic coverage, AST depth, fan-out and
termination measure.
"""

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

from .fixed_template_generation import generate_fixed_template_value


_TYPES = ["boolean", "int", "long", "double", "string", "object"]

_HOST_UNRESOLVED = object()


def _decode_int_literal(raw_value: Any, *, scope: str) -> int:
    try:
        text = str(raw_value).strip() if not isinstance(raw_value, int) else raw_value
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TYPED_PLAN_LITERAL_INT_INVALID: {scope} got {raw_value!r}"
        ) from exc
    if not -(2**31) <= value <= 2**31 - 1:
        raise ValueError(
            f"TYPED_PLAN_LITERAL_INT_RANGE: {scope} value {value} out of range"
        )
    return value


def _decode_long_literal(raw_value: Any, *, scope: str) -> int:
    try:
        text = str(raw_value).strip() if not isinstance(raw_value, int) else raw_value
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TYPED_PLAN_LITERAL_LONG_INVALID: {scope} got {raw_value!r}"
        ) from exc
    if not -(2**63) <= value <= 2**63 - 1:
        raise ValueError(
            f"TYPED_PLAN_LITERAL_LONG_RANGE: {scope} value {value} out of range"
        )
    return value


def _decode_double_literal(raw_value: Any, *, scope: str) -> float:
    import math

    try:
        text = str(raw_value).strip() if not isinstance(raw_value, (int, float)) else raw_value
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TYPED_PLAN_LITERAL_DOUBLE_INVALID: {scope} got {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"TYPED_PLAN_LITERAL_DOUBLE_FINITE: {scope} value {value} not finite"
        )
    return value



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

    annotation_keys = {"title", "description", "$comment", "default", "examples"}
    for combinator in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(combinator)
        if (
            isinstance(branches, Sequence)
            and not isinstance(branches, (str, bytes, bytearray))
            and len(branches) == 1
            and set(schema) <= {combinator, *annotation_keys}
            and isinstance(branches[0], Mapping)
        ):
            return _host_resolved_schema_value(branches[0])

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
        budget: Any = None,
    ) -> None:
        self.router = router
        self.source_text = str(source_text)
        self.structured_sections = dict(structured_sections or {})
        self.capabilities = dict(capabilities or {})
        self.max_calls = max(1, int(max_calls))
        self.call_count = 0
        self.budget = budget
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

        if self.budget is not None:
            self.budget.consume(f"typed_plan.{scope}.{field}")
        if self.call_count >= self.max_calls:
            raise ValueError(
                f"TYPED_PLAN_AUTHORING_LIMIT: exceeded {self.max_calls} bounded decisions"
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
        result = generate_fixed_template_value(
            self.router,
            "planner",
            (
                {
                    "role": "system",
                    "content": (
                        "Author executable behavior through the host typed PlanIR only. "
                        "Return exactly one bounded host-requested typed decision."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ),
            response_schema=parameters,
            enable_tools=False,
            description="Emit exactly one host-requested typed PlanIR value.",
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
            # Wide JSON numeric lexical surfaces are not model-owned. The model emits
            # a bounded decimal spelling and the host parses/range-checks it below.
            "int": {
                "type": "string",
                "pattern": r"^-?(?:0|[1-9][0-9]{0,9})$",
                "maxLength": 11,
                "description": "Base-10 signed 32-bit integer spelling.",
            },
            "long": {
                "type": "string",
                "pattern": r"^-?(?:0|[1-9][0-9]{0,18})$",
                "maxLength": 20,
                "description": "Base-10 signed 64-bit integer spelling.",
            },
            "double": {
                "type": "string",
                "pattern": r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$",
                "maxLength": 32,
                "description": "Finite decimal floating-point spelling.",
            },
            "string": {"type": "string", "maxLength": 512},
            "object": {
                "type": "string",
                "maxLength": 256,
                "description": "Compact JSON encoding of the literal value.",
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
            literal_type = str(head["type"])
            raw_value = head["value"]
            if literal_type == "int":
                value = _decode_int_literal(raw_value, scope=scope)
            elif literal_type == "long":
                value = _decode_long_literal(raw_value, scope=scope)
            elif literal_type == "double":
                value = _decode_double_literal(raw_value, scope=scope)
            elif literal_type == "object":
                try:
                    value = json.loads(str(raw_value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"TYPED_PLAN_LITERAL_OBJECT_JSON: {scope}"
                    ) from exc
            else:
                value = raw_value
            return {
                "op": "literal",
                "type": literal_type,
                "value": value,
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
                            "type": "string",
                            "pattern": r"^[1-9][0-9]{0,6}$",
                            "maxLength": 7,
                            "description": "Base-10 integer from 1 through 1000000.",
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
            try:
                max_iterations = int(str(head["max_iterations"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"TYPED_PLAN_WHILE_ITERATION_INVALID: {scope} got {head['max_iterations']!r}"
                ) from exc
            if not 1 <= max_iterations <= 1_000_000:
                raise ValueError("TYPED_PLAN_WHILE_ITERATION_RANGE")
            return {
                "op": "while",
                "condition": expr(
                    scope + ".condition",
                    bindings,
                ),
                "max_iterations": max_iterations,
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


def _extract_state_variable_types(
    structured_sections: Mapping[str, Any] | None,
) -> dict[str, str]:
    from .authored_structured_design import active_concern_records

    records = active_concern_records(structured_sections, "state_model")
    variables = records.get("variables", [])
    type_map: dict[str, str] = {}
    for var in variables:
        if isinstance(var, Mapping):
            name = str(var.get("name") or "").strip()
            if not name:
                continue
            raw_type = str(var.get("type") or "int").strip().lower()
            if raw_type in ("integer", "int", "long"):
                type_map[name] = "int"
            elif raw_type in ("double", "float", "number"):
                type_map[name] = "double"
            elif raw_type in ("boolean", "bool"):
                type_map[name] = "boolean"
            elif raw_type == "string":
                type_map[name] = "string"
            else:
                type_map[name] = "int"
    return type_map


SEMANTIC_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "trigger_event": {
                        "type": "string",
                        "enum": [
                            "player_join",
                            "player_disconnect",
                            "player_respawn",
                            "server_started",
                            "server_stopping",
                            "server_tick",
                            "command",
                            "mod_initialize",
                            "any",
                        ],
                    },
                    "state_key": {"type": "string", "maxLength": 64},
                    "action_kind": {
                        "type": "string",
                        "enum": [
                            "increment_state",
                            "set_state",
                            "call_capability",
                            "assert_condition",
                            "none",
                        ],
                    },
                    "int_value": {
                        "type": "string",
                        "pattern": r"^-?(?:0|[1-9][0-9]{0,9})$",
                        "maxLength": 11,
                    },
                    "capability_id": {"type": "string", "maxLength": 64},
                    "message": {"type": "string", "maxLength": 128},
                },
                "required": [
                    "trigger_event",
                    "state_key",
                    "action_kind",
                    "int_value",
                    "capability_id",
                    "message",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rules"],
    "additionalProperties": False,
}


def lower_semantic_game_dispatch_to_ir(
    raw_rules: Sequence[Mapping[str, Any]],
    state_types: Mapping[str, str],
    capabilities: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    statements: list[dict[str, Any]] = []

    for rule_idx, rule in enumerate(raw_rules):
        if not isinstance(rule, Mapping):
            continue
        kind = str(rule.get("action_kind") or "none")
        if kind == "none":
            continue

        action_statements: list[dict[str, Any]] = []
        if kind == "increment_state":
            key = str(rule.get("state_key") or "counter")
            st_type = state_types.get(key, "int")
            raw_int = rule.get("int_value") or "1"
            delta = _decode_int_literal(raw_int, scope=f"rule[{rule_idx}].increment")
            action_statements.append({
                "op": "state_set",
                "key": {"op": "literal", "type": "string", "value": key},
                "value": {
                    "op": "binary",
                    "operator": "+",
                    "left": {
                        "op": "state_get",
                        "key": {"op": "literal", "type": "string", "value": key},
                        "type": st_type,
                        "context": {"op": "map", "entries": []},
                    },
                    "right": {"op": "literal", "type": st_type, "value": delta},
                },
                "context": {"op": "map", "entries": []},
            })
        elif kind == "set_state":
            key = str(rule.get("state_key") or "counter")
            st_type = state_types.get(key, "int")
            raw_int = rule.get("int_value") or "0"
            val = _decode_int_literal(raw_int, scope=f"rule[{rule_idx}].set")
            action_statements.append({
                "op": "state_set",
                "key": {"op": "literal", "type": "string", "value": key},
                "value": {"op": "literal", "type": st_type, "value": val},
                "context": {"op": "map", "entries": []},
            })
        elif kind == "call_capability":
            cap_id = str(rule.get("capability_id") or "")
            if capabilities and cap_id in capabilities:
                contract = capabilities[cap_id]
                params = contract.get("parameters", [])

                def _default_cap_arg(ptype: str) -> dict[str, Any]:
                    if ptype == "int":
                        return {"op": "literal", "type": "int", "value": 0}
                    if ptype == "long":
                        return {"op": "literal", "type": "long", "value": 0}
                    if ptype == "double":
                        return {"op": "literal", "type": "double", "value": 0.0}
                    if ptype == "boolean":
                        return {"op": "literal", "type": "boolean", "value": False}
                    if ptype == "string":
                        return {"op": "literal", "type": "string", "value": ""}
                    return {"op": "literal", "type": "object", "value": None}

                action_statements.append({
                    "op": "expr",
                    "value": {
                        "op": "capability",
                        "id": cap_id,
                        "args": [_default_cap_arg(p) for p in params],
                    },
                })
        elif kind == "assert_condition":
            msg = str(rule.get("message") or "assertion passed")
            action_statements.append({
                "op": "assert",
                "condition": {"op": "literal", "type": "boolean", "value": True},
                "message": msg,
            })

        if not action_statements:
            continue

        trigger = str(rule.get("trigger_event") or "any")
        if trigger != "any":
            statements.append({
                "op": "if",
                "condition": {
                    "op": "binary",
                    "operator": "==",
                    "left": {"op": "ref", "name": "event"},
                    "right": {"op": "literal", "type": "string", "value": trigger},
                },
                "then": action_statements,
                "else": [],
            })
        else:
            statements.extend(action_statements)

    statements.append({
        "op": "return",
        "value": {"op": "literal", "type": "int", "value": 0},
    })
    return statements


def author_semantic_game_dispatch(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any] | None,
    capabilities: Mapping[str, Any] | None,
    *,
    budget: Any = None,
) -> list[dict[str, Any]]:
    state_types = _extract_state_variable_types(structured_sections)
    try:
        if budget is not None:
            budget.consume("typed.semantic_dispatch")
        payload = {
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "available_states": list(state_types.keys()),
            "available_capabilities": sorted(capabilities.keys()) if capabilities else [],
            "instruction": "Author bounded game event rules and state actions for runtime logic dispatch.",
        }
        raw = generate_fixed_template_value(
            router,
            "planner",
            (
                {
                    "role": "system",
                    "content": (
                        "Author game logic dispatch rules as bounded semantic operations. "
                        "Do not author AST syntax, loops, or Java."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ),
            response_schema=SEMANTIC_DISPATCH_SCHEMA,
            enable_tools=False,
            description="Author bounded semantic game rules for runtime dispatch.",
        )
        if isinstance(raw, Mapping) and "rules" in raw and isinstance(raw["rules"], Sequence):
            return lower_semantic_game_dispatch_to_ir(raw["rules"], state_types, capabilities)
    except Exception:
        pass

    return [{
        "op": "return",
        "value": {"op": "literal", "type": "int", "value": 0},
    }]


def _platform_schema(properties: Mapping[str, Any], *, required: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _content_config_author_schema(kind: str) -> dict[str, Any]:
    common = {
        "display_name_en": {"type": "string", "minLength": 1, "maxLength": 64},
        "display_name_ko": {"type": "string", "minLength": 1, "maxLength": 64},
        "ingredients": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "string",
                "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                "maxLength": 64,
            },
        },
    }
    if kind == "item":
        return _platform_schema(common)
    if kind == "block":
        return _platform_schema({
            **common,
            "hardness": {"type": "number", "minimum": 0, "maximum": 100},
        })
    if kind == "food":
        return _platform_schema({
            **common,
            "hunger": {"type": "integer", "minimum": 0, "maximum": 20},
            "saturation": {"type": "number", "minimum": 0, "maximum": 20},
        })
    if kind in {"weapon", "tool"}:
        return _platform_schema({
            **common,
            "attack_damage": {"type": "integer", "minimum": 0, "maximum": 100},
            "attack_speed": {"type": "number", "minimum": 0, "maximum": 10},
        })
    if kind == "armor":
        return _platform_schema({
            **common,
            "slot": {
                "type": "string",
                "enum": ["helmet", "chestplate", "leggings", "boots"],
            },
        })
    if kind == "machine":
        return _platform_schema({
            **common,
            "input_item": {
                "type": "string",
                "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                "maxLength": 64,
            },
            "output_item": {
                "type": "string",
                "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                "maxLength": 64,
            },
            "output_count": {"type": "integer", "minimum": 1, "maximum": 64},
            "processing_ticks": {"type": "integer", "minimum": 1, "maximum": 72000},
        })
    if kind == "crop":
        return _platform_schema(common)
    if kind == "effect":
        return _platform_schema({
            **common,
            "color": {
                "type": "string",
                "pattern": r"^#[0-9A-Fa-f]{6}$",
                "maxLength": 7,
            },
        })
    if kind == "enchantment":
        return _platform_schema({
            **common,
            "max_level": {"type": "integer", "minimum": 1, "maximum": 10},
        })
    raise ValueError(f"TYPED_PLATFORM_CONTENT_KIND_UNSUPPORTED: {kind!r}")


def _normalize_content_config(kind: str, raw: Any, module_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        raw = {}

    default_name = module_id.replace("_", " ").title()
    name_en = str(raw.get("display_name_en") or default_name).strip()[:128]
    config["display_name_en"] = name_en or default_name
    name_ko = str(raw.get("display_name_ko") or config["display_name_en"]).strip()[:128]
    config["display_name_ko"] = name_ko or config["display_name_en"]

    if "ingredients" in raw and isinstance(raw["ingredients"], Sequence) and not isinstance(raw["ingredients"], (str, bytes, bytearray)):
        valid_ings = [
            str(ing) for ing in raw["ingredients"]
            if isinstance(ing, str) and re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", ing)
        ]
        if valid_ings:
            config["ingredients"] = valid_ings[:64]

    if kind == "block":
        hardness = 1.5
        if "hardness" in raw:
            try:
                val = float(raw["hardness"])
                if math.isfinite(val) and val >= 0:
                    hardness = val
            except (TypeError, ValueError):
                pass
        config["hardness"] = hardness
    elif kind == "food":
        hunger = 4
        if "hunger" in raw and type(raw["hunger"]) is int and raw["hunger"] >= 0:
            hunger = raw["hunger"]
        saturation = 2.0
        if "saturation" in raw:
            try:
                val = float(raw["saturation"])
                if math.isfinite(val) and val >= 0:
                    saturation = val
            except (TypeError, ValueError):
                pass
        config["hunger"] = hunger
        config["saturation"] = saturation
    elif kind in {"weapon", "tool"}:
        damage = 6 if kind == "weapon" else 3
        if "attack_damage" in raw and type(raw["attack_damage"]) is int:
            damage = raw["attack_damage"]
        speed = 1.6 if kind == "weapon" else 1.2
        if "attack_speed" in raw:
            try:
                val = float(raw["attack_speed"])
                if math.isfinite(val):
                    speed = val
            except (TypeError, ValueError):
                pass
        config["attack_damage"] = damage
        config["attack_speed"] = speed
    elif kind == "armor":
        slot = raw.get("slot")
        if slot in {"helmet", "chestplate", "leggings", "boots"}:
            config["slot"] = slot
        else:
            config["slot"] = "chestplate"
    elif kind == "machine":
        for field, default in (("input_item", "minecraft:iron_ingot"), ("output_item", "minecraft:gold_ingot")):
            val = str(raw.get(field) or default)
            config[field] = val if re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", val) else default
        count = 1
        if "output_count" in raw and type(raw["output_count"]) is int and raw["output_count"] >= 1:
            count = raw["output_count"]
        ticks = 100
        if "processing_ticks" in raw and type(raw["processing_ticks"]) is int and raw["processing_ticks"] >= 1:
            ticks = raw["processing_ticks"]
        config["output_count"] = count
        config["processing_ticks"] = ticks
    elif kind == "effect":
        color = str(raw.get("color") or "#336699")
        config["color"] = color if re.fullmatch(r"^#[0-9A-Fa-f]{6}$", color) else "#336699"
    elif kind == "enchantment":
        max_lvl = 1
        if "max_level" in raw and type(raw["max_level"]) is int and raw["max_level"] >= 1:
            max_lvl = raw["max_level"]
        config["max_level"] = max_lvl

    return config


def _normalize_entity_config(raw: Any, module_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raw = {}

    def _float_val(key: str, default: float, positive: bool = True) -> float:
        try:
            val = float(raw[key])
            if math.isfinite(val) and ((val > 0) if positive else (val >= 0)):
                return val
        except (KeyError, TypeError, ValueError):
            pass
        return default

    archetype = str(raw.get("archetype") or "biped")
    if archetype not in {"biped", "quadruped", "flying", "serpentine", "construct"}:
        archetype = "biped"

    behavior = str(raw.get("behavior") or "hostile_melee")
    if behavior not in {"hostile_melee", "neutral_melee", "passive", "npc"}:
        behavior = "hostile_melee"

    spawn_group = str(raw.get("spawn_group") or "monster")
    if spawn_group not in {"monster", "creature", "ambient", "water_creature", "misc"}:
        spawn_group = "monster"

    color = str(raw.get("main_color") or "#FF0000")
    if not re.fullmatch(r"^#[0-9A-Fa-f]{6}$", color):
        color = "#FF0000"

    attack_damage = _float_val("attack_damage", 2.0, positive=False)
    if behavior in {"hostile_melee", "neutral_melee"} and attack_damage <= 0:
        attack_damage = 2.0

    return {
        "max_health": _float_val("max_health", 20.0),
        "attack_damage": attack_damage,
        "movement_speed": _float_val("movement_speed", 0.25),
        "follow_range": _float_val("follow_range", 16.0),
        "archetype": archetype,
        "behavior": behavior,
        "entity_width": _float_val("entity_width", 0.6),
        "entity_height": _float_val("entity_height", 1.8),
        "spawn_group": spawn_group,
        "main_color": color,
    }


def _normalize_quest_config(raw: Any, module_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raw = {}
    obj = str(raw.get("objective") or "manual")
    if obj not in {"kill", "break", "manual"}:
        obj = "manual"
    target = str(raw.get("target") or module_id)
    if obj in {"kill", "break"} and not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", target):
        target = "minecraft:zombie" if obj == "kill" else "minecraft:stone"
    elif obj == "manual":
        target = module_id

    reward_item = str(raw.get("reward_item") or "")
    if reward_item and not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", reward_item):
        reward_item = ""

    return {
        "objective": obj,
        "target": target,
        "required": max(1, int(raw.get("required", 1))),
        "reward_item": reward_item,
        "reward_count": max(1, int(raw.get("reward_count", 1))),
        "reward_currency": max(0.0, float(raw.get("reward_currency", 0.0))),
    }


def _normalize_skill_config(raw: Any, module_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raw = {}
    effect = str(raw.get("effect") or "minecraft:speed")
    if not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", effect):
        effect = "minecraft:speed"
    cfg: dict[str, Any] = {
        "effect": effect,
        "duration_ticks": max(1, int(raw.get("duration_ticks", 100))),
        "amplifier": max(0, min(255, int(raw.get("amplifier", 0)))),
        "cooldown_ticks": max(1, int(raw.get("cooldown_ticks", 100))),
    }
    req_class = str(raw.get("required_class") or "")
    if req_class and re.fullmatch(r"^[a-z][a-z0-9_]{1,63}$", req_class):
        cfg["required_class"] = req_class
    return cfg


def _normalize_display_config(raw: Any, module_id: str) -> dict[str, Any]:
    if isinstance(raw, Mapping) and "display_name" in raw and isinstance(raw["display_name"], str) and raw["display_name"].strip():
        return {"display_name": raw["display_name"].strip()[:128]}
    return {"display_name": module_id.title()}


def _normalize_economy_config(raw: Any) -> dict[str, Any]:
    balance = 0.0
    if isinstance(raw, Mapping) and "initial_balance" in raw:
        try:
            val = float(raw["initial_balance"])
            if math.isfinite(val) and val >= 0:
                balance = val
        except (TypeError, ValueError):
            pass
    return {"initial_balance": balance}


def _author_platform_config(
    author: TypedOperationAuthor,
    kind: str,
    *,
    scope: str,
    structured_sections: Mapping[str, Any] | None,
    module_id: str,
    uncovered: set[str],
) -> dict[str, Any]:
    from .typed_platform_ir import (
        PLATFORM_CONTENT_KINDS,
        PLATFORM_ENTITY_KINDS,
        platform_config_schema,
    )

    if kind == "state_store":
        return _state_store_config_from_structured(
            structured_sections,
            transfer_required=("persistence.transfers" in uncovered),
        )
    if kind in {"network_sync", "resource_policy"}:
        return {}
    if kind == "recipe":
        return {"json": {"type": "minecraft:crafting_shapeless"}}
    if kind == "advancement":
        return {"json": {"display": {"title": module_id}}}
    if kind == "loot":
        return {"json": {"type": "minecraft:empty"}}

    if kind == "shop":
        entry_schema = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": r"^[a-z][a-z0-9_]{1,63}$",
                    "maxLength": 64,
                },
                "item": {
                    "type": "string",
                    "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                    "maxLength": 128,
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                },
                "price": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100000,
                },
            },
            "required": ["id", "item", "price"],
            "additionalProperties": False,
        }
        shop_schema = {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": entry_schema,
                },
            },
            "additionalProperties": False,
        }
        raw = author._ask("platform_config", shop_schema, scope=scope)
        entries: list[dict[str, Any]] = []
        seen_entry_ids: set[str] = set()
        raw_list: list[Any] = []
        if isinstance(raw, Mapping):
            if "entries" in raw and isinstance(raw["entries"], Sequence) and not isinstance(raw["entries"], (str, bytes, bytearray)):
                raw_list = list(raw["entries"])
            elif "id" in raw and "item" in raw:
                raw_list = [raw]
        for item in raw_list:
            if not isinstance(item, Mapping):
                continue
            raw_id = str(item.get("id") or "").strip()
            cleaned_id = re.sub(r"[^a-z0-9_]+", "", raw_id.lower())
            if not cleaned_id or not cleaned_id[0].isalpha():
                cleaned_id = f"entry_{len(entries) + 1}"
            if cleaned_id in seen_entry_ids:
                cleaned_id = f"{cleaned_id}_{len(entries) + 1}"
            seen_entry_ids.add(cleaned_id)
            raw_item = str(item.get("item") or "").strip()
            if not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", raw_item):
                raw_item = "minecraft:iron_ingot"
            count = max(1, int(item.get("count", 1)))
            price = max(0.0, float(item.get("price", 10.0)))
            entries.append({
                "id": cleaned_id,
                "item": raw_item,
                "count": count,
                "price": price,
            })
        if not entries:
            entries = [{
                "id": f"{module_id}_entry_1",
                "item": "minecraft:iron_ingot",
                "count": 1,
                "price": 10.0,
            }]
        return {"entries": entries}

    if kind == "networking":
        action_schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$", "maxLength": 64},
                        "type": {"const": "message"},
                        "message": {"type": "string", "minLength": 1, "maxLength": 128},
                    },
                    "required": ["id", "type", "message"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$", "maxLength": 64},
                        "type": {"const": "grant_item"},
                        "item": {"type": "string", "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", "maxLength": 128},
                        "count": {"type": "integer", "minimum": 1, "maximum": 64},
                    },
                    "required": ["id", "type", "item"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$", "maxLength": 64},
                        "type": {"const": "status_effect"},
                        "effect": {"type": "string", "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", "maxLength": 128},
                        "duration_ticks": {"type": "integer", "minimum": 1, "maximum": 72000},
                        "amplifier": {"type": "integer", "minimum": 0, "maximum": 255},
                    },
                    "required": ["id", "type", "effect"],
                    "additionalProperties": False,
                },
            ],
        }
        net_schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": action_schema,
                },
            },
            "additionalProperties": False,
        }
        raw = author._ask("platform_config", net_schema, scope=scope)
        actions: list[dict[str, Any]] = []
        seen_action_ids: set[str] = set()
        raw_list: list[Any] = []
        if isinstance(raw, Mapping):
            if "actions" in raw and isinstance(raw["actions"], Sequence) and not isinstance(raw["actions"], (str, bytes, bytearray)):
                raw_list = list(raw["actions"])
            elif "id" in raw and "type" in raw:
                raw_list = [raw]
        for item in raw_list:
            if not isinstance(item, Mapping):
                continue
            raw_id = str(item.get("id") or "").strip()
            cleaned_id = re.sub(r"[^a-z0-9_]+", "", raw_id.lower())
            if not cleaned_id or not cleaned_id[0].isalpha():
                cleaned_id = f"action_{len(actions) + 1}"
            if cleaned_id in seen_action_ids:
                cleaned_id = f"{cleaned_id}_{len(actions) + 1}"
            seen_action_ids.add(cleaned_id)
            action_type = str(item.get("type") or "message")
            if action_type not in {"message", "grant_item", "status_effect"}:
                action_type = "message"
            if action_type == "message":
                actions.append({
                    "id": cleaned_id,
                    "type": "message",
                    "message": str(item.get("message") or "action").strip() or "action",
                })
            elif action_type == "grant_item":
                item_res = str(item.get("item") or "minecraft:iron_ingot")
                if not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", item_res):
                    item_res = "minecraft:iron_ingot"
                actions.append({
                    "id": cleaned_id,
                    "type": "grant_item",
                    "item": item_res,
                    "count": max(1, int(item.get("count", 1))),
                })
            elif action_type == "status_effect":
                effect_res = str(item.get("effect") or "minecraft:speed")
                if not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", effect_res):
                    effect_res = "minecraft:speed"
                actions.append({
                    "id": cleaned_id,
                    "type": "status_effect",
                    "effect": effect_res,
                    "duration_ticks": max(1, int(item.get("duration_ticks", 100))),
                    "amplifier": max(0, min(255, int(item.get("amplifier", 0)))),
                })
        if not actions:
            actions = [{
                "id": f"{module_id}_action",
                "type": "message",
                "message": "action received",
            }]
        return {
            "template": "validated_action_channel",
            "actions": actions,
        }

    if kind == "gui":
        gui_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 64},
                "rows": {"type": "integer", "minimum": 1, "maximum": 6},
                "entries": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "slot": {"type": "integer", "minimum": 0, "maximum": 53},
                            "item": {"type": "string", "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", "maxLength": 128},
                            "count": {"type": "integer", "minimum": 1, "maximum": 64},
                        },
                        "required": ["slot", "item"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        }
        raw = author._ask("platform_config", gui_schema, scope=scope)
        title = str(raw.get("title") or "Menu").strip() if isinstance(raw, Mapping) else "Menu"
        rows = max(1, min(6, int(raw.get("rows", 3)))) if isinstance(raw, Mapping) else 3
        entries: list[dict[str, Any]] = []
        seen_slots: set[int] = set()
        raw_list: list[Any] = []
        if isinstance(raw, Mapping):
            if "entries" in raw and isinstance(raw["entries"], Sequence) and not isinstance(raw["entries"], (str, bytes, bytearray)):
                raw_list = list(raw["entries"])
            elif "slot" in raw and "item" in raw:
                raw_list = [raw]
        for item in raw_list:
            if not isinstance(item, Mapping):
                continue
            slot = int(item.get("slot", 0))
            if not 0 <= slot < rows * 9 or slot in seen_slots:
                continue
            seen_slots.add(slot)
            item_res = str(item.get("item") or "minecraft:iron_ingot")
            if not re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", item_res):
                item_res = "minecraft:iron_ingot"
            entries.append({
                "slot": slot,
                "item": item_res,
                "count": max(1, int(item.get("count", 1))),
            })
        return {
            "template": "read_only_menu",
            "title": title or "Menu",
            "rows": rows,
            "entries": entries,
        }

    if kind == "tag":
        tag_schema = {
            "type": "object",
            "properties": {
                "registry": {
                    "type": "string",
                    "enum": ["items", "blocks", "entity_types", "fluids", "functions"],
                },
                "values": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": {"type": "string", "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", "maxLength": 128},
                },
                "replace": {"type": "boolean"},
            },
            "additionalProperties": False,
        }
        raw = author._ask("platform_config", tag_schema, scope=scope)
        registry = "items"
        values = ["minecraft:iron_ingot"]
        replace = False
        if isinstance(raw, Mapping):
            reg = str(raw.get("registry") or "")
            if reg in {"items", "blocks", "entity_types", "fluids", "functions"}:
                registry = reg
            raw_vals = raw.get("values")
            if isinstance(raw_vals, Sequence) and not isinstance(raw_vals, (str, bytes, bytearray)):
                filtered = [
                    str(v)
                    for v in raw_vals
                    if isinstance(v, str) and re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", v)
                ]
                if filtered:
                    values = filtered
            elif isinstance(raw_vals, str) and re.fullmatch(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$", raw_vals):
                values = [raw_vals]
            if "replace" in raw and isinstance(raw["replace"], bool):
                replace = raw["replace"]
        return {
            "registry": registry,
            "values": values,
            "replace": replace,
        }

    if kind == "command":
        cmd_schema = {
            "type": "object",
            "properties": {
                "literal": {"type": "string", "pattern": r"^[a-z0-9_]+$", "minLength": 1, "maxLength": 32},
                "message": {"type": "string", "maxLength": 128},
                "permission_level": {"type": "integer", "minimum": 0, "maximum": 4},
            },
            "additionalProperties": False,
        }
        raw = author._ask("platform_config", cmd_schema, scope=scope)
        literal = module_id
        message = f"Executed {module_id}"
        perm = 0
        if isinstance(raw, Mapping):
            raw_lit = str(raw.get("literal") or "")
            cleaned = re.sub(r"[^a-z0-9_]+", "", raw_lit.lower())
            if cleaned:
                literal = cleaned[:32]
            if "message" in raw and isinstance(raw["message"], str):
                message = raw["message"][:128]
            if "permission_level" in raw and type(raw["permission_level"]) is int:
                perm = max(0, min(4, raw["permission_level"]))
        return {
            "literal": literal,
            "message": message,
            "permission_level": perm,
        }

    if kind in PLATFORM_CONTENT_KINDS:
        content_schema = _content_config_author_schema(kind)
        raw = author._ask("platform_config", content_schema, scope=scope)
        return _normalize_content_config(kind, raw, module_id)

    if kind in PLATFORM_ENTITY_KINDS:
        entity_schema = {**platform_config_schema(kind), "required": []}
        raw = author._ask("platform_config", entity_schema, scope=scope)
        return _normalize_entity_config(raw, module_id)

    if kind == "quest":
        quest_schema = {**platform_config_schema(kind), "required": []}
        raw = author._ask("platform_config", quest_schema, scope=scope)
        return _normalize_quest_config(raw, module_id)

    if kind == "skill":
        skill_schema = {**platform_config_schema(kind), "required": []}
        raw = author._ask("platform_config", skill_schema, scope=scope)
        return _normalize_skill_config(raw, module_id)

    if kind in {"class", "party", "guild"}:
        disp_schema = {**platform_config_schema(kind), "required": []}
        raw = author._ask("platform_config", disp_schema, scope=scope)
        return _normalize_display_config(raw, module_id)

    if kind == "economy":
        econ_schema = {**platform_config_schema(kind), "required": []}
        raw = author._ask("platform_config", econ_schema, scope=scope)
        return _normalize_economy_config(raw)

    raise ValueError(f"TYPED_PLATFORM_KIND_UNSUPPORTED: {kind!r}")


def author_typed_plan_ir(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    *,
    max_calls: int | None = None,
    budget: Any = None,
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
        budget=budget,
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
                trigger_text = str(row.get("trigger") or "").strip()
                cmd_slug = re.sub(r"[^a-z0-9_]+", "", trigger_text.lower())
                literal_name = cmd_slug[:32] if cmd_slug else f"cmd_{entry_point_index + 1}"
                config = {"literal": literal_name, "permission_level": 0}

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
        logic_dispatch_body = author_semantic_game_dispatch(
            router,
            source_text,
            structured_sections,
            capabilities,
            budget=budget,
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
            module_id = f"typed_{kind}_{platform_index + 1}"
        if module_id in seen_platform_ids:
            suffix = 1
            base_id = module_id
            while module_id in seen_platform_ids:
                module_id = f"{base_id}_{suffix}"
                suffix += 1
        seen_platform_ids.add(module_id)

        config = _author_platform_config(
            author,
            kind,
            scope=scope,
            structured_sections=structured_sections,
            module_id=module_id,
            uncovered=uncovered,
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
