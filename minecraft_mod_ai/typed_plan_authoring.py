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
from collections.abc import Iterable, Mapping, Sequence
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


def _state_store_schema_version(
    structured_sections: Mapping[str, Any] | None,
) -> str:
    """Derive the current persisted-state schema identity from host-canonical state."""

    from .authored_structured_design import active_concern_records

    state = active_concern_records(
        structured_sections,
        "state_model",
    )
    variables = tuple(
        dict(row)
        for row in state.get("variables", ())
        if isinstance(row, Mapping)
    )
    payload = json.dumps(
        variables,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "state-" + hashlib.sha256(payload).hexdigest()[:16]


def _state_store_config_from_structured(
    structured_sections: Mapping[str, Any] | None,
    *,
    transfer_required: bool,
) -> dict[str, Any]:
    """Compile persistence policy with host-owned migration topology.

    Worksheet migration records may describe legacy sources and migration
    operations, but they do not own the destination graph. Every accepted
    migration converges on the canonical current state-schema identity so cycles,
    branching targets, and ambiguous sinks cannot be introduced by model output.
    """

    from .authored_structured_design import active_concern_records

    persistence = active_concern_records(
        structured_sections,
        "persistence",
    )
    rows = tuple(persistence.get("migration", ()))
    migrations: list[dict[str, Any]] = []
    schema_version = _state_store_schema_version(structured_sections)
    seen_operations: set[tuple[str, str, str, str, str]] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_RECORD_INVALID: {index}"
            )
        source = str(row.get("source_version") or "").strip()
        operation = str(row.get("operation") or "").strip()
        if not source:
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

        # A row whose source already is the current canonical schema cannot be a
        # migration edge. Ignore it rather than constructing a self-loop.
        if source == schema_version:
            continue

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

        operation_key = (
            source,
            operation,
            str(source_key or ""),
            str(destination_key or ""),
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
        )
        if operation_key in seen_operations:
            continue
        seen_operations.add(operation_key)
        migrations.append({
            "from_version": source,
            "to_version": schema_version,
            "operation": operation,
            "source_key": source_key,
            "destination_key": destination_key,
            "value": value,
        })

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


def assert_model_atomic_decision_schema(
    schema: Mapping[str, Any],
    *,
    field: str = "",
) -> None:
    """Validate that a schema queried from the model is an atomic decision schema.

    Model decision schemas must represent single bounded semantic choices or leaf
    records only. Passing non-atomic schemas (e.g. containing arrays, nested objects,
    unbounded properties, or canonical storage schemas) is strictly prohibited.
    Host logic owns cardinality, loops, repetition, IDs, and collection assembly.
    """
    if not isinstance(schema, Mapping):
        raise TypeError(f"NON_ATOMIC_MODEL_SCHEMA: schema must be a mapping: {field}")

    if field == "platform_config":
        raise ValueError(
            "NON_ATOMIC_MODEL_SCHEMA: 'platform_config' cannot be used as an authoring field name; "
            "decompose into bounded leaf decisions."
        )

    from .typed_platform_ir import PLATFORM_KINDS, platform_config_schema

    for k in PLATFORM_KINDS:
        canonical = platform_config_schema(k)
        if canonical.get("properties") and schema == canonical:
            raise ValueError(
                f"NON_ATOMIC_MODEL_SCHEMA: passing canonical storage schema for {k!r} to model is forbidden; "
                "use small bounded semantic authoring schemas."
            )

    def _check_node(node: Any, path: str, depth: int) -> None:
        if not isinstance(node, Mapping):
            return

        schema_type = node.get("type")
        if schema_type == "array" or (isinstance(schema_type, Sequence) and "array" in schema_type):
            raise ValueError(
                f"NON_ATOMIC_MODEL_SCHEMA: array at {path!r} is forbidden in model decision schemas; "
                "host must own loops, cardinality, and collection assembly."
            )

        for combinator in ("oneOf", "anyOf"):
            if combinator in node:
                branches = node[combinator]
                if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes, bytearray)):
                    raise ValueError(f"NON_ATOMIC_MODEL_SCHEMA: {combinator} must be a sequence at {path!r}")
                for b_idx, branch in enumerate(branches):
                    if not isinstance(branch, Mapping):
                        raise ValueError(f"NON_ATOMIC_MODEL_SCHEMA: branch {b_idx} at {path!r} must be an object")
                    _check_node(branch, f"{path}.{combinator}[{b_idx}]", depth + 1)

        if schema_type == "object" or "properties" in node:
            props = node.get("properties")
            if not isinstance(props, Mapping):
                raise ValueError(
                    f"NON_ATOMIC_MODEL_SCHEMA: object at {path!r} must declare explicit properties."
                )
            if len(props) > 12:
                raise ValueError(
                    f"NON_ATOMIC_MODEL_SCHEMA: object at {path!r} has {len(props)} properties, "
                    "exceeding atomic limit of 12."
                )
            if node.get("additionalProperties", True) is not False:
                raise ValueError(
                    f"NON_ATOMIC_MODEL_SCHEMA: object at {path!r} must set additionalProperties: False."
                )
            for prop_name, prop_schema in props.items():
                if not isinstance(prop_schema, Mapping):
                    continue
                prop_type = prop_schema.get("type")
                if prop_type == "array" or (isinstance(prop_type, Sequence) and "array" in prop_type):
                    raise ValueError(
                        f"NON_ATOMIC_MODEL_SCHEMA: property {prop_name!r} at {path!r} is an array; "
                        "host must own loops, cardinality, and collection assembly."
                    )
                if prop_type == "object" or "properties" in prop_schema:
                    raise ValueError(
                        f"NON_ATOMIC_MODEL_SCHEMA: property {prop_name!r} at {path!r} is a nested object; "
                        "model decisions must be flat leaf records only."
                    )
                _check_node(prop_schema, f"{path}.{prop_name}", depth + 1)

    _check_node(schema, "$", 0)


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

        assert_model_atomic_decision_schema(schema, field=field)

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
                raise ValueError(
                    "TYPED_PLAN_STATE_TYPE_UNSUPPORTED: "
                    f"{name!r} declares {raw_type!r}"
                )
    return type_map


SEMANTIC_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "object"},
        },
    },
    "required": ["rules"],
    "additionalProperties": False,
}


_SEMANTIC_TRIGGER_EVENTS = (
    "player_join",
    "player_disconnect",
    "player_respawn",
    "server_started",
    "server_stopping",
    "server_tick",
    "command",
    "mod_initialize",
    "any",
)


def _semantic_state_value_schema(type_name: str) -> dict[str, Any]:
    if type_name == "int":
        return {
            "type": "integer",
            "minimum": -(2**31),
            "maximum": 2**31 - 1,
        }
    if type_name == "long":
        return {
            "type": "integer",
            "minimum": -(2**63),
            "maximum": 2**63 - 1,
        }
    if type_name == "double":
        return {"type": "number"}
    if type_name == "boolean":
        return {"type": "boolean"}
    if type_name == "string":
        return {"type": "string", "maxLength": 128}
    raise ValueError(
        f"TYPED_PLAN_SEMANTIC_STATE_TYPE_UNSUPPORTED: {type_name!r}"
    )


def semantic_dispatch_schema(
    state_types: Mapping[str, str],
    capabilities: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build semantic actions only from host-known state/capability namespaces."""

    branches: list[dict[str, Any]] = []
    for state_key, state_type in sorted(state_types.items()):
        value_schema = _semantic_state_value_schema(state_type)
        branches.append({
            "type": "object",
            "properties": {
                "trigger_event": {
                    "type": "string",
                    "enum": list(_SEMANTIC_TRIGGER_EVENTS),
                },
                "action_kind": {"const": "set_state"},
                "state_key": {"const": state_key},
                "value": value_schema,
            },
            "required": [
                "trigger_event",
                "action_kind",
                "state_key",
                "value",
            ],
            "additionalProperties": False,
        })
        if state_type in {"int", "long", "double"}:
            branches.append({
                "type": "object",
                "properties": {
                    "trigger_event": {
                        "type": "string",
                        "enum": list(_SEMANTIC_TRIGGER_EVENTS),
                    },
                    "action_kind": {"const": "increment_state"},
                    "state_key": {"const": state_key},
                    "value": value_schema,
                },
                "required": [
                    "trigger_event",
                    "action_kind",
                    "state_key",
                    "value",
                ],
                "additionalProperties": False,
            })

    for capability_id in sorted((capabilities or {}).keys()):
        branches.append({
            "type": "object",
            "properties": {
                "trigger_event": {
                    "type": "string",
                    "enum": list(_SEMANTIC_TRIGGER_EVENTS),
                },
                "action_kind": {"const": "call_capability"},
                "capability_id": {"const": capability_id},
            },
            "required": [
                "trigger_event",
                "action_kind",
                "capability_id",
            ],
            "additionalProperties": False,
        })

    if not branches:
        raise ValueError(
            "TYPED_PLAN_SEMANTIC_ACTION_UNAVAILABLE: active executable logic has "
            "no canonical state or host capability target"
        )

    return {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"oneOf": branches},
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
    """Lower validated semantic actions without inventing identifiers or values."""

    statements: list[dict[str, Any]] = []
    semantic_action_count = 0

    for rule_idx, rule in enumerate(raw_rules):
        if not isinstance(rule, Mapping):
            raise ValueError(
                f"TYPED_PLAN_SEMANTIC_RULE_INVALID: rule[{rule_idx}] must be an object"
            )
        kind = str(rule.get("action_kind") or "").strip()
        action_statements: list[dict[str, Any]] = []

        if kind in {"increment_state", "set_state"}:
            key = str(rule.get("state_key") or "").strip()
            if key not in state_types:
                raise ValueError(
                    f"TYPED_PLAN_SEMANTIC_STATE_UNKNOWN: rule[{rule_idx}] {key!r}"
                )
            st_type = state_types[key]
            raw_value = (
                rule["value"]
                if "value" in rule
                else rule.get("int_value")
            )
            if kind == "increment_state":
                if st_type not in {"int", "long", "double"}:
                    raise ValueError(
                        "TYPED_PLAN_SEMANTIC_INCREMENT_TYPE: "
                        f"{key!r} has non-numeric type {st_type!r}"
                    )
                if st_type == "int":
                    value = _decode_int_literal(
                        raw_value, scope=f"rule[{rule_idx}].increment"
                    )
                elif st_type == "long":
                    value = _decode_long_literal(
                        raw_value, scope=f"rule[{rule_idx}].increment"
                    )
                else:
                    value = _decode_double_literal(
                        raw_value, scope=f"rule[{rule_idx}].increment"
                    )
                value_expr = {
                    "op": "binary",
                    "operator": "+",
                    "left": {
                        "op": "state_get",
                        "key": {"op": "literal", "type": "string", "value": key},
                        "type": st_type,
                        "context": {"op": "map", "entries": []},
                    },
                    "right": {"op": "literal", "type": st_type, "value": value},
                }
            else:
                if st_type == "int":
                    value = _decode_int_literal(
                        raw_value, scope=f"rule[{rule_idx}].set"
                    )
                elif st_type == "long":
                    value = _decode_long_literal(
                        raw_value, scope=f"rule[{rule_idx}].set"
                    )
                elif st_type == "double":
                    value = _decode_double_literal(
                        raw_value, scope=f"rule[{rule_idx}].set"
                    )
                elif st_type == "boolean":
                    if type(raw_value) is not bool:
                        raise ValueError(
                            f"TYPED_PLAN_SEMANTIC_BOOLEAN_INVALID: rule[{rule_idx}]"
                        )
                    value = raw_value
                elif st_type == "string":
                    if not isinstance(raw_value, str):
                        raise ValueError(
                            f"TYPED_PLAN_SEMANTIC_STRING_INVALID: rule[{rule_idx}]"
                        )
                    value = raw_value
                else:
                    raise ValueError(
                        f"TYPED_PLAN_SEMANTIC_STATE_TYPE_UNSUPPORTED: {st_type!r}"
                    )
                value_expr = {
                    "op": "literal",
                    "type": st_type,
                    "value": value,
                }

            action_statements.append({
                "op": "state_set",
                "key": {"op": "literal", "type": "string", "value": key},
                "value": value_expr,
                "context": {"op": "map", "entries": []},
            })

        elif kind == "call_capability":
            cap_id = str(rule.get("capability_id") or "").strip()
            contract = (capabilities or {}).get(cap_id)
            if not isinstance(contract, Mapping):
                raise ValueError(
                    f"TYPED_PLAN_SEMANTIC_CAPABILITY_UNKNOWN: {cap_id!r}"
                )
            params = contract.get("parameters", [])

            def _default_cap_arg(ptype: str, index: int) -> dict[str, Any]:
                if ptype == "object" and index == 0:
                    return {"op": "ref", "name": "primary"}
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
                    "args": [
                        _default_cap_arg(str(ptype), index)
                        for index, ptype in enumerate(params)
                    ],
                },
            })
        else:
            raise ValueError(
                f"TYPED_PLAN_SEMANTIC_ACTION_UNSUPPORTED: {kind!r}"
            )

        semantic_action_count += len(action_statements)
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

    if semantic_action_count == 0:
        raise ValueError(
            "TYPED_PLAN_SEMANTIC_BODY_EMPTY: executable logic cannot claim coverage "
            "without at least one host-bound action"
        )

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
    """Author executable semantic actions and fail closed on any contract error."""

    state_types = _extract_state_variable_types(structured_sections)
    if budget is not None:
        budget.consume("typed.semantic_dispatch")

    payload = {
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "available_states": {
            name: state_types[name]
            for name in sorted(state_types)
        },
        "available_capabilities": sorted(capabilities.keys()) if capabilities else [],
        "instruction": (
            "Choose one or more host-bound runtime actions. Every state_key and "
            "capability_id is closed by the supplied schema; do not invent identifiers."
        ),
    }
    schema = semantic_dispatch_schema(state_types, capabilities)
    raw = generate_fixed_template_value(
        router,
        "planner",
        (
            {
                "role": "system",
                "content": (
                    "Author bounded game logic as semantic host actions. "
                    "Do not author AST syntax, loops, Java, or no-op coverage."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ),
        response_schema=schema,
        enable_tools=False,
        description="Author bounded semantic game rules for runtime dispatch.",
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "TYPED_PLAN_SEMANTIC_RESPONSE_INVALID: expected an object"
        )
    rules = raw.get("rules")
    if not isinstance(rules, Sequence) or isinstance(
        rules, (str, bytes, bytearray)
    ):
        raise ValueError(
            "TYPED_PLAN_SEMANTIC_RESPONSE_INVALID: rules must be an array"
        )
    return lower_semantic_game_dispatch_to_ir(
        rules,
        state_types,
        capabilities,
    )


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


def _require_authored_mapping(raw: Any, scope: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"TYPED_PLATFORM_AUTHORING_RESPONSE_INVALID: {scope} must be an object"
        )
    return raw


def _normalize_content_config(kind: str, raw: Any, module_id: str) -> dict[str, Any]:
    """Apply only deterministic defaults for omitted, already-validated fields."""

    values = _require_authored_mapping(raw, f"{module_id}.{kind}")
    config: dict[str, Any] = {}

    default_name = module_id.replace("_", " ").title()
    config["display_name_en"] = str(
        values.get("display_name_en", default_name)
    ).strip()
    config["display_name_ko"] = str(
        values.get("display_name_ko", config["display_name_en"])
    ).strip()

    if "ingredients" in values:
        ingredients = values["ingredients"]
        if not isinstance(ingredients, Sequence) or isinstance(
            ingredients, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"TYPED_PLATFORM_AUTHORING_RESPONSE_INVALID: {module_id}.ingredients"
            )
        config["ingredients"] = list(ingredients)

    if kind == "block":
        config["hardness"] = float(values.get("hardness", 1.5))
    elif kind == "food":
        config["hunger"] = int(values.get("hunger", 4))
        config["saturation"] = float(values.get("saturation", 2.0))
    elif kind in {"weapon", "tool"}:
        config["attack_damage"] = int(
            values.get("attack_damage", 6 if kind == "weapon" else 3)
        )
        config["attack_speed"] = float(
            values.get("attack_speed", 1.6 if kind == "weapon" else 1.2)
        )
    elif kind == "armor":
        config["slot"] = str(values.get("slot", "chestplate"))
    elif kind == "machine":
        config["input_item"] = str(
            values.get("input_item", "minecraft:iron_ingot")
        )
        config["output_item"] = str(
            values.get("output_item", "minecraft:gold_ingot")
        )
        config["output_count"] = int(values.get("output_count", 1))
        config["processing_ticks"] = int(values.get("processing_ticks", 100))
    elif kind == "effect":
        config["color"] = str(values.get("color", "#336699"))
    elif kind == "enchantment":
        config["max_level"] = int(values.get("max_level", 1))

    return config


def _normalize_entity_config(raw: Any, module_id: str) -> dict[str, Any]:
    """Fill host-owned physical defaults without repairing authored enum/color fields."""

    values = _require_authored_mapping(raw, f"{module_id}.entity")
    behavior = str(values.get("behavior", "hostile_melee"))
    attack_damage = 2.0
    if behavior in {"passive", "npc"}:
        attack_damage = 0.0

    return {
        "max_health": 20.0,
        "attack_damage": attack_damage,
        "movement_speed": 0.25,
        "follow_range": 16.0,
        "archetype": str(values.get("archetype", "biped")),
        "behavior": behavior,
        "entity_width": 0.6,
        "entity_height": 1.8,
        "spawn_group": str(values.get("spawn_group", "monster")),
        "main_color": str(values.get("main_color", "#FF0000")),
    }


def _normalize_quest_config(raw: Any, module_id: str) -> dict[str, Any]:
    values = _require_authored_mapping(raw, f"{module_id}.quest")
    objective = str(values.get("objective", "manual"))
    default_target = (
        "minecraft:zombie"
        if objective == "kill"
        else "minecraft:stone"
        if objective == "break"
        else module_id
    )
    return {
        "objective": objective,
        "target": str(values.get("target", default_target)),
        "required": int(values.get("required", 1)),
        "reward_item": str(values.get("reward_item", "minecraft:air")),
        "reward_count": int(values.get("reward_count", 1)),
        "reward_currency": float(values.get("reward_currency", 0.0)),
    }


def _normalize_skill_config(raw: Any, module_id: str) -> dict[str, Any]:
    values = _require_authored_mapping(raw, f"{module_id}.skill")
    cfg: dict[str, Any] = {
        "effect": str(values.get("effect", "minecraft:speed")),
        "duration_ticks": int(values.get("duration_ticks", 100)),
        "amplifier": int(values.get("amplifier", 0)),
        "cooldown_ticks": int(values.get("cooldown_ticks", 100)),
    }
    if "required_class" in values:
        cfg["required_class"] = str(values["required_class"])
    return cfg


def _normalize_display_config(raw: Any, module_id: str) -> dict[str, Any]:
    values = _require_authored_mapping(raw, f"{module_id}.display")
    return {
        "display_name": str(
            values.get("display_name", module_id.replace("_", " ").title())
        ).strip()
    }


def _normalize_economy_config(raw: Any) -> dict[str, Any]:
    values = _require_authored_mapping(raw, "economy")
    return {
        "initial_balance": float(values.get("initial_balance", 0.0))
    }


_SHOP_ENTRY_SCHEMA = _platform_schema({
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
}, required=("item", "price"))

_NET_ACTION_MESSAGE_SCHEMA = _platform_schema({
    "message": {"type": "string", "minLength": 1, "maxLength": 128},
}, required=("message",))

_NET_ACTION_GRANT_ITEM_SCHEMA = _platform_schema({
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
}, required=("item",))

_NET_ACTION_STATUS_EFFECT_SCHEMA = _platform_schema({
    "effect": {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
    },
    "duration_ticks": {
        "type": "integer",
        "minimum": 1,
        "maximum": 72000,
    },
    "amplifier": {
        "type": "integer",
        "minimum": 0,
        "maximum": 255,
    },
}, required=("effect",))

_GUI_TITLE_SCHEMA = _platform_schema({
    "title": {"type": "string", "minLength": 1, "maxLength": 64},
}, required=("title",))

_GUI_ENTRY_SCHEMA = _platform_schema({
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
}, required=("item",))

_TAG_VALUE_SCHEMA = _platform_schema({
    "value": {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
    },
}, required=("value",))

_COMMAND_PARAMS_AUTHOR_SCHEMA = _platform_schema({
    "literal": {
        "type": "string",
        "pattern": r"^[a-z0-9_]+$",
        "minLength": 1,
        "maxLength": 32,
    },
    "message": {"type": "string", "maxLength": 128},
    "permission_level": {"type": "integer", "minimum": 0, "maximum": 4},
})

_ENTITY_TRAITS_AUTHOR_SCHEMA = _platform_schema({
    "archetype": {
        "type": "string",
        "enum": ["biped", "quadruped", "flying", "serpentine", "construct"],
    },
    "behavior": {
        "type": "string",
        "enum": ["hostile_melee", "neutral_melee", "passive", "npc"],
    },
    "spawn_group": {
        "type": "string",
        "enum": ["monster", "creature", "ambient", "water_creature", "misc"],
    },
    "main_color": {
        "type": "string",
        "pattern": r"^#[0-9A-Fa-f]{6}$",
        "maxLength": 7,
    },
})

_QUEST_PARAMS_AUTHOR_SCHEMA = _platform_schema({
    "objective": {
        "type": "string",
        "enum": ["kill", "break", "manual"],
    },
    "target": {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
    },
    "required": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
    },
    "reward_item": {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
    },
    "reward_count": {
        "type": "integer",
        "minimum": 1,
        "maximum": 64,
    },
    "reward_currency": {
        "type": "number",
        "minimum": 0,
        "maximum": 10000,
    },
})

_SKILL_PARAMS_AUTHOR_SCHEMA = _platform_schema({
    "effect": {
        "type": "string",
        "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
        "maxLength": 128,
    },
    "duration_ticks": {
        "type": "integer",
        "minimum": 1,
        "maximum": 72000,
    },
    "amplifier": {
        "type": "integer",
        "minimum": 0,
        "maximum": 255,
    },
    "cooldown_ticks": {
        "type": "integer",
        "minimum": 1,
        "maximum": 72000,
    },
    "required_class": {
        "type": "string",
        "pattern": r"^[a-z][a-z0-9_]{1,63}$",
        "maxLength": 64,
    },
})

_DISPLAY_PARAMS_AUTHOR_SCHEMA = _platform_schema({
    "display_name": {
        "type": "string",
        "minLength": 1,
        "maxLength": 64,
    },
})

_ECONOMY_PARAMS_AUTHOR_SCHEMA = _platform_schema({
    "initial_balance": {
        "type": "number",
        "minimum": 0,
        "maximum": 1000000,
    },
})


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
        raw = _require_authored_mapping(
            author._ask(
                "shop_entry",
                _SHOP_ENTRY_SCHEMA,
                scope=f"{scope}.entries[0]",
            ),
            f"{scope}.entries[0]",
        )
        return {
            "entries": [
                {
                    "id": f"{module_id}_entry_1",
                    "item": str(raw["item"]),
                    "count": int(raw.get("count", 1)),
                    "price": float(raw["price"]),
                }
            ]
        }

    if kind == "networking":
        action_type = author._enum(
            "action_type",
            ["message", "grant_item", "status_effect"],
            scope=f"{scope}.actions[0]",
        )
        action_id = f"{module_id}_action_1"
        if action_type == "message":
            raw = _require_authored_mapping(
                author._ask(
                    "message_action",
                    _NET_ACTION_MESSAGE_SCHEMA,
                    scope=f"{scope}.actions[0]",
                ),
                f"{scope}.actions[0]",
            )
            action = {
                "id": action_id,
                "type": "message",
                "message": str(raw["message"]),
            }
        elif action_type == "grant_item":
            raw = _require_authored_mapping(
                author._ask(
                    "grant_item_action",
                    _NET_ACTION_GRANT_ITEM_SCHEMA,
                    scope=f"{scope}.actions[0]",
                ),
                f"{scope}.actions[0]",
            )
            action = {
                "id": action_id,
                "type": "grant_item",
                "item": str(raw["item"]),
                "count": int(raw.get("count", 1)),
            }
        else:
            raw = _require_authored_mapping(
                author._ask(
                    "status_effect_action",
                    _NET_ACTION_STATUS_EFFECT_SCHEMA,
                    scope=f"{scope}.actions[0]",
                ),
                f"{scope}.actions[0]",
            )
            action = {
                "id": action_id,
                "type": "status_effect",
                "effect": str(raw["effect"]),
                "duration_ticks": int(raw.get("duration_ticks", 100)),
                "amplifier": int(raw.get("amplifier", 0)),
            }
        return {
            "template": "validated_action_channel",
            "actions": [action],
        }

    if kind == "gui":
        raw_title = _require_authored_mapping(
            author._ask(
                "gui_title",
                _GUI_TITLE_SCHEMA,
                scope=f"{scope}.title",
            ),
            f"{scope}.title",
        )
        raw_entry = _require_authored_mapping(
            author._ask(
                "gui_entry",
                _GUI_ENTRY_SCHEMA,
                scope=f"{scope}.entries[0]",
            ),
            f"{scope}.entries[0]",
        )
        return {
            "template": "read_only_menu",
            "title": str(raw_title["title"]),
            "rows": 3,
            "entries": [
                {
                    "slot": 0,
                    "item": str(raw_entry["item"]),
                    "count": int(raw_entry.get("count", 1)),
                }
            ],
        }

    if kind == "tag":
        registry = author._enum(
            "registry",
            ["items", "blocks", "entity_types", "fluids", "functions"],
            scope=f"{scope}.registry",
        )
        raw_val = _require_authored_mapping(
            author._ask(
                "tag_value",
                _TAG_VALUE_SCHEMA,
                scope=f"{scope}.values[0]",
            ),
            f"{scope}.values[0]",
        )
        return {
            "registry": registry,
            "values": [str(raw_val["value"])],
            "replace": False,
        }

    if kind == "command":
        raw = _require_authored_mapping(
            author._ask(
                "command_params",
                _COMMAND_PARAMS_AUTHOR_SCHEMA,
                scope=scope,
            ),
            scope,
        )
        return {
            "literal": str(raw.get("literal", module_id)),
            "message": str(raw.get("message", f"Executed {module_id}")),
            "permission_level": int(raw.get("permission_level", 0)),
        }

    if kind in PLATFORM_CONTENT_KINDS:
        content_schema = _content_config_author_schema(kind)
        raw = author._ask("content_traits", content_schema, scope=scope)
        return _normalize_content_config(kind, raw, module_id)

    if kind in PLATFORM_ENTITY_KINDS:
        raw = author._ask("entity_traits", _ENTITY_TRAITS_AUTHOR_SCHEMA, scope=scope)
        return _normalize_entity_config(raw, module_id)

    if kind == "quest":
        raw = author._ask("quest_params", _QUEST_PARAMS_AUTHOR_SCHEMA, scope=scope)
        return _normalize_quest_config(raw, module_id)

    if kind == "skill":
        raw = author._ask("skill_params", _SKILL_PARAMS_AUTHOR_SCHEMA, scope=scope)
        return _normalize_skill_config(raw, module_id)

    if kind in {"class", "party", "guild"}:
        raw = author._ask("display_name", _DISPLAY_PARAMS_AUTHOR_SCHEMA, scope=scope)
        return _normalize_display_config(raw, module_id)

    if kind == "economy":
        raw = author._ask("initial_balance", _ECONOMY_PARAMS_AUTHOR_SCHEMA, scope=scope)
        return _normalize_economy_config(raw)

    raise ValueError(f"TYPED_PLATFORM_KIND_UNSUPPORTED: {kind!r}")


def author_typed_plan_ir(
    router: Any,
    source_text: str,
    structured_sections: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    *,
    deterministic_module_kinds: Iterable[str] | None = None,
    allowed_platform_kinds: Iterable[str] | None = None,
    externally_covered_refs: Iterable[str] = (),
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
        PLATFORM_HOST_MODULE_IDS,
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

        entry_point_covers = ["integration.entry_points"]
        if (
            event == "command"
            and "resources_and_ui.interactions" in coverage_refs
        ):
            entry_point_covers.append("resources_and_ui.interactions")
        specs.append({
            "id": function_id,
            "parameters": parameters,
            "return_type": return_type,
            "covers": entry_point_covers,
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
    host_covered_refs = {
        str(cover)
        for function in functions
        for cover in function.get("covers", ())
        if isinstance(cover, str)
    }
    external_coverage = {
        str(ref).strip()
        for ref in externally_covered_refs
        if str(ref).strip()
    }
    unknown_external = external_coverage - set(coverage_refs)
    if unknown_external:
        raise ValueError(
            "TYPED_PLAN_EXTERNAL_COVERAGE_UNKNOWN: "
            + ", ".join(sorted(unknown_external))
        )
    uncovered = {
        ref
        for ref in coverage_refs
        if ref.startswith((
            "authority_and_network.",
            "persistence.",
            "resources_and_ui.",
        ))
        and ref not in host_covered_refs
        and ref not in external_coverage
    }
    # None means the target is not bound yet (AUTO planning may still choose a
    # compatible target later). An empty iterable is different: the bound target
    # explicitly exposes zero deterministic content backends and must therefore
    # not fall back to every syntactically valid platform kind.
    target_deterministic_kinds = (
        frozenset(
            str(kind).strip()
            for kind in deterministic_module_kinds
            if str(kind).strip()
        )
        if deterministic_module_kinds is not None
        else None
    )
    if target_deterministic_kinds is not None:
        from .platform_backend_contract import production_backend_is_supported

        effective_allowed_platform_kinds = set(PLATFORM_HOST_KINDS)
        effective_allowed_platform_kinds.update(
            kind
            for kind in PLATFORM_KINDS
            if kind not in PLATFORM_HOST_KINDS
            and production_backend_is_supported(
                target_deterministic_kinds,
                kind,
            )
        )
    else:
        effective_allowed_platform_kinds = set(PLATFORM_KINDS)

    if allowed_platform_kinds is not None:
        host_selected_kinds = {
            str(kind).strip()
            for kind in allowed_platform_kinds
            if str(kind).strip()
        }
        effective_allowed_platform_kinds.intersection_update(
            host_selected_kinds | set(PLATFORM_HOST_KINDS)
        )

    has_command_binding = any(
        binding.get("event") == "command"
        for binding in event_bindings
    )
    if has_command_binding:
        effective_allowed_platform_kinds.discard("command")

    platform_index = 0
    while uncovered:
        previous_uncovered = len(uncovered)
        scope = f"platform[{platform_index}]"
        available_kinds = [
            kind
            for kind in sorted(effective_allowed_platform_kinds)
            if platform_coverable_refs(kind, sorted(uncovered))
        ]
        if not available_kinds:
            raise ValueError(
                "TYPED_PLAN_PLATFORM_KIND_UNAVAILABLE: active concerns have "
                "no deterministic platform backend; "
                f"uncovered={sorted(uncovered)!r}; "
                f"allowed_platform_kinds={sorted(effective_allowed_platform_kinds)!r}; "
                "target_deterministic_kinds="
                f"{sorted(target_deterministic_kinds) if target_deterministic_kinds is not None else None!r}"
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
            module_id = PLATFORM_HOST_MODULE_IDS[kind]
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
