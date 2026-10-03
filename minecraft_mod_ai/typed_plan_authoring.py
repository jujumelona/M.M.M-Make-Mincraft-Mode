from __future__ import annotations

"""Native-tool authoring for typed PlanIR.

The planner emits one scalar typed decision at a time. It never emits Java or an
opaque JSON blob, and a hard call bound prevents recursive authoring from running
without a host-observed termination measure.
"""

import hashlib
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
        context = self._scope_semantic_context(scope)
        units = 0
        for concerns in context.values():
            if not isinstance(concerns, Mapping):
                continue
            for rows in concerns.values():
                if isinstance(rows, Sequence) and not isinstance(
                    rows, (str, bytes, bytearray)
                ):
                    units += max(1, len(rows))
        # This is a fail-closed emergency ceiling derived from actual semantic
        # records, not a normal completion target.  Normal bodies stop on done or
        # an unconditional terminal statement well before it.
        return min(24, max(2, 2 + units * 2))

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
            "scope": scope,
            "field": field,
            "decision_index": self.call_count,
            "policy": (
                "Choose only the requested bounded typed value. Do not emit Java, prose, "
                "markdown, or additional fields. Stop at the compile-ready semantic unit; "
                "the host owns iteration and completion."
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
        if not values:
            raise ValueError(f"TYPED_PLAN_AUTHORING_ENUM_EMPTY: {field}")
        if len(values) == 1:
            return str(values[0])
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
        max_statements: int | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        bindings = dict(env or {})
        limit = (
            self._semantic_statement_budget(scope)
            if max_statements is None
            else max(1, int(max_statements))
        )
        for _ in range(limit):
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
    max_calls: int = 2048,
) -> dict[str, Any]:
    """Author a complete typed PlanIR using bounded native decisions only."""

    import hashlib

    from .typed_event_ir import (
        EVENT_PARAMETERS,
        EVENT_SIGNATURES,
        event_config_schema,
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

    author = TypedOperationAuthor(
        router,
        source_text,
        structured_sections,
        capabilities,
        max_calls=max_calls,
    )
    coverage_refs = _active_concern_refs(structured_sections)

    specs: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    event_bindings: list[dict[str, Any]] = []

    # Event handler structure is host-owned. The planner only chooses the typed
    # event enum and, for command events, the bounded command configuration.
    for entry_point_index, row in enumerate(
        _integration_entry_points(structured_sections)
    ):
        if is_mod_initialize_trigger(row.get("trigger")):
            continue

        event_scope = f"integration.entry_points[{entry_point_index}]"
        event = author._enum(
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

    uncovered_logic = {
        ref
        for ref in coverage_refs
        if ref.startswith((
            "behavior_contract.",
            "algorithm.",
            "failure_and_limits.",
        ))
    }
    logic_index = 0
    while uncovered_logic:
        if len(specs) >= 64:
            raise ValueError(
                "TYPED_PLAN_FUNCTION_LIMIT: logic coverage did not converge."
            )
        scope = f"function[{logic_index}]"
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
        available_covers = tuple(sorted(uncovered_logic))
        coverage_count = int(author._ask(
            "coverage_count",
            {
                "type": "integer",
                "minimum": 1,
                "maximum": min(128, len(available_covers)),
            },
            scope=scope,
        ))
        covers: list[str] = []
        for coverage_index in range(coverage_count):
            cover = author._enum(
                "coverage_ref",
                list(available_covers),
                scope=f"{scope}.coverage[{coverage_index}]",
            )
            if cover not in covers:
                covers.append(cover)
        if not covers:
            raise ValueError(
                "TYPED_PLAN_FUNCTION_COVERAGE_REQUIRED: function made no progress."
            )
        uncovered_logic.difference_update(covers)

        specs.append({
            "id": function_id,
            "parameters": parameters,
            "return_type": return_type,
            "covers": covers,
        })
        logic_index += 1

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

    signature_map = {
        str(spec["id"]): (
            tuple(
                str(parameter["type"])
                for parameter in spec["parameters"]
            ),
            str(spec["return_type"]),
        )
        for spec in specs
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
        if platform_index >= 64:
            raise ValueError(
                "TYPED_PLAN_PLATFORM_MODULE_LIMIT: host coverage did not converge."
            )
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
        kind = author._enum(
            "platform_kind",
            available_kinds,
            scope=scope,
        )
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
        coverable_refs = platform_coverable_refs(
            kind,
            sorted(uncovered),
        )
        if not coverable_refs:
            raise ValueError(
                f"TYPED_PLAN_PLATFORM_COVERAGE_REQUIRED: {kind} has no "
                "uncovered canonical concern to implement."
            )
        covers = list(coverable_refs)
        uncovered.difference_update(covers)
        platform_modules.append({
            "module_id": module_id,
            "kind": kind,
            "config": dict(config),
            "covers": covers,
        })
        platform_index += 1

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
        "event_bindings": event_bindings,
    }
    from .typed_plan_ir import validate_typed_plan_ir

    return validate_typed_plan_ir(plan, capabilities=capabilities)

__all__ = ["TypedOperationAuthor", "author_typed_plan_ir"]
