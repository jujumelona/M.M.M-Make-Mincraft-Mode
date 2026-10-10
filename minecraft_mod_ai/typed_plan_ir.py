from __future__ import annotations

"""Typed, model-independent executable PlanIR validation.

The planner may author only the operations defined here. Production consumes the
validated structure; it never accepts Java/source fragments from the plan.
"""

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

TYPED_PLAN_IR_SCHEMA_VERSION = "mmm/typed-plan-ir-v1"
_SCALAR_TYPES = frozenset({"boolean", "int", "long", "double", "string", "object"})
# Shared producer/validator/authoring whitelist for capability argument metadata.
CAPABILITY_ARGUMENT_CONSTRAINT_FIELDS = frozenset({
    "description", "minimum", "maximum", "minLength", "maxLength", "pattern", "enum",
})
_NAME = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_JAVA_RESERVED = frozenset({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "exports", "extends", "false", "final", "finally", "float", "for", "goto",
    "if", "implements", "import", "instanceof", "int", "interface", "long",
    "module", "native", "new", "non-sealed", "null", "open", "opens", "package",
    "permits", "private", "protected", "provides", "public", "record", "requires",
    "return", "sealed", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "to", "transient", "transitive",
    "true", "try", "uses", "var", "void", "volatile", "when", "while", "with",
    "yield", "_",
})


def _error(message: str) -> ValueError:
    return ValueError(message)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{where}: expected object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(f"{where}: expected array")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise _error(f"{where}: unknown fields {unknown!r}")
    missing = sorted(key for key in required if key not in value)
    if missing:
        raise _error(f"{where}: missing fields {missing!r}")


def _name(value: Any, where: str) -> str:
    text = str(value or "")
    if not _NAME.fullmatch(text) or text in _JAVA_RESERVED:
        raise _error(f"{where}: invalid identifier {text!r}")
    return text


def _symbol(value: Any, where: str) -> str:
    text = str(value or "")
    if not _NAME.fullmatch(text):
        raise _error(f"{where}: invalid symbol {text!r}")
    return text


def _type(value: Any, where: str, *, allow_void: bool = False) -> str:
    text = str(value or "")
    if text in _SCALAR_TYPES or (allow_void and text == "void"):
        return text
    raise _error(f"{where}: unsupported type {text!r}")


def _assignable(expected: str, actual: str) -> bool:
    return expected == actual or expected == "object"


def _require_type(expected: str, actual: str, where: str) -> None:
    if not _assignable(expected, actual):
        raise _error(f"{where}: type mismatch, expected {expected}, got {actual}")


def _literal_type(node: Mapping[str, Any], where: str) -> str:
    _keys(node, {"op", "type", "value"}, {"op", "type", "value"}, where)
    kind = _type(node["type"], where + ".type")
    value = node["value"]
    if kind == "boolean":
        if type(value) is not bool:
            raise _error(f"{where}: type mismatch for boolean literal")
    elif kind == "int":
        if type(value) is not int:
            raise _error(f"{where}: type mismatch for int literal")
        if not -(2**31) <= value <= 2**31 - 1:
            raise _error(f"{where}: int literal out of range")
    elif kind == "long":
        if type(value) is not int:
            raise _error(f"{where}: type mismatch for long literal")
        if not -(2**63) <= value <= 2**63 - 1:
            raise _error(f"{where}: long literal out of range")
    elif kind == "double":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _error(f"{where}: type mismatch for double literal")
        if not math.isfinite(float(value)):
            raise _error(f"{where}: double literal must be finite")
    elif kind == "string":
        if not isinstance(value, str):
            raise _error(f"{where}: type mismatch for string literal")
    elif kind == "object":
        if isinstance(value, (Mapping, list, tuple, set)):
            raise _error(f"{where}: object literal must be a scalar")
    return kind


def _capability_contracts(capabilities: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cap_id, raw in dict(capabilities or {}).items():
        if not isinstance(raw, Mapping):
            raise _error(f"capability {cap_id!r}: contract must be an object")
        _keys(
            raw,
            {
                "owner",
                "method",
                "parameters",
                "parameter_constraints",
                "return_type",
                "gameplay_mutation",
            },
            {"owner", "method", "parameters", "return_type"},
            f"capability {cap_id!r}",
        )
        # The planner uses this host-owned metadata to exclude notifications
        # and reads from gameplay-mutation decisions. It is not a Java method
        # argument, but it is part of the same authoritative capability contract.
        mutation = raw.get("gameplay_mutation", False)
        if type(mutation) is not bool:
            raise _error(
                f"capability {cap_id!r}.gameplay_mutation: expected boolean"
            )
        owner = str(raw["owner"])
        method = str(raw["method"])
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*", owner):
            raise _error(f"capability {cap_id!r}: invalid owner")
        _name(method, f"capability {cap_id!r}.method")
        params = [
            _type(item, f"capability {cap_id!r}.parameters")
            for item in _sequence(
                raw["parameters"],
                f"capability {cap_id!r}.parameters",
            )
        ]
        raw_constraints = raw.get("parameter_constraints")
        if raw_constraints is not None:
            constraints = list(
                _sequence(
                    raw_constraints,
                    f"capability {cap_id!r}.parameter_constraints",
                )
            )
            if len(constraints) != len(params):
                raise _error(
                    f"capability {cap_id!r}: parameter constraint count mismatch"
                )
            for index, constraint in enumerate(constraints):
                where = f"capability {cap_id!r}.parameter_constraints[{index}]"
                if not isinstance(constraint, Mapping):
                    raise _error(f"{where}: constraint must be an object")
                unknown = set(constraint) - CAPABILITY_ARGUMENT_CONSTRAINT_FIELDS
                if unknown:
                    raise _error(f"{where}: unsupported fields {sorted(unknown)!r}")
                argument_type = params[index]
                if argument_type == "object" and constraint:
                    raise _error(f"{where}: object-bound parameter cannot have scalar constraints")
                if any(field in constraint for field in ("minimum", "maximum")):
                    if argument_type not in {"int", "long", "double"}:
                        raise _error(f"{where}: numeric bounds require numeric parameter")
                    for field in ("minimum", "maximum"):
                        if field in constraint and (
                            type(constraint[field]) not in {float, int}
                            or not math.isfinite(constraint[field])
                        ):
                            raise _error(f"{where}.{field}: finite numeric value required")
                    if (
                        "minimum" in constraint and "maximum" in constraint
                        and constraint["minimum"] > constraint["maximum"]
                    ):
                        raise _error(f"{where}: minimum exceeds maximum")
                if any(field in constraint for field in ("minLength", "maxLength", "pattern")):
                    if argument_type != "string":
                        raise _error(f"{where}: string constraints require string parameter")
                    for field in ("minLength", "maxLength"):
                        if field in constraint and (
                            type(constraint[field]) is not int or constraint[field] < 0
                        ):
                            raise _error(f"{where}.{field}: nonnegative integer required")
                    if (
                        "minLength" in constraint and "maxLength" in constraint
                        and constraint["minLength"] > constraint["maxLength"]
                    ):
                        raise _error(f"{where}: minLength exceeds maxLength")
                    if "pattern" in constraint:
                        if not isinstance(constraint["pattern"], str):
                            raise _error(f"{where}.pattern: expected string")
                        try:
                            re.compile(constraint["pattern"])
                        except re.error as exc:
                            raise _error(f"{where}.pattern: invalid regular expression") from exc
                if "enum" in constraint and (
                    not isinstance(constraint["enum"], list) or not constraint["enum"]
                ):
                    raise _error(f"{where}.enum: nonempty list required")
                if "description" in constraint and not isinstance(constraint["description"], str):
                    raise _error(f"{where}.description: expected string")
        return_type = _type(
            raw["return_type"],
            f"capability {cap_id!r}.return_type",
            allow_void=True,
        )
        result[str(cap_id)] = {
            "owner": owner,
            "method": method,
            "parameters": params,
            "return_type": return_type,
            "gameplay_mutation": mutation,
        }
    return result


def validate_typed_host_capability_contracts(
    capabilities: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Fail before any LLM calls if the planner's host capability ABI diverges.

    Planner schema generation and final Typed PlanIR validation must consume
    exactly the same normalized contract, including semantic metadata.
    """
    return _capability_contracts(capabilities)


class _Validator:
    def __init__(self, plan: Mapping[str, Any], capabilities: Mapping[str, Any] | None) -> None:
        self.plan = plan
        self.capabilities = _capability_contracts(capabilities)
        self.signatures: dict[str, tuple[tuple[str, ...], str]] = {}
        self.calls: dict[str, set[str]] = {}

    def expression(self, raw: Any, env: Mapping[str, str], where: str, caller: str) -> str:
        node = _mapping(raw, where)
        op = str(node.get("op") or "")
        if op == "literal":
            return _literal_type(node, where)
        if op == "ref":
            _keys(node, {"op", "name"}, {"op", "name"}, where)
            name = _name(node["name"], where + ".name")
            if name not in env:
                raise _error(f"{where}: unknown reference {name!r}")
            return env[name]
        if op == "unary":
            _keys(node, {"op", "operator", "value"}, {"op", "operator", "value"}, where)
            operator = str(node["operator"])
            value_type = self.expression(node["value"], env, where + ".value", caller)
            if operator == "!":
                _require_type("boolean", value_type, where)
                return "boolean"
            if operator == "-" and value_type in {"int", "long", "double"}:
                return value_type
            raise _error(f"{where}: unsupported unary operator {operator!r}")
        if op == "binary":
            _keys(node, {"op", "operator", "left", "right"},
                  {"op", "operator", "left", "right"}, where)
            operator = str(node["operator"])
            left = self.expression(node["left"], env, where + ".left", caller)
            right = self.expression(node["right"], env, where + ".right", caller)
            if operator in {"&&", "||"}:
                _require_type("boolean", left, where + ".left")
                _require_type("boolean", right, where + ".right")
                return "boolean"
            if operator in {"==", "!="}:
                if not (_assignable(left, right) or _assignable(right, left)):
                    raise _error(f"{where}: type mismatch in equality")
                return "boolean"
            if operator in {"<", "<=", ">", ">="}:
                if left != right or left not in {"int", "long", "double"}:
                    raise _error(f"{where}: type mismatch in comparison")
                return "boolean"
            if operator == "+" and left == right == "string":
                return "string"
            if operator in {"+", "-", "*", "/", "%"}:
                if left != right or left not in {"int", "long", "double"}:
                    raise _error(f"{where}: type mismatch in arithmetic")
                return left
            raise _error(f"{where}: unsupported binary operator {operator!r}")
        if op == "list":
            _keys(node, {"op", "items"}, {"op", "items"}, where)
            items = list(_sequence(node["items"], where + ".items"))
            if not items:
                return "list<object>"
            item_types = [self.expression(item, env, f"{where}.items[{i}]", caller)
                          for i, item in enumerate(items)]
            first = item_types[0]
            if any(item != first for item in item_types[1:]):
                raise _error(f"{where}: list items have incompatible types")
            return f"list<{first}>"
        if op == "map":
            _keys(node, {"op", "entries"}, {"op", "entries"}, where)
            entries = _sequence(node["entries"], where + ".entries")
            for index, raw_entry in enumerate(entries):
                entry = _mapping(raw_entry, f"{where}.entries[{index}]")
                _keys(entry, {"key", "value"}, {"key", "value"}, f"{where}.entries[{index}]")
                key_type = self.expression(entry["key"], env, f"{where}.entries[{index}].key", caller)
                _require_type("string", key_type, f"{where}.entries[{index}].key")
                self.expression(entry["value"], env, f"{where}.entries[{index}].value", caller)
            return "map"
        if op == "call":
            _keys(node, {"op", "function", "args"}, {"op", "function", "args"}, where)
            function = _symbol(node["function"], where + ".function")
            if function not in self.signatures:
                raise _error(f"{where}: unknown function {function!r}")
            args = list(_sequence(node["args"], where + ".args"))
            params, result = self.signatures[function]
            if len(args) != len(params):
                raise _error(f"{where}: function argument count mismatch")
            for index, (arg, expected) in enumerate(zip(args, params)):
                actual = self.expression(arg, env, f"{where}.args[{index}]", caller)
                _require_type(expected, actual, f"{where}.args[{index}]")
            self.calls.setdefault(caller, set()).add(function)
            return result
        if op == "capability":
            _keys(node, {"op", "id", "args"}, {"op", "id", "args"}, where)
            cap_id = str(node["id"])
            contract = self.capabilities.get(cap_id)
            if contract is None:
                raise _error(f"{where}: unknown capability {cap_id!r}")
            args = list(_sequence(node["args"], where + ".args"))
            if len(args) != len(contract["parameters"]):
                raise _error(f"{where}: capability argument count mismatch")
            for index, (arg, expected) in enumerate(zip(args, contract["parameters"])):
                actual = self.expression(arg, env, f"{where}.args[{index}]", caller)
                _require_type(expected, actual, f"{where}.args[{index}]")
            return contract["return_type"]
        if op == "state_get":
            _keys(node, {"op", "key", "type", "context"},
                  {"op", "key", "type", "context"}, where)
            key_type = self.expression(node["key"], env, where + ".key", caller)
            _require_type("string", key_type, where + ".key")
            context_type = self.expression(node["context"], env, where + ".context", caller)
            if context_type != "map":
                raise _error(f"{where}.context: type mismatch, expected map, got {context_type}")
            return _type(node["type"], where + ".type")
        raise _error(f"{where}: unsupported expression op {op!r}")

    def block(self, raw: Any, env: dict[str, str], where: str, caller: str,
              return_type: str, *, initialize: bool = False) -> bool:
        statements = _sequence(raw, where)
        returned = False
        for index, raw_statement in enumerate(statements):
            statement_where = f"{where}[{index}]"
            if returned:
                raise _error(f"{statement_where}: unreachable statement")
            statement = _mapping(raw_statement, statement_where)
            op = str(statement.get("op") or "")
            if op == "let":
                _keys(statement, {"op", "name", "type", "value"},
                      {"op", "name", "type", "value"}, statement_where)
                name = _name(statement["name"], statement_where + ".name")
                if name in env:
                    raise _error(f"{statement_where}: duplicate local {name!r}")
                declared = _type(statement["type"], statement_where + ".type")
                actual = self.expression(statement["value"], env, statement_where + ".value", caller)
                _require_type(declared, actual, statement_where + ".value")
                env[name] = declared
            elif op == "set":
                _keys(statement, {"op", "name", "value"}, {"op", "name", "value"}, statement_where)
                name = _name(statement["name"], statement_where + ".name")
                if name not in env:
                    raise _error(f"{statement_where}: unknown local {name!r}")
                actual = self.expression(statement["value"], env, statement_where + ".value", caller)
                _require_type(env[name], actual, statement_where + ".value")
            elif op == "return":
                _keys(statement, {"op", "value"}, {"op"}, statement_where)
                if initialize:
                    raise _error(f"{statement_where}: return is not allowed in initialize")
                if return_type == "void":
                    if "value" in statement and statement["value"] is not None:
                        raise _error(f"{statement_where}: type mismatch returning value from void")
                else:
                    if "value" not in statement:
                        raise _error(f"{statement_where}: missing return value")
                    actual = self.expression(statement["value"], env, statement_where + ".value", caller)
                    _require_type(return_type, actual, statement_where + ".value")
                returned = True
            elif op == "assert":
                _keys(statement, {"op", "condition", "message"},
                      {"op", "condition", "message"}, statement_where)
                actual = self.expression(statement["condition"], env, statement_where + ".condition", caller)
                _require_type("boolean", actual, statement_where + ".condition")
                if not isinstance(statement["message"], str):
                    raise _error(f"{statement_where}.message: expected string")
            elif op == "if":
                _keys(statement, {"op", "condition", "then", "else"},
                      {"op", "condition", "then", "else"}, statement_where)
                actual = self.expression(statement["condition"], env, statement_where + ".condition", caller)
                _require_type("boolean", actual, statement_where + ".condition")
                then_returns = self.block(statement["then"], dict(env), statement_where + ".then",
                                          caller, return_type, initialize=initialize)
                else_returns = self.block(statement["else"], dict(env), statement_where + ".else",
                                          caller, return_type, initialize=initialize)
                returned = then_returns and else_returns
            elif op == "while":
                _keys(statement, {"op", "condition", "max_iterations", "body"},
                      {"op", "condition", "max_iterations", "body"}, statement_where)
                actual = self.expression(statement["condition"], env, statement_where + ".condition", caller)
                _require_type("boolean", actual, statement_where + ".condition")
                limit = statement["max_iterations"]
                if type(limit) is not int or limit < 1 or limit > 1_000_000:
                    raise _error(f"{statement_where}.max_iterations: invalid bounded loop limit")
                self.block(statement["body"], dict(env), statement_where + ".body",
                           caller, return_type, initialize=initialize)
            elif op == "foreach":
                _keys(statement, {"op", "name", "type", "collection", "body"},
                      {"op", "name", "type", "collection", "body"}, statement_where)
                name = _name(statement["name"], statement_where + ".name")
                item_type = _type(statement["type"], statement_where + ".type")
                collection_type = self.expression(statement["collection"], env,
                                                  statement_where + ".collection", caller)
                if collection_type not in {f"list<{item_type}>", "list<object>"}:
                    raise _error(f"{statement_where}.collection: type mismatch for foreach")
                nested = dict(env)
                if name in nested:
                    raise _error(f"{statement_where}: duplicate local {name!r}")
                nested[name] = item_type
                self.block(statement["body"], nested, statement_where + ".body",
                           caller, return_type, initialize=initialize)
            elif op == "state_set":
                _keys(statement, {"op", "key", "value", "context"},
                      {"op", "key", "value", "context"}, statement_where)
                key_type = self.expression(statement["key"], env, statement_where + ".key", caller)
                _require_type("string", key_type, statement_where + ".key")
                self.expression(statement["value"], env, statement_where + ".value", caller)
                context_type = self.expression(statement["context"], env,
                                               statement_where + ".context", caller)
                if context_type != "map":
                    raise _error(f"{statement_where}.context: type mismatch, expected map")
            elif op == "expr":
                _keys(statement, {"op", "value"}, {"op", "value"}, statement_where)
                self.expression(statement["value"], env, statement_where + ".value", caller)
            else:
                raise _error(f"{statement_where}: unsupported statement op {op!r}")
        return returned

    def run(self) -> dict[str, Any]:
        plan = _mapping(self.plan, "typed_plan_ir")
        _keys(
            plan,
            {
                "schema_version",
                "source_sha256",
                "functions",
                "initialize",
                "platform_modules",
                "event_bindings",
            },
            {"schema_version", "source_sha256", "functions", "initialize"},
            "typed_plan_ir",
        )
        if plan["schema_version"] != TYPED_PLAN_IR_SCHEMA_VERSION:
            raise _error("typed_plan_ir: unsupported schema_version")
        source_sha = str(plan["source_sha256"] or "")
        if not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", source_sha):
            raise _error("typed_plan_ir.source_sha256: invalid sha256")
        functions = list(_sequence(plan["functions"], "typed_plan_ir.functions"))
        for index, raw_function in enumerate(functions):
            function = _mapping(raw_function, f"typed_plan_ir.functions[{index}]")
            _keys(function, {"id", "parameters", "return_type", "body", "covers"},
                  {"id", "parameters", "return_type", "body", "covers"},
                  f"typed_plan_ir.functions[{index}]")
            function_id = _symbol(function["id"], f"typed_plan_ir.functions[{index}].id")
            if function_id in self.signatures:
                raise _error(f"typed_plan_ir.functions[{index}]: duplicate function {function_id!r}")
            params = []
            param_names: set[str] = set()
            for p_index, raw_param in enumerate(_sequence(function["parameters"],
                                                          f"typed_plan_ir.functions[{index}].parameters")):
                param = _mapping(raw_param, f"typed_plan_ir.functions[{index}].parameters[{p_index}]")
                _keys(param, {"name", "type"}, {"name", "type"},
                      f"typed_plan_ir.functions[{index}].parameters[{p_index}]")
                param_name = _name(param["name"], f"typed_plan_ir.functions[{index}].parameters[{p_index}].name")
                if param_name in param_names:
                    raise _error(f"typed_plan_ir.functions[{index}]: duplicate parameter {param_name!r}")
                param_names.add(param_name)
                params.append(_type(param["type"], f"typed_plan_ir.functions[{index}].parameters[{p_index}].type"))
            self.signatures[function_id] = (tuple(params), _type(function["return_type"],
                                                                 f"typed_plan_ir.functions[{index}].return_type",
                                                                 allow_void=True))
        for index, raw_function in enumerate(functions):
            function = _mapping(raw_function, f"typed_plan_ir.functions[{index}]")
            function_id = str(function["id"])
            param_types, return_type = self.signatures[function_id]
            env = {
                str(param["name"]): param_type
                for param, param_type in zip(function["parameters"], param_types)
            }
            covers = _sequence(function["covers"], f"typed_plan_ir.functions[{index}].covers")
            if any(not isinstance(item, str) or not item.strip() for item in covers):
                raise _error(f"typed_plan_ir.functions[{index}].covers: expected non-empty strings")
            definitely_returns = self.block(function["body"], env,
                                              f"typed_plan_ir.functions[{index}].body",
                                              function_id, return_type)
            if return_type != "void" and not definitely_returns:
                raise _error(f"typed_plan_ir.functions[{index}]: non-void function must return")
        self.block(plan["initialize"], {}, "typed_plan_ir.initialize", "<initialize>",
                   "void", initialize=True)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(function: str) -> None:
            if function in visiting:
                raise _error(f"typed_plan_ir: recursive function call involving {function!r}")
            if function in visited:
                return
            visiting.add(function)
            for target in self.calls.get(function, ()):
                visit(target)
            visiting.remove(function)
            visited.add(function)

        for function in self.signatures:
            visit(function)

        normalized = deepcopy(dict(plan))
        if "platform_modules" in plan:
            from .typed_platform_ir import validate_platform_modules

            normalized["platform_modules"] = validate_platform_modules(
                plan.get("platform_modules")
            )
        if "event_bindings" in plan:
            from .typed_event_ir import validate_event_bindings

            normalized["event_bindings"] = validate_event_bindings(
                plan.get("event_bindings"),
                signatures=self.signatures,
            )
        return normalized


def validate_typed_plan_ir(
    plan: Mapping[str, Any], *, capabilities: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate without mutating the caller's structure and return a deep copy."""

    return _Validator(plan, capabilities).run()


def _walk_nodes(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_nodes(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_nodes(item)


def _execution_ast(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "functions": plan.get("functions", ()),
        "initialize": plan.get("initialize", ()),
    }


def typed_plan_uses_state(plan: Mapping[str, Any]) -> bool:
    return any(
        node.get("op") in {"state_get", "state_set"}
        for node in _walk_nodes(_execution_ast(plan))
    )


def typed_plan_reachable_function_ids(
    plan: Mapping[str, Any],
    *,
    runtime_only: bool = False,
) -> tuple[str, ...]:
    """Return functions reachable from runtime event bindings or initialize."""

    functions = {
        str(function.get("id") or ""): function
        for function in plan.get("functions", ())
        if isinstance(function, Mapping) and str(function.get("id") or "")
    }

    def calls(value: Any) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            str(node.get("function") or "")
            for node in _walk_nodes(value)
            if node.get("op") == "call" and str(node.get("function") or "")
        ))

    roots = [
        str(binding.get("function") or "")
        for binding in plan.get("event_bindings", ())
        if isinstance(binding, Mapping) and str(binding.get("function") or "")
    ]
    if not runtime_only:
        roots.extend(calls(plan.get("initialize", ())))

    reachable: list[str] = []
    pending = list(dict.fromkeys(root for root in roots if root in functions))
    seen: set[str] = set()
    while pending:
        function_id = pending.pop(0)
        if function_id in seen:
            continue
        seen.add(function_id)
        reachable.append(function_id)
        for target in calls(functions[function_id].get("body", ())):
            if target in functions and target not in seen:
                pending.append(target)
    return tuple(reachable)


def typed_plan_runtime_mutations(
    plan: Mapping[str, Any],
) -> tuple[str, ...]:
    """Host-proven mutations reachable through real runtime event bindings.

    Merely authoring a function with a state_set is insufficient: uncalled
    functions and initialization-only code are not gameplay implementations.
    Read the mutating host capability registry rather than guessing from names.
    """
    from .typed_host_capabilities import typed_host_capability_contracts

    mutating = {
        identifier for identifier, contract in typed_host_capability_contracts().items()
        if contract.get("gameplay_mutation") is True
    }
    reachable = set(typed_plan_reachable_function_ids(plan, runtime_only=True))
    operations: set[str] = set()
    for function in plan.get("functions", ()):
        if (
            not isinstance(function, Mapping)
            or str(function.get("id") or "") not in reachable
        ):
            continue
        for node in _walk_nodes(function.get("body", ())):
            if node.get("op") == "state_set":
                operations.add("state_set")
            elif (
                node.get("op") == "capability"
                and str(node.get("id") or "") in mutating
            ):
                operations.add("capability:" + str(node["id"]))
    return tuple(sorted(operations))



def typed_plan_mutating_events(
    plan: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Prove which bound runtime events can reach an actual gameplay mutation.

    A non-bootstrap event registration is not proof that it executes a writer:
    the generated logic_dispatch may mutate only for player_join. Evaluate
    host-authored event equality guards using the literal event passed by the
    wrapper. Unknown runtime conditions are explored on both branches.
    """
    from .typed_host_capabilities import typed_host_capability_contracts

    mutating_caps = {
        key for key, contract in typed_host_capability_contracts().items()
        if contract.get("gameplay_mutation") is True
    }
    functions = {
        str(row.get("id") or ""): row
        for row in plan.get("functions", ())
        if isinstance(row, Mapping) and row.get("id")
    }

    def tested_event(condition: Any, event: str) -> bool | None:
        if not isinstance(condition, Mapping) or condition.get("op") != "binary":
            return None
        op = condition.get("operator")
        if op not in {"==", "!="}:
            return None
        left, right = condition.get("left"), condition.get("right")
        for variable, constant in ((left, right), (right, left)):
            if (
                isinstance(variable, Mapping)
                and variable.get("op") == "ref"
                and variable.get("name") == "event"
                and isinstance(constant, Mapping)
                and constant.get("op") == "literal"
                and constant.get("type") == "string"
            ):
                equal = constant.get("value") == event
                return equal if op == "==" else not equal
        return None

    result: dict[str, tuple[str, ...]] = {}
    for binding in plan.get("event_bindings", ()):
        if not isinstance(binding, Mapping):
            continue
        bound_event = str(binding.get("event") or "")
        function_id = str(binding.get("function") or "")
        if not bound_event or function_id not in functions:
            continue
        operations: set[str] = set()

        def visit(node: Any, event: str, stack: frozenset[str]) -> None:
            if isinstance(node, (list, tuple)):
                for child in node:
                    visit(child, event, stack)
                return
            if not isinstance(node, Mapping):
                return
            op = node.get("op")
            if op == "if":
                selected = tested_event(node.get("condition"), event)
                if selected is True:
                    visit(node.get("then", ()), event, stack)
                elif selected is False:
                    visit(node.get("else", ()), event, stack)
                else:
                    visit(node.get("condition"), event, stack)
                    visit(node.get("then", ()), event, stack)
                    visit(node.get("else", ()), event, stack)
                return
            if op == "state_set":
                operations.add("state_set")
            elif op == "capability":
                cap_id = str(node.get("id") or "")
                if cap_id in mutating_caps:
                    operations.add("capability:" + cap_id)
            elif op == "call":
                called = str(node.get("function") or "")
                callee = functions.get(called)
                if callee is not None and called not in stack:
                    callee_event = event
                    params = callee.get("parameters", ())
                    args = node.get("args", ())
                    if (
                        isinstance(params, (list, tuple)) and params
                        and isinstance(params[0], Mapping)
                        and params[0].get("name") == "event"
                        and isinstance(args, (list, tuple)) and args
                        and isinstance(args[0], Mapping)
                        and args[0].get("op") == "literal"
                        and args[0].get("type") == "string"
                    ):
                        callee_event = str(args[0].get("value") or "")
                    visit(callee.get("body", ()), callee_event, stack | {called})
            for child in node.values():
                if op == "call" and child is node.get("function"):
                    continue
                visit(child, event, stack)

        visit(functions[function_id].get("body", ()), bound_event, frozenset({function_id}))
        result[bound_event] = tuple(sorted(set(result.get(bound_event, ())) | operations))
    return result


def typed_plan_capability_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(node.get("id") or "")
        for node in _walk_nodes(_execution_ast(plan))
        if node.get("op") == "capability" and str(node.get("id") or "")
    ))


__all__ = [
    "TYPED_PLAN_IR_SCHEMA_VERSION",
    "typed_plan_capability_ids",
    "typed_plan_reachable_function_ids",
    "typed_plan_runtime_mutations",
    "typed_plan_mutating_events",
    "typed_plan_uses_state",
    "validate_typed_plan_ir",
]
