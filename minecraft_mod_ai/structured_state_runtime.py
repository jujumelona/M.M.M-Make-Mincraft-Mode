from __future__ import annotations

"""Host compiler for structured state-model records.

Structured state records are compiled directly to Java. No text-model Java generation
or compiler-repair loop is used on this path.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any



_STATE_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_STATE_OPERAND_PATTERN = (
    r'(?:[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'
    r'|"(?:\\.|[^"\\])*"|true|false|null|[A-Za-z_][A-Za-z0-9_]*)'
)
_STATE_ARITHMETIC_PATTERN = (
    _STATE_OPERAND_PATTERN
    + r"(?:[ \t]*[+\-*/%][ \t]*"
    + _STATE_OPERAND_PATTERN
    + r")*"
)
_STATE_COMPARISON_PATTERN = (
    _STATE_ARITHMETIC_PATTERN
    + r"(?:[ \t]*(?:==|!=|>=|<=|>|<)[ \t]*"
    + _STATE_ARITHMETIC_PATTERN
    + r")?"
)
_STATE_TERM_PATTERN = r"(?:![ \t]*)?" + _STATE_COMPARISON_PATTERN
STATE_EXPRESSION_PATTERN = (
    r"^"
    + _STATE_TERM_PATTERN
    + r"(?:[ \t]*(?:&&|\|\|)[ \t]*"
    + _STATE_TERM_PATTERN
    + r")*$"
)


def _state_mutation_pattern(target_pattern: str = r"[A-Za-z_][A-Za-z0-9_]*") -> str:
    expression = (
        _STATE_TERM_PATTERN
        + r"(?:[ \t]*(?:&&|\|\|)[ \t]*"
        + _STATE_TERM_PATTERN
        + r")*"
    )
    assignment = (
        target_pattern
        + r"[ \t]*(?:\+=|-=|\*=|/=|=)[ \t]*"
        + expression
    )
    return r"^" + assignment + r"(?:[ \t]*;[ \t]*" + assignment + r")*$"


STATE_MUTATION_PATTERN = _state_mutation_pattern()

_STATE_EXECUTABLE_FIELDS = {
    "transitions": {
        "guard": STATE_EXPRESSION_PATTERN,
        "mutation": STATE_MUTATION_PATTERN,
    },
    "invariants": {"condition": STATE_EXPRESSION_PATTERN},
    "initialization": {"initial_state": STATE_MUTATION_PATTERN},
    "updates": {"mutation": STATE_MUTATION_PATTERN},
    "cleanup": {"action": STATE_MUTATION_PATTERN},
}


def constrain_state_record_schema(
    concern: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the compiler grammar into the planner schema from one SSOT."""
    result = deepcopy(dict(schema))
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    if concern == "variables" and isinstance(properties.get("name"), dict):
        properties["name"]["pattern"] = _STATE_IDENTIFIER_PATTERN
        properties["name"]["description"] = (
            "Stable ASCII internal state identifier consumed by the host state compiler."
        )
    for field, pattern in _STATE_EXECUTABLE_FIELDS.get(concern, {}).items():
        target = properties.get(field)
        if not isinstance(target, dict):
            continue
        target["pattern"] = pattern
        target["description"] = (
            "Host state-compiler DSL. Use identifiers/literals/operators only; "
            "never natural-language pseudocode or Java method/member syntax."
        )
    return result


def _prior_state_variable_names(
    prior_chunks: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    names: list[str] = []
    for chunk in prior_chunks:
        rows = chunk.get("variables")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "").strip()
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                and name not in names
            ):
                names.append(name)
    return tuple(names)


def constrain_state_chunk_schema(
    schema: Mapping[str, Any],
    prior_chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Narrow mutation targets to variables already authored in earlier chunks."""
    result = deepcopy(dict(schema))
    names = _prior_state_variable_names(prior_chunks)
    if not names:
        return result
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    target_pattern = "(?:" + "|".join(re.escape(name) for name in names) + ")"
    mutation_pattern = _state_mutation_pattern(target_pattern)
    for concern in ("transitions", "initialization", "updates", "cleanup"):
        concern_schema = properties.get(concern)
        if not isinstance(concern_schema, dict):
            continue
        items = concern_schema.get("items")
        item_properties = items.get("properties") if isinstance(items, dict) else None
        if not isinstance(item_properties, dict):
            continue
        field = {
            "transitions": "mutation",
            "initialization": "initial_state",
            "updates": "mutation",
            "cleanup": "action",
        }[concern]
        field_schema = item_properties.get(field)
        if isinstance(field_schema, dict):
            field_schema["pattern"] = mutation_pattern
            field_schema["description"] = (
                "Host mutation DSL. Assignment targets are restricted to already-authored "
                "state variables: " + ", ".join(names) + "."
            )
    return result

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)
      | (?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
      | (?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)
      | (?P<operator>\|\||&&|==|!=|>=|<=|[()!<>+\-*/%])
    )""",
    re.VERBOSE,
)
_ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*(\+=|-=|\*=|/=|=)\s*(.*?)\s*$"
)


def _java_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


class _Expression:
    def __init__(self, text: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        cursor = 0
        source = str(text or "")
        while cursor < len(source):
            match = _TOKEN.match(source, cursor)
            if match is None:
                raise ValueError(
                    "STRUCTURED_STATE_EXPRESSION: unsupported token near "
                    + repr(source[cursor:cursor + 32])
                )
            kind = next(key for key, value in match.groupdict().items() if value is not None)
            self.tokens.append((kind, match.group(kind)))
            cursor = match.end()
        self.index = 0

    def _peek(self, value: str | None = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        return value is None or self.tokens[self.index][1] == value

    def _take(self, value: str | None = None) -> tuple[str, str]:
        if not self._peek(value):
            expected = value or "token"
            found = self.tokens[self.index][1] if self.index < len(self.tokens) else "<end>"
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: expected {expected!r}, found {found!r}"
            )
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self) -> tuple:
        if not self.tokens:
            return ("bool", True)
        node = self._or()
        if self.index != len(self.tokens):
            raise ValueError(
                "STRUCTURED_STATE_EXPRESSION: trailing token "
                + repr(self.tokens[self.index][1])
            )
        return node

    def _or(self) -> tuple:
        node = self._and()
        while self._peek("||"):
            self._take()
            node = ("binary", "||", node, self._and())
        return node

    def _and(self) -> tuple:
        node = self._compare()
        while self._peek("&&"):
            self._take()
            node = ("binary", "&&", node, self._compare())
        return node

    def _compare(self) -> tuple:
        node = self._add()
        if self._peek() and self.tokens[self.index][1] in {
            "==", "!=", ">=", "<=", ">", "<"
        }:
            op = self._take()[1]
            node = ("binary", op, node, self._add())
        return node

    def _add(self) -> tuple:
        node = self._multiply()
        while self._peek() and self.tokens[self.index][1] in {"+", "-"}:
            op = self._take()[1]
            node = ("binary", op, node, self._multiply())
        return node

    def _multiply(self) -> tuple:
        node = self._unary()
        while self._peek() and self.tokens[self.index][1] in {"*", "/", "%"}:
            op = self._take()[1]
            node = ("binary", op, node, self._unary())
        return node

    def _unary(self) -> tuple:
        if self._peek("!"):
            self._take()
            return ("unary", "!", self._unary())
        if self._peek("-"):
            self._take()
            return ("binary", "-", ("number", "0"), self._unary())
        return self._primary()

    def _primary(self) -> tuple:
        if self._peek("("):
            self._take()
            node = self._or()
            self._take(")")
            return node
        kind, value = self._take()
        if kind == "number":
            return ("number", value)
        if kind == "string":
            if value.startswith("'"):
                value = json.dumps(bytes(value[1:-1], "utf-8").decode("unicode_escape"))
            return ("string", json.loads(value))
        if kind == "identifier":
            lowered = value.casefold()
            if lowered == "true":
                return ("bool", True)
            if lowered == "false":
                return ("bool", False)
            if lowered == "null":
                return ("null",)
            return ("identifier", value)
        raise ValueError(f"STRUCTURED_STATE_EXPRESSION: invalid primary {value!r}")


def _value(node: tuple, context: str = "context") -> str:
    kind = node[0]
    if kind == "number":
        return f"Double.valueOf({_java_string(node[1])})"
    if kind == "string":
        return _java_string(node[1])
    if kind == "bool":
        return "Boolean.TRUE" if node[1] else "Boolean.FALSE"
    if kind == "null":
        return "null"
    if kind == "identifier":
        return f"$mmmRead({_java_string(node[1])}, {context})"
    if kind == "unary":
        return f"Boolean.valueOf({_condition(node, context)})"
    if kind == "binary":
        op = node[1]
        if op in {"+", "-", "*", "/", "%"}:
            return (
                f"$mmmArithmetic({_java_string(op)}, "
                f"{_value(node[2], context)}, {_value(node[3], context)})"
            )
        return f"Boolean.valueOf({_condition(node, context)})"
    raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unknown node {kind!r}")


def _condition(node: tuple, context: str = "context") -> str:
    kind = node[0]
    if kind == "unary" and node[1] == "!":
        return f"(!$mmmTruthy({_value(node[2], context)}))"
    if kind == "binary":
        op = node[1]
        if op in {"&&", "||"}:
            left = _condition(node[2], context)
            right = _condition(node[3], context)
            java_op = "&&" if op == "&&" else "||"
            return f"(({left}) {java_op} ({right}))"
        if op == "==":
            return f"$mmmEquals({_value(node[2], context)}, {_value(node[3], context)})"
        if op == "!=":
            return f"(!$mmmEquals({_value(node[2], context)}, {_value(node[3], context)}))"
        if op in {">=", "<=", ">", "<"}:
            return (
                f"($mmmCompare({_value(node[2], context)}, "
                f"{_value(node[3], context)}) {op} 0)"
            )
    return f"$mmmTruthy({_value(node, context)})"


def _compile_condition(text: str, context: str = "context") -> str:
    raw = str(text or "").strip()
    if not raw:
        return "true"
    return _condition(_Expression(raw).parse(), context)


def _compile_mutation(
    script: str,
    *,
    declared: set[str],
    context: str = "context",
) -> str:
    text = str(script or "").strip()
    if text.casefold() in {"noop", "no-op", "no_op"}:
        return ""

    rows: list[str] = []
    for raw in text.split(";"):
        statement = raw.strip()
        if not statement:
            continue
        match = _ASSIGNMENT.fullmatch(statement)
        if match is None:
            raise ValueError(
                "STRUCTURED_STATE_MUTATION: expected assignment, got "
                + repr(statement)
            )
        name, operator, expression = match.groups()
        if name not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {name!r}"
            )
        right = _value(_Expression(expression).parse(), context)
        if operator == "=":
            rows.append(f"setState({_java_string(name)}, {right});")
        else:
            arithmetic = operator[0]
            rows.append(
                f"setState({_java_string(name)}, "
                f"$mmmArithmetic({_java_string(arithmetic)}, "
                f"$mmmRead({_java_string(name)}, {context}), {right}));"
            )
    return " ".join(rows)


def validate_structured_state_section(section: Mapping[str, Any]) -> None:
    """Fail closed on canonical state semantics before any production code is generated."""

    raw_specification = section.get("specification")
    specification = (
        raw_specification if isinstance(raw_specification, Mapping) else section
    )
    variables = specification.get("variables", [])
    declared: set[str] = set()
    if isinstance(variables, Sequence) and not isinstance(
        variables, (str, bytes, bytearray)
    ):
        for record in variables:
            if not isinstance(record, Mapping):
                continue
            name = str(record.get("name") or "").strip()
            if re.fullmatch(_STATE_IDENTIFIER_PATTERN, name) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_NAME: {name!r} is not a stable identifier"
                )
            if name in declared:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DUPLICATE: {name!r}"
                )
            declared.add(name)

    for record in specification.get("transitions", []) or []:
        if not isinstance(record, Mapping):
            continue
        _Expression(str(record.get("guard") or "")).parse()
        _compile_mutation(
            str(record.get("mutation") or ""),
            declared=declared,
        )
    for record in specification.get("invariants", []) or []:
        if isinstance(record, Mapping):
            _Expression(str(record.get("condition") or "")).parse()
    for concern, field in (
        ("initialization", "initial_state"),
        ("updates", "mutation"),
        ("cleanup", "action"),
    ):
        for record in specification.get(concern, []) or []:
            if not isinstance(record, Mapping):
                continue
            _compile_mutation(
                str(record.get(field) or ""),
                declared=declared,
            )


def _obligations(task: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    raw = task.get("implementation_obligations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return {}
    result: dict[str, list[dict[str, str]]] = {}
    for item in raw:
        try:
            outer = json.loads(str(item))
            instruction = json.loads(str(outer.get("instruction") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if "structured_records" not in outer:
            continue
        name = str(instruction.get("concern") or "").strip()
        records = outer.get("structured_records")
        if not name or not isinstance(records, list):
            continue
        result[name] = [
            {str(key): str(value) for key, value in record.items()}
            for record in records
            if isinstance(record, Mapping)
        ]
    return result


def has_complete_structured_state(
    task: Mapping[str, Any],
    concerns: Sequence[Mapping[str, Any]],
) -> bool:
    available = _obligations(task)
    return bool(concerns) and all(
        str(item.get("concern") or "") in available
        for item in concerns
    )


def _default_value(record: Mapping[str, str]) -> str:
    type_name = str(record.get("type") or "").strip().casefold()
    value = str(record.get("default") or "").strip()
    lowered = value.casefold()
    if type_name in {"bool", "boolean"} and lowered in {"true", "false"}:
        return "Boolean.TRUE" if lowered == "true" else "Boolean.FALSE"
    if type_name in {"byte", "short", "int", "integer", "long"} and re.fullmatch(
        r"[-+]?\d+", value
    ):
        return f"Long.valueOf({_java_string(value)})"
    if type_name in {"float", "double"} and re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value
    ):
        return f"Double.valueOf({_java_string(value)})"
    if lowered in {"[]", "empty_list"} or type_name in {"list", "collection"}:
        return "new java.util.ArrayList<>()"
    if lowered in {"{}", "empty_map"} or type_name == "map":
        return "new java.util.LinkedHashMap<>()"
    if lowered == "empty_set" or type_name in {"set", "enumset"}:
        return "new java.util.LinkedHashSet<>()"
    if lowered == "null":
        return "null"
    return _java_string(value)


_COMMON = r"""
private static final java.util.Map<String, Object> $mmmState =
        new java.util.concurrent.ConcurrentHashMap<>();

@FunctionalInterface
private interface $mmmGuard {
    boolean test(java.util.Map<String, Object> context);
}

@FunctionalInterface
private interface $mmmAction {
    void apply(java.util.Map<String, Object> context);
}

private record $mmmTransition(
        String fromState,
        String trigger,
        $mmmGuard guard,
        $mmmAction action,
        String toState
) {}

private record $mmmInvariant($mmmGuard guard, String enforcement) {}
private record $mmmTriggeredAction(String trigger, $mmmAction action, String owner) {}
private record $mmmCleanupAction(
        String event,
        $mmmAction action,
        String retainedState
) {}

private static final java.util.List<$mmmTransition> $mmmTransitions =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmInvariant> $mmmInvariants =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmTriggeredAction> $mmmInitializers =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmTriggeredAction> $mmmUpdates =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<$mmmCleanupAction> $mmmCleanup =
        new java.util.concurrent.CopyOnWriteArrayList<>();
private static final java.util.List<String[]> $mmmConcurrency =
        new java.util.concurrent.CopyOnWriteArrayList<>();

public static synchronized Object getState(String name) {
    return $mmmState.get(name);
}

public static synchronized void setState(String name, Object value) {
    if (name == null || name.isBlank()) {
        return;
    }
    if (value == null) {
        $mmmState.remove(name);
    } else {
        $mmmState.put(name, value);
    }
}

private static Object $mmmRead(
        String name,
        java.util.Map<String, Object> context
) {
    if (context != null && context.containsKey(name)) {
        return context.get(name);
    }
    return $mmmState.get(name);
}

private static double $mmmNumber(Object value) {
    if (value instanceof Number number) {
        return number.doubleValue();
    }
    if (value instanceof Boolean bool) {
        return bool ? 1.0d : 0.0d;
    }
    try {
        return Double.parseDouble(String.valueOf(value));
    } catch (RuntimeException ignored) {
        return 0.0d;
    }
}

private static boolean $mmmTruthy(Object value) {
    if (value == null) {
        return false;
    }
    if (value instanceof Boolean bool) {
        return bool;
    }
    if (value instanceof Number number) {
        return number.doubleValue() != 0.0d;
    }
    String text = String.valueOf(value);
    return !text.isBlank() && !"false".equalsIgnoreCase(text) && !"0".equals(text);
}

private static Object $mmmArithmetic(String op, Object left, Object right) {
    if ("+".equals(op) && (left instanceof String || right instanceof String)) {
        return String.valueOf(left) + String.valueOf(right);
    }
    double l = $mmmNumber(left);
    double r = $mmmNumber(right);
    return switch (op) {
        case "+" -> l + r;
        case "-" -> l - r;
        case "*" -> l * r;
        case "/" -> r == 0.0d ? l : l / r;
        case "%" -> r == 0.0d ? l : l % r;
        default -> l;
    };
}

private static boolean $mmmEquals(Object left, Object right) {
    if (left instanceof Number && right instanceof Number) {
        return Double.compare($mmmNumber(left), $mmmNumber(right)) == 0;
    }
    return java.util.Objects.equals(left, right);
}

private static int $mmmCompare(Object left, Object right) {
    if (left instanceof Number || right instanceof Number) {
        return Double.compare($mmmNumber(left), $mmmNumber(right));
    }
    return String.valueOf(left).compareTo(String.valueOf(right));
}

public static synchronized String transition(
        String fromState,
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTransition transition : $mmmTransitions) {
        if (java.util.Objects.equals(transition.fromState(), fromState)
                && java.util.Objects.equals(transition.trigger(), trigger)
                && transition.guard().test(context)) {
            transition.action().apply(context);
            return transition.toState();
        }
    }
    return fromState;
}

public static synchronized boolean invariantsHold(
        java.util.Map<String, Object> context
) {
    for ($mmmInvariant invariant : $mmmInvariants) {
        if (!invariant.guard().test(context)) {
            return false;
        }
    }
    return true;
}

public static synchronized java.util.List<String> invariantFailures(
        java.util.Map<String, Object> context
) {
    java.util.List<String> failures = new java.util.ArrayList<>();
    for ($mmmInvariant invariant : $mmmInvariants) {
        if (!invariant.guard().test(context)) {
            failures.add(invariant.enforcement());
        }
    }
    return java.util.List.copyOf(failures);
}

public static synchronized void initializeState(
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTriggeredAction row : $mmmInitializers) {
        if (java.util.Objects.equals(row.trigger(), trigger)) {
            row.action().apply(context);
        }
    }
}

public static synchronized void applyUpdate(
        String trigger,
        java.util.Map<String, Object> context
) {
    for ($mmmTriggeredAction row : $mmmUpdates) {
        if (java.util.Objects.equals(row.trigger(), trigger)) {
            row.action().apply(context);
        }
    }
}

public static synchronized void cleanupState(
        String event,
        java.util.Map<String, Object> context
) {
    for ($mmmCleanupAction row : $mmmCleanup) {
        if (java.util.Objects.equals(row.event(), event)) {
            row.action().apply(context);
        }
    }
}

public static synchronized java.util.List<String[]> concurrencyRules() {
    java.util.List<String[]> result = new java.util.ArrayList<>();
    for (String[] row : $mmmConcurrency) {
        result.add(row.clone());
    }
    return java.util.List.copyOf(result);
}
""".strip()


def render_state_model_concern(
    task: Mapping[str, Any],
    concern: str,
    *,
    include_runtime: bool,
) -> str | None:
    records_by_concern = _obligations(task)
    if concern not in records_by_concern:
        return None
    records = records_by_concern[concern]
    variables = records_by_concern.get("variables", [])
    declared = {
        str(record.get("name") or "").strip()
        for record in variables
        if str(record.get("name") or "").strip()
    }
    parts: list[str] = [_COMMON] if include_runtime else []

    if concern == "variables":
        lines = ["static {"]
        for record in records:
            name = str(record.get("name") or "").strip()
            if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_NAME: {name!r} is not a stable identifier"
                )
            default = _default_value(record)
            if default != "null":
                lines.append(
                    f"    $mmmState.put({_java_string(name)}, {default});"
                )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "transitions":
        lines = ["static {"]
        for record in records:
            guard = _compile_condition(record.get("guard", ""))
            mutation = _compile_mutation(
                record.get("mutation", ""),
                declared=declared,
            )
            lines.append(
                "    $mmmTransitions.add(new $mmmTransition("
                + ", ".join((
                    _java_string(record.get("from_state", "")),
                    _java_string(record.get("trigger", "")),
                    f"context -> ({guard})",
                    f"context -> {{ {mutation} }}",
                    _java_string(record.get("to_state", "")),
                ))
                + "));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "invariants":
        lines = ["static {"]
        for record in records:
            condition = _compile_condition(record.get("condition", ""))
            lines.append(
                "    $mmmInvariants.add(new $mmmInvariant("
                f"context -> ({condition}), "
                f"{_java_string(record.get('enforcement', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "initialization":
        lines = ["static {"]
        for record in records:
            action = _compile_mutation(
                record.get("initial_state", ""),
                declared=declared,
            )
            lines.append(
                "    $mmmInitializers.add(new $mmmTriggeredAction("
                f"{_java_string(record.get('trigger', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('owner', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "updates":
        lines = ["static {"]
        for record in records:
            action = _compile_mutation(
                record.get("mutation", ""),
                declared=declared,
            )
            lines.append(
                "    $mmmUpdates.add(new $mmmTriggeredAction("
                f"{_java_string(record.get('trigger', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('owner', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "cleanup":
        lines = ["static {"]
        for record in records:
            action = _compile_mutation(
                record.get("action", ""),
                declared=declared,
            )
            lines.append(
                "    $mmmCleanup.add(new $mmmCleanupAction("
                f"{_java_string(record.get('event', ''))}, "
                f"context -> {{ {action} }}, "
                f"{_java_string(record.get('retained_state', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "concurrency":
        lines = ["static {"]
        for record in records:
            values = ", ".join(
                _java_string(record.get(field, ""))
                for field in ("entry_path", "ownership", "reentrancy_rule")
            )
            lines.append(
                f"    $mmmConcurrency.add(new String[]{{{values}}});"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    else:
        return None
    return "\n\n".join(part for part in parts if part).strip()


PUBLIC_API = (
    "public static synchronized Object getState(String name)",
    "public static synchronized void setState(String name, Object value)",
    "public static synchronized String transition(String fromState, String trigger, java.util.Map<String, Object> context)",
    "public static synchronized boolean invariantsHold(java.util.Map<String, Object> context)",
    "public static synchronized java.util.List<String> invariantFailures(java.util.Map<String, Object> context)",
    "public static synchronized void initializeState(String trigger, java.util.Map<String, Object> context)",
    "public static synchronized void applyUpdate(String trigger, java.util.Map<String, Object> context)",
    "public static synchronized void cleanupState(String event, java.util.Map<String, Object> context)",
    "public static synchronized java.util.List<String[]> concurrencyRules()",
)


__all__ = [
    "PUBLIC_API",
    "STATE_EXPRESSION_PATTERN",
    "STATE_MUTATION_PATTERN",
    "constrain_state_chunk_schema",
    "constrain_state_record_schema",
    "has_complete_structured_state",
    "render_state_model_concern",
]
