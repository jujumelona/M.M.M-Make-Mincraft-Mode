from __future__ import annotations

"""Host compiler for structured state-model records.

Structured state records are compiled directly to Java. No text-model Java generation
or compiler-repair loop is used on this path.
"""

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any


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
    rows: list[str] = []
    for raw in str(script or "").split(";"):
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
    "has_complete_structured_state",
    "render_state_model_concern",
]
