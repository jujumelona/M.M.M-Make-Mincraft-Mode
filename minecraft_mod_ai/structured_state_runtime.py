from __future__ import annotations

"""Host compiler for structured state-model records.

Structured state records are compiled directly to Java. No text-model Java generation
or compiler-repair loop is used on this path.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import ast
import json
import re
from typing import Any

from lark import Lark, Transformer, UnexpectedInput, v_args
from lark.exceptions import VisitError



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
    return r"^(?:" + assignment + r"(?:[ \t]*;[ \t]*" + assignment + r")*)?$"


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
        if field in {"mutation", "initial_state", "action"}:
            target["minLength"] = 0
            target["description"] = (
                "Host state-mutation DSL. Empty string means no state mutation. "
                "Non-empty values must contain only assignments to declared state "
                "variables; external subsystem actions do not belong in this field."
            )
        else:
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

_STATE_DSL_GRAMMAR = r"""
?expr: implication
?implication: or_expr
            | or_expr IMPLIES implication      -> implication
?or_expr: and_expr (OR and_expr)*              -> or_expr
?and_expr: comparison (AND comparison)*        -> and_expr
?comparison: sum_expr
           | sum_expr COMP_OP sum_expr         -> comparison
?sum_expr: product (ADD_OP product)*           -> sum_expr
?product: unary (MUL_OP unary)*                -> product
?unary: NOT unary                              -> logical_not
      | "-" unary                              -> negate
      | atom
?atom: NUMBER                                  -> number
     | STRING                                  -> string
     | TRUE                                    -> true
     | FALSE                                   -> false
     | NULL                                    -> null
     | EMPTY_MAP                               -> empty_map
     | EMPTY_LIST                              -> empty_list
     | function_call
     | NAME                                    -> identifier
     | "(" implication ")"                     -> grouped

function_call: NAME "(" [arguments] ")"         -> function_call
arguments: implication ("," implication)*       -> arguments
assignment: NAME ASSIGN_OP expr                 -> assignment

OR.5: "||" | /(?i:OR)\b/
AND.5: "&&" | /(?i:AND)\b/
NOT.5: "!" | /(?i:NOT)\b/
IMPLIES.5: "->" | /(?i:IMPLIES)\b/
COMP_OP: "==" | "!=" | ">=" | "<=" | ">" | "<" | "="
ADD_OP: "+" | "-"
MUL_OP: "*" | "/" | "%"
ASSIGN_OP: "+=" | "-=" | "*=" | "/=" | "=" | ":"
TRUE.6: /(?i:true)\b/
FALSE.6: /(?i:false)\b/
NULL.6: /(?i:null)\b/
EMPTY_MAP: "{}"
EMPTY_LIST: "[]"
NAME: /[A-Za-z_$][A-Za-z0-9_$.]*/
NUMBER: /[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?/
STRING: ESCAPED_STRING | /'(?:\\.|[^'\\])*'/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

_STATE_DSL_PARSER = Lark(
    _STATE_DSL_GRAMMAR,
    parser="lalr",
    lexer="contextual",
    start=["expr", "assignment"],
    maybe_placeholders=False,
)


def _fold_binary(first: tuple, tail: tuple[Any, ...], op_map: Mapping[str, str] | None = None) -> tuple:
    node = first
    if len(tail) % 2:
        raise ValueError("STRUCTURED_STATE_EXPRESSION: malformed operator sequence")
    for index in range(0, len(tail), 2):
        raw_op = str(tail[index])
        right = tail[index + 1]
        op = op_map.get(raw_op.casefold(), raw_op) if op_map else raw_op
        node = ("binary", op, node, right)
    return node


@v_args(inline=True)
class _StateDslTransformer(Transformer):
    def number(self, token):
        return ("number", str(token))

    def string(self, token):
        text = str(token)
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: invalid string literal {text!r}"
            ) from exc
        return ("string", str(value))

    def true(self, _token):
        return ("bool", True)

    def false(self, _token):
        return ("bool", False)

    def null(self, _token):
        return ("null",)

    def empty_map(self, _token):
        return ("empty_map",)

    def empty_list(self, _token):
        return ("empty_list",)

    def identifier(self, token):
        return ("identifier", str(token))

    def grouped(self, node):
        return node

    def arguments(self, *nodes):
        return tuple(nodes)

    def function_call(self, name, arguments=()):
        args = arguments if isinstance(arguments, tuple) else (arguments,)
        return ("call", str(name), args)

    def logical_not(self, _operator, node):
        return ("unary", "!", node)

    def negate(self, node):
        return ("binary", "-", ("number", "0"), node)

    def product(self, first, *tail):
        return _fold_binary(first, tail)

    def sum_expr(self, first, *tail):
        return _fold_binary(first, tail)

    def comparison(self, left, operator, right):
        op = "==" if str(operator) == "=" else str(operator)
        return ("binary", op, left, right)

    def and_expr(self, first, *tail):
        return _fold_binary(first, tail, {"and": "&&", "&&": "&&"})

    def or_expr(self, first, *tail):
        return _fold_binary(first, tail, {"or": "||", "||": "||"})

    def implication(self, left, _operator, right):
        return ("binary", "->", left, right)

    def assignment(self, name, operator, expression):
        op = "=" if str(operator) == ":" else str(operator)
        return (str(name), op, expression)


_STATE_DSL_TRANSFORMER = _StateDslTransformer()


def _state_dsl_error(source: str, exc: BaseException) -> ValueError:
    if isinstance(exc, UnexpectedInput):
        context = exc.get_context(source, span=48).strip().replace("\n", " ")
        return ValueError(
            "STRUCTURED_STATE_EXPRESSION: parse error at "
            f"{exc.line}:{exc.column}: {context}"
        )
    if isinstance(exc, VisitError) and isinstance(exc.orig_exc, ValueError):
        return exc.orig_exc
    return ValueError(f"STRUCTURED_STATE_EXPRESSION: {exc}")


def _parse_state_expression(text: str) -> tuple:
    source = str(text or "").strip()
    if not source:
        return ("bool", True)
    try:
        tree = _STATE_DSL_PARSER.parse(source, start="expr")
        return _STATE_DSL_TRANSFORMER.transform(tree)
    except (UnexpectedInput, VisitError, ValueError) as exc:
        raise _state_dsl_error(source, exc) from exc


def _parse_state_assignment(text: str) -> tuple[str, str, tuple]:
    source = str(text or "").strip()
    try:
        tree = _STATE_DSL_PARSER.parse(source, start="assignment")
        result = _STATE_DSL_TRANSFORMER.transform(tree)
    except (UnexpectedInput, VisitError, ValueError) as exc:
        raise _state_dsl_error(source, exc) from exc
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("STRUCTURED_STATE_MUTATION: invalid assignment")
    return result


class _Expression:
    """Compatibility facade backed entirely by the Lark grammar."""

    def __init__(self, text: str) -> None:
        self.text = str(text or "")

    def parse(self) -> tuple:
        return _parse_state_expression(self.text)


def _java_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


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
    if kind == "empty_map":
        return "new java.util.LinkedHashMap<>()"
    if kind == "empty_list":
        return "new java.util.ArrayList<>()"
    if kind == "call":
        name = str(node[1]).casefold()
        if name not in {"sum", "min", "max", "abs", "count", "size", "len"}:
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: unsupported function {node[1]!r}"
            )
        args = ", ".join(_value(arg, context) for arg in node[2])
        return (
            f"$mmmFunction({_java_string(name)}, "
            f"java.util.Arrays.asList({args}), {context})"
        )
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
        if op == "->":
            left = _condition(node[2], context)
            right = _condition(node[3], context)
            return f"((!({left})) || ({right}))"
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


def validate_state_expression(text: str) -> None:
    """Validate the canonical host expression DSL without generating Java."""

    _Expression(str(text or "")).parse()


def _compile_condition(text: str, context: str = "context") -> str:
    raw = str(text or "").strip()
    if not raw:
        return "true"
    return _condition(_Expression(raw).parse(), context)


def _split_mutation_statements(script: str) -> list[str]:
    source = str(script or "")
    rows: list[str] = []
    buffer: list[str] = []
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            buffer.append(char)
            if char == "\\" and index + 1 < len(source):
                index += 1
                buffer.append(source[index])
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
        elif char == ";":
            rows.append("".join(buffer))
            buffer.clear()
        else:
            buffer.append(char)
        index += 1
    rows.append("".join(buffer))
    return rows


def _compile_mutation(
    script: str,
    *,
    declared: set[str],
    context: str = "context",
) -> str:
    text = str(script or "").strip()
    rows: list[str] = []
    for raw in _split_mutation_statements(text):
        statement = raw.strip()
        if not statement:
            continue
        try:
            name, operator, expression = _parse_state_assignment(statement)
        except ValueError as exc:
            raise ValueError(
                "STRUCTURED_STATE_MUTATION: expected assignment, got "
                + repr(statement)
            ) from exc
        if name not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {name!r}"
            )
        right = _value(expression, context)
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

public static synchronized Object getState(
        String name,
        java.util.Map<String, Object> context
) {
    return $mmmRead(name, context);
}

public static synchronized void setState(
        String name,
        Object value,
        java.util.Map<String, Object> context
) {
    setState(name, value);
    if (context == null || name == null || name.isBlank()) {
        return;
    }
    try {
        if (value == null) {
            context.remove(name);
        } else {
            context.put(name, value);
        }
    } catch (UnsupportedOperationException ignored) {
        // Immutable event contexts remain valid read overlays; persistent state
        // was already committed above.
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

private static double $mmmSumValue(Object value) {
    if (value == null) {
        return 0.0d;
    }
    if (value instanceof java.lang.Iterable<?> iterable) {
        double total = 0.0d;
        for (Object item : iterable) {
            total += $mmmSumValue(item);
        }
        return total;
    }
    if (value instanceof java.util.Map<?, ?> map) {
        double total = 0.0d;
        for (Object item : map.values()) {
            total += $mmmSumValue(item);
        }
        return total;
    }
    if (value.getClass().isArray()) {
        double total = 0.0d;
        int length = java.lang.reflect.Array.getLength(value);
        for (int index = 0; index < length; index++) {
            total += $mmmSumValue(java.lang.reflect.Array.get(value, index));
        }
        return total;
    }
    return $mmmNumber(value);
}

private static int $mmmSize(Object value) {
    if (value == null) {
        return 0;
    }
    if (value instanceof java.util.Collection<?> collection) {
        return collection.size();
    }
    if (value instanceof java.util.Map<?, ?> map) {
        return map.size();
    }
    if (value instanceof java.lang.CharSequence text) {
        return text.length();
    }
    if (value.getClass().isArray()) {
        return java.lang.reflect.Array.getLength(value);
    }
    return 1;
}

private static Object $mmmFunction(
        String name,
        java.util.List<Object> args,
        java.util.Map<String, Object> context
) {
    return switch (name) {
        case "sum" -> {
            double total = 0.0d;
            for (Object arg : args) {
                total += $mmmSumValue(arg);
            }
            yield total;
        }
        case "min" -> {
            double result = Double.POSITIVE_INFINITY;
            for (Object arg : args) {
                result = Math.min(result, $mmmNumber(arg));
            }
            yield result == Double.POSITIVE_INFINITY ? 0.0d : result;
        }
        case "max" -> {
            double result = Double.NEGATIVE_INFINITY;
            for (Object arg : args) {
                result = Math.max(result, $mmmNumber(arg));
            }
            yield result == Double.NEGATIVE_INFINITY ? 0.0d : result;
        }
        case "abs" -> Math.abs(args.isEmpty() ? 0.0d : $mmmNumber(args.get(0)));
        case "count", "size", "len" ->
                args.isEmpty() ? 0 : $mmmSize(args.get(0));
        default -> throw new IllegalArgumentException(
                "Unsupported structured-state function: " + name
        );
    };
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
    "public static synchronized Object getState(String name, java.util.Map<String, Object> context)",
    "public static synchronized void setState(String name, Object value)",
    "public static synchronized void setState(String name, Object value, java.util.Map<String, Object> context)",
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
    "validate_state_expression",
]
