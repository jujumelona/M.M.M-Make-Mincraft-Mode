from __future__ import annotations

"""Host compiler for structured state-model records.

Supported state expressions and mutations compile directly to Java. Anything
outside the host DSL fails closed before source generation; there is no model or
Java callback fallback on this path.
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
STATE_EXPRESSION_PATTERN = r"^.*$"
STATE_MUTATION_PATTERN = r"^.*$"

_STATE_EXECUTABLE_FIELDS = {
    "transitions": ("guard", "mutation", "mutations"),
    "invariants": ("condition",),
    "initialization": ("initial_state", "mutations"),
    "updates": ("mutation", "mutations"),
    "cleanup": ("action", "mutations"),
}


def constrain_state_record_schema(
    concern: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Decorate state-model record schemas with guidance descriptions."""
    result = deepcopy(dict(schema))
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    if concern == "variables" and isinstance(properties.get("name"), dict):
        properties["name"]["description"] = (
            "Stable ASCII internal state identifier consumed by the host state compiler."
        )
    for field in _STATE_EXECUTABLE_FIELDS.get(concern, ()):
        target = properties.get(field)
        if not isinstance(target, dict):
            continue
        if field in {"mutation", "mutations", "initial_state", "action"}:
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

_SUPPORTED_STATE_FUNCTIONS = frozenset({
    "sum",
    "min",
    "max",
    "abs",
    "count",
    "size",
    "len",
})


def _validate_state_ast(node: tuple) -> None:
    """Validate semantic nodes accepted by the host state-expression compiler."""
    kind = node[0]
    if kind == "call":
        name = str(node[1]).casefold()
        if name not in _SUPPORTED_STATE_FUNCTIONS:
            raise ValueError(
                f"STRUCTURED_STATE_EXPRESSION: unsupported function {node[1]!r}"
            )
        for argument in node[2]:
            _validate_state_ast(argument)
        return
    if kind == "unary":
        _validate_state_ast(node[2])
        return
    if kind == "binary":
        _validate_state_ast(node[2])
        _validate_state_ast(node[3])
        return


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
        if name not in _SUPPORTED_STATE_FUNCTIONS:
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
    """Validate the exact semantic subset that the Java lowering can compile."""

    node = _Expression(str(text or "")).parse()
    _validate_state_ast(node)


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


def validate_state_expr_ir(
    expr: Any,
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate expression IR or legacy expression against symbols."""
    if expr is None or isinstance(expr, bool):
        return
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = None

    if isinstance(expr, str):
        raw = expr.strip()
        if not raw or raw.casefold() in {"true", "false", "null"}:
            return
        node = _Expression(raw).parse()
        _validate_state_ast(node)
        return

    if isinstance(expr, (int, float)):
        return

    if not isinstance(expr, Mapping):
        raise ValueError(f"STRUCTURED_STATE_EXPRESSION: expected object or string, got {type(expr).__name__}")

    kind = expr.get("kind")
    if not kind:
        if "terms" in expr:
            kind = "and"
        elif "op" in expr and "left" in expr and "right" in expr:
            kind = "compare" if expr["op"] in {"==", "!=", ">=", "<=", ">", "<", "="} else "arithmetic"
        elif "name" in expr:
            kind = "state_ref"
        elif "value" in expr:
            kind = "literal"
        else:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: missing 'kind' in expression object {expr!r}")

    if kind in {"and", "or"}:
        terms = expr.get("terms")
        if terms is None:
            terms = ()
        elif not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, bytearray)):
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: {kind} terms must be a list")
        for term in terms:
            validate_state_expr_ir(term, symbols=declared)
        return

    if kind == "compare":
        op = expr.get("op", "==")
        if op not in {"==", "!=", ">=", "<=", ">", "<", "="}:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported comparison op {op!r}")
        left = expr.get("left")
        right = expr.get("right")
        if left is None or right is None:
            raise ValueError("STRUCTURED_STATE_EXPRESSION: compare requires left and right operands")
        validate_state_expr_ir(left, symbols=declared)
        validate_state_expr_ir(right, symbols=declared)
        return

    if kind == "not":
        term = expr.get("term") or expr.get("operand") or expr.get("left")
        if term is None:
            raise ValueError("STRUCTURED_STATE_EXPRESSION: not requires a term")
        validate_state_expr_ir(term, symbols=declared)
        return

    if kind == "implies":
        validate_state_expr_ir(expr.get("left"), symbols=declared)
        validate_state_expr_ir(expr.get("right"), symbols=declared)
        return

    if kind == "state_ref":
        name = expr.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("STRUCTURED_STATE_EXPRESSION: state_ref requires string 'name'")
        name = name.strip()
        if declared is not None and len(declared) > 0 and name not in declared:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: undeclared state variable {name!r}")
        return

    if kind == "context_ref":
        name = expr.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("STRUCTURED_STATE_EXPRESSION: context_ref requires string 'name'")
        return

    if kind == "number":
        val = str(expr.get("value") or "").strip()
        if not re.fullmatch(r"^-?[0-9]+([.][0-9]+)?$", val):
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: invalid number literal {val!r}")
        return

    if kind == "literal":
        return

    if kind == "arithmetic":
        op = expr.get("op", "+")
        if op not in {"+", "-", "*", "/", "%"}:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported arithmetic op {op!r}")
        validate_state_expr_ir(expr.get("left"), symbols=declared)
        validate_state_expr_ir(expr.get("right"), symbols=declared)
        return

    if kind == "call":
        name = str(expr.get("name") or "").casefold()
        if name not in _SUPPORTED_STATE_FUNCTIONS:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unsupported function {name!r}")
        args = expr.get("args") or ()
        for arg in args:
            validate_state_expr_ir(arg, symbols=declared)
        return

    raise ValueError(f"STRUCTURED_STATE_EXPRESSION: unknown expression kind {kind!r}")


def compile_state_expr_ir(
    expr: Any,
    *,
    declared: set[str] | None = None,
    context: str = "context",
) -> str:
    """Compile expression IR or legacy string to Java."""
    if expr is None:
        return "true"
    if isinstance(expr, bool):
        return "true" if expr else "false"
    if isinstance(expr, (int, float)):
        return f"Double.valueOf({expr})"
    if isinstance(expr, str):
        raw = expr.strip()
        if not raw or raw.casefold() == "true":
            return "true"
        return _compile_condition(raw, context=context)

    if not isinstance(expr, Mapping):
        return "true"

    kind = expr.get("kind")
    if not kind:
        if "terms" in expr:
            kind = "and"
        elif "op" in expr and "left" in expr and "right" in expr:
            kind = "compare" if expr["op"] in {"==", "!=", ">=", "<=", ">", "<", "="} else "arithmetic"
        elif "name" in expr:
            kind = "state_ref"
        elif "value" in expr:
            kind = "literal"
        else:
            kind = "literal"

    if kind == "literal":
        val = expr.get("value")
        if val is None or val == "null":
            return "null"
        if isinstance(val, bool):
            return "Boolean.TRUE" if val else "Boolean.FALSE"
        if isinstance(val, (int, float)):
            return f"Double.valueOf({val})"
        return _java_string(str(val))

    if kind == "state_ref":
        name = str(expr.get("name") or "").strip()
        if declared is not None and len(declared) > 0 and name not in declared:
            raise ValueError(f"STRUCTURED_STATE_EXPRESSION: undeclared state variable {name!r}")
        return f"$mmmRead({_java_string(name)}, {context})"

    if kind == "context_ref":
        name = str(expr.get("name") or "").strip()
        return f"$mmmRead({_java_string(name)}, {context})"

    if kind == "and":
        terms = expr.get("terms") or []
        if not terms:
            return "true"
        compiled_terms = [compile_state_expr_ir(t, declared=declared, context=context) for t in terms]
        if len(compiled_terms) == 1:
            return compiled_terms[0]
        return "(" + " && ".join(f"({t})" for t in compiled_terms) + ")"

    if kind == "or":
        terms = expr.get("terms") or []
        if not terms:
            return "false"
        compiled_terms = [compile_state_expr_ir(t, declared=declared, context=context) for t in terms]
        if len(compiled_terms) == 1:
            return compiled_terms[0]
        return "(" + " || ".join(f"({t})" for t in compiled_terms) + ")"

    if kind == "not":
        term = expr.get("term") or expr.get("operand") or expr.get("left")
        compiled = compile_state_expr_ir(term, declared=declared, context=context)
        return f"(!$mmmTruthy({compiled}))"

    if kind == "implies":
        left = compile_state_expr_ir(expr.get("left"), declared=declared, context=context)
        right = compile_state_expr_ir(expr.get("right"), declared=declared, context=context)
        return f"((!({left})) || ({right}))"

    if kind == "compare":
        op = expr.get("op", "==")
        op = "==" if op == "=" else op
        left = compile_state_expr_ir(expr.get("left"), declared=declared, context=context)
        right = compile_state_expr_ir(expr.get("right"), declared=declared, context=context)
        if op == "==":
            return f"$mmmEquals({left}, {right})"
        if op == "!=":
            return f"(!$mmmEquals({left}, {right}))"
        if op in {">=", "<=", ">", "<"}:
            return f"($mmmCompare({left}, {right}) {op} 0)"
        return f"Boolean.valueOf({left} {op} {right})"

    if kind == "arithmetic":
        op = expr.get("op", "+")
        left = compile_state_expr_ir(expr.get("left"), declared=declared, context=context)
        right = compile_state_expr_ir(expr.get("right"), declared=declared, context=context)
        return f"$mmmArithmetic({_java_string(op)}, {left}, {right})"

    if kind == "call":
        name = str(expr.get("name") or "").casefold()
        args = expr.get("args") or ()
        args_compiled = ", ".join(compile_state_expr_ir(a, declared=declared, context=context) for a in args)
        return f"$mmmFunction({_java_string(name)}, java.util.Arrays.asList({args_compiled}), {context})"

    return "true"


def validate_mutation_ir(
    mutation: Any,
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate mutation IR or legacy mutation string against declared symbols."""
    if mutation is None:
        return
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = None

    if isinstance(mutation, str):
        text = mutation.strip()
        if not text:
            return
        for raw in _split_mutation_statements(text):
            stmt = raw.strip()
            if not stmt:
                continue
            try:
                name, op, expression = _parse_state_assignment(stmt)
            except ValueError as exc:
                raise ValueError(
                    f"STRUCTURED_STATE_MUTATION: expected assignment, got {stmt!r}"
                ) from exc
            if declared is not None and len(declared) > 0 and name not in declared:
                raise ValueError(
                    f"STRUCTURED_STATE_MUTATION: undeclared state variable {name!r}"
                )
        return

    if isinstance(mutation, Mapping):
        mutations = [mutation]
    elif isinstance(mutation, Sequence) and not isinstance(mutation, (str, bytes, bytearray)):
        mutations = list(mutation)
    else:
        raise ValueError(
            f"STRUCTURED_STATE_MUTATION: expected list or object, got {type(mutation).__name__}"
        )

    for item in mutations:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: mutation item must be an object, got {type(item).__name__}"
            )
        target = str(item.get("target") or "").strip()
        if not target:
            raise ValueError("STRUCTURED_STATE_MUTATION: mutation requires non-empty string 'target'")
        if declared is not None and len(declared) > 0 and target not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {target!r}"
            )
        op = item.get("operator", "=")
        if op not in {"=", "+=", "-=", "*=", "/=", ":"}:
            raise ValueError(f"STRUCTURED_STATE_MUTATION: invalid operator {op!r}")
        val = item.get("value")
        if val is not None:
            validate_state_expr_ir(val, symbols=declared)


def compile_mutation_ir(
    mutation: Any,
    *,
    declared: set[str] | None = None,
    context: str = "context",
) -> str:
    """Compile mutation IR or legacy string to Java."""
    if mutation is None:
        return ""
    if isinstance(mutation, str):
        return _compile_mutation(mutation, declared=declared or set(), context=context)

    if isinstance(mutation, Mapping):
        mutations = [mutation]
    elif isinstance(mutation, Sequence) and not isinstance(mutation, (str, bytes, bytearray)):
        mutations = list(mutation)
    else:
        return ""

    rows: list[str] = []
    for item in mutations:
        if not isinstance(item, Mapping):
            continue
        target = str(item.get("target") or "").strip()
        if not target:
            continue
        if declared is not None and len(declared) > 0 and target not in declared:
            raise ValueError(
                f"STRUCTURED_STATE_MUTATION: undeclared state variable {target!r}"
            )
        operator = item.get("operator", "=")
        operator = "=" if operator == ":" else operator
        val = item.get("value")
        right = compile_state_expr_ir(val, declared=declared, context=context)
        if operator == "=":
            rows.append(f"setState({_java_string(target)}, {right});")
        else:
            arithmetic = operator[0]
            rows.append(
                f"setState({_java_string(target)}, "
                f"$mmmArithmetic({_java_string(arithmetic)}, "
                f"$mmmRead({_java_string(target)}, {context}), {right}));"
            )
    return " ".join(rows)


def mutations_schema(allowed_state_symbols: Any = None) -> dict[str, Any]:
    symbols = (
        sorted(allowed_state_symbols.declared_names)
        if isinstance(allowed_state_symbols, StateSymbolTable)
        else sorted(set(allowed_state_symbols))
        if isinstance(allowed_state_symbols, (set, list, tuple)) and allowed_state_symbols
        else []
    )
    target_schema = {
        "type": "string",
        "maxLength": 32,
        **({"enum": symbols} if symbols else {"pattern": r"^[A-Za-z_][A-Za-z0-9_]*$", "minLength": 1}),
    }
    name_schema = {
        "type": "string",
        "maxLength": 32,
        **({"enum": symbols} if symbols else {"pattern": r"^[A-Za-z_][A-Za-z0-9_]*$", "minLength": 1}),
    }
    value_operand = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["literal", "number", "state_ref", "context_ref"],
                "maxLength": 16,
            },
            "name": name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    value_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["literal", "number", "state_ref", "context_ref", "arithmetic", "call"],
                "maxLength": 16,
            },
            "name": name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
            "op": {"type": "string", "enum": ["+", "-", "*", "/", "%"], "maxLength": 2},
            "left": value_operand,
            "right": value_operand,
            "args": {
                "type": "array",
                "maxItems": 2,
                "items": value_operand,
            },
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    return {
        "type": "array",
        "maxItems": 2,
        "items": {
            "type": "object",
            "properties": {
                "target": target_schema,
                "operator": {
                    "type": "string",
                    "enum": ["=", "+=", "-=", "*=", "/="],
                    "maxLength": 2,
                },
                "value": value_schema,
            },
            "required": ["target", "operator", "value"],
            "additionalProperties": False,
        },
    }


def state_expr_schema(allowed_state_symbols: Any = None) -> dict[str, Any]:
    symbols = (
        sorted(allowed_state_symbols.declared_names)
        if isinstance(allowed_state_symbols, StateSymbolTable)
        else sorted(set(allowed_state_symbols))
        if isinstance(allowed_state_symbols, (set, list, tuple)) and allowed_state_symbols
        else []
    )
    name_schema = {
        "type": "string",
        "maxLength": 24,
        **({"enum": symbols} if symbols else {"pattern": r"^[A-Za-z_][A-Za-z0-9_]*$", "minLength": 1}),
    }
    fn_or_name_schema = {
        "type": "string",
        "maxLength": 24,
    }
    leaf_operand = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["state_ref", "context_ref", "literal"],
                "maxLength": 16,
            },
            "name": name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    operand_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["state_ref", "context_ref", "literal", "arithmetic", "call"],
                "maxLength": 16,
            },
            "name": fn_or_name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
            "op": {"type": "string", "enum": ["+", "-", "*", "/", "%"], "maxLength": 2},
            "left": leaf_operand,
            "right": leaf_operand,
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    compare_term_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["compare", "state_ref", "literal", "call"], "maxLength": 16},
            "op": {"type": "string", "enum": ["==", "!=", ">=", "<=", ">", "<", "="], "maxLength": 2},
            "left": operand_schema,
            "right": leaf_operand,
            "name": fn_or_name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["and", "or", "compare", "not", "literal", "call", "state_ref", "context_ref"], "maxLength": 16},
            "terms": {
                "type": "array",
                "maxItems": 2,
                "items": compare_term_schema,
            },
            "op": {"type": "string", "enum": ["==", "!=", ">=", "<=", ">", "<", "="], "maxLength": 2},
            "left": leaf_operand,
            "right": leaf_operand,
            "name": fn_or_name_schema,
            "value": {"type": ["string", "null"], "maxLength": 24},
            "args": {
                "type": "array",
                "maxItems": 2,
                "items": leaf_operand,
            },
        },
        "required": ["kind"],
        "additionalProperties": False,
    }


def state_concern_schema(
    concern: str,
    *,
    allowed_state_symbols: Any = None,
) -> dict[str, Any]:
    symbols = (
        sorted(allowed_state_symbols.declared_names)
        if isinstance(allowed_state_symbols, StateSymbolTable)
        else sorted(set(allowed_state_symbols))
        if isinstance(allowed_state_symbols, (set, list, tuple)) and allowed_state_symbols
        else []
    )
    if concern == "variables":
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                    "description": "Stable ASCII internal state identifier consumed by the host state compiler.",
                },
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
                "type": {"type": "string", "minLength": 1, "maxLength": 128},
                "unit": {"type": "string", "minLength": 1, "maxLength": 128},
                "default": {"type": "string", "minLength": 1, "maxLength": 128},
                "domain": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["name", "owner", "type", "unit", "default", "domain"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "transitions":
        schema = {
            "type": "object",
            "properties": {
                "from_state": {"type": "string", "minLength": 1, "maxLength": 128},
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "guard": state_expr_schema(symbols),
                "mutations": mutations_schema(symbols),
                "mutation": mutations_schema(symbols),
                "to_state": {"type": "string", "minLength": 1, "maxLength": 128},
            },
            "required": ["from_state", "trigger", "guard", "to_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "invariants":
        schema = {
            "type": "object",
            "properties": {
                "condition": state_expr_schema(symbols),
                "enforcement": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["condition", "enforcement"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "initialization":
        schema = {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "initial_state": mutations_schema(symbols),
                "mutations": mutations_schema(symbols),
            },
            "required": ["owner", "trigger", "initial_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "updates":
        schema = {
            "type": "object",
            "properties": {
                "trigger": {"type": "string", "minLength": 1, "maxLength": 128},
                "mutation": mutations_schema(symbols),
                "mutations": mutations_schema(symbols),
                "owner": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["trigger", "mutation", "owner"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "cleanup":
        schema = {
            "type": "object",
            "properties": {
                "event": {"type": "string", "minLength": 1, "maxLength": 128},
                "action": mutations_schema(symbols),
                "mutations": mutations_schema(symbols),
                "retained_state": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["event", "action", "retained_state"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    if concern == "concurrency":
        schema = {
            "type": "object",
            "properties": {
                "entry_path": {"type": "string", "minLength": 1, "maxLength": 256},
                "ownership": {"type": "string", "minLength": 1, "maxLength": 256},
                "reentrancy_rule": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["entry_path", "ownership", "reentrancy_rule"],
            "additionalProperties": False,
        }
        return constrain_state_record_schema(concern, schema)
    raise ValueError(f"Unknown state concern: {concern!r}")


class StateSymbolTable:
    """Canonical symbol table for state variables declared in state_model."""

    def __init__(
        self,
        variables: Sequence[Mapping[str, Any]] | Sequence[str] | set[str] | StateSymbolTable = (),
    ) -> None:
        if isinstance(variables, StateSymbolTable):
            self.declared_names: set[str] = set(variables.declared_names)
            self.variables: dict[str, Mapping[str, Any]] = dict(variables.variables)
            return

        self.declared_names = set()
        self.variables = {}
        for item in variables or ():
            if isinstance(item, Mapping):
                name = str(item.get("name") or "").strip()
                if name:
                    self.variables[name] = item
                    self.declared_names.add(name)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    self.declared_names.add(name)

    def contains(self, name: str) -> bool:
        return name in self.declared_names

    def __contains__(self, name: str) -> bool:
        return name in self.declared_names

    def __iter__(self):
        return iter(sorted(self.declared_names))

    def __len__(self) -> int:
        return len(self.declared_names)

    def prompt_text(self) -> str:
        if not self.declared_names:
            return ""
        lines = ["Canonical state symbols:", "variables:"]
        for name in sorted(self.declared_names):
            lines.append(f"- {name}")
        return "\n".join(lines)


def validate_state_concern(
    concern: str,
    records: Sequence[Mapping[str, Any]],
    *,
    symbols: StateSymbolTable | set[str] | Sequence[str] | None = None,
) -> None:
    """Validate a single concern of state_model against the declared symbol table."""
    if isinstance(symbols, StateSymbolTable):
        declared = symbols.declared_names
    elif isinstance(symbols, set):
        declared = symbols
    elif isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
        declared = set(symbols)
    else:
        declared = set()

    rows = records if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)) else ()

    if concern == "variables":
        seen: set[str] = set()
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            name = str(record.get("name") or "").strip()
            if re.fullmatch(_STATE_IDENTIFIER_PATTERN, name) is None:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_NAME: {name!r} is not a stable identifier"
                )
            if name in seen:
                raise ValueError(
                    f"STRUCTURED_STATE_VARIABLE_DUPLICATE: {name!r}"
                )
            seen.add(name)
        return

    if concern == "transitions":
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            guard_val = record.get("guard")
            if guard_val is not None:
                validate_state_expr_ir(guard_val, symbols=declared)
            mut_val = record.get("mutations") if "mutations" in record else record.get("mutation")
            if mut_val is not None:
                validate_mutation_ir(mut_val, symbols=declared)
        return

    if concern == "invariants":
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            cond_val = record.get("condition")
            if cond_val is not None:
                validate_state_expr_ir(cond_val, symbols=declared)
        return

    if concern in {"initialization", "updates", "cleanup"}:
        field = {
            "initialization": "initial_state",
            "updates": "mutation",
            "cleanup": "action",
        }[concern]
        for record in rows:
            if not isinstance(record, Mapping):
                continue
            mut_val = record.get("mutations") if "mutations" in record else record.get(field)
            if mut_val is not None:
                validate_mutation_ir(mut_val, symbols=declared)
        return


def validate_structured_state_section(section: Mapping[str, Any]) -> None:
    """Fail closed on canonical state semantics before any production code is generated."""

    raw_specification = section.get("specification")
    specification = (
        raw_specification if isinstance(raw_specification, Mapping) else section
    )
    variables = specification.get("variables", [])
    validate_state_concern("variables", variables)
    symbols = StateSymbolTable(variables)

    for concern in ("transitions", "invariants", "initialization", "updates", "cleanup", "concurrency"):
        records = specification.get(concern, [])
        if records:
            validate_state_concern(concern, records, symbols=symbols)


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

    def executable(index: int, record: Mapping[str, Any], field: str) -> str:
        del index
        if field in {"guard", "condition"}:
            return compile_state_expr_ir(record.get(field), declared=declared)
        mut_val = record.get("mutations") if "mutations" in record else record.get(field)
        return compile_mutation_ir(mut_val, declared=declared)

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
        for index, record in enumerate(records):
            guard = executable(index, record, "guard")
            mutation = executable(index, record, "mutations" if "mutations" in record else "mutation")
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
        for index, record in enumerate(records):
            condition = executable(index, record, "condition")
            lines.append(
                "    $mmmInvariants.add(new $mmmInvariant("
                f"context -> ({condition}), "
                f"{_java_string(record.get('enforcement', ''))}));"
            )
        lines.append("}")
        parts.append("\n".join(lines))
    elif concern == "initialization":
        lines = ["static {"]
        for index, record in enumerate(records):
            action = executable(index, record, "initial_state")
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
        for index, record in enumerate(records):
            action = executable(index, record, "mutation")
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
        for index, record in enumerate(records):
            action = executable(index, record, "action")
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
    "StateSymbolTable",
    "compile_mutation_ir",
    "compile_state_expr_ir",
    "constrain_state_chunk_schema",
    "constrain_state_record_schema",
    "has_complete_structured_state",
    "mutations_schema",
    "render_state_model_concern",
    "state_concern_schema",
    "state_expr_schema",
    "validate_mutation_ir",
    "validate_state_concern",
    "validate_state_expr_ir",
    "validate_state_expression",
    "validate_structured_state_section",
]
