"""Deterministic, host-owned Java syntax for incremental authoring.

Only expression operands cross the source boundary. Declarations and control-flow
syntax are assembled here; javac remains responsible for names, types and returns.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import re
from textwrap import dedent
from typing import Any

from .first_pass_java_declarations import declaration_header
from .java_region_parser import JavaRegionParseError, _class_body, _parse, _text

_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")
_RESERVED = frozenset((
    "abstract assert boolean break byte case catch char class const continue default "
    "do double else enum extends final finally float for goto if implements import "
    "instanceof int interface long native new package private protected public return "
    "short static strictfp super switch synchronized this throw throws transient try "
    "void volatile while true false null record sealed permits yield var _"
).split())
_COMMENTS = frozenset({"line_comment", "block_comment"})
_TYPES = frozenset({"class", "record", "enum"})
_ASSIGNMENTS = frozenset({"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>="})
_MEMBER_NODES = {"field": "field_declaration", "method": "method_declaration",
                 "constructor": "constructor_declaration", "class": "class_declaration",
                 "record": "record_declaration", "enum": "enum_declaration"}


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in _RESERVED:
        raise ValueError(f"INCREMENTAL_JAVA_IDENTIFIER_INVALID: {value!r}")
    return value


def _qualified_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("INCREMENTAL_JAVA_QUALIFIED_NAME_INVALID")
    for part in value.split("."):
        _identifier(part)
    return value


def _noncomments(node: Any) -> list[Any]:
    return [child for child in node.named_children if child.type not in _COMMENTS]


def _unicode_translation(value: str) -> str:
    """Apply Java's pre-lexing Unicode escapes before trusting parser boundaries.

    An escaped quote/newline in a literal or comment must not hide syntax from
    tree-sitter. Consecutive raw backslashes follow Java's eligibility parity;
    a backslash produced by an escape makes the following raw one eligible.
    """
    result: list[str] = []
    backslashes = 0
    escaped_backslash = False
    index = 0
    while index < len(value):
        char = value[index]
        eligible = backslashes % 2 == 0 or escaped_backslash
        if char == "\\" and eligible:
            end = index + 1
            while end < len(value) and value[end] == "u":
                end += 1
            digits = value[end:end + 4]
            if end > index + 1 and len(digits) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                char = chr(int(digits, 16))
                result.append(char)
                backslashes = backslashes + 1 if char == "\\" else 0
                escaped_backslash = char == "\\"
                index = end + 4
                continue
        result.append(char)
        backslashes = backslashes + 1 if char == "\\" else 0
        escaped_backslash = False
        index += 1
    # A supplementary Java character may be spelled as two surrogate escapes.
    return "".join(result).encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")


def _operand_text(value: Any, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"INCREMENTAL_JAVA_{kind}_EMPTY")
    return value.strip()


def _wrapper_field(source: str) -> tuple[bytes, Any, Any]:
    raw, root = _parse(source)
    declarations = _noncomments(root)
    if len(declarations) != 1 or declarations[0].type != "class_declaration":
        raise JavaRegionParseError("operand escaped the host class", category="scope")
    members = _noncomments(_class_body(root))
    if len(members) != 1 or members[0].type != "field_declaration":
        raise JavaRegionParseError("operand changed the host field", category="scope")
    fields = [child for child in members[0].named_children if child.type == "variable_declarator"]
    if len(fields) != 1 or _text(raw, fields[0].child_by_field_name("name")) != "value":
        raise JavaRegionParseError("operand changed the host variable", category="scope")
    return raw, members[0], fields[0]


def _expression_node(text: str, *, allow_update: bool) -> tuple[str, bytes, Any]:
    original = _operand_text(text, "EXPRESSION")
    translated = _unicode_translation(original)
    wrapped = "(\n" + translated + "\n)"
    try:
        raw, _, variable = _wrapper_field("class MmmOperand { Object value = " + wrapped + "; }\n")
    except UnicodeError as exc:
        raise ValueError("INCREMENTAL_JAVA_EXPRESSION_INVALID_UNICODE") from exc
    value = variable.child_by_field_name("value")
    if value is None or value.type != "parenthesized_expression" or _text(raw, value) != wrapped:
        raise JavaRegionParseError("expression escaped its host parentheses", category="scope")
    expressions = _noncomments(value)
    if len(expressions) != 1:
        raise JavaRegionParseError("operand must be exactly one expression", category="scope")
    stack = [expressions[0]]
    while stack:
        child = stack.pop()
        if child.type in {"lambda_expression", "class_body", "block", "switch_expression"} or child.type.endswith("_statement"):
            raise JavaRegionParseError("expression contains an authored body or statement", category="scope")
        if not allow_update and child.type in {"assignment_expression", "update_expression"}:
            raise JavaRegionParseError("updates require an explicit assignment or update operation", category="scope")
        stack.extend(child.named_children)
    return original, raw, expressions[0]


def validate_expression(text: str, *, allow_update: bool = False) -> str:
    """Return one Java expression, rejecting statements, lambdas and class bodies."""
    return _expression_node(text, allow_update=allow_update)[0]


def _reject_bodies(node: Any, *, skip: Any = None) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current == skip:
            continue
        if current.type in {"lambda_expression", "class_body", "block", "switch_expression"} or current.type.endswith("_statement"):
            raise JavaRegionParseError("operand contains an authored body or statement", category="scope")
        stack.extend(current.named_children)


def _catch_parameter(translated: str) -> tuple[bytes, Any]:
    raw, root = _parse("class MmmType { void method() { try {} catch (" + translated + " value) {} } }")
    members = _noncomments(_class_body(root))
    if len(_noncomments(root)) != 1 or len(members) != 1 or members[0].type != "method_declaration":
        raise JavaRegionParseError("catch type escaped its method", category="scope")
    statements = _noncomments(members[0].child_by_field_name("body"))
    if len(statements) != 1 or statements[0].type != "try_statement":
        raise JavaRegionParseError("catch type changed the host try", category="scope")
    children = _noncomments(statements[0])
    if len(children) != 2 or children[0].type != "block" or _noncomments(children[0]) or children[1].type != "catch_clause":
        raise JavaRegionParseError("catch type changed the host catch", category="scope")
    catch = children[1]
    parameters = [child for child in catch.named_children if child.type == "catch_formal_parameter"]
    if len(parameters) != 1 or _noncomments(catch.child_by_field_name("body")):
        raise JavaRegionParseError("catch type supplied an implementation", category="scope")
    return raw, parameters[0]


def _union_type(translated: str) -> None:
    raw, parameter = _catch_parameter(translated)
    types = [child for child in parameter.named_children if child.type == "catch_type"]
    if (len(types) != 1 or _text(raw, types[0]) != translated
            or _text(raw, parameter.child_by_field_name("name")) != "value"):
        raise JavaRegionParseError("catch type changed its parameter", category="scope")
    _reject_bodies(parameter)


def _varargs_type(translated: str) -> None:
    raw, root = _parse("class MmmType { void method(" + translated + " value) {} }")
    members = _noncomments(_class_body(root))
    if len(_noncomments(root)) != 1 or len(members) != 1 or members[0].type != "method_declaration":
        raise JavaRegionParseError("type escaped its parameter", category="scope")
    children = _noncomments(members[0].child_by_field_name("parameters"))
    if len(children) != 1 or children[0].type != "spread_parameter" or _text(raw, children[0]) != translated + " value":
        raise JavaRegionParseError("type changed its parameter", category="scope")
    _reject_bodies(children[0])


def _annotated_type(raw: bytes, field: Any, type_node: Any, translated: str) -> bool:
    modifiers = next((child for child in field.named_children if child.type == "modifiers"), None)
    if modifiers is None or type_node is None:
        return False
    annotations = {"annotation", "marker_annotation"}
    return (all(child.type in annotations | _COMMENTS for child in modifiers.children)
            and raw[modifiers.start_byte:type_node.end_byte].decode("utf-8") == translated)


def _field_type(translated: str) -> None:
    raw, field, variable = _wrapper_field("class MmmType { " + translated + " value; }\n")
    type_node = field.child_by_field_name("type")
    if (type_node is None or (_text(raw, type_node) != translated and not _annotated_type(raw, field, type_node, translated))
            or variable.child_by_field_name("value") is not None):
        raise JavaRegionParseError("type changed the host declaration", category="scope")
    _reject_bodies(field)
    # Diamond syntax belongs to constructor creation, never a type slot.
    stack = [type_node]
    while stack:
        node = stack.pop()
        if node.type == "type_arguments" and not _noncomments(node):
            raise JavaRegionParseError("type arguments must not be empty")
        stack.extend(node.named_children)


def validate_type(text: str) -> str:
    """Return a parser-backed Java type, including generics, void and catch unions."""
    original = _operand_text(text, "TYPE")
    translated = _unicode_translation(original)
    if translated == "void":
        _parse("class MmmType { void value() {} }")
        return original
    if translated.endswith("..."):
        _varargs_type(translated)
        return original
    try:
        _field_type(translated)
    except JavaRegionParseError:
        if "|" not in translated:
            raise
        _union_type(translated)
    except UnicodeError as exc:
        raise ValueError("INCREMENTAL_JAVA_TYPE_INVALID_UNICODE") from exc
    return original


def validate_value_type(text: str, *, allow_void: bool = False, allow_varargs: bool = False,
                        allow_union: bool = False) -> str:
    """Validate a type in its operand position before asking for any source."""
    value = validate_type(text)
    translated = _unicode_translation(value)
    if allow_void and translated == "void":
        return value
    if allow_varargs and translated.endswith("..."):
        return value
    if allow_union and "|" in translated:
        return value
    if translated == "void":
        raise ValueError("void is a method return type, not a value or parameter type")
    _field_type(translated)
    return value


def _parameters(parameters: Any) -> list[Mapping[str, Any]]:
    if not isinstance(parameters, list):
        raise ValueError("INCREMENTAL_JAVA_PARAMETERS_REQUIRE_ARRAY")
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            raise ValueError("INCREMENTAL_JAVA_PARAMETER_INVALID")
        validate_type(parameter["type"])
        _identifier(parameter["name"])
    return parameters


def _metadata(member: Mapping[str, Any]) -> Mapping[str, Any]:
    parts = member.get("metadata", {})
    if not isinstance(parts, Mapping):
        raise ValueError("INCREMENTAL_JAVA_METADATA_INVALID")
    if any(parts.get(key) for key in ("source", "body", "initializer")):
        raise ValueError("INCREMENTAL_JAVA_METADATA_CANNOT_CONTAIN_IMPLEMENTATION")
    if parts.get("visibility", "") not in {"", "public", "protected", "private"}:
        raise ValueError("INCREMENTAL_JAVA_VISIBILITY_INVALID")
    for flag in ("static", "final"):
        if flag in parts and not isinstance(parts[flag], bool):
            raise ValueError("INCREMENTAL_JAVA_MODIFIER_INVALID")
    return parts


def _type_metadata(kind: str, parts: Mapping[str, Any]) -> None:
    if any(parts.get(key) for key in ("fields", "constructors", "methods")):
        raise ValueError("INCREMENTAL_JAVA_NESTED_MEMBERS_MUST_BE_FLAT")
    for value in parts.get("interfaces", []):
        validate_type(value)
    if parts.get("superclass"):
        validate_type(parts["superclass"])
    if kind == "record":
        _parameters(parts.get("components", []))


def _signature_metadata(kind: str, name: str, parts: Mapping[str, Any], owner: str) -> None:
    if kind == "field":
        validate_type(parts["type"])
    elif kind in {"method", "constructor"}:
        _parameters(parts.get("parameters", []))
        if kind == "method":
            validate_type(parts["return_type"])
        elif name != owner:
            raise ValueError("INCREMENTAL_JAVA_CONSTRUCTOR_OWNER_MISMATCH")
        for value in parts.get("throws", []):
            validate_type(value)
    else:
        _type_metadata(kind, parts)


def _header_identity(raw: bytes, node: Any, kind: str, name: str) -> None:
    actual_name = node.child_by_field_name("name")
    if kind == "field":
        variables = [child for child in node.named_children if child.type == "variable_declarator"]
        if len(variables) != 1 or variables[0].child_by_field_name("value") is not None:
            raise JavaRegionParseError("metadata supplied a field implementation", category="scope")
        actual_name = variables[0].child_by_field_name("name")
    else:
        body = node.child_by_field_name("body")
        if body is None or _noncomments(body):
            raise JavaRegionParseError("metadata supplied a declaration body", category="scope")
    if actual_name is None or _text(raw, actual_name) != name:
        raise JavaRegionParseError("metadata changed the host identifier", category="scope")
    _reject_bodies(node, skip=node.child_by_field_name("body"))


def _validate_header(header: str, kind: str, name: str, owner: str) -> None:
    suffix = ";" if kind == "field" else " {}"
    raw, root = _parse(_unicode_translation("class " + owner + " {\n" + header + suffix + "\n}"))
    members = _noncomments(_class_body(root))
    if len(_noncomments(root)) != 1 or len(members) != 1 or members[0].type != _MEMBER_NODES[kind]:
        raise JavaRegionParseError("metadata changed its declaration shape", category="scope")
    _header_identity(raw, members[0], kind, name)


def _header(member: Mapping[str, Any], owner: str) -> str:
    kind, name = member["kind"], _identifier(member["name"])
    if kind not in _MEMBER_NODES:
        raise ValueError(f"INCREMENTAL_JAVA_MEMBER_KIND_INVALID: {kind}")
    parts = _metadata(member)
    _signature_metadata(kind, name, parts, owner)
    header = declaration_header(kind, name, parts)
    if kind in {"class", "record"} and parts.get("type_parameters"):
        marker = kind + " " + name
        header = header.replace(marker, marker + "<" + parts["type_parameters"] + ">", 1)
    _validate_header(header, kind, name, owner)
    return header


class _Renderer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.line_map: dict[int, dict[str, str]] = {}

    def emit(self, text: str, indent: int, member_id: str = "", node_id: str = "", slot: str = "declaration") -> None:
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            self.lines.append("    " * indent + line if line else "")
            if member_id:
                self.line_map[len(self.lines)] = {"member_id": member_id, "node_id": node_id, "slot": slot}

    def expression(self, value: str, indent: int, member_id: str, node_id: str, slot: str,
                   *, allow_update: bool = False) -> None:
        self.emit(validate_expression(value, allow_update=allow_update), indent, member_id, node_id, slot)

    def body(self, nodes: Any, indent: int, member_id: str) -> None:
        if not isinstance(nodes, list):
            raise ValueError("INCREMENTAL_JAVA_BODY_REQUIRES_OPERATION_ARRAY")
        for node in nodes:
            self.node(node, indent, member_id)

    def node(self, node: Mapping[str, Any], indent: int, member_id: str) -> None:
        _Operation(self, node, indent, member_id).render()


class _Operation:
    def __init__(self, renderer: _Renderer, node: Mapping[str, Any], indent: int, member_id: str) -> None:
        if not isinstance(node, Mapping):
            raise ValueError("INCREMENTAL_JAVA_OPERATION_INVALID")
        self.renderer, self.node, self.indent, self.member_id = renderer, node, indent, member_id
        self.node_id = _operand_text(node.get("id"), "NODE_ID")

    def emit(self, text: str, slot: str = "operation", *, extra: int = 0) -> None:
        self.renderer.emit(text, self.indent + extra, self.member_id, self.node_id, slot)

    def expr(self, key: str, *, extra: int = 1, allow_update: bool = False) -> None:
        self.renderer.expression(self.node[key], self.indent + extra, self.member_id, self.node_id, key,
                                 allow_update=allow_update)

    def body(self, key: str = "body") -> None:
        self.renderer.body(self.node.get(key, []), self.indent + 1, self.member_id)

    def arguments(self, *, extra: int = 1) -> None:
        arguments = self.node.get("arguments", [])
        for index, argument in enumerate(arguments):
            self.renderer.expression(argument, self.indent + extra, self.member_id, self.node_id, f"argument:{index}")
            if index < len(arguments) - 1:
                self.emit(",", f"argument:{index}", extra=extra)

    def render(self) -> None:
        handlers = {
            "local": self.local, "assign": self.assign, "call": self.call,
            "if": self.conditional, "while": self.conditional, "do": self.do,
            "foreach": self.foreach, "for": self.for_loop,
            "return": self.terminal, "throw": self.terminal,
            "break": self.jump, "continue": self.jump,
            "block": self.block, "synchronized": self.synchronized,
            "switch": self.switch, "try": self.try_block, "lambda": self.lambda_body,
            "register": self.register, "constructor_call": self.constructor_call,
        }
        handler = handlers.get(self.node.get("op"))
        if handler is None:
            raise ValueError(f"INCREMENTAL_JAVA_OPERATION_UNKNOWN: {self.node.get('op')!r}")
        handler()

    def local(self) -> None:
        self.emit(validate_type(self.node["type"]) + " " + _identifier(self.node["name"]) + " =", "type")
        self.expr("value")
        self.emit(";", "value")

    def assign(self) -> None:
        if self.node["operator"] not in _ASSIGNMENTS:
            raise ValueError("INCREMENTAL_JAVA_ASSIGNMENT_OPERATOR_INVALID")
        self.expr("target", extra=0)
        self.emit(self.node["operator"], "operator", extra=1)
        self.expr("value")
        self.emit(";", "value")

    def receiver(self, receiver: str) -> None:
        if receiver == "super":
            receiver_kind = "super"
        else:
            receiver, _, receiver_node = _expression_node(receiver, allow_update=False)
            receiver_kind = receiver_node.type
        # A name may denote a static class. Other receivers need parentheses
        # so a cast or conditional retains its authored precedence.
        if receiver_kind in {"identifier", "field_access", "super", "this"}:
            self.emit(receiver, "receiver", extra=1)
        else:
            self.emit("(", "receiver", extra=1)
            self.emit(receiver, "receiver", extra=2)
            self.emit(")", "receiver", extra=1)

    def call(self) -> None:
        if self.node.get("name"):
            _identifier(self.node["name"])
        if self.node.get("result_type"):
            self.emit(validate_type(self.node["result_type"]) + " " + _identifier(self.node["name"]) + " =", "result_type")
        receiver = self.node.get("receiver", "")
        method = _identifier(self.node["method"])
        if receiver:
            self.receiver(receiver)
            method = "." + method
        self.emit(method + "(", "method", extra=1)
        self.arguments(extra=2)
        self.emit(");")

    def conditional(self) -> None:
        self.emit(self.node["op"] + " (")
        self.expr("condition")
        self.emit(") {", "condition")
        self.body()
        if self.node["op"] == "if" and self.node.get("otherwise"):
            self.emit("} else {")
            self.body("otherwise")
        self.emit("}")

    def do(self) -> None:
        self.emit("do {")
        self.body()
        self.emit("} while (")
        self.expr("condition")
        self.emit(");", "condition")

    def foreach(self) -> None:
        self.emit("for (" + validate_type(self.node["type"]) + " " + _identifier(self.node["name"]) + " :", "type")
        self.expr("iterable")
        self.emit(") {", "iterable")
        self.body()
        self.emit("}")

    def for_loop(self) -> None:
        self.emit("for (" + validate_type(self.node["type"]) + " " + _identifier(self.node["name"]) + " =", "type")
        self.expr("value")
        self.emit(";", "value", extra=1)
        self.expr("condition")
        self.emit(";", "condition", extra=1)
        self.expr("update", allow_update=True)
        self.emit(") {", "update")
        self.body()
        self.emit("}")

    def terminal(self) -> None:
        op = self.node["op"]
        self.emit(op)
        if self.node.get("value", ""):
            self.expr("value")
        elif op == "throw":
            raise ValueError("INCREMENTAL_JAVA_THROW_REQUIRES_VALUE")
        self.emit(";", "value")

    def jump(self) -> None:
        self.emit(self.node["op"] + ";")

    def block(self) -> None:
        self.emit("{")
        self.body()
        self.emit("}")

    def synchronized(self) -> None:
        self.emit("synchronized (")
        self.expr("monitor")
        self.emit(") {", "monitor")
        self.body()
        self.emit("}")

    def switch(self) -> None:
        self.emit("switch (")
        self.expr("selector")
        self.emit(") {", "selector")
        for index, case in enumerate(self.node.get("cases", [])):
            label = case.get("label", "")
            slot = f"case:{index}:label"
            if label:
                self.emit("case", slot, extra=1)
                self.renderer.expression(label, self.indent + 2, self.member_id, self.node_id, slot)
                self.emit(":", slot, extra=1)
            else:
                self.emit("default:", slot, extra=1)
            self.renderer.body(case.get("body", []), self.indent + 2, self.member_id)
        self.emit("}")

    def try_block(self) -> None:
        self.emit("try {")
        self.body()
        catches = self.node.get("catches", [])
        for index, catch in enumerate(catches):
            type_name = validate_type(catch["type"])
            self.emit("} catch (" + type_name + " " + _identifier(catch["name"]) + ") {", f"catch:{index}:type")
            self.renderer.body(catch.get("body", []), self.indent + 1, self.member_id)
        if self.node.get("finally") or not catches:
            self.emit("} finally {")
            self.body("finally")
        self.emit("}")

    def lambda_body(self) -> None:
        target_type = validate_type(self.node["type"])
        name = _identifier(self.node["name"])
        validate_type(self.node.get("return_type", "void"))
        self.emit(target_type + " " + name + " = (", "type")
        parameters = self.node.get("parameters", [])
        for index, parameter in enumerate(parameters):
            parameter_type = parameter.get("type", "")
            prefix = validate_type(parameter_type) + " " if parameter_type else ""
            self.emit(prefix + _identifier(parameter["name"]) + ("," if index < len(parameters) - 1 else ""),
                      f"parameter:{index}:type", extra=1)
        self.emit(") -> {")
        self.body()
        self.emit("};")

    def register(self) -> None:
        event_owner = _qualified_name(self.node["event_owner"])
        event_field = _identifier(self.node["event_field"])
        handler = _identifier(self.node["handler"])
        self.emit(event_owner, "event_owner")
        self.emit("." + event_field, "event_field", extra=1)
        self.emit(".register(this::" + handler + ");", "handler", extra=1)

    def constructor_call(self) -> None:
        target = self.node["target"]
        if target not in {"super", "this"}:
            raise ValueError("INCREMENTAL_JAVA_CONSTRUCTOR_TARGET_INVALID")
        self.emit(target + "(", "target")
        self.arguments()
        self.emit(");")


def _enum_arguments(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(validate_expression(item) for item in value)
    if not isinstance(value, str):
        raise ValueError("INCREMENTAL_JAVA_ENUM_ARGUMENTS_INVALID")
    expression, raw, node = _expression_node("MmmArguments(" + value + ")", allow_update=False)
    name = node.child_by_field_name("name")
    arguments = node.child_by_field_name("arguments")
    if (node.type != "method_invocation" or name is None or _text(raw, name) != "MmmArguments"
            or arguments is None or _text(raw, arguments) != "(" + _unicode_translation(value) + ")"):
        raise JavaRegionParseError("enum arguments escaped their host call", category="scope")
    return expression[len("MmmArguments("):-1]


def _member_groups(members: Any, class_name: str) -> tuple[dict[str, list[Mapping[str, Any]]], set[str]]:
    if not isinstance(members, list):
        raise ValueError("INCREMENTAL_JAVA_MEMBERS_REQUIRE_ARRAY")
    by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    type_names: set[str] = {class_name}
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("INCREMENTAL_JAVA_MEMBER_INVALID")
        member_id = _operand_text(member.get("id"), "MEMBER_ID")
        if member_id in ids:
            raise ValueError("INCREMENTAL_JAVA_DUPLICATE_MEMBER_ID: " + member_id)
        ids.add(member_id)
        name = _identifier(member["name"])
        owner = _identifier(member.get("owner") or class_name)
        by_owner[owner].append(member)
        if member["kind"] in _TYPES:
            if name in type_names:
                raise ValueError("INCREMENTAL_JAVA_AMBIGUOUS_TYPE_OWNER: " + name)
            type_names.add(name)
    unknown = set(by_owner) - type_names
    if unknown:
        raise ValueError("INCREMENTAL_JAVA_UNKNOWN_MEMBER_OWNER: " + ", ".join(sorted(unknown)))
    return by_owner, ids


class _ProgramRenderer(_Renderer):
    def __init__(self, by_owner: Mapping[str, list[Mapping[str, Any]]]) -> None:
        super().__init__()
        self.by_owner = by_owner
        self.rendered: set[str] = set()

    def lifecycle(self, class_name: str, lifecycle: str) -> None:
        self.emit("@Override", 1)
        self.emit("public void " + lifecycle + "() {", 1)
        for member in self.by_owner[class_name]:
            parts = _metadata(member)
            registrar = member.get("registrar") is True or parts.get("host_registrar") is True
            if member.get("kind") == "method" and member.get("behavior") and registrar:
                self.emit("this." + _identifier(member["name"]) + "();", 2, member["id"], "", "registrar")
        self.emit("}", 1)

    def members(self, owner: str, indent: int, ancestors: frozenset[str]) -> None:
        if owner in ancestors:
            raise ValueError("INCREMENTAL_JAVA_TYPE_OWNER_CYCLE: " + owner)
        for member in self.by_owner[owner]:
            self.member(member, owner, indent, ancestors)
            self.rendered.add(member["id"])

    def member(self, member: Mapping[str, Any], owner: str, indent: int, ancestors: frozenset[str]) -> None:
        kind = member["kind"]
        header = _header(member, owner)
        self.emit("", indent)
        if kind == "field":
            self.field(member, header, indent)
        elif kind in {"method", "constructor"}:
            self.callable(member, header, indent)
        else:
            self.nested_type(member, header, indent, ancestors | {owner})

    def field(self, member: Mapping[str, Any], header: str, indent: int) -> None:
        member_id = member["id"]
        initializer = member.get("initializer", "")
        self.emit(header + (" =" if initializer else ";"), indent, member_id)
        if initializer:
            self.expression(initializer, indent + 1, member_id, "", "initializer")
            self.emit(";", indent, member_id, "", "initializer")

    def callable(self, member: Mapping[str, Any], header: str, indent: int) -> None:
        member_id = member["id"]
        self.emit(header + " {", indent, member_id)
        self.body(member.get("body", []), indent + 1, member_id)
        self.emit("}", indent, member_id)

    def enum_constants(self, member: Mapping[str, Any], indent: int) -> None:
        constants = member.get("metadata", {}).get("constants", [])
        for index, constant in enumerate(constants):
            arguments = _enum_arguments(constant.get("arguments", ""))
            text = _identifier(constant["name"]) + ("(" + arguments + ")" if arguments else "")
            self.emit(text + ("," if index < len(constants) - 1 else ";"), indent, member["id"])
        if not constants:
            self.emit(";", indent, member["id"])

    def nested_type(self, member: Mapping[str, Any], header: str, indent: int, ancestors: frozenset[str]) -> None:
        if member.get("body"):
            raise ValueError("INCREMENTAL_JAVA_TYPE_BODY_MUST_USE_OWNED_DECLARATIONS")
        self.emit(header + " {", indent, member["id"])
        if member["kind"] == "enum":
            self.enum_constants(member, indent + 1)
        self.members(member["name"], indent + 1, ancestors)
        self.emit("}", indent, member["id"])


def render_program(program: Mapping[str, Any]) -> tuple[str, dict[int, dict[str, str]]]:
    """Render a Program v1 without gameplay admission gates or source fallbacks."""
    if not isinstance(program, Mapping) or program.get("schema") != "mmm/incremental-java-v1":
        raise ValueError("INCREMENTAL_JAVA_PROGRAM_SCHEMA_INVALID")
    class_name = _identifier(program["class_name"])
    package = program.get("package", "")
    if package:
        _qualified_name(package)
    interface = validate_type(program["interface"])
    lifecycle = _identifier(program["lifecycle"])
    by_owner, ids = _member_groups(program.get("members", []), class_name)
    renderer = _ProgramRenderer(by_owner)
    if package:
        renderer.emit("package " + package + ";", 0)
        renderer.emit("", 0)
    renderer.emit("public final class " + class_name + " implements " + interface + " {", 0)
    renderer.lifecycle(class_name, lifecycle)
    renderer.members(class_name, 1, frozenset())
    if renderer.rendered != ids:
        raise ValueError("INCREMENTAL_JAVA_UNREACHABLE_MEMBER_OWNER")
    renderer.emit("}", 0)
    return "\n".join(renderer.lines) + "\n", renderer.line_map


def render_member_fragment(program: Mapping[str, Any], member_id: str) -> str:
    """Return one owned declaration, retaining overload identity from its id."""
    source, line_map = render_program(program)
    rows = [line for line, location in line_map.items()
            if location["member_id"] == member_id and location["slot"] != "registrar"]
    if not rows:
        raise ValueError("INCREMENTAL_JAVA_MEMBER_NOT_FOUND: " + member_id)
    return dedent("\n".join(source.split("\n")[min(rows) - 1:max(rows)])) + "\n"
