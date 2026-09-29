"""Tree-sitter-backed Java region admission for production generation.

Java syntax belongs to a real parser, not regexes or model-authored scalar AST slots.
The host still owns policy: concern regions may contribute class members, but never
an outer class constructor, lifecycle initializer, package/import wrapper, or
non-private nested type.
"""
from __future__ import annotations

from threading import local
from typing import Any

import tree_sitter_java
from markdown_it import MarkdownIt
from tree_sitter import Language, Parser


class JavaRegionParseError(ValueError):
    """A Java region could not be safely admitted into the host-owned scaffold."""


_JAVA = Language(tree_sitter_java.language())
_THREAD = local()
_NESTED_TYPES = frozenset(
    {
        "annotation_type_declaration",
        "class_declaration",
        "enum_declaration",
        "interface_declaration",
        "record_declaration",
    }
)
_MEMBER_TYPES = frozenset({"field_declaration", "method_declaration"}) | _NESTED_TYPES
_HOST_OWNED_MEMBER_TYPES = frozenset(
    {"block", "constructor_declaration", "compact_constructor_declaration", "static_initializer"}
)
_COMMENT_TYPES = frozenset({"line_comment", "block_comment"})


def _parser() -> Parser:
    parser = getattr(_THREAD, "java_parser", None)
    if parser is None:
        parser = Parser(_JAVA)
        _THREAD.java_parser = parser
    return parser


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _first_error(node: Any, source: bytes) -> str:
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        snippet = _text(source, node).strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        row, column = node.start_point
        kind = "missing syntax" if getattr(node, "is_missing", False) else "syntax error"
        return f"{kind} at {row + 1}:{column + 1}: {snippet!r}"
    for child in node.children:
        issue = _first_error(child, source)
        if issue:
            return issue
    return ""


def _parse(source_text: str) -> tuple[bytes, Any]:
    source = source_text.encode("utf-8")
    tree = _parser().parse(source)
    issue = _first_error(tree.root_node, source) if tree.root_node.has_error else ""
    if issue:
        raise JavaRegionParseError(issue)
    return source, tree.root_node


def _modifiers(node: Any, source: bytes) -> frozenset[str]:
    modifiers = next(
        (child for child in node.named_children if child.type == "modifiers"),
        None,
    )
    if modifiers is None:
        return frozenset()
    return frozenset(_text(source, modifiers).replace("\n", " ").split())


def _name(node: Any, source: bytes) -> str:
    name = node.child_by_field_name("name")
    return _text(source, name).strip() if name is not None else ""


def _is_host_initialize(node: Any, source: bytes) -> bool:
    if node.type != "method_declaration" or _name(node, source) != "initialize":
        return False
    return_type = node.child_by_field_name("type")
    parameters = node.child_by_field_name("parameters")
    return (
        return_type is not None
        and _text(source, return_type).strip() == "void"
        and parameters is not None
        and _text(source, parameters).strip() == "()"
        and "static" in _modifiers(node, source)
    )


def _validate_member_node(node: Any, source: bytes) -> None:
    if node.type in _HOST_OWNED_MEMBER_TYPES:
        raise JavaRegionParseError(
            f"host-owned lifecycle member {node.type!r} is not admissible in a concern region"
        )
    if node.type not in _MEMBER_TYPES:
        raise JavaRegionParseError(
            f"Java class-body node {node.type!r} is not an admissible concern member"
        )
    if node.type in _NESTED_TYPES and "private" not in _modifiers(node, source):
        raise JavaRegionParseError(
            f"nested type {_name(node, source)!r} must be private because outer type ownership is host-owned"
        )
    if _is_host_initialize(node, source):
        raise JavaRegionParseError(
            "static void initialize() is host-owned lifecycle structure and cannot be declared by a concern"
        )


def _class_body(root: Any) -> Any:
    declaration = next(
        (child for child in root.named_children if child.type == "class_declaration"),
        None,
    )
    if declaration is None:
        raise JavaRegionParseError("Tree-sitter did not produce the host wrapper class")
    body = declaration.child_by_field_name("body")
    if body is None:
        raise JavaRegionParseError("Tree-sitter did not produce the host wrapper class body")
    return body


def _chunks_from_body(body: Any, source: bytes, *, drop_host_lifecycle: bool) -> tuple[str, ...]:
    chunks: list[str] = []
    for node in body.named_children:
        if node.type in _COMMENT_TYPES:
            continue
        if drop_host_lifecycle and node.type in _HOST_OWNED_MEMBER_TYPES:
            continue
        if drop_host_lifecycle and _is_host_initialize(node, source):
            continue
        _validate_member_node(node, source)
        chunks.append(_text(source, node).strip())
    return tuple(chunk for chunk in chunks if chunk)


def class_body_chunks(value: str) -> tuple[str, ...]:
    """Split a syntactically valid Java class body without applying ownership policy."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    return tuple(
        _text(source, node).strip()
        for node in body.named_children
        if node.type not in _COMMENT_TYPES and _text(source, node).strip()
    )


def class_body_member_kinds(value: str) -> tuple[str, ...]:
    """Return Tree-sitter node kinds for one syntactically valid Java class-body region."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    return tuple(
        node.type
        for node in body.named_children
        if node.type not in _COMMENT_TYPES and _text(source, node).strip()
    )


def strict_member_chunks(value: str) -> tuple[str, ...]:
    """Parse model-authored class-body members and enforce concern ownership policy."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    return _chunks_from_body(body, source, drop_host_lifecycle=False)


def _parameter_contracts(parameters: Any, source: bytes) -> tuple[dict[str, str], ...]:
    if parameters is None:
        return ()
    rows: list[dict[str, str]] = []
    for node in parameters.named_children:
        if node.type not in {
            "formal_parameter",
            "spread_parameter",
            "receiver_parameter",
        }:
            continue
        type_node = node.child_by_field_name("type")
        name_node = node.child_by_field_name("name")
        if type_node is None:
            continue
        java_type = _text(source, type_node).strip()
        if node.type == "spread_parameter" and not java_type.endswith("..."):
            java_type += "..."
        rows.append(
            {
                "type": java_type,
                "name": _text(source, name_node).strip() if name_node is not None else "",
            }
        )
    return tuple(rows)


def _node_contracts(node: Any, source: bytes) -> tuple[dict[str, Any], ...]:
    modifiers = _modifiers(node, source)
    visibility = next(
        (item for item in ("public", "protected", "private") if item in modifiers),
        "package",
    )
    common = {
        "visibility": visibility,
        "static": "static" in modifiers,
        "final": "final" in modifiers,
    }

    if node.type == "method_declaration":
        type_node = node.child_by_field_name("type")
        parameters = node.child_by_field_name("parameters")
        name = _name(node, source)
        if not name or type_node is None:
            return ()
        return (
            {
                **common,
                "kind": "method",
                "symbol": name,
                "return_type": _text(source, type_node).strip(),
                "parameters": list(_parameter_contracts(parameters, source)),
                "declaration": _text(source, node).strip(),
            },
        )

    if node.type == "constructor_declaration":
        parameters = node.child_by_field_name("parameters")
        name = _name(node, source)
        if not name:
            return ()
        return (
            {
                **common,
                "kind": "constructor",
                "symbol": name,
                "parameters": list(_parameter_contracts(parameters, source)),
                "declaration": _text(source, node).strip(),
            },
        )

    if node.type == "field_declaration":
        type_node = node.child_by_field_name("type")
        if type_node is None:
            return ()
        java_type = _text(source, type_node).strip()
        rows: list[dict[str, Any]] = []
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            rows.append(
                {
                    **common,
                    "kind": "field",
                    "symbol": _text(source, name_node).strip(),
                    "declared_type": java_type,
                    "mutable": "final" not in modifiers,
                    "initialized": child.child_by_field_name("value") is not None,
                    "declaration": _text(source, node).strip(),
                }
            )
        return tuple(rows)

    if node.type in _NESTED_TYPES:
        name = _name(node, source)
        return (
            {
                **common,
                "kind": "type",
                "symbol": name,
                "declaration_kind": node.type,
                "declaration": _text(source, node).strip(),
            },
        ) if name else ()

    return ()


def class_body_member_contracts(value: str) -> tuple[dict[str, Any], ...]:
    """Return Tree-sitter-derived declaration contracts for one Java class body."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    rows: list[dict[str, Any]] = []
    for node in body.named_children:
        if node.type in _COMMENT_TYPES:
            continue
        rows.extend(_node_contracts(node, source))
    return tuple(rows)


def public_source_member_contracts(value: str) -> tuple[dict[str, Any], ...]:
    """Extract public/protected API contracts from one complete Java source unit.

    Tree-sitter owns syntax/declaration structure. This is intentionally not a
    symbol resolver; project/JDK binding still belongs to javac/JDT.
    """
    text = str(value or "").strip()
    if not text:
        return ()
    source, root = _parse(text)
    declaration = next(
        (child for child in root.named_children if child.type in _NESTED_TYPES),
        None,
    )
    if declaration is None:
        return ()
    body = declaration.child_by_field_name("body")
    if body is None:
        return ()
    rows: list[dict[str, Any]] = []
    for node in body.named_children:
        if node.type in _COMMENT_TYPES:
            continue
        for contract in _node_contracts(node, source):
            if contract.get("visibility") in {"public", "protected"}:
                rows.append(contract)
    return tuple(rows)


def _walk_named(node: Any):
    yield node
    for child in node.named_children:
        yield from _walk_named(child)


def class_body_assignment_targets(value: str) -> tuple[str, ...]:
    """Return simple identifiers assigned/updated by executable class-body code."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    targets: list[str] = []
    for node in _walk_named(body):
        target = None
        if node.type == "assignment_expression":
            target = node.child_by_field_name("left")
        elif node.type == "update_expression":
            target = node.child_by_field_name("operand")
            if target is None and node.named_children:
                target = node.named_children[0]
        if target is None:
            continue
        rendered = _text(source, target).strip()
        if rendered.isidentifier() and rendered not in targets:
            targets.append(rendered)
    return tuple(targets)


def _method_invocation_contract(node: Any, source: bytes) -> dict[str, Any] | None:
    if node.type != "method_invocation":
        return None
    name = node.child_by_field_name("name")
    arguments = node.child_by_field_name("arguments")
    receiver = node.child_by_field_name("object")
    if name is None:
        return None
    return {
        "receiver": _text(source, receiver).strip() if receiver is not None else "",
        "symbol": _text(source, name).strip(),
        "argument_count": (
            len(arguments.named_children) if arguments is not None else 0
        ),
    }


def class_body_method_invocations(value: str) -> tuple[dict[str, Any], ...]:
    """Return receiver/name/arity for method invocations in a Java class body."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    rows: list[dict[str, Any]] = []
    for node in _walk_named(body):
        row = _method_invocation_contract(node, source)
        if row is not None:
            rows.append(row)
    return tuple(rows)


def class_body_direct_return_calls(value: str) -> tuple[dict[str, Any], ...]:
    """Return direct method calls used as return expressions with enclosing return type.

    This covers the high-value unsafe shape where a method directly returns an API
    call. Complex expressions are intentionally left to javac/JDT.
    """
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    rows: list[dict[str, Any]] = []
    for method in body.named_children:
        if method.type != "method_declaration":
            continue
        method_name = method.child_by_field_name("name")
        return_type = method.child_by_field_name("type")
        method_body = method.child_by_field_name("body")
        if method_name is None or return_type is None or method_body is None:
            continue
        for node in _walk_named(method_body):
            if node.type != "return_statement":
                continue
            expression = next(iter(node.named_children), None)
            if expression is None or expression.type != "method_invocation":
                continue
            call = _method_invocation_contract(expression, source)
            if call is None:
                continue
            rows.append(
                {
                    "method": _text(source, method_name).strip(),
                    "declared_return_type": _text(source, return_type).strip(),
                    **call,
                }
            )
    return tuple(rows)

def _outer_type_candidates(node: Any) -> tuple[Any, ...]:
    """Find outermost Java type declarations even when Tree-sitter nests them in ERROR."""
    found: list[Any] = []

    def visit(current: Any) -> None:
        if current.type in _NESTED_TYPES:
            found.append(current)
            return
        for child in current.named_children:
            visit(child)

    visit(node)
    return tuple(found)


def _unwrap_single_outer_class(value: str) -> tuple[str, ...]:
    """Salvage one structurally sound outer class from a noisy model envelope.

    Tree-sitter is intentionally error-tolerant. Noise outside the one class
    declaration (for example prose that became an ERROR node) is ignored, but
    any syntax error inside the class itself remains terminal.
    """
    source = str(value or "").encode("utf-8")
    tree = _parser().parse(source)
    root = tree.root_node
    outer_types = _outer_type_candidates(root)
    if len(outer_types) != 1 or outer_types[0].type != "class_declaration":
        raise JavaRegionParseError(
            "output is neither a class-body region nor one unambiguous outer-class envelope"
        )
    outer = outer_types[0]
    issue = _first_error(outer, source) if outer.has_error else ""
    if issue:
        raise JavaRegionParseError(f"outer-class envelope is malformed: {issue}")
    if "private" in _modifiers(outer, source):
        raise JavaRegionParseError(
            "a private top-level class cannot be treated as an accidental host wrapper"
        )
    body = outer.child_by_field_name("body")
    if body is None:
        raise JavaRegionParseError("outer-class envelope has no class body")
    chunks = _chunks_from_body(body, source, drop_host_lifecycle=True)
    if not chunks:
        raise JavaRegionParseError(
            "outer-class envelope contained no admissible concern members after host lifecycle removal"
        )
    return chunks


def _markdown_java_candidates(value: str) -> tuple[str, ...]:
    """Extract Java fenced blocks from noisy model Markdown with markdown-it."""
    tokens = MarkdownIt("commonmark").parse(str(value or ""))
    candidates: list[str] = []
    for token in tokens:
        if token.type != "fence":
            continue
        language = str(token.info or "").strip().split(maxsplit=1)[0].casefold()
        if language not in {"java", "javac"}:
            continue
        candidate = str(token.content or "").strip()
        if candidate:
            candidates.append(candidate)
    return tuple(candidates)


def _admit_member_candidate(region: str) -> tuple[str, ...]:
    """Admit one candidate Java payload; host-owned lifecycle nodes are discarded."""
    candidate = str(region or "").strip()
    if not candidate:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    try:
        source, root = _parse(prefix + candidate + "\n}\n")
        body = _class_body(root)
        chunks = _chunks_from_body(body, source, drop_host_lifecycle=True)
        if chunks:
            return chunks
    except JavaRegionParseError:
        pass
    return _unwrap_single_outer_class(candidate)


def admit_member_region(value: str) -> str:
    """Normalize noisy model output into host-admissible Java members.

    The raw output may already be Java, may accidentally wrap members in one outer
    class, or may be Markdown containing several Java drafts. Markdown parsing is
    delegated to markdown-it and Java parsing to Tree-sitter. When several fenced
    Java drafts exist, the last structurally admissible candidate wins because local
    models commonly reason through earlier drafts before emitting their final answer.
    """
    region = str(value or "").strip()
    if not region:
        return ""

    direct_error: JavaRegionParseError | None = None
    try:
        chunks = _admit_member_candidate(region)
        if chunks:
            return "\n\n".join(chunks).strip()
    except JavaRegionParseError as exc:
        direct_error = exc

    fenced = _markdown_java_candidates(region)
    fence_errors: list[str] = []
    for candidate in reversed(fenced):
        try:
            chunks = _admit_member_candidate(candidate)
            if chunks:
                return "\n\n".join(chunks).strip()
        except JavaRegionParseError as exc:
            fence_errors.append(str(exc))

    detail = str(direct_error or "raw output contained no admissible Java members")
    if fenced:
        detail += f"; {len(fenced)} Java fence(s) found but none were admissible"
        if fence_errors:
            detail += f"; last fence error: {fence_errors[0]}"
    raise JavaRegionParseError(detail)


def strict_initialize_statements(value: str) -> tuple[str, ...]:
    """Parse statements that will live inside the host-owned initialize() body."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost { static void __mmmInitialize() {\n"
    source, root = _parse(prefix + region + "\n} }\n")
    outer_body = _class_body(root)
    method = next(
        (node for node in outer_body.named_children if node.type == "method_declaration"),
        None,
    )
    if method is None:
        raise JavaRegionParseError("Tree-sitter did not produce the initialize wrapper method")
    body = method.child_by_field_name("body")
    if body is None:
        raise JavaRegionParseError("Tree-sitter did not produce the initialize wrapper body")
    return tuple(
        _text(source, node).strip()
        for node in body.named_children
        if node.type not in _COMMENT_TYPES and _text(source, node).strip()
    )


def admit_initialize_region(value: str) -> str:
    return "\n".join(strict_initialize_statements(value)).strip()
