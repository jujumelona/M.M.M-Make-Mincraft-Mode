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
    if node.type == "method_declaration" and _name(node, source) == "initialize":
        raise JavaRegionParseError(
            "initialize() is host-owned lifecycle structure and cannot be declared by a concern"
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
        if drop_host_lifecycle and node.type in _HOST_OWNED_MEMBER_TYPES:
            continue
        if (
            drop_host_lifecycle
            and node.type == "method_declaration"
            and _name(node, source) == "initialize"
        ):
            continue
        _validate_member_node(node, source)
        chunks.append(_text(source, node).strip())
    return tuple(chunk for chunk in chunks if chunk)


def strict_member_chunks(value: str) -> tuple[str, ...]:
    """Parse a Java class-body fragment and return structural top-level members."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    return _chunks_from_body(body, source, drop_host_lifecycle=False)


def _unwrap_single_outer_class(value: str) -> tuple[str, ...]:
    """Salvage an accidental compilation-unit/class envelope using the Java AST."""
    source, root = _parse(str(value or ""))
    declarations = [
        node
        for node in root.named_children
        if node.type not in {"package_declaration", "import_declaration"}
    ]
    if len(declarations) != 1 or declarations[0].type != "class_declaration":
        raise JavaRegionParseError(
            "output is neither a class-body region nor one unambiguous outer-class envelope"
        )
    outer = declarations[0]
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


def admit_member_region(value: str) -> str:
    """Normalize model Java into host-admissible members without regex parsing.

    Direct class-body fragments are accepted as-is. If the model accidentally emits
    package/import lines plus one outer class wrapper, Tree-sitter unwraps that single
    class and drops only host-owned constructor/initialize/static-initializer nodes.
    """
    region = str(value or "").strip()
    if not region:
        return ""
    try:
        chunks = strict_member_chunks(region)
    except JavaRegionParseError as direct_error:
        try:
            chunks = _unwrap_single_outer_class(region)
        except JavaRegionParseError as envelope_error:
            raise JavaRegionParseError(
                f"class-body parse failed ({direct_error}); envelope admission failed ({envelope_error})"
            ) from envelope_error
    return "\n\n".join(chunks).strip()


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
        if _text(source, node).strip()
    )


def admit_initialize_region(value: str) -> str:
    return "\n".join(strict_initialize_statements(value)).strip()
