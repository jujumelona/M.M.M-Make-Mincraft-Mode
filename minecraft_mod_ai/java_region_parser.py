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

from .java_generation_policy import (
    EXPLICIT_JDK_IMPORT_PATTERN,
    JAVA_FENCE_LANGUAGES,
    imported_jdk_use_can_be_qualified,
    initialize_wrapper_allowed,
    localize_initialize_field_modifiers,
    member_jdk_import_allowed,
    region_recovery_shapes,
)


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


def _declaration_summary(node: Any, source: bytes) -> str:
    body = node.child_by_field_name("body")
    if body is None:
        return _text(source, node).strip()
    header = source[node.start_byte:body.start_byte].decode(
        "utf-8",
        errors="replace",
    ).strip()
    return (header + " { ... }").strip()


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
                "declaration": _declaration_summary(node, source),
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
                "declaration": _declaration_summary(node, source),
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
                    "declaration": _declaration_summary(node, source),
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
                "declaration": _declaration_summary(node, source),
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


def class_body_simple_type_occurrences(value: str) -> tuple[dict[str, Any], ...]:
    """Return byte ranges of unqualified simple type identifiers in one class body.

    Only Tree-sitter type_identifier nodes are returned. Identifiers nested inside
    scoped_type_identifier are omitted so already-qualified FQCNs are never doubled.
    Byte ranges are relative to the original class-body region.
    """

    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    prefix_bytes = prefix.encode("utf-8")
    region_bytes = region.encode("utf-8")
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    rows: list[dict[str, Any]] = []
    for node in _walk_named(body):
        if node.type != "type_identifier":
            continue
        parent = getattr(node, "parent", None)
        if parent is not None and parent.type in {
            "scoped_type_identifier",
            "scoped_identifier",
        }:
            continue
        start = node.start_byte - len(prefix_bytes)
        end = node.end_byte - len(prefix_bytes)
        if start < 0 or end > len(region_bytes) or start >= end:
            continue
        rows.append(
            {
                "name": _text(source, node).strip(),
                "start_byte": start,
                "end_byte": end,
            }
        )
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

def class_body_method_invocation_details(value: str) -> tuple[dict[str, Any], ...]:
    """Return invocation arguments and method-name byte spans relative to the class body."""
    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    prefix_bytes = len(prefix.encode("utf-8"))
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    region_size = len(region.encode("utf-8"))
    rows: list[dict[str, Any]] = []
    for node in _walk_named(body):
        if node.type != "method_invocation":
            continue
        name = node.child_by_field_name("name")
        arguments = node.child_by_field_name("arguments")
        receiver = node.child_by_field_name("object")
        if name is None:
            continue
        name_start = int(name.start_byte) - prefix_bytes
        name_end = int(name.end_byte) - prefix_bytes
        if name_start < 0 or name_end < name_start or name_end > region_size:
            continue
        argument_nodes = tuple(arguments.named_children) if arguments is not None else ()
        rows.append(
            {
                "receiver": _text(source, receiver).strip() if receiver is not None else "",
                "symbol": _text(source, name).strip(),
                "argument_count": len(argument_nodes),
                "arguments": tuple(_text(source, item).strip() for item in argument_nodes),
                "name_start_byte": name_start,
                "name_end_byte": name_end,
            }
        )
    return tuple(rows)


def class_body_object_creations(value: str) -> tuple[dict[str, Any], ...]:
    """Return constructed type spelling and argument count for object creation expressions."""

    region = str(value or "").strip()
    if not region:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + region + "\n}\n")
    body = _class_body(root)
    rows: list[dict[str, Any]] = []
    for node in _walk_named(body):
        if node.type != "object_creation_expression":
            continue
        type_node = node.child_by_field_name("type")
        arguments = node.child_by_field_name("arguments")
        if type_node is None:
            type_node = next(
                (
                    child
                    for child in node.named_children
                    if child.type in {
                        "type_identifier",
                        "scoped_type_identifier",
                        "generic_type",
                    }
                ),
                None,
            )
        if type_node is None:
            continue
        prefix_bytes = len(prefix.encode("utf-8"))
        start_byte = int(node.start_byte) - prefix_bytes
        end_byte = int(node.end_byte) - prefix_bytes
        argument_nodes = tuple(arguments.named_children) if arguments is not None else ()
        rows.append(
            {
                "type": _text(source, type_node).strip(),
                "argument_count": len(argument_nodes),
                "arguments": tuple(
                    _text(source, item).strip() for item in argument_nodes
                ),
                "start_byte": start_byte,
                "end_byte": end_byte,
            }
        )
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
            expression = (
                node.child_by_field_name("expression")
                or node.child_by_field_name("value")
                or next(iter(node.named_children), None)
            )
            if expression is None:
                continue
            if expression.type == "parenthesized_expression":
                invocations = [
                    child
                    for child in _walk_named(expression)
                    if child.type == "method_invocation"
                ]
                expression = invocations[0] if len(invocations) == 1 else expression
            if expression.type != "method_invocation":
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

    imports: dict[str, str] = {}
    for node in root.named_children:
        if node.type != "import_declaration":
            continue
        rendered = _text(source, node).strip()
        match = _EXPLICIT_JDK_IMPORT.fullmatch(rendered)
        if match is None or not member_jdk_import_allowed(match.group(1)):
            raise JavaRegionParseError(
                "outer-class member recovery only admits explicit java.* imports"
            )
        fqcn = match.group(1)
        simple = fqcn.rsplit(".", 1)[-1]
        existing = imports.get(simple)
        if existing is not None and existing != fqcn:
            raise JavaRegionParseError(
                f"ambiguous JDK imports for simple type {simple!r}"
            )
        imports[simple] = fqcn

    if not imports:
        return chunks
    canonical = _qualify_imported_jdk_names("\n\n".join(chunks), imports)
    return strict_member_chunks(canonical)


def _fence_language(info: object) -> str:
    """Normalize a Markdown fence info string without assuming a language token."""
    parts = str(info or "").strip().split(maxsplit=1)
    return parts[0].casefold() if parts else ""


def _model_java_candidates(value: str) -> tuple[str, ...]:
    """Return Java-shaped payload candidates from an arbitrary model envelope.

    This function is the single normalization boundary for model-authored Java.
    It is intentionally total for ordinary text inputs: Markdown metadata is never
    indexed blindly, bare fences are accepted as candidates, explicit non-Java
    fences are ignored, and the raw payload remains a final fallback. Candidate
    syntax is still decided only by Tree-sitter downstream.
    """
    raw = str(value or "").strip()
    if not raw:
        return ()

    fenced: list[str] = []
    try:
        tokens = MarkdownIt("commonmark").parse(raw)
    except Exception:
        tokens = ()
    for token in tokens:
        if token.type != "fence":
            continue
        language = _fence_language(token.info)
        if language not in JAVA_FENCE_LANGUAGES:
            continue
        candidate = str(token.content or "").strip()
        if candidate and candidate not in fenced:
            fenced.append(candidate)

    # Models often emit earlier draft fences followed by a final fence. Prefer
    # later fenced candidates; raw text is only the final fallback for direct Java
    # or accidental single-class envelopes.
    ordered = list(reversed(fenced))
    if raw not in ordered:
        ordered.append(raw)
    return tuple(ordered)



_EXPLICIT_JDK_IMPORT = EXPLICIT_JDK_IMPORT_PATTERN


def _split_leading_jdk_imports(
    region: str,
) -> tuple[str, dict[str, str]] | None:
    """Detach explicit leading java.* imports from a model-authored member region."""
    lines = str(region or "").splitlines(keepends=True)
    imports: dict[str, str] = {}
    import_lines: set[int] = set()
    saw_import = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = _EXPLICIT_JDK_IMPORT.fullmatch(stripped)
        if match is not None:
            fqcn = match.group(1)
            if not member_jdk_import_allowed(fqcn):
                raise JavaRegionParseError(
                    f"member-region JDK import is not allowed by production policy: {fqcn!r}"
                )
            simple = fqcn.rsplit(".", 1)[-1]
            existing = imports.get(simple)
            if existing is not None and existing != fqcn:
                raise JavaRegionParseError(
                    f"ambiguous JDK imports for simple type {simple!r}: "
                    f"{existing!r} and {fqcn!r}"
                )
            imports[simple] = fqcn
            import_lines.add(index)
            saw_import = True
            continue
        if stripped.startswith("import "):
            raise JavaRegionParseError(
                "member-region imports must be explicit non-static java.* imports"
            )
        break

    if not saw_import:
        return None

    body = "".join(
        line for index, line in enumerate(lines) if index not in import_lines
    ).strip()
    if not body:
        raise JavaRegionParseError(
            "member-region imports were present without any class-body members"
        )
    return body, imports


def _qualify_imported_jdk_names(
    region: str,
    imports: dict[str, str],
) -> str:
    """Rewrite imported JDK class references to FQCNs using Tree-sitter spans."""
    candidate = str(region or "").strip()
    prefix = "final class __MMMRegionHost {\n"
    prefix_size = len(prefix.encode("utf-8"))
    region_bytes = candidate.encode("utf-8")
    source, root = _parse(prefix + candidate + "\n}\n")
    body = _class_body(root)
    edits: list[tuple[int, int, str]] = []

    for node in _walk_named(body):
        rendered = _text(source, node).strip()
        replacement = imports.get(rendered)
        if replacement is None:
            continue

        role = ""
        if node.type == "type_identifier":
            role = "type"
        elif node.type == "identifier":
            parent = getattr(node, "parent", None)
            receiver = (
                parent.child_by_field_name("object")
                if parent is not None
                and parent.type in {"field_access", "method_invocation"}
                else None
            )
            if (
                receiver is not None
                and int(receiver.start_byte) == int(node.start_byte)
                and int(receiver.end_byte) == int(node.end_byte)
            ):
                role = "static_receiver"
        if not imported_jdk_use_can_be_qualified(role):
            continue

        start = int(node.start_byte) - prefix_size
        end = int(node.end_byte) - prefix_size
        if start < 0 or end > len(region_bytes) or start >= end:
            continue
        edits.append((start, end, replacement))

    encoded = region_bytes
    for start, end, replacement in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement.encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8")


def _qualify_imported_jdk_names_in_initialize(
    region: str,
    imports: dict[str, str],
) -> str:
    """Rewrite imported JDK references inside an initialize statement list."""

    candidate = str(region or "").strip()
    prefix = "final class __MMMRegionHost { static void __mmmInitialize() {\n"
    prefix_size = len(prefix.encode("utf-8"))
    region_bytes = candidate.encode("utf-8")
    source, root = _parse(prefix + candidate + "\n} }\n")
    outer_body = _class_body(root)
    method = next(
        (node for node in outer_body.named_children if node.type == "method_declaration"),
        None,
    )
    if method is None:
        raise JavaRegionParseError(
            "Tree-sitter did not produce the initialize wrapper method"
        )
    body = method.child_by_field_name("body")
    if body is None:
        raise JavaRegionParseError(
            "Tree-sitter did not produce the initialize wrapper body"
        )

    edits: list[tuple[int, int, str]] = []
    for node in _walk_named(body):
        rendered = _text(source, node).strip()
        replacement = imports.get(rendered)
        if replacement is None:
            continue
        role = ""
        if node.type == "type_identifier":
            role = "type"
        elif node.type == "identifier":
            parent = getattr(node, "parent", None)
            receiver = (
                parent.child_by_field_name("object")
                if parent is not None
                and parent.type in {"field_access", "method_invocation"}
                else None
            )
            if (
                receiver is not None
                and int(receiver.start_byte) == int(node.start_byte)
                and int(receiver.end_byte) == int(node.end_byte)
            ):
                role = "static_receiver"
        if not imported_jdk_use_can_be_qualified(role):
            continue
        start = int(node.start_byte) - prefix_size
        end = int(node.end_byte) - prefix_size
        if start < 0 or end > len(region_bytes) or start >= end:
            continue
        edits.append((start, end, replacement))

    encoded = region_bytes
    for start, end, replacement in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement.encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8")


def _canonicalize_initialize_jdk_imports(region: str) -> str | None:
    split = _split_leading_jdk_imports(region)
    if split is None:
        return None
    body, imports = split
    canonical = _qualify_imported_jdk_names_in_initialize(body, imports)
    strict_initialize_statements(canonical)
    return canonical


def _canonicalize_member_jdk_imports(region: str) -> str | None:
    split = _split_leading_jdk_imports(region)
    if split is None:
        return None
    body, imports = split
    canonical = _qualify_imported_jdk_names(body, imports)

    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + canonical + "\n}\n")
    class_body = _class_body(root)
    for node in _walk_named(class_body):
        if node.type not in {"identifier", "type_identifier"}:
            continue
        rendered = _text(source, node).strip()
        if rendered not in imports:
            continue
        parent = getattr(node, "parent", None)
        if parent is not None and parent.type in {
            "scoped_identifier",
            "scoped_type_identifier",
        }:
            scoped = _text(source, parent).strip()
            if scoped.startswith(imports[rendered] + ".") or scoped == imports[rendered]:
                continue
        raise JavaRegionParseError(
            f"JDK import {imports[rendered]!r} has an unqualified use that "
            "cannot be safely canonicalized inside a member region"
        )
    return canonical


def _host_initialize_only_member_candidate(region: str) -> bool:
    """Return whether a valid class-body candidate contains only host initialize()."""
    candidate = str(region or "").strip()
    if not candidate:
        return False
    prefix = "final class __MMMRegionHost {\n"
    try:
        source, root = _parse(prefix + candidate + "\n}\n")
        body = _class_body(root)
    except JavaRegionParseError:
        return False
    nodes = tuple(
        node for node in body.named_children if node.type not in _COMMENT_TYPES
    )
    return bool(nodes) and all(_is_host_initialize(node, source) for node in nodes)


def _raw_compilation_unit_envelope_kinds(region: str) -> frozenset[str]:
    """Return host-envelope node kinds from raw Java without lexical guessing."""

    source = str(region or "").encode("utf-8")
    tree = _parser().parse(source)
    return frozenset(
        child.type
        for child in tree.root_node.named_children
        if child.type in {
            "package_declaration",
            "import_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }
    )


def _admit_member_candidate(region: str) -> tuple[str, ...]:
    """Admit one candidate Java payload; host-owned lifecycle nodes are discarded."""
    candidate = str(region or "").strip()
    if not candidate:
        return ()
    prefix = "final class __MMMRegionHost {\n"
    envelope_kinds = _raw_compilation_unit_envelope_kinds(candidate)
    has_outer_type = bool(
        envelope_kinds
        & {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }
    )

    # A package/import/type envelope is compilation-unit structure, never a class
    # member. Do not feed it through the class-body parser where tolerant parsing
    # can reinterpret keywords as identifiers. Recovery below handles JDK imports
    # and one accidental outer class structurally.
    if not envelope_kinds:
        try:
            source, root = _parse(prefix + candidate + "\n}\n")
            body = _class_body(root)
            chunks = _chunks_from_body(body, source, drop_host_lifecycle=True)
            if chunks:
                return chunks
        except JavaRegionParseError:
            pass

    if "package_declaration" in envelope_kinds and not has_outer_type:
        raise JavaRegionParseError(
            "package declaration is compilation-unit structure and cannot be a concern member"
        )

    if "package_declaration" not in envelope_kinds and not has_outer_type:
        try:
            canonical = _canonicalize_member_jdk_imports(candidate)
            if canonical is not None:
                source, root = _parse(prefix + canonical + "\n}\n")
                body = _class_body(root)
                chunks = _chunks_from_body(body, source, drop_host_lifecycle=True)
                if chunks:
                    return chunks
        except JavaRegionParseError:
            pass

    return _unwrap_single_outer_class(candidate)


def admit_member_region(
    value: str,
    *,
    allow_host_initialize_only_empty: bool = False,
) -> str:
    """Normalize arbitrary model output into host-admissible Java members."""
    candidates = _model_java_candidates(value)
    if not candidates:
        return ""

    errors: list[str] = []
    for candidate in candidates:
        if (
            allow_host_initialize_only_empty
            and _host_initialize_only_member_candidate(candidate)
        ):
            return ""
        try:
            chunks = _admit_member_candidate(candidate)
            if chunks:
                return "\n\n".join(chunks).strip()
        except JavaRegionParseError as exc:
            errors.append(str(exc))

    detail = (
        f"{len(candidates)} model Java candidate(s) were structurally inadmissible"
    )
    if errors:
        detail += f"; last candidate error: {errors[-1]}"
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
    statements: list[str] = []
    for node in body.named_children:
        if node.type in _COMMENT_TYPES:
            continue
        if node.type in _NESTED_TYPES:
            raise JavaRegionParseError(
                f"initialize body cannot declare local type {_name(node, source)!r}"
            )
        rendered = _text(source, node).strip()
        if rendered:
            statements.append(rendered)
    return tuple(statements)


def _initialize_wrapper_method(node: Any, source: bytes) -> bool:
    if node.type != "method_declaration":
        return False
    name = node.child_by_field_name("name")
    return_type = node.child_by_field_name("type")
    parameters = node.child_by_field_name("parameters")
    return initialize_wrapper_allowed(
        _text(source, name).strip() if name is not None else "",
        _text(source, return_type).strip() if return_type is not None else "",
        _text(source, parameters).strip() if parameters is not None else "",
    )


def _localize_initialize_field(node: Any, source: bytes) -> str:
    if node.type != "field_declaration":
        raise JavaRegionParseError(
            f"initialize wrapper sibling {node.type!r} cannot be localized"
        )
    rendered = _text(source, node).strip()
    modifiers = next(
        (child for child in node.named_children if child.type == "modifiers"),
        None,
    )
    modifier_text = _text(source, modifiers).strip() if modifiers is not None else ""
    preserved = localize_initialize_field_modifiers(modifier_text)
    if preserved is None:
        raise JavaRegionParseError(
            f"initialize field modifiers cannot be safely localized: {modifier_text!r}"
        )
    if modifiers is None:
        return rendered
    # Tree-sitter offsets are byte-based; node-relative slicing keeps Unicode safe.
    raw_node = _text(source, node)
    relative_start = int(modifiers.end_byte) - int(node.start_byte)
    tail = raw_node.encode("utf-8")[relative_start:].decode("utf-8").lstrip()
    prefix = (" ".join(preserved) + " ") if preserved else ""
    localized = prefix + tail
    strict_initialize_statements(localized)
    return localized


def _recover_initialize_from_class_body(candidate: str) -> str:
    prefix = "final class __MMMRegionHost {\n"
    source, root = _parse(prefix + str(candidate or "").strip() + "\n}\n")
    body = _class_body(root)
    nodes = tuple(
        node for node in body.named_children if node.type not in _COMMENT_TYPES
    )
    wrappers = tuple(
        node for node in nodes if _initialize_wrapper_method(node, source)
    )
    if len(wrappers) != 1:
        raise JavaRegionParseError(
            "initialize recovery requires exactly one zero-argument void initialize() wrapper"
        )
    wrapper = wrappers[0]
    locals_: list[str] = []
    for node in nodes:
        if node is wrapper:
            continue
        locals_.append(_localize_initialize_field(node, source))

    method_body = wrapper.child_by_field_name("body")
    if method_body is None:
        raise JavaRegionParseError("initialize wrapper has no method body")
    statements = [
        _text(source, node).strip()
        for node in method_body.named_children
        if node.type not in _COMMENT_TYPES and _text(source, node).strip()
    ]
    recovered = "\n".join([*locals_, *statements]).strip()
    strict_initialize_statements(recovered)
    return recovered


def _recover_initialize_from_outer_class(candidate: str) -> str:
    raw = str(candidate or "").strip()
    source = raw.encode("utf-8")
    tree = _parser().parse(source)
    root = tree.root_node
    outer_types = _outer_type_candidates(root)
    if len(outer_types) != 1 or outer_types[0].type != "class_declaration":
        raise JavaRegionParseError(
            "initialize recovery found no unambiguous outer-class envelope"
        )
    outer = outer_types[0]
    issue = _first_error(outer, source) if outer.has_error else ""
    if issue:
        raise JavaRegionParseError(f"outer-class initialize envelope is malformed: {issue}")

    imports: dict[str, str] = {}
    for node in root.named_children:
        if node.type != "import_declaration":
            continue
        rendered = _text(source, node).strip()
        match = _EXPLICIT_JDK_IMPORT.fullmatch(rendered)
        if match is None or not member_jdk_import_allowed(match.group(1)):
            raise JavaRegionParseError(
                "outer-class initialize recovery only admits explicit java.* imports"
            )
        fqcn = match.group(1)
        simple = fqcn.rsplit(".", 1)[-1]
        existing = imports.get(simple)
        if existing is not None and existing != fqcn:
            raise JavaRegionParseError(
                f"ambiguous JDK imports for simple type {simple!r}"
            )
        imports[simple] = fqcn

    body = outer.child_by_field_name("body")
    if body is None:
        raise JavaRegionParseError("outer-class initialize envelope has no class body")
    class_body = "\n".join(
        _text(source, node).strip()
        for node in body.named_children
        if node.type not in _COMMENT_TYPES and _text(source, node).strip()
    )
    if imports:
        class_body = _qualify_imported_jdk_names(class_body, imports)
    return _recover_initialize_from_class_body(class_body)


def _recover_initialize_shape(candidate: str, shape: str) -> str:
    if shape == "direct_statements":
        return "\n".join(strict_initialize_statements(candidate)).strip()
    if shape == "jdk_imported_statements":
        canonical = _canonicalize_initialize_jdk_imports(candidate)
        if canonical is None:
            raise JavaRegionParseError("initialize candidate has no leading JDK imports")
        return "\n".join(strict_initialize_statements(canonical)).strip()
    if shape == "initialize_wrapper":
        return _recover_initialize_from_class_body(candidate)
    if shape == "jdk_imported_initialize_wrapper":
        canonical = _canonicalize_member_jdk_imports(candidate)
        if canonical is None:
            raise JavaRegionParseError(
                "initialize wrapper candidate has no leading JDK imports"
            )
        return _recover_initialize_from_class_body(canonical)
    if shape == "outer_class_initialize":
        return _recover_initialize_from_outer_class(candidate)
    raise JavaRegionParseError(f"unsupported initialize recovery shape {shape!r}")


def admit_initialize_region(value: str) -> str:
    """Recover arbitrary model output into host-owned initialize statements.

    Recovery is policy-driven and shape-based: direct statement lists, explicit
    JDK imports, accidental initialize() wrappers, and full outer-class envelopes
    are all attempted before the response is classified as a terminal scope error.
    """
    candidates = _model_java_candidates(value)
    if not candidates:
        return ""

    errors: list[str] = []
    shapes = region_recovery_shapes("initialize")
    for candidate in candidates:
        for shape in shapes:
            try:
                recovered = _recover_initialize_shape(candidate, shape)
                # A successfully parsed initialize region may intentionally lower
                # to no statements (for example, an initialize() wrapper whose
                # body contains comments only). Empty is therefore a valid no-op,
                # not evidence that recovery failed.
                return recovered
            except JavaRegionParseError as exc:
                detail = f"{shape}: {exc}"
                if detail not in errors:
                    errors.append(detail)

    detail = (
        f"{len(candidates)} model initialize candidate(s) were structurally inadmissible"
    )
    if errors:
        detail += "; recovery attempts: " + " | ".join(errors[-5:])
    raise JavaRegionParseError(detail)
