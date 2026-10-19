"""Host-owned Java import resolution against inspected Fabric classpath symbols.

Use the installed tree-sitter-java parser for type references and the existing
FileAPIEvidence accessibility index. Never ask the coder to emit imports or
pick an arbitrary collision winner. The project compiler remains authoritative.
"""
from __future__ import annotations

import re

from .construction_file_api_evidence import FileAPIEvidence
from .java_region_parser import _parse, _text


# Unqualified JDK names that javac resolves from java.lang automatically.
_JAVA_LANG = frozenset({
    "Object", "String", "StringBuilder", "StringBuffer", "Class",
    "Enum", "Record", "Exception", "RuntimeException", "Error",
    "IllegalArgumentException", "IllegalStateException", "Override",
    "SuppressWarnings", "Deprecated", "FunctionalInterface", "Integer",
    "Long", "Boolean", "Double", "Float", "Short", "Byte", "Character",
    "Void", "Math", "System", "Iterable", "Comparable", "Number",
    "Thread", "Throwable", "AutoCloseable",
})
# These are known Java SE types; javac validates their presence and signatures.
_JDK_IMPORTS = {
    name: "java.util." + name
    for name in (
        "List", "ArrayList", "Map", "HashMap", "LinkedHashMap", "Set",
        "HashSet", "LinkedHashSet", "Collection", "Collections", "Objects",
        "Optional", "UUID", "Iterator", "Comparator", "Arrays", "Deque",
        "ArrayDeque", "Queue", "PriorityQueue", "TreeMap", "TreeSet",
        "Random", "Spliterator", "EnumMap", "EnumSet", "LinkedList",
    )
}


def _type_mentions(source: str) -> set[str]:
    raw, tree = _parse(source)
    own_types: set[str] = set()
    references: set[str] = set()

    def visit(node):
        if node.type in {"class_declaration", "interface_declaration",
                         "enum_declaration", "record_declaration",
                         "annotation_type_declaration"}:
            own = node.child_by_field_name("name")
            if own is not None:
                own_types.add(_text(raw, own))
        if node.type == "type_identifier":
            # A qualified spelling already resolves without an import.
            parent = node.parent
            while parent is not None and parent.type in {
                "scoped_type_identifier", "scoped_identifier",
            }:
                spelling = _text(raw, parent)
                if "." in spelling and spelling.split(".", 1)[0][:1].islower():
                    break
                parent = parent.parent
            else:
                references.add(_text(raw, node))
        for child in node.named_children:
            visit(child)

    visit(tree)
    return references - own_types


def host_add_verified_imports(source: str, *, symbols: dict,
                              owner: str, target_path: str) -> str:
    """Add only unique and accessible type owners to a host-assembled file.

    Any ambiguous simple name is a hard error, not a guess. Unknown names are
    left for javac's concrete error (e.g. same-package sibling declarations).
    """
    if re.search(r"(?m)^\\s*import\\s", source):
        raise ValueError("JAVA_HOST_IMPORTS_MUST_BE_HOST_OWNED")
    package_match = re.match(r"\\s*package\\s+([A-Za-z_$][\\w$]*(?:\\.[A-Za-z_$][\\w$]*)*)\\s*;", source)
    if not package_match:
        raise ValueError("JAVA_HOST_IMPORTS_PACKAGE_REQUIRED")
    package = package_match.group(1)
    index = FileAPIEvidence(symbols, target_path=target_path, owner=owner)
    imports: set[str] = set()
    for simple in sorted(_type_mentions(source)):
        if simple in _JAVA_LANG:
            continue
        candidates = sorted(set(name for name in index.simple.get(simple, ())
                                if index._accessible(name)))
        # Types in this package or nested inside this owner need no imports.
        if package + "." + simple in candidates or owner + "." + simple in candidates:
            continue
        if len(candidates) > 1:
            raise ValueError("JAVA_HOST_IMPORT_AMBIGUOUS: " + simple + ": " +
                             ", ".join(candidates))
        if len(candidates) == 1:
            if not candidates[0].startswith(package + "."):
                imports.add(candidates[0])
        elif simple in _JDK_IMPORTS:
            imports.add(_JDK_IMPORTS[simple])
    if not imports:
        return source
    header = "\\n".join("import " + item + ";" for item in sorted(imports))
    return source[:package_match.end()] + "\\n\\n" + header + source[package_match.end():]
