"""Compiler declaration provenance and immutable binary member authority."""
from __future__ import annotations

from pathlib import Path


def is_project_compiled_output(path, project_root):
    if project_root is None:
        return False
    candidate = Path(path).resolve()
    return candidate.is_dir() and candidate.is_relative_to(
        Path(project_root).resolve() / "build/classes")


def read_binary_classpath(classpath, *, java_version, classpath_sides):
    from .jar_api_extractor import inspect_jar, parse_class

    classes, locations = {}, {}
    for path in classpath:
        default_side = classpath_sides.get(str(path), "COMMON")
        found = ({c.name: c for c in (parse_class(p.read_bytes(), default_side=default_side)
                                     for p in path.rglob("*.class"))} if path.is_dir() else
                 inspect_jar(path, java_version=int(java_version), default_side=default_side,
                             require_classes=False))
        for name, item in found.items():
            if name == "module-info":
                continue
            if name in classes and classes[name].content_hash != item.content_hash:
                raise ValueError("AMBIGUOUS_CLASSPATH_CLASS: " + name)
            classes.setdefault(name, item)
            locations.setdefault(name, []).append(path)
    return classes, locations


def resolved_declaration_side(row, classes, locations, *, source_names, project_root):
    """Use javac's selected declaration, never infer an owner from source text."""
    owner = classes.get(row["owner"])
    declaration = row.get("declaration_source", "")
    if declaration:
        if declaration not in source_names:
            raise ValueError("JAVA_DECLARATION_SOURCE_UNVERIFIED: " + declaration)
        if owner and not all(is_project_compiled_output(path, project_root)
                             for path in locations[row["owner"]]):
            raise ValueError("JAVA_SOURCE_BINARY_OWNER_CONFLICT: " + row["owner"])
        return row["side"]
    if owner is None:
        return row["side"]  # Platform/JDK declarations resolved by the target compiler.
    if row["kind"] == "CLASS":
        return owner.side
    members = [m for m in owner.members if (m.name, m.descriptor, m.kind, m.is_static) == (
        row["name"], row["descriptor"], row["kind"], row["is_static"])]
    if len(members) != 1:
        raise ValueError("MEMBER_NOT_IN_INSPECTED_CLASSPATH: " +
            row["owner"] + "#" + row["name"] + row["descriptor"] +
            " kind=" + row["kind"] + " origin=binary reference=" +
            row.get("reference_source", "") + ":" + str(row["line"]))
    return members[0].side
