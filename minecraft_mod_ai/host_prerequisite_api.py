"""Complete declaration evidence from materialized Java dependencies.

Source prefixes are not API contracts: an initializer can consume the entire
budget before a single usable method, or hide the fifth dependency entirely.
Tree-sitter removes implementation bodies rather than truncating declarations.
The project compiler remains the authority for type checking these APIs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .java_region_parser import _parse, _text

_TYPES = frozenset({
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
})


def _is_private(node) -> bool:
    modifiers = next((child for child in node.named_children if child.type == "modifiers"), None)
    return modifiers is not None and any(child.type == "private" for child in modifiers.children)


def _members(body):
    for child in body.named_children:
        if child.type == "enum_body_declarations":
            yield from child.named_children
        else:
            yield child


def java_api_declarations(source: str, *, path: str) -> dict:
    raw, root = _parse(source)
    package = next((
        _text(raw, identifier)
        for child in root.named_children if child.type == "package_declaration"
        for identifier in child.named_children
        if identifier.type in {"identifier", "scoped_identifier"}
    ), "")
    imports = [_text(raw, child) for child in root.named_children
               if child.type == "import_declaration"]
    declarations = []

    def visit(node, parent_owner: str):
        if _is_private(node):
            return
        name = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name is None or body is None:
            return
        owner = ".".join(part for part in (parent_owner, _text(raw, name)) if part)
        members = []
        for member in _members(body):
            if _is_private(member):
                continue
            if member.type in _TYPES:
                visit(member, owner)
            elif member.type in {"method_declaration", "constructor_declaration",
                                 "compact_constructor_declaration", "annotation_type_element_declaration"}:
                implementation = member.child_by_field_name("body")
                end = implementation.start_byte if implementation else member.end_byte
                header = raw[member.start_byte:end].decode("utf-8").strip().rstrip(";")
                if member.type == "compact_constructor_declaration":
                    parameters = node.child_by_field_name("parameters")
                    header += _text(raw, parameters)
                members.append(header + ";")
            elif member.type in {"field_declaration", "constant_declaration"}:
                field_type = member.child_by_field_name("type")
                if field_type is None:
                    continue
                prefix = raw[member.start_byte:field_type.end_byte].decode("utf-8")
                names = []
                for declarator in member.named_children:
                    if declarator.type != "variable_declarator":
                        continue
                    field_name = declarator.child_by_field_name("name")
                    if field_name is not None:
                        dimensions = declarator.child_by_field_name("dimensions")
                        names.append(_text(raw, field_name) + (
                            _text(raw, dimensions) if dimensions is not None else ""))
                if names:
                    members.append(prefix + " " + ", ".join(names) + ";")
            elif member.type == "enum_constant":
                enum_name = member.child_by_field_name("name")
                if enum_name is not None:
                    members.append(_text(raw, enum_name))
        declarations.append({
            "owner": owner,
            "declaration": raw[node.start_byte:body.start_byte].decode("utf-8").strip(),
            "members": members,
        })

    for child in root.named_children:
        if child.type in _TYPES:
            visit(child, package)
    return {
        "path": path, "authority": "materialized_prerequisite_declarations",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "imports": imports, "types": declarations,
    }


def read_prerequisite_apis(context: dict, *, target_path: str, source_context=()) -> list[dict]:
    """Read all dependency APIs once, preserving overloads and generic types."""
    root = Path(context["project_root"]).resolve()
    rows, seen = [], set()
    for item in source_context:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative.endswith(".java"):
            continue
        requested = root / relative
        absolute = requested.resolve()
        if (not absolute.is_relative_to(root) or not absolute.is_file()
                or requested.is_symlink() or relative == target_path or absolute in seen):
            continue
        seen.add(absolute)
        rows.append(java_api_declarations(absolute.read_text(encoding="utf-8"), path=relative))
    return rows



def inspected_prerequisite_symbols(declarations: list[dict]) -> dict:
    """Index callable Java declarations from already materialized prerequisites.

    The Minecraft classpath is indexed before generated dependencies exist.
    A generated entity/registry therefore remains invisible unless the host
    imports its *actual* declaration ABI. Names, signatures, and visibility
    are parsed from saved source, never inferred from a planner label.
    """
    import re

    primitive = {"void", "boolean", "byte", "short", "int", "long",
                 "float", "double", "char"}
    java_lang = {"String", "Object", "Class", "Boolean", "Integer", "Long",
                 "Double", "Float", "Byte", "Short", "Character", "Exception",
                 "RuntimeException", "Throwable"}
    all_types = {
        entry.get("owner")
        for page in declarations if isinstance(page, dict)
        for entry in page.get("types", ()) if isinstance(entry, dict)
    }
    symbols = {}

    def erase(value):
        depth, chars = 0, []
        for char in value:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif depth == 0:
                chars.append(char)
        return "".join(chars).strip().replace("...", "[]")

    def resolve(value, *, owner, aliases):
        plain = erase(str(value or ""))
        suffix = ""
        while plain.endswith("[]"):
            suffix = "[]" + suffix
            plain = plain[:-2].strip()
        if not plain:
            return None
        if plain in primitive:
            return plain + suffix
        if plain in aliases:
            return aliases[plain] + suffix
        if plain in java_lang:
            return "java.lang." + plain + suffix
        if "." in plain:
            return plain + suffix if plain.startswith(
                ("java.", "javax.", "net.", "org.", "com.", "dev.", "io.")
            ) or plain in all_types else None
        local = owner.rpartition(".")[0] + "." + plain
        return local + suffix if local in all_types else None

    for page in declarations:
        if not isinstance(page, dict):
            continue
        aliases = {}
        for statement in page.get("imports", ()):
            if not isinstance(statement, str):
                continue
            match = re.fullmatch(
                r"\s*import\s+((?:[A-Za-z_$][\w$]*\.)+[A-Za-z_$][\w$]*)\s*;\s*",
                statement,
            )
            if match:
                fqcn = match.group(1)
                aliases[fqcn.rpartition(".")[2]] = fqcn
        for declaration in page.get("types", ()):
            if not isinstance(declaration, dict):
                continue
            owner = declaration.get("owner")
            if not isinstance(owner, str) or owner not in all_types:
                continue
            class_header = str(declaration.get("declaration") or "")
            if not re.search(r"\bpublic\b", class_header):
                continue
            parent = re.search(r"\bextends\s+([A-Za-z_$][\w$.<>]*)", class_header)
            superclass = (resolve(parent.group(1), owner=owner, aliases=aliases)
                          if parent else None)
            symbols[owner] = {
                "kind": "CLASS", "owner": owner, "access": 1,
                "provenance": "materialized_prerequisite_declaration",
                **({"superclass": superclass} if superclass else {}),
            }
            for header in declaration.get("members", ()):
                if not isinstance(header, str):
                    continue
                try:
                    raw, root = _parse("class MmmHeader {\n" + header + "\n}")
                except (ValueError, TypeError):
                    # Some legal signatures require source-level context
                    # (generic type variables, nested owners, etc.).
                    # Never invent a callable signature from a failed parse.
                    continue
                outer = next((node for node in root.named_children
                              if node.type == "class_declaration"), None)
                if outer is None:
                    continue
                members = outer.child_by_field_name("body")
                if members is None:
                    continue
                for member in members.named_children:
                    if member.type not in {"method_declaration", "field_declaration"}:
                        continue
                    modifiers = next((node for node in member.named_children
                                      if node.type == "modifiers"), None)
                    modifiers_text = _text(raw, modifiers) if modifiers is not None else ""
                    if not re.search(r"\bpublic\b", modifiers_text):
                        continue
                    static = bool(re.search(r"\bstatic\b", modifiers_text))
                    access = 1 | (8 if static else 0)
                    type_node = member.child_by_field_name("type")
                    if type_node is None:
                        continue
                    return_type = resolve(_text(raw, type_node), owner=owner, aliases=aliases)
                    if return_type is None:
                        continue
                    if member.type == "field_declaration":
                        for child in member.named_children:
                            if child.type != "variable_declarator":
                                continue
                            field_name = child.child_by_field_name("name")
                            if field_name is None:
                                continue
                            name = _text(raw, field_name)
                            key = "materialized:" + owner + "#" + name
                            symbols[key] = {
                                "kind": "FIELD", "owner": owner, "name": name,
                                "return_type": return_type, "access": access,
                                "static": static,
                                "provenance": "materialized_prerequisite_declaration",
                            }
                    else:
                        # Unknown checked exceptions must not leak into a rule
                        # whose callback contract does not declare them.
                        if re.search(r"\bthrows\b", header):
                            continue
                        name_node = member.child_by_field_name("name")
                        parameters_node = member.child_by_field_name("parameters")
                        if name_node is None or parameters_node is None:
                            continue
                        types = []
                        for param in parameters_node.named_children:
                            if param.type != "formal_parameter":
                                types = []
                                break
                            param_type = param.child_by_field_name("type")
                            resolved = (resolve(_text(raw, param_type),
                                                owner=owner, aliases=aliases)
                                        if param_type is not None else None)
                            if resolved is None:
                                types = []
                                break
                            types.append(resolved)
                        else:
                            name = _text(raw, name_node)
                            key = ("materialized:" + owner + "#" + name
                                   + "(" + ",".join(types) + ")")
                            symbols[key] = {
                                "kind": "METHOD", "owner": owner, "name": name,
                                "return_type": return_type, "parameters": types,
                                "access": access, "static": static, "throws": [],
                                "provenance": "materialized_prerequisite_declaration",
                            }
    return symbols
