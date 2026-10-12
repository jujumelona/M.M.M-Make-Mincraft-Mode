"""Host-rendered Java declarations from typed metadata and implementation bodies."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 80,
              "pattern": "^[A-Za-z_$][A-Za-z0-9_$]*$"}
TYPE = {"type": "string", "minLength": 1, "maxLength": 240}
BODY = {"type": "string", "maxLength": 2600}
MEMBER_KINDS = ("field", "method", "constructor", "class", "record", "enum")


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": deepcopy(dict(properties)),
            "required": list(properties), "additionalProperties": False}


def _array(items: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "array", "maxItems": 32, "items": deepcopy(dict(items))}


PARAMETER = _object({"type": TYPE, "name": IDENTIFIER})
VISIBILITY = {"type": "string", "enum": ["public", "protected", "private", ""]}
FLAG = {"type": "boolean"}
ANNOTATIONS = _array({"type": "string", "maxLength": 240})
FIELD = _object({"name": IDENTIFIER, "visibility": VISIBILITY, "static": FLAG,
                 "final": FLAG, "type": TYPE, "initializer": BODY})
METHOD = _object({"name": IDENTIFIER, "visibility": VISIBILITY, "static": FLAG,
                  "return_type": TYPE, "type_parameters": {"type": "string", "maxLength": 240},
                  "parameters": _array(PARAMETER), "throws": _array(TYPE),
                  "annotations": ANNOTATIONS, "body": BODY})
CONSTRUCTOR = _object({"visibility": VISIBILITY, "parameters": _array(PARAMETER),
                       "throws": _array(TYPE), "annotations": ANNOTATIONS, "body": BODY})


def declaration_schema(kind: str, name: str) -> dict[str, Any]:
    """The selected kind/name never come back as executable declaration text."""
    if kind == "field":
        schema = deepcopy(FIELD)
    elif kind == "method":
        schema = deepcopy(METHOD)
    elif kind == "constructor":
        schema = deepcopy(CONSTRUCTOR)
    elif kind in {"class", "record", "enum"}:
        properties: dict[str, Any] = {
            "visibility": VISIBILITY, "static": FLAG,
            "interfaces": _array(TYPE), "fields": _array(FIELD),
            "constructors": _array(CONSTRUCTOR), "methods": _array(METHOD),
        }
        if kind == "class":
            properties["superclass"] = {"type": "string", "maxLength": 240}
        elif kind == "record":
            properties["components"] = _array(PARAMETER)
        else:
            properties["constants"] = _array(_object({
                "name": IDENTIFIER, "arguments": {"type": "string", "maxLength": 240},
            }))
        schema = _object(properties)
    else:
        raise ValueError(f"JAVA_DECLARATION_KIND_UNKNOWN: {kind}")
    # The host has already selected the symbol, including constructor identity.
    schema["properties"].pop("name", None)
    schema["required"] = list(schema["properties"])
    return schema


def metadata_schema(kind: str, name: str) -> dict[str, Any]:
    schema = declaration_schema(kind, name)
    def remove_implementation(node: dict[str, Any]) -> None:
        properties = node.get("properties")
        if isinstance(properties, dict):
            for field in ("body", "initializer"):
                properties.pop(field, None)
            node["required"] = list(properties)
            for child in properties.values():
                remove_implementation(child)
        if isinstance(node.get("items"), dict):
            remove_implementation(node["items"])
    remove_implementation(schema)
    return schema


def _parameters(values: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(f"{item['type']} {item['name']}" for item in values)


def _annotations(values: Sequence[str]) -> str:
    return "".join("@" + value.removeprefix("@") + "\n" for value in values if value)


def _throws(values: Sequence[str]) -> str:
    return " throws " + ", ".join(values) if values else ""


def declaration_header(kind: str, name: str, parts: Mapping[str, Any]) -> str:
    modifiers = [parts.get("visibility", "")]
    if parts.get("static"):
        modifiers.append("static")
    prefix = " ".join(value for value in modifiers if value)
    prefix = prefix + " " if prefix else ""
    if kind == "field":
        return prefix + ("final " if parts.get("final") else "") + parts["type"] + " " + name
    annotations = _annotations(parts.get("annotations", ()))
    if kind == "method":
        type_parameters = str(parts.get("type_parameters") or "").strip()
        generic = "<" + type_parameters + "> " if type_parameters else ""
        return (annotations + prefix + generic + parts["return_type"] + " " + name
                + "(" + _parameters(parts.get("parameters", ())) + ")"
                + _throws(parts.get("throws", ())))
    if kind == "constructor":
        return (annotations + prefix + name + "(" + _parameters(parts.get("parameters", ()))
                + ")" + _throws(parts.get("throws", ())))
    header = prefix + kind + " " + name
    if kind == "record":
        header += "(" + _parameters(parts.get("components", ())) + ")"
    if kind == "class" and parts.get("superclass"):
        header += " extends " + parts["superclass"]
    if parts.get("interfaces"):
        header += " implements " + ", ".join(parts["interfaces"])
    return header


def _one_member(source: str, *, kind: str, name: str, owner: str) -> str:
    """Construct one parser-backed declaration; implementation cannot escape its scope."""
    from .java_region_parser import JavaRegionParseError, _class_body, _parse, _text

    raw, root = _parse(f"class {owner} {{\n{source}\n}}")
    top_level = [node for node in root.named_children if node.type not in {"line_comment", "block_comment"}]
    if len(top_level) != 1:
        raise JavaRegionParseError("implementation escaped the host-owned class", category="scope")
    members = [node for node in _class_body(root).named_children
               if node.type not in {"line_comment", "block_comment"}]
    expected = {"field": "field_declaration", "method": "method_declaration",
                "constructor": "constructor_declaration", "class": "class_declaration",
                "record": "record_declaration", "enum": "enum_declaration"}[kind]
    if len(members) != 1 or members[0].type != expected:
        raise JavaRegionParseError(f"typed {kind} implementation changed its declaration shape", category="scope")
    node = members[0]
    actual_name = node.child_by_field_name("name")
    if kind == "field":
        variables = [child for child in node.named_children if child.type == "variable_declarator"]
        if len(variables) != 1:
            raise JavaRegionParseError("initializer declared an additional field", category="scope")
        actual_name = variables[0].child_by_field_name("name")
    if actual_name is None or _text(raw, actual_name) != name:
        raise JavaRegionParseError("implementation changed the host-selected symbol", category="scope")
    return _text(raw, node)


def render_declaration(kind: str, name: str, parts: Mapping[str, Any], *, owner: str) -> str:
    header = declaration_header(kind, name, parts)
    if kind == "field":
        initializer = str(parts.get("initializer") or "").strip()
        source = header + (" = " + initializer if initializer else "") + ";"
    elif kind in {"method", "constructor"}:
        source = header + " {\n" + str(parts["body"]) + "\n}"
    else:
        members = []
        if kind == "enum":
            members.append(", ".join(item["name"] + ("(" + item["arguments"] + ")" if item["arguments"] else "")
                                     for item in parts["constants"]) + ";")
        members.extend(render_declaration("field", item["name"], item, owner=name) for item in parts["fields"])
        members.extend(render_declaration("constructor", name, item, owner=name) for item in parts["constructors"])
        members.extend(render_declaration("method", item["name"], item, owner=name) for item in parts["methods"])
        source = header + " {\n" + "\n\n".join(members) + "\n}"
    return _one_member(source, kind=kind, name=name, owner=owner)
