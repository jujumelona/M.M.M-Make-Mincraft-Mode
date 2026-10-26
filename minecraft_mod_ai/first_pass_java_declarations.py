"""Host-rendered Java declarations from typed metadata and implementation bodies."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 80,
              "pattern": "^[A-Za-z_$][A-Za-z0-9_$]*$"}
TYPE = {"type": "string", "minLength": 1, "maxLength": 240}
BODY = {"type": "string"}  # Source bodies are not model-bounded by a string cap.
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


def declaration_frame(kind: str, name: str, parts: Mapping[str, Any], *, allow_uninitialized: bool = True):
    """One complete Java declaration with a host-owned literal signature."""
    header = declaration_header(kind, name, parts)
    if kind == "field":
        prefix, suffix = header + " = ", ";"
        empty = header + ";" if allow_uninitialized else None
    elif kind in {"method", "constructor"}:
        prefix, suffix, empty = header + " {\n", "\n}", None
    else:
        raise ValueError("JAVA_SOURCE_FRAME_REQUIRES_FIELD_METHOD_OR_CONSTRUCTOR")
    frame = {"kind": kind, "prefix": prefix, "suffix": suffix}
    if empty is not None:
        frame["empty_value"] = empty
    return frame


def callback_parameter_metadata(event_callback):
    """Keep callback input names unique while preserving authored UI input names."""
    from .java_operation_ir import KEYWORDS

    names, used = [], set()
    for index, type_name in enumerate(event_callback["parameters"]):
        simple = type_name.rsplit(".", 1)[-1].replace("[]", "Array")
        proposed = simple[:1].lower() + simple[1:]
        name = (proposed if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", simple) and proposed not in KEYWORDS
                else f"eventArg{index}")
        if event_callback.get("provenance") == "host_typed_ui_action":
            name = event_callback["parameter_names"][index]
        while name in used:
            name += str(index)
        used.add(name)
        names.append({"type": type_name, "name": name})
    return names





def canonicalize_field_type_metadata(
    kind: str, name: str, parts: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover a typed Java field from an accidentally repeated symbol.

    The host already owns the field name. Some small decoders nevertheless
    supply a valid *declaration* in the 'type' slot, e.g. 'int mmmMember1'.
    Use the Java parser to extract its actual type, but only when the AST
    proves exactly one field and its identifier is the selected host name.
    Do not infer or substitute unknown Minecraft types.
    """
    result = dict(parts)
    if kind != "field" or not isinstance(result.get("type"), str):
        return result
    value = result["type"].strip()
    # Avoid parsing every well-formed simple type in the common case.
    if not value.endswith(" " + name):
        return result
    from .java_region_parser import JavaRegionParseError, _class_body, _parse, _text
    try:
        raw, root = _parse("class MmmMetadataField {\n" + value + ";\n}")
    except JavaRegionParseError:
        return result
    members = [
        node for node in _class_body(root).named_children
        if node.type not in {"line_comment", "block_comment"}
    ]
    if len(members) != 1 or members[0].type != "field_declaration":
        return result
    field = members[0]
    variables = [
        child for child in field.named_children
        if child.type == "variable_declarator"
    ]
    type_node = field.child_by_field_name("type")
    selected_name = (
        variables[0].child_by_field_name("name")
        if len(variables) == 1 else None
    )
    if type_node is None or selected_name is None or _text(raw, selected_name) != name:
        return result
    result["type"] = _text(raw, type_node).strip()
    return result


def validate_declaration_metadata(kind: str, name: str, parts: Mapping[str, Any], *, owner: str) -> None:
    """Parse fixed declaration syntax before asking for any implementation."""
    stub = deepcopy(dict(parts))
    if kind == "field":
        stub["initializer"] = ""
    elif kind in {"method", "constructor"}:
        stub["body"] = ""
    else:
        for child in stub["fields"]:
            child["initializer"] = ""
        for child in (*stub["methods"], *stub["constructors"]):
            child["body"] = ""
    render_declaration(kind, name, stub, owner=owner, phase="metadata")


def _one_member(source: str, *, kind: str, name: str, owner: str, phase: str) -> str:
    from .java_region_parser import JavaRegionParseError

    try:
        return _parse_one_member(source, kind=kind, name=name, owner=owner)
    except JavaRegionParseError as exc:
        _record_source_failure(source, kind=kind, name=name, owner=owner, phase=phase, exc=exc)
        raise


def _record_source_failure(source: str, *, kind: str, name: str, owner: str, phase: str, exc: Exception) -> None:
    from .planner_trace_artifacts import save_trace_artifact
    from .root_cause_trace import durable_trace_path, emit_root_cause

    details: dict[str, Any] = {"kind": kind, "member": name, "owner": owner, "phase": phase,
                               "source_chars": len(source), "member_preview": source[:1024]}
    try:
        details["source_artifact"] = save_trace_artifact(
            {**details, "member_source": source, "parser_source": f"class {owner} {{\n{source}\n}}",
             "error": str(exc)}, durable_trace_path().parent / "artifacts",
        )
    except Exception as artifact_error:
        details["source_artifact_error"] = type(artifact_error).__name__
    # The console receives a bounded member preview and the artifact path. The
    # complete parser input stays in the existing downloadable failure ZIP.
    emit_root_cause("java_declaration_failure", stage="generate:content",
                    operation=name, gate="java_declaration", result="FAIL",
                    reason=str(exc), details=details, exc=exc)


def _parse_one_member(source: str, *, kind: str, name: str, owner: str) -> str:
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


def validate_source_frame(source: str, frame: Mapping[str, Any]) -> None:
    """Prove that unconstrained source contains exactly its host-owned member.

    A matching prefix/suffix alone permits an extra static initializer, sibling
    method or escaped class. Derive the immutable identity from the frame's
    syntax, then use the same parser-backed declaration gate as final rendering.
    No guessed return type, import, or target API is introduced by this check.
    """
    from .java_region_parser import JavaRegionParseError, _class_body, _parse, _text

    kind = frame.get("kind")
    if kind not in {"field", "method", "constructor"}:
        raise JavaRegionParseError("unknown source declaration kind", category="scope")
    stub = (str(frame["prefix"]) + ("null" if kind == "field" else "")
            + str(frame["suffix"]))
    raw, root = _parse("class MmmSourceFrame {\n" + stub + "\n}")
    members = [node for node in _class_body(root).named_children
               if node.type not in {"line_comment", "block_comment"}]
    if len(members) != 1:
        raise JavaRegionParseError("host source frame is not one member", category="scope")
    node = members[0]
    if kind == "field":
        variables = [child for child in node.named_children if child.type == "variable_declarator"]
        if len(variables) != 1:
            raise JavaRegionParseError("host field frame is not one variable", category="scope")
        node = variables[0]
    name = node.child_by_field_name("name")
    if name is None:
        raise JavaRegionParseError("host source frame has no member identity", category="scope")
    member_name = _text(raw, name)
    owner = member_name if kind == "constructor" else "MmmSourceFrame"
    _parse_one_member(source, kind=kind, name=member_name, owner=owner)


def render_declaration(kind: str, name: str, parts: Mapping[str, Any], *, owner: str,
                       phase: str = "implementation") -> str:
    header = declaration_header(kind, name, parts)
    if "source" in parts:
        source = parts["source"]
        frame = declaration_frame(kind, name, parts)
        if source != frame.get("empty_value") and not (
            source.startswith(frame["prefix"]) and source.endswith(frame["suffix"])
        ):
            from .java_region_parser import JavaRegionParseError

            exc = JavaRegionParseError("source changed the immutable host declaration", category="scope")
            _record_source_failure(source, kind=kind, name=name, owner=owner, phase=phase, exc=exc)
            raise exc
        # The response is already a complete declaration. Never concatenate it
        # after an '=' or insert it as statements inside another declaration.
        return _one_member(source, kind=kind, name=name, owner=owner, phase=phase)
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
        members.extend(render_declaration("field", item["name"], item, owner=name, phase=phase) for item in parts["fields"])
        members.extend(render_declaration("constructor", name, item, owner=name, phase=phase) for item in parts["constructors"])
        members.extend(render_declaration("method", item["name"], item, owner=name, phase=phase) for item in parts["methods"])
        source = header + " {\n" + "\n\n".join(members) + "\n}"
    return _one_member(source, kind=kind, name=name, owner=owner, phase=phase)
