"""Choose Java types as types, with host-owned generic and array syntax."""
from __future__ import annotations

from copy import deepcopy
import re

from .incremental_java_context import _erase, _formals
from .incremental_java_history import PAGE
from .incremental_java_render import validate_value_type
from .incremental_java_schema import choice, obj
from .java_operation_ir import PRIMITIVES

_LANGUAGE_TYPES = ("String", "Object", "java.lang.String", "java.lang.Object", "Runnable",
                   "java.lang.Boolean", "java.lang.Integer", "java.lang.Long", "java.lang.Double",
                   "java.lang.Class", "java.lang.Throwable", "java.lang.Exception", "java.lang.RuntimeException")


def _generated_types(builder):
    root = builder.program["class_name"]
    owners = {root: root}
    for row in builder.program["members"]:
        if row["kind"] in {"class", "record", "enum"}:
            owners[row["name"]] = owners.get(row["owner"], row["owner"]) + "." + row["name"]
    return list(dict.fromkeys([*owners, *owners.values(),
                              *(builder.program["package"] + "." + value for value in owners.values())]))


def _variables(builder):
    members = [builder.member, *[row for row in builder.program["members"] if row["name"] == builder.member["owner"]]]
    return {str(formal).split()[0] for member in members for formal in
            _formals(member.get("metadata", {}).get("type_parameters"))}


def _page_types(builder):
    page = builder.latest_api or {}
    for row in [*page.get("declarations", ()), *page.get("generated_declarations", ())]:
        if row.get("kind", "CLASS") in {"CLASS", "class", "record", "enum"}:
            yield row.get("owner") if row.get("kind", "CLASS") == "CLASS" else row.get("name")
            yield row.get("generic_superclass", row.get("superclass"))
            yield from row.get("generic_interfaces", row.get("interfaces", ()))
    for row in page.get("members", ()):
        yield row.get("generic_return_type", row.get("return_type"))
        for part in row.get("generic_parameters", row.get("parameters", ())):
            yield part.get("type") if isinstance(part, dict) else part


def _known_type(builder, value):
    visible = {*PRIMITIVES, *_LANGUAGE_TYPES, *_generated_types(builder), *_variables(builder)}
    visible.update(_erase(str(row.get("owner") or "")) for row in (builder.latest_api or {}).get("declarations", ()))
    spelling = re.sub(r"@[\w.$]+(?:\([^)]*\))?", "", value)
    names = re.findall(r"[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*", spelling)
    return all(name in visible or name in {"extends", "super"} or builder.api._owner(name)
               for name in names)


def _eligible(builder, value, options):
    if not isinstance(value, str) or not value:
        return False
    try:
        validate_value_type(value, **{k: options.get(k, False) for k in ("allow_void", "allow_varargs", "allow_union")})
    except ValueError:
        return False
    if options.get("reference_only") and value in PRIMITIVES:
        return False
    if options.get("non_array") and value.endswith("[]"):
        return False
    if options.get("scalar_only") and value.removeprefix("java.lang.") not in {
            "boolean", "byte", "short", "char", "int", "long", "float", "double", "String",
            "Boolean", "Byte", "Short", "Character", "Integer", "Long", "Float", "Double"}:
        return False
    if options.get("named_only") and any(token in value for token in ("<", "[", "|", "?", "...")):
        return False
    return _known_type(builder, value)


def suggested_types(builder, expected="", **options):
    parts = builder.member.get("metadata", {})
    preferred = [expected, parts.get("return_type"), parts.get("type")]
    requested = [value for value in _page_types(builder) if _eligible(builder, value, options)]
    locals = [row.get("type") for row in builder.scope.get("locals", ())]
    root = builder.program["class_name"]
    baseline = [*sorted(PRIMITIVES), *_LANGUAGE_TYPES, *sorted(_variables(builder)),
                root, builder.program["package"] + "." + root]
    candidates = [*baseline, *preferred, *requested[:PAGE], *locals[-PAGE:], *_generated_types(builder)[-PAGE:]]
    return list(dict.fromkeys(value for value in candidates if _eligible(builder, value, options)))


def _options(builder, expected, settings, *, empty=False):
    values = suggested_types(builder, expected, **settings)
    if empty:
        values.append("")
    values.append("@lookup")
    if not (settings.get("named_only") or settings.get("scalar_only")):
        values.extend(["@parameterized"] if settings.get("non_array") else ["@array", "@parameterized"])
    if settings.get("allow_union"):
        values.append("@union")
    if settings.get("allow_wildcard"):
        values.append("@wildcard")
    if settings.get("allow_varargs"):
        values.append("@varargs")
    return values


def _count(builder, suffix, minimum=1, maximum=None):
    prop = {"type": "integer", "minimum": minimum}
    if maximum is not None:
        prop["maximum"] = maximum
    return builder.ask(obj({"count": prop}), suffix)["count"]


def _type_parameters(builder, java_type):
    owner = builder.api._owner(java_type)
    if owner:
        return _formals(builder.api.symbols[builder.api.evidence.classes[owner]].get("type_parameters"))
    for row in (builder.latest_api or {}).get("declarations", ()):
        if row.get("owner") == java_type and "type_parameters" in row:
            return _formals(row["type_parameters"])
    return None


def _array_type(builder, marker, suffix):
    frame = builder.type_stack[-1]
    frame["construction"] = marker
    base = select_java_type(builder, suffix + "_component")
    frame["component"] = base
    rank = 1 if marker == "@varargs" else _count(builder, suffix + "_dimensions", maximum=255)
    frame["dimensions"] = rank
    return base + ("..." if marker == "@varargs" else "[]" * rank)


def _wildcard_type(builder, suffix):
    frame = builder.type_stack[-1]
    frame["construction"] = "@wildcard"
    bound = builder.ask(choice("bound", ["unbounded", "extends", "super"]), suffix + "_bound")["bound"]
    frame["bound"] = bound
    return "?" if bound == "unbounded" else "? " + bound + " " + select_java_type(builder, suffix + "_bound_type", reference_only=True)


def _union_type(builder, suffix):
    frame = builder.type_stack[-1]
    frame.update(construction="@union", alternatives=[])
    count = _count(builder, suffix + "_alternatives", minimum=2)
    frame["alternative_count"] = count
    for index in range(count):
        frame["alternative_index"] = index
        frame["alternatives"].append(select_java_type(builder, suffix + "_alternative", reference_only=True))
    return " | ".join(frame["alternatives"])


def _parameterized_type(builder, suffix):
    frame = builder.type_stack[-1]
    frame["construction"] = "@parameterized"
    base = select_java_type(builder, suffix + "_base", reference_only=True, named_only=True)
    formals = _type_parameters(builder, base)
    frame.update(base=base, formals=formals, arguments=[])
    count = _count(builder, suffix + "_type_arguments") if formals is None else len(formals)
    frame["argument_count"] = count
    if not count:
        return base
    for index in range(count):
        frame["argument_index"] = index
        frame["arguments"].append(select_java_type(builder, suffix + "_type_argument", reference_only=True, allow_wildcard=True))
    return base + "<" + ", ".join(frame["arguments"]) + ">"


def _constructed_type(builder, marker, suffix):
    if marker in {"@array", "@varargs"}:
        return _array_type(builder, marker, suffix)
    if marker == "@wildcard":
        return _wildcard_type(builder, suffix)
    if marker == "@union":
        return _union_type(builder, suffix)
    return _parameterized_type(builder, suffix)


def select_java_type(builder, suffix, *, expected="", initial=None, field="type", metadata=None, **settings):
    stack = getattr(builder, "type_stack", [])
    request = {"field": field, "purpose": suffix, "expected": expected, "reference_only": settings.get("reference_only", False)}
    if metadata is not None:
        request["metadata"] = deepcopy(metadata)
    stack.append(request)
    builder.type_stack = stack
    selected = initial
    try:
        while True:
            if selected is None:
                selected = builder.ask(choice("type", _options(builder, expected, settings)), suffix,
                                       {"current_type_request": request, "type_stack": stack})["type"]
            if selected == "@lookup":
                builder.lookup()
                selected = None
                continue
            if selected.startswith("@"):
                selected = _constructed_type(builder, selected, suffix)
            if selected.startswith("?") and settings.get("allow_wildcard"):
                return selected
            validate_value_type(selected, **{k: settings.get(k, False) for k in ("allow_void", "allow_varargs", "allow_union")})
            return selected
    finally:
        stack.pop()


def author_typed_metadata(builder, schema, suffix, fields, *, extra=None, **settings):
    effective = deepcopy(schema)
    for field in fields:
        original = effective["properties"][field]
        options = {**settings, "reference_only": field == "superclass" or suffix in {"lambda", "catch_type", "interface"}}
        effective["properties"][field] = {"type": "string", "enum": _options(builder, "", options,
                                          empty=not original.get("minLength"))}
    result = builder.ask(effective, suffix, extra)
    for field in fields:
        value = result[field]
        if not value:
            continue
        options = {**settings, "reference_only": field == "superclass" or suffix in {"lambda", "catch_type", "interface"}}
        result[field] = select_java_type(builder, suffix + "_" + field, initial=value, field=field, metadata=result, **options)
    return result
