"""Declare new supporting code immediately and queue its implementation."""
from __future__ import annotations

from .incremental_java_schema import SCHEMAS, COUNT, TYPE, obj, choice
from .incremental_java_render import validate_type


def _field(builder):
    values = builder.typed(SCHEMAS["field"], "field", "type")
    builder.declaration_draft.update(values)
    validate_type(values["type"])
    identifier = builder.allocate("f")
    member = {"id": identifier, "owner": _owner(builder), "kind": "field",
              "name": "mmmField" + identifier[1:], "purpose": values.pop("purpose"),
              "behavior": builder.member["behavior"], "metadata": {"visibility": "private", **values}}
    builder.declaration_draft.update(name=member["name"], purpose=member["purpose"])
    with builder.declaration_scope(member):
        member["initializer"] = builder.expression("initializer", values["type"])
    builder.program["members"].append(member)
    return member


def _owner(builder):
    return builder.member["name"] if builder.member["kind"] in {"class", "record", "enum"} else builder.member["owner"]


def _method(builder):
    from .incremental_java_model import method_metadata
    values = builder.typed(SCHEMAS["method"], "method", "return_type", allow_void=True)
    builder.declaration_draft.update(values)
    validate_type(values["return_type"])
    return builder.new_method(owner=_owner(builder), purpose=values["purpose"], behavior=builder.member["behavior"],
        metadata=method_metadata(values["return_type"], builder.parameters(), static=values["static"]))


def _type(builder):
    values = builder.typed(SCHEMAS["type"], "type", "superclass")
    builder.declaration_draft.update(values)
    if values["superclass"]:
        validate_type(values["superclass"])
    from .incremental_java_progress import MAX_HELPER_DEPENDENCY_DEPTH
    depth = int((builder.member or {}).get("generation_depth", 0)) + 1
    if depth > MAX_HELPER_DEPENDENCY_DEPTH:
        raise RuntimeError("INCREMENTAL_JAVA_HELPER_DEPENDENCY_CYCLE: "
                           f"member={builder.member['id']}; depth={depth}")
    identifier = builder.allocate("t")
    builder.declaration_draft.update(name="MmmType" + identifier[1:], owner=_owner(builder))
    metadata = {"visibility": "private", "static": True, "interfaces": [],
                "fields": [], "constructors": [], "methods": []}
    count = builder.ask(COUNT, "interfaces_count")["count"]
    for index in range(count):
        interface = builder.typed(obj({"type": TYPE}), "interface", "type", extra={"interface_index": index})["type"]
        validate_type(interface)
        metadata["interfaces"].append(interface)
    _type_layout(builder, values, metadata)
    member = {"id": identifier, "owner": _owner(builder), "kind": values["kind"], "generation_depth": depth,
              "name": "MmmType" + identifier[1:], "purpose": values["purpose"],
              "behavior": builder.member["behavior"], "metadata": metadata, "body": []}
    builder.program["members"].append(member)
    builder.pending.append(member)
    return member


def _type_layout(builder, values, metadata):
    kind = values["kind"]
    if kind == "class":
        metadata["superclass"] = values["superclass"]
    elif kind == "record":
        metadata["components"] = builder.parameters()
    else:
        metadata["constants"] = []
        for _ in range(builder.ask(COUNT, "constants_count")["count"]):
            constant = builder.ask(SCHEMAS["constant"], "constant")
            metadata["constants"].append({"name": constant["name"], "arguments": ""})


def _constructor(builder):
    from .incremental_java_model import method_metadata
    owner = _owner(builder)
    metadata = method_metadata(parameters=builder.parameters())
    metadata.pop("return_type")
    return builder.new_method(owner=owner, purpose="Initialize this new type according to its purpose.",
        behavior=builder.member["behavior"], metadata=metadata, name=owner, kind="constructor")


def _override(builder):
    from .incremental_java_model import method_metadata
    metadata = builder.member["metadata"]
    parents = [metadata.get("superclass", ""), *metadata.get("interfaces", ())]
    choices = []
    for parent in filter(None, parents):
        choices.extend(builder.api.override(parent))
    choices = [row for row in choices if not (row.get("static") or int(row.get("access", 0)) & 0x12)]
    if not choices:
        raise ValueError("Inspect the parent type's overridable API first, or declare a new helper method.")
    selected = _select_override(builder, choices)
    types = selected.get("generic_parameters", selected.get("parameters", ()))
    parameters = [{"type": value, "name": "arg" + str(index)} for index, value in enumerate(types)]
    declaration = method_metadata(selected.get("generic_return_type", selected["return_type"]), parameters)
    declaration.update(visibility="public", annotations=["Override"],
        type_parameters=", ".join(selected.get("type_parameters", ())),
        throws=selected.get("generic_throws", selected.get("throws", [])))
    return builder.new_method(owner=_owner(builder), purpose="Implement " + selected["name"] + " for " + builder.member["purpose"],
        behavior=builder.member["behavior"], metadata=declaration, name=selected["name"])


def _select_override(builder, rows):
    offset, size = 0, 10
    while True:
        indexed = {"override" + str(index): row for index, row in enumerate(rows[offset:offset + size], offset)}
        options = list(indexed)
        if offset + size < len(rows):
            options.append("@next")
        if offset:
            options.append("@previous")
        selected = builder.ask(choice("override_id", options), "override", {
            "override_candidates": indexed, "override_offset": offset, "override_total": len(rows)})["override_id"]
        if selected in indexed:
            return indexed[selected]
        offset += size if selected == "@next" else -size


_HANDLERS = {"field": _field, "method": _method, "type": _type, "constructor": _constructor, "override": _override}


def author_declaration(builder, operation):
    from .incremental_java_progress import record_declaration
    previous = builder.declaration_draft
    builder.declaration_draft = {"kind": operation}
    count = len(builder.program["members"])
    try:
        member = _HANDLERS[operation](builder)
        record_declaration(builder, member, operation, reused=len(builder.program["members"]) == count)
        return member
    finally:
        builder.declaration_draft = previous
