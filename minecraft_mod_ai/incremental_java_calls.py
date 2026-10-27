"""Resolve call contracts and author their arguments one expression at a time."""
from __future__ import annotations

import re

from .incremental_java_context import _erase, _ground_argument
from .incremental_java_schema import COUNT, choice, obj
from .java_operation_ir import parameter_types

_PAGE = 10
_SIGNATURE_FIELDS = ("kind", "owner", "name", "parameters", "generic_parameters", "return_type",
                     "generic_return_type", "type_parameters", "throws", "generic_throws",
                     "static", "access", "varargs")


def _owner_matches(left, right):
    left, right = _erase(str(left)).replace("$", "."), _erase(str(right)).replace("$", ".")
    return (left == right or "." not in right and left.rsplit(".", 1)[-1] == right
            or "." not in left and right.rsplit(".", 1)[-1] == left)


def _current_owner(builder):
    member = builder.member or {}
    if member.get("kind") in {"class", "record", "enum"}:
        return member["name"]
    return member.get("owner") or builder.program["class_name"]


def _receiver_type(builder, receiver):
    if receiver in {"", "this"}:
        return _current_owner(builder)
    for local in reversed(builder.scope.get("locals", ())):
        if local["name"] == receiver:
            return local["type"]
    field_name = receiver.removeprefix("this.")
    for member in builder.program.get("members", ()):
        if (member["kind"] == "field" and member["name"] == field_name
                and _owner_matches(member["owner"], _current_owner(builder))):
            return member["metadata"]["type"]
    return receiver if re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", receiver) else ""


def _generated_methods(builder, owner, method):
    return [{"kind": "METHOD", "owner": row["owner"], "name": row["name"], **row["metadata"]}
            for row in builder.program.get("members", ())
            if row["kind"] == "method" and row["name"] == method and _owner_matches(row["owner"], owner)]


def _lookup_pages(api, query):
    offset, seen = 0, set()
    while offset is not None and offset not in seen:
        seen.add(offset)
        page = api.lookup(query, offset)
        yield page
        offset = page.get("next_offset")


def _page_methods(page, owner, method):
    owners = [row.get("owner", row.get("name", "")) for row in page.get("declarations", ())]
    exact_owner = any(_owner_matches(value, owner) for value in owners)
    return [row for row in page.get("members", ())
            if row.get("kind") == "METHOD" and row.get("name") == method
            and (exact_owner or _owner_matches(row.get("owner", ""), owner))]


def _method_key(row):
    return row["name"], tuple(parameter_types(row))


def _api_methods(builder, owner, method):
    indexed = {}
    for page in _lookup_pages(builder.api, _erase(owner) + "#" + method):
        indexed.update((_method_key(row), row) for row in _page_methods(page, owner, method))
    for row in builder.api.override(owner, method):
        key = _method_key(row)
        if key in indexed:
            indexed[key] = row
    return list(indexed.values())


def _call_candidates(builder, node):
    method = node.get("method")
    if not method:
        return []
    receiver = str(node.get("receiver") or "").strip()
    owner = _receiver_type(builder, receiver)
    if not owner:
        return []
    generated = _generated_methods(builder, owner, method)
    if not generated and not receiver:
        generated = _generated_methods(builder, builder.program["class_name"], method)
    return generated or _api_methods(builder, owner, method)


def _signature(row):
    return {field: row[field] for field in _SIGNATURE_FIELDS if field in row}


def _select_overload(builder, candidates):
    if len(candidates) == 1:
        return candidates[0]
    offset = 0
    while True:
        page = {"overload" + str(index): _signature(row)
                for index, row in enumerate(candidates[offset:offset + _PAGE], offset)}
        options = list(page)
        if offset + _PAGE < len(candidates):
            options.append("next_page")
        if offset:
            options.append("previous_page")
        selected = builder.ask(choice("overload_id", options), "call_overload", {
            "overload_candidates": page, "overload_offset": offset, "overload_total": len(candidates)})["overload_id"]
        if selected in page:
            return candidates[int(selected.removeprefix("overload"))]
        offset += _PAGE if selected == "next_page" else -_PAGE


def _source_parameters(row):
    values = row.get("generic_parameters", row.get("parameters", ()))
    return [value["type"] if isinstance(value, dict) else value for value in values]


def _is_varargs(row, parameters):
    return bool(parameters and (row.get("varargs") or int(row.get("access") or 0) & 0x80
                               or parameters[-1].endswith("...")))


def _argument_types(builder, contract):
    if contract is None:
        return ["java.lang.Object"] * builder.ask(COUNT, "arguments_count")["count"]
    parameters = _source_parameters(contract)
    if not _is_varargs(contract, parameters):
        return parameters
    fixed = parameters[:-1]
    schema = obj({"count": {"type": "integer", "minimum": len(fixed), "maximum": 255}})
    count = builder.ask(schema, "arguments_count", {"varargs_component_type": parameters[-1],
                                                  "fixed_argument_count": len(fixed)})["count"]
    component = re.sub(r"(?:\[\s*\]|\.\.\.)\s*$", "", parameters[-1]).strip()
    return [*fixed, *([component] * (count - len(fixed)))]


def author_call_arguments(builder, node) -> None:
    """Use an exact selected signature when known; unknown calls remain authorable."""
    contract = _call_contract(builder, node)
    if contract is not None:
        node["call_contract"] = _signature(contract)
    types = _argument_types(builder, contract)
    node["arguments"] = []
    for index, expected in enumerate(types):
        node["arguments"].append(builder.expression("argument:" + str(index), _ground_argument(expected, "")))


def _call_contract(builder, node):
    if node.get("call_contract") is not None:
        return node["call_contract"]
    candidates = _call_candidates(builder, node)
    return _select_overload(builder, candidates) if candidates else None
