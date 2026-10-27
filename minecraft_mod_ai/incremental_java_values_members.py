"""Paginated generated/indexed member selection and exact invocation contracts."""
from __future__ import annotations

from copy import deepcopy

from .incremental_java_calls import _signature
from .incremental_java_context import _erase
from .incremental_java_values_catalog import (
    generated_members, generated_type, implicit_constructor, indexed_members, superclass_type,
)
from .incremental_java_schema import choice, obj
from .incremental_java_values_model import JavaReceiver, current_owner, receiver_source, request

_PAGE = 10


def _static(row):
    return bool(row.get("static") or int(row.get("access") or 0) & 8)


def _final(row):
    return bool(row.get("final") or int(row.get("access") or 0) & 16)


def _generated(builder, owner, kind):
    return generated_members(builder, owner, kind)


def _default_constructor(builder, owner):
    return implicit_constructor(builder, owner)


def _eligible(row, receiver, kind, *, nonvoid=False, writable=False):
    if row.get("kind") != kind:
        return False
    if kind != "CONSTRUCTOR" and not receiver.unbound and _static(row) != receiver.static:
        return False
    if nonvoid and row.get("generic_return_type", row.get("return_type", "void")) == "void":
        return False
    return not writable or not _final(row)


def _generated_page(builder, receiver, kind, query, offset, options):
    rows = _generated(builder, receiver.java_type, kind)
    if kind == "CONSTRUCTOR" and not rows:
        rows = _default_constructor(builder, receiver.java_type)
    rows = [row for row in rows if _eligible(row, receiver, kind, **options)
            and query.casefold() in (row["name"] + " " + row.get("purpose", "")).casefold()]
    return {row["id"]: row for row in rows[offset:offset + _PAGE]}, len(rows)


def _api_page(builder, receiver, kind, query, offset, options):
    if receiver.java_type.endswith("[]") and kind == "FIELD":
        row = {"id": "array:length", "kind": "FIELD", "owner": receiver.java_type,
               "name": "length", "return_type": "int", "final": True}
        rows = {"array:length": row} if _eligible(row, receiver, kind, **options) and query in {"", "length"} else {}
        return rows, {"query": query, "offset": 0, "total": len(rows), "next_offset": None}
    candidates = indexed_members(builder, receiver, kind)
    candidates = [row for row in candidates if _eligible(row, receiver, kind, **options)
                  and query.casefold() in row["name"].casefold()]
    rows = {"api:" + str(index): row for index, row in enumerate(candidates[offset:offset + _PAGE], offset)}
    total = len(candidates)
    return rows, {"query": query, "offset": offset, "total": total,
                  "next_offset": offset + _PAGE if offset + _PAGE < total else None}


def _member_page(builder, receiver, kind, query, location, options):
    source, offset = location
    generated, total = _generated_page(builder, receiver, kind, query, offset, options)
    if source == "generated" and total:
        next_page = ("generated", offset + _PAGE) if offset + _PAGE < total else ("api", 0)
        return generated, {"offset": offset, "total": total, "catalog": "generated"}, next_page
    rows, page = _api_page(builder, receiver, kind, query, 0 if source == "generated" else offset, options)
    following = page.get("next_offset")
    return rows, page, ("api", following) if following is not None else None


def _specialize(builder, receiver, selected):
    return deepcopy(selected)


def select_member(builder, receiver, kind, suffix, *, nonvoid=False, writable=False, allow_helper=False):
    query, location, history = "", ("generated", 0), []
    filters = {"nonvoid": nonvoid, "writable": writable}
    while True:
        rows, page, following = _member_page(builder, receiver, kind, query, location, filters)
        options = [*rows, "search", "change_receiver"]
        if allow_helper:
            options.append("new_helper")
        if following is not None:
            options.append("next_page")
        if history:
            options.append("previous_page")
        picked = request(builder, choice("member_id", options), suffix, {
            "member_candidates": rows, "member_page": {key: page[key] for key in
                ("query", "offset", "total", "catalog", "next_offset") if key in page},
            "selected_receiver": {"source": receiver.source, "type": receiver.java_type}})["member_id"]
        if picked in rows:
            return _specialize(builder, receiver, rows[picked])
        if picked in {"new_helper", "change_receiver"}:
            return picked
        if picked == "search":
            query = request(builder, obj({"query": {"type": "string"}}), "member_search")["query"]
            location, history = ("generated", 0), []
        elif picked == "next_page":
            history.append(location)
            location = following
        else:
            location = history.pop()


def _super_type(builder):
    return superclass_type(builder)


def select_receiver(author, suffix, *, reference=False):
    builder = author.builder
    options = ["static", "value"]
    if not builder.scope.get("static"):
        options = ["this", "super", *options]
    selected = request(builder, choice("receiver_kind", options), suffix)["receiver_kind"]
    if selected == "static":
        java_type = builder.select_java_type("static_owner", reference_only=True)
        return JavaReceiver(_erase(java_type), java_type, static=True, unbound=reference)
    if selected == "value":
        value = author.value("receiver", "java.lang.Object")
        return JavaReceiver(receiver_source(value), value.java_type)
    return JavaReceiver(selected, current_owner(builder) if selected == "this" else _super_type(builder))


def _can_declare(builder, receiver):
    return generated_type(builder, receiver.java_type) is not None


def select_call(author, node, *, nonvoid=False, reference=False):
    builder = author.builder
    while True:
        receiver = select_receiver(author, "call_receiver", reference=reference)
        node.update(receiver=receiver.source, receiver_type=receiver.java_type)
        selected = select_member(builder, receiver, "METHOD", "call_member", nonvoid=nonvoid,
                                 allow_helper=_can_declare(builder, receiver))
        if selected == "change_receiver":
            continue
        if selected == "new_helper":
            selected = author.helper_contract(nonvoid=nonvoid, owner=receiver.java_type, static=receiver.static)
        node.update(method=selected["name"], call_contract=_signature(selected))
        return receiver
