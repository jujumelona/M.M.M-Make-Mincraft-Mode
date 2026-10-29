"""Finish declared work promptly and make repeated helper requests idempotent."""
from __future__ import annotations

import re
from contextlib import contextmanager

from .incremental_java_context import _generated

# Bound recursive helper dependencies, not total helpers or feature size.
MAX_HELPER_DEPENDENCY_DEPTH = 32


@contextmanager
def _helper_dependency_frame(builder):
    previous = getattr(builder, "_active_member_depth", 0)
    if previous >= MAX_HELPER_DEPENDENCY_DEPTH:
        raise RuntimeError("JAVA_HELPER_DEPENDENCY_NESTING_EXHAUSTED")
    builder._active_member_depth = previous + 1
    try:
        yield
    finally:
        builder._active_member_depth = previous
from .incremental_java_model import initial_scope


def _purpose(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _contract(metadata):
    result = dict(metadata)
    result["parameters"] = [
        {key: _purpose(value) if key == "purpose" else value
         for key, value in parameter.items() if key != "name"}
        for parameter in metadata.get("parameters", ())]
    return result


def existing_method(builder, *, owner, purpose, behavior, metadata, name, kind):
    """Compare intent and contract, never just an automatically allocated name."""
    for member in builder.program["members"]:
        if member.get("registrar") or member.get("event"):
            continue
        if member["owner"] != owner or member["kind"] != kind:
            continue
        if name is not None and member["name"] != name:
            continue
        if member.get("behavior") != behavior or _purpose(member.get("purpose")) != _purpose(purpose):
            continue
        if _contract(member.get("metadata") or {}) == _contract(metadata):
            return member
    return None


def declare_method(builder, *, owner, purpose, behavior, metadata, name=None, kind="method", reuse=True):
    if reuse:
        existing = existing_method(builder, owner=owner, purpose=purpose, behavior=behavior,
                                   metadata=metadata, name=name, kind=kind)
        if existing is not None:
            record_declaration(builder, existing, kind, reused=True)
            return existing
    identifier = builder.allocate("m")
    method = {"id": identifier, "owner": owner, "kind": kind,
              "name": name or "mmmHelper" + identifier[1:], "metadata": metadata,
              "purpose": purpose, "behavior": behavior, "body": [], "implementation_status": "pending"}
    builder.program["members"].append(method)
    builder.pending.append(method)
    record_declaration(builder, method, kind)
    return method


def record_declaration(builder, member, operation, *, reused=False):
    outcome = "created"
    if reused:
        outcome = "active" if member.get("implementation_status") == "authoring" else "reused"
    builder.last_declaration = {"op": operation, "outcome": outcome,
                                "member_id": member["id"]}


def declaration_feedback(builder):
    if not builder.last_declaration:
        return {}
    outcome = builder.last_declaration
    member = next(row for row in builder.program["members"] if row["id"] == outcome["member_id"])
    if outcome["outcome"] == "active":
        action, instruction = "implement_current_work", (
            "This work is already being implemented. Implement the current body directly; "
            "do not delegate it back to this active signature or call it as a substitute for its implementation.")
    elif member.get("event"):
        action, instruction = "continue_caller", (
            "The accepted register operation already connects this implemented listener to its event. "
            "Continue the caller's remaining work, or complete its block if the activation is installed.")
    elif builder.member["kind"] in {"class", "record", "enum"}:
        action, instruction = "continue_type_layout", (
            "The declared member's body has been implemented. Continue the type's remaining distinct declarations, "
            "or complete the type layout when all required members exist.")
    elif member["kind"] == "method":
        action, instruction = "use_declaration", (
            "Continue the caller with operations using this exact helper signature where needed. "
            "Repeating its declaration does not invoke it.")
    else:
        action, instruction = "use_declaration", (
            "Use this declared field or type through the appropriate expression or constructor; "
            "continue with distinct remaining work instead of repeating this declaration.")
    return {**outcome, "declaration": _generated(member), "next_action": action, "next_step": instruction}


def finish_member(builder, member):
    """A child's locals, open operations, and diagnostics cannot leak to its caller."""
    # Reject excessive dependency depth before reading or modifying caller
    # state. The depth guard must also cover declaration_scope setup.
    with _helper_dependency_frame(builder):
        previous = (builder.current_operation, builder.declaration_draft, builder.latest_api,
                    builder.rejection, builder.last_declaration)
        with builder.declaration_scope(member):
            builder.current_operation, builder.declaration_draft = "", {}
            builder.latest_api, builder.rejection, builder.last_declaration = {}, {}, {}
            member["implementation_status"] = "authoring"
            try:
                # Publish the actual list before filling it, so accepted child work is
                # visible through the program even while its parent is still active.
                body = member.setdefault("body", [])
                builder.author_block(member, initial_scope(member), [], body=body)
                member["implementation_status"] = "complete"
            finally:
                (builder.current_operation, builder.declaration_draft, builder.latest_api,
                 builder.rejection, builder.last_declaration) = previous


def finish_new_members(builder, pending_before):
    # Initial sibling registrars remain queued. Only declarations made by this
    # operation are dependencies of the current caller and take precedence.
    while len(builder.pending) > pending_before:
        finish_member(builder, builder.pending.pop())
