"""Finish declared work promptly and make repeated helper requests idempotent."""
from __future__ import annotations

import re

from .incremental_java_context import _generated
from .incremental_java_model import initial_scope

# A cyclic helper dependency is an implementation error, not a reason to
# recursively generate another layer forever. Independent sibling helpers are
# not limited by this bound.
MAX_HELPER_DEPENDENCY_DEPTH = 32


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
    depth = int((builder.member or {}).get("generation_depth", 0)) + 1
    if depth > MAX_HELPER_DEPENDENCY_DEPTH:
        raise RuntimeError(
            "INCREMENTAL_JAVA_HELPER_DEPENDENCY_CYCLE: "
            f"member={(builder.member or {}).get('id')}; depth={depth}; "
            "generate an implementation using existing helpers instead of another wrapper"
        )
    identifier = builder.allocate("m")
    method = {"id": identifier, "owner": owner, "kind": kind, "generation_depth": depth,
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
            "The accepted register operation already connects the queued listener to its event. "
            "Continue the caller's remaining work, or complete its block if the activation is installed.")
    elif builder.member["kind"] in {"class", "record", "enum"}:
        action, instruction = "continue_type_layout", (
            "The declared member is queued for authoring. Continue the type's remaining distinct declarations, "
            "or complete the type layout when all required members exist.")
    elif member["kind"] == "method":
        action, instruction = "use_declaration", (
            "Continue the caller with operations using this exact helper signature where needed. "
            "Its implementation is scheduled and must complete before rendering. "
            "Repeating its declaration does not invoke it.")
    else:
        action, instruction = "use_declaration", (
            "Use this declared field or type through the appropriate expression or constructor; "
            "continue with distinct remaining work instead of repeating this declaration.")
    return {**outcome, "declaration": _generated(member), "next_action": action, "next_step": instruction}


def finish_member(builder, member):
    """A child's locals, open operations, and diagnostics cannot leak to its caller."""
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
    """Legacy hook: leave declarations queued for iterative builder.finish().

    No caller may synchronously author pending members while an operation is
    active: that recreates the helper recursion this work queue prevents.
    """
    return None
