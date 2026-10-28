"""Finish declared work promptly and make repeated helper requests idempotent."""
from __future__ import annotations

import re

from .incremental_java_context import _generated
from .incremental_java_model import initial_scope


_REFERENCE_FIELDS = frozenset((
    "body", "otherwise", "finally", "cases", "catches", "value",
    "condition", "target", "iterable", "update", "monitor", "selector",
    "arguments", "initializer", "receiver", "label",
))
_BLOCK_FIELDS = frozenset(("body", "otherwise", "finally", "cases", "catches"))
# Protect Python and JSON-schema traversal without capping total generated work.
_MAX_CONTROL_NESTING = 24
_NESTED_CONTROL_OPS = frozenset((
    "if", "while", "do", "foreach", "for", "block", "synchronized",
    "switch", "try", "lambda",
))
_JAVA_LITERAL = re.compile(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""")


def _operation_effect(node, members, visited):
    operation = node.get("op")
    if operation in {"assign", "register", "throw", "constructor_call"}:
        return True
    if operation == "return":
        return bool(node.get("value"))
    if operation != "call":
        return False
    callee = next((row for row in members
                   if row.get("kind") == "method" and row.get("name") == node.get("method")), None)
    if callee is None:
        return True  # External API calls remain subject to compiler verification.
    if callee["id"] in visited:
        return False
    return _observable_work(callee.get("body"), members, visited | {callee["id"]})


def _nested_effect(node, members, visited):
    for key, child in node.items():
        if key not in _BLOCK_FIELDS:
            continue
        if isinstance(child, dict):
            child = [child]
        if isinstance(child, list) and _observable_work(child, members, visited):
            return True
    return False


def _observable_work(body, members, visited=frozenset()):
    """Cyclic generated-call chains cannot fake executable implementation work."""
    for node in body or ():
        if isinstance(node, dict) and (
            _operation_effect(node, members, visited)
            or _nested_effect(node, members, visited)
        ):
            return True
    return False


def _java_reference(expression, name):
    source = _JAVA_LITERAL.sub("", expression)
    escaped = re.escape(name)
    return bool(re.search(r"(?<![\w$])" + escaped + r"\s*\("
                          r"|::" + escaped + r"(?![\w$])", source))


def _linked_in_ast(node, name):
    """Only accepted statement and expression fields prove a helper reference."""
    if isinstance(node, str):
        return _java_reference(node, name)
    if isinstance(node, list):
        return any(_linked_in_ast(item, name) for item in node)
    if not isinstance(node, dict):
        return False
    if node.get("op") == "call" and node.get("method") == name:
        return True
    if node.get("op") == "register" and node.get("handler") == name:
        return True
    return any(_linked_in_ast(value, name) for key, value in node.items()
               if key in _REFERENCE_FIELDS)


def _unconnected(builder, body, declared_ids):
    """Track every helper created by this member, including nested lexical blocks."""
    members = builder.program["members"]
    created = set(declared_ids)
    created.update(row["id"] for row in members
                   if (row.get("created_by") == builder.member["id"]
                       and row.get("kind") == "method" and not row.get("event")))
    outstanding = []
    for target in members:
        if target["id"] not in created:
            continue
        # A helper may be connected from a sibling listener, not only its
        # lexical parent. Ignore the helper's own body as proof of connection.
        sources = [body]
        sources.extend(row.get("body", []) for row in members if row is not target)
        if not any(_linked_in_ast(source, target["name"]) for source in sources):
            outstanding.append(target["id"])
    return outstanding


def _waiting_ancestor(builder, current):
    """An incomplete ancestor cannot delegate to another unimplemented helper."""
    members = builder.program["members"]
    return next((row for row in members
                 if row is not current and row.get("implementation_status") == "authoring"
                 and not row.get("registrar") and row.get("kind") in {"method", "constructor"}
                 and not _observable_work(row.get("body"), members, frozenset({row["id"]}))), None)


def frontier_options(builder, current, body, options, declared_ids):
    """Host-owned liveness gate; independent of feature names and model prompts.

    There is deliberately no arbitrary number-of-operations limit. New helpers
    remain possible after their predecessors are connected. The host refuses
    recursive delegation without implementation and orphan-declaration growth.
    """
    outstanding = _unconnected(builder, body, declared_ids)
    ancestor = _waiting_ancestor(builder, current)
    # A nested helper must implement its own body; it cannot indefinitely
    # delegate all of its work into another unfinished helper.
    nested_member = bool(current.get("created_by"))
    # Anonymous lexical blocks nested inside another anonymous lexical block
    # create no new behavior. Refuse recursive structural growth at its source.
    drafts = getattr(builder, "drafts", ())
    nested_block = any(row.get("op") == "block" for row in drafts)
    at_depth_limit = len(drafts) >= _MAX_CONTROL_NESTING
    builder.frontier_unconnected = (outstanding if current["kind"] not in
                                    {"class", "record", "enum"} else [])
    builder.frontier_feedback = {}
    if (outstanding or ancestor or nested_member) and current["kind"] not in {"class", "record", "enum"}:
        excluded = {"field", "method", "type", "constructor", "override"}
        if nested_member:
            # Event callbacks must not register fresh listeners recursively.
            excluded.add("register")
        if outstanding:
            excluded.add("complete")
        options = [op for op in options if op not in excluded]
        builder.frontier_feedback = {
            "state": "implementation_required",
            "unconnected_member_ids": outstanding,
            "waiting_ancestor_id": ancestor["id"] if ancestor else None,
            "next_action": ("Reference an already completed helper in an executable "
                            "operation, or implement this method directly; a new "
                            "declaration is not an implementation."),
        }
    if nested_member and current["kind"] in {"class", "record", "enum"}:
        options = [op for op in options if op != "type"]
    if nested_block:
        options = [op for op in options if op != "block"]
    if at_depth_limit:
        options = [op for op in options if op not in _NESTED_CONTROL_OPS]
    return options


def helper_delegation_allowed(builder):
    """Use the same dependency invariant for expression-level new_helper."""
    return (not getattr(builder, "frontier_unconnected", ())
            and not (builder.member or {}).get("created_by")
            and _waiting_ancestor(builder, builder.member) is None)


def constrain_helper_value_choices(builder, options):
    if not helper_delegation_allowed(builder):
        return [name for name in options if name != "new_helper"]
    return options


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
              "purpose": purpose, "behavior": behavior, "body": [], "implementation_status": "pending",
              "created_by": builder.member["id"]}
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


def assert_member_connections(builder, member):
    """Terminal returns cannot bypass the dependency frontier."""
    if member["kind"] in {"class", "record", "enum"}:
        return
    disconnected = _unconnected(builder, member.get("body", []), set())
    if disconnected:
        raise ValueError("JAVA_INCREMENTAL_UNCONNECTED_DECLARATIONS: " + ",".join(disconnected))


def finish_member(builder, member):
    """A child's locals, open operations, and diagnostics cannot leak to its caller."""
    previous = (builder.current_operation, builder.declaration_draft, builder.latest_api,
                builder.rejection, builder.last_declaration,
                builder.frontier_unconnected, builder.frontier_feedback)
    with builder.declaration_scope(member):
        builder.current_operation, builder.declaration_draft = "", {}
        builder.latest_api, builder.rejection, builder.last_declaration = {}, {}, {}
        member["implementation_status"] = "authoring"
        try:
            # Publish the actual list before filling it, so accepted child work is
            # visible through the program even while its parent is still active.
            body = member.setdefault("body", [])
            builder.author_block(member, initial_scope(member), [], body=body)
            assert_member_connections(builder, member)
            member["implementation_status"] = "complete"
        finally:
            (builder.current_operation, builder.declaration_draft, builder.latest_api,
             builder.rejection, builder.last_declaration,
             builder.frontier_unconnected, builder.frontier_feedback) = previous


def finish_new_members(builder, pending_before):
    # Initial sibling registrars remain queued. Only declarations made by this
    # operation are dependencies of the current caller and take precedence.
    while len(builder.pending) > pending_before:
        finish_member(builder, builder.pending.pop())
