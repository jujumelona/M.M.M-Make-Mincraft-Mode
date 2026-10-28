"""Liveness gate for structured Java authoring.

Only the host decides which model actions can extend work. This module has no
imports back into authoring implementations, preventing new dependency cycles.
"""
from __future__ import annotations

from .incremental_java_schema import DECLARATION_OPS, choice, operation_options

# Safety valves only: reaching them never marks incomplete source as verified.
_MAX_MODEL_PAGES = 4096
_MAX_CONSECUTIVE_DECLARATIONS = 8
_MAX_CONSECUTIVE_LOOKUPS = 8
_MAX_ACTIVE_MEMBER_DEPTH = 8
_MAX_CONTROL_FLOW_DEPTH = 16
_MAX_BLOCK_OPERATIONS = 256


def check_page_budget(builder):
    if builder.serial >= _MAX_MODEL_PAGES:
        raise RuntimeError(
            "JAVA_INCREMENTAL_MODEL_PAGE_BUDGET_EXHAUSTED: "
            f"member={builder.member['id']} path={builder.path!r}; "
            "no Java source was certified."
        )


def pending_feedback(member, action, instruction):
    if member.get("implementation_status") == "pending":
        return "implement_current_work", (
            "This new helper is queued behind its active dependencies. "
            "Implement the current body; do not delegate it to unfinished work."
        )
    return action, instruction


def author_member_body(builder, member, scope, body):
    try:
        builder.author_block(member, scope, [], body=body)
    except BaseException:
        member["implementation_status"] = "failed"
        raise
    else:
        member["implementation_status"] = "complete"


def drain_new_members(builder, pending_before, finish_member):
    # Avoid recursion through unbounded helper generation. Deferred members
    # stay in the existing queue and are drained when parent work returns.
    while (len(builder.pending) > pending_before
           and len(builder.active_members) < _MAX_ACTIVE_MEMBER_DEPTH):
        finish_member(builder, builder.pending.pop())


def run_block(builder, member, scope, path, body=None, *,
              author_operation, author_declaration, finish_new_members):
    previous = (builder.member, builder.scope, builder.path,
                builder.block, builder.current_operation)
    body = [] if body is None else body
    builder.member, builder.scope, builder.path, builder.block = member, scope, path, body
    blocked_declarations = set()
    declaration_streak, lookup_streak = 0, 0
    depth = len(path) // 2
    try:
        if depth > _MAX_CONTROL_FLOW_DEPTH:
            raise RuntimeError(
                "JAVA_INCREMENTAL_CONTROL_DEPTH_EXCEEDED: "
                f"member={member['id']} path={path!r}"
            )
        while True:
            if len(body) >= _MAX_BLOCK_OPERATIONS:
                raise RuntimeError(
                    "JAVA_INCREMENTAL_BLOCK_OPERATION_BUDGET_EXHAUSTED: "
                    f"member={member['id']} path={path!r}; "
                    "no Java source was certified."
                )
            options = operation_options(member, scope, body, blocked_declarations)
            declarations_closed = (
                declaration_streak >= _MAX_CONSECUTIVE_DECLARATIONS
                and member["kind"] not in {"class", "record", "enum"}
            )
            member_depth_closed = len(builder.active_members) >= _MAX_ACTIVE_MEMBER_DEPTH
            if declarations_closed or member_depth_closed:
                options = [
                    op for op in options
                    if op not in {*DECLARATION_OPS, "constructor", "override"}
                ]
            if member_depth_closed:
                options = [op for op in options if op != "register"]
            if lookup_streak >= _MAX_CONSECUTIVE_LOOKUPS:
                options = [op for op in options if op != "lookup"]
            if depth >= _MAX_CONTROL_FLOW_DEPTH:
                options = [
                    op for op in options
                    if op not in {"if", "while", "do", "foreach", "for",
                                  "switch", "try", "lambda", "synchronized"}
                ]
            if not options:
                raise RuntimeError(
                    "JAVA_INCREMENTAL_NO_ADMISSIBLE_OPERATION: "
                    f"member={member['id']} path={path!r}"
                )
            builder.current_operation = ""
            operation = builder.ask(
                choice("op", options), "kind",
                {"authoring_progress": {
                    "consecutive_declarations_without_statement": declaration_streak,
                    "consecutive_lookups_without_statement": lookup_streak,
                    "active_helper_depth": len(builder.active_members),
                    "control_flow_depth": depth,
                    "must_implement_current_body": declarations_closed or member_depth_closed,
                }}
            )["op"]
            builder.current_operation = operation
            if operation == "complete":
                if declarations_closed and not body:
                    raise RuntimeError(
                        "JAVA_INCREMENTAL_DECLARATION_ONLY_STALL: "
                        f"member={member['id']} declarations={declaration_streak}; "
                        "model created supporting members but no executable body."
                    )
                break
            if operation == "lookup":
                builder.lookup()
                lookup_streak += 1
                continue
            if operation in {*DECLARATION_OPS, "constructor", "override"}:
                pending_before = len(builder.pending)
                author_declaration(builder, operation)
                finish_new_members(builder, pending_before)
                declaration_streak += 1
                if builder.last_declaration.get("outcome") in {"reused", "active"}:
                    blocked_declarations.add(operation)
                continue
            pending_before = len(builder.pending)
            node = author_operation(builder, operation)
            if node is None:
                blocked_declarations.add(operation)
                continue
            body.append(node)
            builder.history[node["id"]] = node
            builder.rejection = {}
            blocked_declarations.clear()
            declaration_streak = lookup_streak = 0
            finish_new_members(builder, pending_before)
            if operation in {"return", "throw", "break", "continue"}:
                break
        return body
    finally:
        builder.member, builder.scope, builder.path, builder.block, builder.current_operation = previous
