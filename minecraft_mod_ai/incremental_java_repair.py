"""Map compiler diagnostics to one accepted AST operation or declaration slot."""
from __future__ import annotations

import re

from .incremental_java_author import IncrementalJavaBuilder
from .incremental_java_model import initial_scope
from .incremental_java_schema import choice
from .incremental_java_scope import scope_at
from .incremental_java_completion import complete_program_block
from .incremental_java_edits import insert_before, relocate, can_relocate
from .incremental_java_history import operation_view
from .incremental_java_repair_slots import (replace_node, replace_declaration, revise_operation, create_helper,
                                          repair_related_declaration)
from .java_member_diagnostics import _first_compiler_error


def _walk(nodes):
    for node in nodes:
        yield node
        for key in ("body", "otherwise", "finally"):
            yield from _walk(node.get(key, []))
        for case in node.get("cases", []):
            yield from _walk(case.get("body", []))
        for caught in node.get("catches", []):
            yield from _walk(caught.get("body", []))


def _mapped_target(builder, diagnostic):
    filename = builder.spec["target_path"].rsplit("/", 1)[-1]
    diagnostic = _first_compiler_error(diagnostic, filename=filename)
    match = re.search(re.escape(filename) + r":(\d+):", diagnostic)
    if match is None:
        raise ValueError("JAVA_INCREMENTAL_REPAIR_NO_SOURCE_LINE")
    line = int(match[1])
    location = builder.line_map.get(line)
    if not location:
        raise ValueError("JAVA_INCREMENTAL_REPAIR_UNMAPPED_LINE: " + str(line))
    member = next(m for m in builder.program["members"] if m["id"] == location["member_id"])
    node = next((n for n in _walk(member.get("body", [])) if n["id"] == location["node_id"]), None)
    return diagnostic, member, node, location["slot"]


def _count_attempt(builder, member, node, slot):
    state = builder.context.setdefault("incremental_state", {})
    attempts = state.setdefault("repair_attempts", {})
    root = node.get("_repair_origin", node["id"]) if node else "declaration"
    key = member["id"] + ":" + root + ":" + slot
    attempts[key] = attempts.get(key, 0) + 1
    acceptance = (builder.context.get("feature_work_order") or {}).get("acceptance") or {}
    limit = acceptance.get("max_operation_repair_attempts", 2)
    if type(limit) is not int or limit < 1:
        raise ValueError("JAVA_INCREMENTAL_OPERATION_REPAIR_BUDGET_INVALID")
    if attempts[key] > limit:
        raise ValueError("JAVA_INCREMENTAL_OPERATION_REPAIR_EXHAUSTED: " + key)


def _perform_repair(builder, member, node, slot, diagnostic):
    if complete_program_block(builder, member, node, diagnostic):
        return
    choices = ["mapped_operand", "declaration"]
    if node:
        choices.append("insert_before")
        if can_relocate(member, node):
            choices.append("relocate")
    if node and node["op"] == "call":
        choices.append("operation")
        if node.get("receiver", "") in {"", "this"}:
            choices.append("new_helper")
    selected = _select_repair(builder, choices)
    if selected == "insert_before":
        insert_before(builder, member, node)
    elif selected == "relocate":
        relocate(builder, member, node)
    elif selected == "new_helper":
        create_helper(builder, node)
    elif selected == "operation":
        revise_operation(builder, node)
    elif selected == "declaration" and node:
        repair_related_declaration(builder, member)
    elif node is None:
        replace_declaration(builder, member, slot)
    else:
        replace_node(builder, node, slot)


def _repair_choice_pages(options):
    # Two evidence reads, then an obligatory concrete correction. Do not
    # count a lookup as a source repair or silently accept an empty mutation.
    from .incremental_java_values import MAX_VALUE_EVIDENCE_LOOKUPS
    for _ in range(MAX_VALUE_EVIDENCE_LOOKUPS):
        yield [*options, "lookup"]
    yield list(options)


def _select_repair(builder, options):
    for choices in _repair_choice_pages(options):
        selected = builder.ask(choice("repair", choices), "repair_target")["repair"]
        if selected != "lookup":
            return selected
        builder.lookup()
    raise AssertionError("unreachable")


def _validate_repaired_call(member, node):
    """Compiler repair cannot reintroduce a direct recursive self-call."""
    if node is not None:
        from .incremental_java_author import _reject_active_self_call
        _reject_active_self_call(member, node)


def repair_incremental_program(router, source, diagnostic, *, program, spec, messages,
                               construction_context, behaviors):
    builder = IncrementalJavaBuilder(router, spec, messages, construction_context, behaviors, program=program)
    reconstructed, _ = builder.render()
    if source != reconstructed:
        raise ValueError("JAVA_INCREMENTAL_CANDIDATE_PROGRAM_MISMATCH")
    diagnostic, member, node, slot = _mapped_target(builder, diagnostic)
    builder.member = member
    builder.scope = scope_at(member, node["id"]) if node else initial_scope(member)
    builder.block = member.get("body", [])
    builder.current_operation = node["op"] if node else "declaration"
    builder.rejection = {"javac_diagnostic": diagnostic, "rejected_operation": operation_view(node) if node else None,
                         "rejected_declaration": {k: member[k] for k in ("id", "name", "metadata", "purpose")},
                         "repair_slot": slot,
                         "repair_instruction": "Correct this exact operand/type using the diagnostic. Preserve the operation, control flow and all other accepted work."}
    if node and node["op"] == "for" and slot in {"condition", "update"}:
        builder.scope["locals"].append({"name": node["name"], "type": node["type"]})
    _count_attempt(builder, member, node, slot)
    builder.drafts = [node] if node else []
    _perform_repair(builder, member, node, slot, diagnostic)
    _validate_repaired_call(member, node)
    from .incremental_java_progress import finish_member
    while builder.pending:
        finish_member(builder, builder.pending.popleft())
    result, revised = builder.render()
    if result == source:
        raise ValueError("JAVA_INCREMENTAL_REPAIR_NO_PROGRESS")
    return result, revised
