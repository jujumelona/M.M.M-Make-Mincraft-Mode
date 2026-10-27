"""Dependency edits around an error retain every unrelated accepted operation."""
from __future__ import annotations

from .incremental_java_history import child_blocks, operation_view, walk_nodes, PAGE
from .incremental_java_scope import scope_at
from .incremental_java_schema import SIMPLE_OPS, choice


def find_location(member, node_id):
    def visit(nodes, ancestors):
        for index, node in enumerate(nodes):
            if node["id"] == node_id:
                return nodes, index, ancestors
            for _, children in child_blocks(node):
                found = visit(children, [*ancestors, node])
                if found:
                    return found
        return None
    result = visit(member.get("body", []), [])
    if result is None:
        raise ValueError("JAVA_INCREMENTAL_EDIT_NODE_MISSING: " + node_id)
    return result


def insert_before(builder, member, node):
    from .incremental_java_operations import author_operation
    container, index, ancestors = find_location(member, node["id"])
    candidates = {item["id"]: item for item in [node, *reversed(ancestors)]}
    selected = builder.ask(choice("before_node", candidates), "repair_insert_location",
                           {"nearby_operations": [operation_view(item) for item in candidates.values()]})["before_node"]
    target = candidates[selected]
    container, index, _ = find_location(member, target["id"])
    builder.scope = scope_at(member, selected)
    builder.block = container[:index]
    builder.drafts = []
    options = [op for op in SIMPLE_OPS if op not in {"return", "throw", "register"}]
    operation = builder.ask(choice("op", options), "repair_insert_kind")["op"]
    builder.current_operation = operation
    inserted = author_operation(builder, operation)
    container.insert(index, inserted)
    builder.history[inserted["id"]] = inserted


def _select_anchor(builder, candidates):
    offset = 0
    while True:
        page = candidates[offset:offset + PAGE]
        options = [node["id"] for node in page]
        if offset + PAGE < len(candidates):
            options.append("@next")
        if offset:
            options.append("@previous")
        selected = builder.ask(choice("anchor_id", options), "repair_move_anchor", {
            "anchor_candidates": [operation_view(node) for node in page],
            "anchor_offset": offset, "anchor_total": len(candidates)})["anchor_id"]
        if selected in {node["id"] for node in page}:
            return selected
        offset += PAGE if selected == "@next" else -PAGE


def can_relocate(member, node):
    excluded = {item["id"] for item in walk_nodes([node])}
    return any(item["id"] not in excluded for item in walk_nodes(member.get("body", [])))


def relocate(builder, member, node):
    excluded = {item["id"] for item in walk_nodes([node])}
    candidates = [item for item in walk_nodes(member["body"]) if item["id"] not in excluded]
    anchor = _select_anchor(builder, candidates)
    position = builder.ask(choice("position", ["before", "after"]), "repair_move_position")["position"]
    container, index, _ = find_location(member, node["id"])
    moved = container.pop(index)
    destination, anchor_index, _ = find_location(member, anchor)
    destination.insert(anchor_index + (position == "after"), moved)
