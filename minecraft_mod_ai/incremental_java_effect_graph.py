"""Conservative potential-effect analysis for generated callback call graphs.

This gate only rejects provable no-op routes. External API calls remain
potential effects because purity cannot be inferred reliably from a signature.
GameTest and behavior acceptance remain the final semantic checks.
"""
from __future__ import annotations

from .incremental_java_history import walk_nodes


def has_reachable_effect(start, members, local_names, predicate):
    """Walk same-owner, generated-helper calls; do not trust empty wrappers.

    The visited set also prevents self/mutually recursive stubs from producing
    a false-positive effect or making verification recurse forever.
    """
    generated = {}
    for member in members:
        if member.get("kind") == "method":
            generated.setdefault((member.get("owner"), member.get("name")), []).append(member)
    pending, visited = [start], set()
    while pending:
        current = pending.pop()
        member_id = current.get("id")
        if member_id in visited:
            continue
        visited.add(member_id)
        nodes = tuple(walk_nodes(current.get("body") or ()))
        locals_in_scope = local_names(current, nodes)
        for node in nodes:
            if node.get("op") == "call":
                receiver = str(node.get("receiver") or "").strip()
                owner = (node.get("call_contract") or {}).get("owner") or current.get("owner")
                same_owner = generated.get((owner, node.get("method")), ())
                if receiver in {"", "this"} and same_owner:
                    pending.extend(same_owner)
                    continue
                return True  # External or dynamically dispatched API call
            if current is not start and node.get("op") == "return":
                # Returning a pure value from a helper is not a gameplay effect.
                # The caller must consume it in a visible state mutation.
                continue
            if predicate(node, locals_in_scope):
                return True
    return False
