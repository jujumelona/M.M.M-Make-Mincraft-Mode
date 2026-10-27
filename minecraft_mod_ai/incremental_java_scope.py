"""Recompute lexical environments from the current program, including repaired types."""
from __future__ import annotations

from copy import deepcopy


def _bind(scope, node):
    if node["op"] in {"local", "lambda"}:
        scope["locals"].append({"name": node["name"], "type": node["type"]})
    elif node["op"] == "call" and node.get("result_type"):
        scope["locals"].append({"name": node["name"], "type": node["result_type"]})


def nested_scopes(node, outer):
    op = node["op"]
    scope = deepcopy(outer)
    if op in {"while", "do", "for", "foreach"}:
        scope["loop_depth"] += 1
    if op in {"for", "foreach"}:
        scope["locals"].append({"name": node["name"], "type": node["type"]})
    if op == "lambda":
        scope.update(return_type=node["return_type"], loop_depth=0, switch_depth=0)
        scope["locals"].extend(deepcopy(node["parameters"]))
    result = [(node.get(key, []), deepcopy(scope)) for key in ("body", "otherwise", "finally")]
    for case in node.get("cases", []):
        case_scope = deepcopy(outer)
        case_scope["switch_depth"] += 1
        result.append((case.get("body", []), case_scope))
    for caught in node.get("catches", []):
        caught_scope = deepcopy(outer)
        caught_scope["locals"].append({"name": caught["name"], "type": caught["type"]})
        result.append((caught.get("body", []), caught_scope))
    return result


def scope_at(member, node_id=None):
    from .incremental_java_model import initial_scope
    def walk(nodes, scope):
        for node in nodes:
            if node["id"] == node_id:
                return deepcopy(scope)
            if node_id is not None:
                for children, child_scope in nested_scopes(node, scope):
                    found = walk(children, child_scope)
                    if found is not None:
                        return found
            _bind(scope, node)
        if node_id is None:
            return scope
        return None
    result = walk(member.get("body", []), initial_scope(member))
    if result is None:
        raise ValueError("JAVA_INCREMENTAL_SCOPE_NODE_MISSING: " + str(node_id))
    return result
