"""Lossless operand views with paginated access to accepted control-flow bodies."""
from __future__ import annotations

PAGE = 10
_CHILDREN = ("body", "otherwise", "finally")


def child_blocks(node):
    result = [(key, node[key]) for key in _CHILDREN if key in node]
    result.extend((f"case:{index}", case.get("body", [])) for index, case in enumerate(node.get("cases", [])))
    result.extend((f"catch:{index}", case.get("body", [])) for index, case in enumerate(node.get("catches", [])))
    return result


def operation_view(node):
    result = {key: value for key, value in node.items()
              if key not in {*_CHILDREN, "cases", "catches", "scope"}}
    result["child_blocks"] = {key: {"count": len(children), "first_id": children[0]["id"] if children else None}
                              for key, children in child_blocks(node)}
    if "cases" in node:
        result["case_labels"] = [case.get("label", "") for case in node["cases"]]
    if "catches" in node:
        result["catch_types"] = [{key: case[key] for key in ("type", "name")} for case in node["catches"]]
    return result


def walk_nodes(nodes):
    for node in nodes:
        yield node
        for _, children in child_blocks(node):
            yield from walk_nodes(children)


def lookup_work(builder, query, offset):
    if query == "@declarations" or query.startswith("@declarations:"):
        return builder.api.generated(builder.program, query.partition(":")[2], offset)
    if query == "@operations":
        values = [("current", node) for node in builder.block]
    elif query.startswith("@node:"):
        identity = query.split(":", 1)[1]
        node = builder.history.get(identity)
        if node is None:
            return {"query": query, "operations": [], "total": 0}
        values = [(key, child) for key, children in child_blocks(node) for child in children]
        page = _page(query, values, offset)
        page["node"] = operation_view(node)
        return page
    else:
        return None
    return _page(query, values, offset)


def _page(query, values, offset):
    return {"query": query, "offset": offset, "total": len(values),
            "next_offset": offset + PAGE if offset + PAGE < len(values) else None,
            "operations": [{"block": key, **operation_view(node)} for key, node in values[offset:offset + PAGE]]}
