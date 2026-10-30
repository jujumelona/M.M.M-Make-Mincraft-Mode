"""Bounded, lossless access to the saved work order and verified state API."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import re


def load_work_state(api):
    api.work_order = api._bound("feature_work_order", "feature_work_order_json", {})
    api.state_access = api.payload.get("host_state_access", api.context.get("host_state_access", [])) or []
    api._state_rows = []
    api._state_selections = {}
    api._state_methods = defaultdict(list)
    for owner_index, owner in enumerate(api.state_access):
        for method_index, method in enumerate(owner.get("methods") or ()):
            api._state_rows.append((owner["owner"], method))
            api._state_methods[method.get("name")].append((owner_index, method_index, method))


def _page(query, offset, total, size):
    return {"query": query, "offset": offset, "total": total,
            "next_offset": offset + size if offset + size < total else None}


def work_order_view(order, behavior, size):
    if not isinstance(order, Mapping):
        return {}
    result, pages = {}, {}
    ids = order.get("behavior_ids") or []
    behavior_id = behavior.get("behavior_id")
    current = ids.index(behavior_id) if behavior_id in ids else 0
    for key, value in order.items():
        if isinstance(value, (list, tuple)):
            offset = current if key in {"behavior_ids", "intents", "effects"} else 0
            result[key] = deepcopy(value[offset:offset + size])
            pages[key] = _page("@work_order:" + key, offset, len(value), size)
        else:
            result[key] = deepcopy(value)
    result["pages"] = pages
    return result


def _state_rows(api, names):
    selected = {}
    for name in names:
        for prefix in ("readState_", "writeState_"):
            for owner_index, method_index, method in api._state_methods.get(prefix + name, ()):
                selected[owner_index, method_index] = (api.state_access[owner_index]["owner"], method)
    return [selected[key] for key in sorted(selected)]


def _state_page(rows, query, offset, size):
    owners = defaultdict(list)
    for owner, method in rows[offset:offset + size]:
        owners[owner].append(deepcopy(method))
    return {**_page(query, offset, len(rows), size), "host_state_access": [
        {"owner": owner, "methods": methods} for owner, methods in owners.items()]}


def state_access_view(api, names):
    order = api.work_order or {}
    names = names | set(order.get("state_reads") or ()) | set(order.get("state_writes") or ())
    rows = _state_rows(api, names)
    query = _state_selection(api, names, rows)
    # Logical observations are often world inputs rather than storage IDs.
    # Zero exact matches do not establish that the existing state API is useless.
    return _state_page(rows or api._state_rows, query, 0, api.page_size)


def _state_selection(api, names, rows):
    if not rows:
        return "@state_access"
    identity = hashlib.sha256("\0".join(sorted(names)).encode()).hexdigest()[:24]
    query = "@state_access:selection_" + identity
    api._state_selections[query] = rows
    return query


def _canonical_names(values):
    return {name for name in values if isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)}


def _state_permissions(api, behavior):
    writable = _canonical_names(behavior.get("state_writes") or ())
    readable = _canonical_names(behavior.get("state_reads") or ()) | writable
    assigned = bool(_state_rows(api, readable | writable))
    return {"mode": "assigned_behavior" if assigned else "declaration_discovery",
            "readable": sorted(readable), "writable": sorted(writable),
            "contract": "State headers are declaration evidence, not additional call permission. "
                        "Only these canonical readable/writable names are authorized for this behavior. "
                        "Unassigned logical names require canonical plan binding or real world inputs; "
                        "never substitute a different storage key."}


def work_context_view(api, behavior, names):
    page = state_access_view(api, names)
    return {"feature_work_order": work_order_view(api.work_order, behavior, api.page_size),
            "state_access_permissions": _state_permissions(api, behavior),
            "host_state_access": page["host_state_access"],
            "host_state_access_page": {key: value for key, value in page.items()
                                       if key != "host_state_access"}}


def lookup_work_evidence(api, query, offset):
    if type(offset) is not int or offset < 0:
        raise ValueError("JAVA_API_LOOKUP_OFFSET_INVALID")
    if query == "@work_order":
        return {"feature_work_order": work_order_view(api.work_order, {}, api.page_size)}
    if query.startswith("@work_order:"):
        key = query.partition(":")[2]
        values = (api.work_order or {}).get(key, [])
        if not isinstance(values, (list, tuple)):
            values = [values]
        return {**_page(query, offset, len(values), api.page_size),
                "values": deepcopy(values[offset:offset + api.page_size])}
    if query.startswith("@state_access"):
        return _lookup_state_page(api, query, offset)
    return None


def _lookup_state_page(api, query, offset):
    if query == "@state_access":
        rows = api._state_rows
    elif query.startswith("@state_access:selection_"):
        rows = api._state_selections.get(query, [])
    else:
        rows = _state_rows(api, query.partition(":")[2].split(","))
    return _state_page(rows, query, offset, api.page_size)
