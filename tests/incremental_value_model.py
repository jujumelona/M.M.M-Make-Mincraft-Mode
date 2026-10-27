"""Translate existing golden model answers into the new structured model pages.

Only fixture answers pass through this adapter. The production value selector,
native page schemas, Java renderer and compiler boundary all run unchanged.
"""
from __future__ import annotations

import json

from minecraft_mod_ai.incremental_java_render import _expression_node
from minecraft_mod_ai.java_region_parser import _text

_PAGES = {"value_kind", "literal_type", "literal", "local_value", "binary_operator", "unary_operator",
          "call_receiver", "field_receiver", "call_member", "field_member", "static_owner",
          "instanceof_type", "cast_type", "call_capture", "call_result_type"}


def _child(node, name):
    return node.child_by_field_name(name)


def _node(raw, node):
    text = _text(raw, node)
    if node.type == "parenthesized_expression":
        return _node(raw, node.named_children[0])
    if node.type in {"identifier", "this", "super"}:
        return {"kind": "this" if text == "this" else "local", "name": text}
    if node.type == "null_literal":
        return {"kind": "null"}
    if node.type in {"true", "false"}:
        return {"kind": "literal", "value": text == "true", "type": "boolean"}
    if node.type == "decimal_integer_literal":
        return {"kind": "literal", "value": int(text.rstrip("lL")), "type": "long" if text[-1:] in {"l", "L"} else "int"}
    if node.type == "string_literal":
        return {"kind": "literal", "value": json.loads(text), "type": "java.lang.String"}
    if node.type in {"binary_expression", "instanceof_expression"}:
        return _binary_node(raw, node)
    if node.type in {"method_invocation", "field_access"}:
        return _member_node(raw, node)
    raise AssertionError("Golden model expression needs an explicit structured fixture: " + text)


def _binary_node(raw, node):
    if node.type == "instanceof_expression":
        return {"kind": "instanceof", "instanceof:value": _node(raw, _child(node, "left")),
                "type": _text(raw, _child(node, "right"))}
    return {"kind": "binary", "operator": _text(raw, _child(node, "operator")),
            "binary:left": _node(raw, _child(node, "left")), "binary:right": _node(raw, _child(node, "right"))}


def _member_node(raw, node):
    receiver = _child(node, "object")
    target = _text(raw, receiver) if receiver is not None else "this"
    if target in {"this", "super"}:
        receiver_kind = target
    else:
        receiver_kind = "static" if target.rsplit(".", 1)[-1][:1].isupper() else "value"
    result = {"kind": "call" if node.type == "method_invocation" else "field", "receiver_kind": receiver_kind,
              "name": _text(raw, _child(node, "name") or _child(node, "field")), "owner": target}
    if receiver_kind == "value":
        result["receiver"] = _node(raw, receiver)
    arguments = _child(node, "arguments")
    for index, argument in enumerate(arguments.named_children if arguments is not None else ()):
        result["argument:" + str(index)] = _node(raw, argument)
    return result


def _golden(coder, task):
    projected = {**task, "expression_slot": task["value_stack"][0]["slot"]}
    projected["expected_type"] = task["value_stack"][0]["expected_type"]
    frame = {"kind": "field", "prefix": "", "suffix": ""}
    source = coder.value_answer("coder", [{"role": "user", "content": json.dumps(projected)}],
                                        data_name="golden_value", source_frame=frame)
    _, raw, node = _expression_node(source, allow_update=False)
    return _node(raw, node)


def _active_node(coder, task, page):
    stack = task.get("value_stack", [])
    if not stack:
        return getattr(coder, "_statement_call", {})
    if len(stack) == 1 and page == "value_kind":
        coder._value_root = _golden(coder, task)
    value = getattr(coder, "_value_root", {})
    for frame in stack[1:]:
        value = value[frame["slot"]]
    return value


def _reply(coder, task, page, parameters, original, role, messages, data_name):
    if page == "call_receiver" and task.get("value_stack", [{}])[0].get("slot") == "call":
        details = getattr(coder, "details", None)
        if not isinstance(details, dict) or "receiver" not in details:
            details = original(coder, role, messages, data_name="fixture_call", parameters={})
        source = (details["receiver"] + "." if details["receiver"] else "") + details["method"] + "()"
        _, raw, node = _expression_node(source, allow_update=False)
        coder._value_root = coder._statement_call = _node(raw, node)
        coder._statement_call["capture"] = bool(details.get("result_type"))
        coder._statement_call["result_type"] = details.get("result_type", "")
    value = _active_node(coder, task, page)
    if page in {"call_member", "field_member"}:
        candidates = task["member_candidates"]
        found = next((key for key, row in candidates.items() if row["name"] == value["name"]), None)
        if found is None:
            return {"member_id": "next_page"} if "next_page" in parameters["properties"]["member_id"]["enum"] else {"member_id": "search"}
        return {"member_id": found}
    return _simple_reply(value, page)


def _simple_reply(value, page):
    if page == "value_kind":
        return {"kind": value["kind"]}
    if page == "literal":
        return {"value": value["value"]}
    if page in {"literal_type", "cast_type", "instanceof_type"}:
        return {"type": value["type"]}
    if page == "static_owner":
        return {"type": value["owner"]}
    if page in {"call_receiver", "field_receiver"}:
        return {"receiver_kind": value["receiver_kind"]}
    if page == "local_value":
        return {"name": value["name"]}
    if page == "call_capture":
        return {"enabled": value.get("capture", False)}
    if page == "call_result_type":
        return {"type": value["result_type"]}
    return {"operator": value["operator"]}


def structured_values(cls):
    original = getattr(cls, "generate_structured_data", None)
    if original is None:
        if not hasattr(cls, "value_answer"):
            return cls
        def original(*args, **kwargs):
            raise AssertionError("This value-only fixture received a metadata request")
    if getattr(original, "_structured_fixture", False):
        return cls
    def generate(self, role, messages, *, data_name, parameters):
        task = json.loads(messages[-1]["content"])
        page = task.get("requested_page", "")
        if page in _PAGES:
            return _reply(self, task, page, parameters, original, role, messages, data_name)
        return original(self, role, messages, data_name=data_name, parameters=parameters)
    generate._structured_fixture = True
    cls.generate_structured_data = generate
    return cls
