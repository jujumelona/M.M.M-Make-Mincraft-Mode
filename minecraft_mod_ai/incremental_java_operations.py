"""Host-owned statement and control-flow assembly from individual choices."""
from __future__ import annotations

from copy import deepcopy

from .incremental_java_schema import SCHEMAS
from .incremental_java_render import validate_type


def _child(builder, node, key, *, scope=None, **changes):
    scope = deepcopy(builder.scope) if scope is None else scope
    scope.update(changes)
    node[key] = builder.author_block(builder.member, scope, [*builder.path, node["id"], key])


def _local(builder, node):
    values = builder.typed(SCHEMAS[node["op"]], node["op"], "type")
    validate_type(values["type"])
    node.update(values, name="mmmValue" + builder.allocate("v")[1:])
    node["value"] = builder.expression("value", values["type"])
    builder.scope["locals"].append({"type": values["type"], "name": node["name"]})


def _call(builder, node):
    from .incremental_java_values import author_call_head
    author_call_head(builder, node)
    builder.arguments(node)
    contract = node["call_contract"]
    returns = contract.get("generic_return_type", contract.get("return_type", "void"))
    node["result_type"] = ""
    if returns != "void" and builder.enabled("call_capture"):
        result_type = builder.select_java_type("call_result_type", expected=returns)
        node["result_type"] = result_type
        node["name"] = "mmmValue" + builder.allocate("v")[1:]
        builder.scope["locals"].append({"type": result_type, "name": node["name"]})


def _assign(builder, node):
    node.update(builder.ask(SCHEMAS["assign"], "assign"))
    node["target"] = builder.expression("target")
    node["value"] = builder.expression("value", getattr(builder, "last_value_type", "java.lang.Object"))


def _conditional(builder, node):
    node["condition"] = builder.expression("condition", "boolean")
    changes = {"loop_depth": builder.scope["loop_depth"] + 1} if node["op"] in {"while", "do"} else {}
    _child(builder, node, "body", **changes)
    if node["op"] == "if":
        node["otherwise"] = []
        if builder.enabled("otherwise"):
            _child(builder, node, "otherwise")


def _iteration(builder, node):
    values = builder.typed(SCHEMAS[node["op"]], node["op"], "type")
    validate_type(values["type"])
    node.update(values, name="mmmValue" + builder.allocate("v")[1:])
    scope = deepcopy(builder.scope)
    scope["locals"].append({"type": values["type"], "name": node["name"]})
    if node["op"] == "foreach":
        node["iterable"] = builder.expression("iterable")
    else:
        node["value"] = builder.expression("value", values["type"])
        original = builder.scope
        builder.scope = scope
        node["condition"] = builder.expression("condition", "boolean")
        node["update"] = builder.expression("update", allow_update=True)
        builder.scope = original
    _child(builder, node, "body", scope=scope, loop_depth=scope["loop_depth"] + 1)


def _terminal(builder, node):
    if node["op"] == "return":
        returns = builder.scope["return_type"]
        node["value"] = "" if returns == "void" else builder.expression("value", returns)
    elif node["op"] == "throw":
        node["value"] = builder.expression("value", "java.lang.Throwable")


def _block(builder, node):
    if node["op"] == "synchronized":
        node["monitor"] = builder.expression("monitor")
    _child(builder, node, "body")


def _switch(builder, node):
    node.update(selector=builder.expression("selector"), cases=[])
    while builder.enabled("case"):
        case = {"label": "" if builder.enabled("default_case") else builder.expression("label")}
        node["active_case_label"] = case["label"]
        scope = deepcopy(builder.scope)
        scope["switch_depth"] += 1
        case["body"] = builder.author_block(builder.member, scope, [*builder.path, node["id"], "case", len(node["cases"])])
        node["cases"].append(case)
    node.pop("active_case_label", None)


def _try(builder, node):
    final = builder.enabled("finally")
    _child(builder, node, "body")
    node["catches"] = []
    more = not final or builder.enabled("catch")
    while more:
        caught = builder.typed(SCHEMAS["catch"], "catch_type", "type", allow_union=True)
        validate_type(caught["type"])
        caught["name"] = "mmmCaught" + builder.allocate("e")[1:]
        node["active_catch"] = {key: caught[key] for key in ("type", "name")}
        scope = deepcopy(builder.scope)
        scope["locals"].append({"type": caught["type"], "name": caught["name"]})
        caught["body"] = builder.author_block(builder.member, scope, [*builder.path, node["id"], "catch", len(node["catches"])])
        node["catches"].append(caught)
        more = builder.enabled("catch")
    node.pop("active_catch", None)
    node["finally"] = []
    if final:
        _child(builder, node, "finally")


def _lambda(builder, node):
    selected = builder.typed(SCHEMAS["lambda"], "lambda", "type")
    validate_type(selected["type"])
    contract = builder.api.sam(selected["type"])
    if contract:
        types = contract["parameters"]
        parameters = [{"type": value, "name": "mmmLambdaArg" + builder.allocate("p")[1:]} for value in types]
    else:
        contract = builder.typed(SCHEMAS["lambda_contract"], "lambda_contract", "return_type", allow_void=True)
        parameters = builder.parameters()
        for parameter in parameters:
            parameter["name"] = "mmmLambdaArg" + builder.allocate("p")[1:]
    validate_type(contract["return_type"])
    node.update(selected, name="mmmValue" + builder.allocate("v")[1:], parameters=parameters,
                return_type=contract["return_type"])
    scope = deepcopy(builder.scope)
    scope["locals"] = [*scope["locals"], *parameters]
    scope.update(return_type=contract["return_type"], loop_depth=0, switch_depth=0)
    _child(builder, node, "body", scope=scope)
    builder.scope["locals"].append({"type": node["type"], "name": node["name"]})


def _register(builder, node):
    from .incremental_java_events import select_event
    event = select_event(builder)
    if event is None:
        return False
    from .incremental_java_model import method_metadata
    from .first_pass_java_declarations import callback_parameter_metadata
    metadata = method_metadata(event["return_type"], callback_parameter_metadata(event))
    metadata["throws"] = event.get("throws", [])
    metadata["type_parameters"] = ", ".join(event.get("type_parameters", []))
    method = builder.new_method(owner=builder.member["owner"],
        purpose="At this event observation, enforce the original trigger and conditions, then implement its effects.",
        behavior=builder.member["behavior"], metadata=metadata, reuse=False)
    method["event"] = event
    node.update(event_owner=event["event_owner"], event_field=event["event_field"], handler=method["name"])


def _constructor_call(builder, node):
    node.update(builder.ask(SCHEMAS["constructor_call"], "constructor_call"))
    builder.arguments(node)


_HANDLERS = {"local": _local, "call": _call, "assign": _assign, "if": _conditional,
             "while": _conditional, "do": _conditional, "for": _iteration, "foreach": _iteration,
             "return": _terminal, "throw": _terminal, "break": _terminal, "continue": _terminal,
             "block": _block, "synchronized": _block, "switch": _switch, "try": _try,
             "lambda": _lambda, "register": _register, "constructor_call": _constructor_call}


def author_operation(builder, operation):
    node = {"id": builder.allocate("n"), "op": operation}
    builder.drafts.append(node)
    try:
        if _HANDLERS[operation](builder, node) is False:
            return None
    finally:
        builder.drafts.pop()
    return node
