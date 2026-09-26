"""Small native decisions assembled into an implementation contribution by the host.

The model never serializes a graph, requirement IDs, checkpoint state or accepted
owners. Each successful decision is durable before the next one is requested.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .model_adapters.base import NativeToolDecisionRejected

_TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
_SYMBOL = {"type": "string", "pattern": r"^[A-Z][A-Za-z0-9_]*$"}
_LIST = {"type": "array", "uniqueItems": True, "items": _TEXT}


def work_packet(requirements: Mapping[str, str], context: Mapping[str, str]) -> dict[str, Any]:
    """Take one authored responsibility block with every subordinate condition.

    Split at peer bullets/paragraphs, never bytes or nested state fields. The full
    enclosing concern remains read-only context, including previously covered
    state and failure rules. Requirement ownership still advances monotonically.
    """
    bullet_pattern = re.compile(r"^(\s*)(?:[-*+] |\d+[.)] )")
    depths = [len(m[1].expandtabs(4)) for line in context.values()
              if (m := bullet_pattern.match(line))]
    root_depth = min(depths) if depths else None
    groups: list[list[str]] = []
    current: list[str] = []
    headers: list[str] = []
    previous_line = 0
    fence = ""
    for ref, line in context.items():
        number = int(ref[1:])
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            current.append(ref)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
            previous_line = number
            continue
        if marker:
            fence = marker[1]
            current.append(ref)
            previous_line = number
            continue
        heading = re.match(r"^\s*#{1,6}\s", line)
        bullet = bullet_pattern.match(line)
        peer = bullet and len(bullet[1].expandtabs(4)) == root_depth
        paragraph = root_depth is None and number > previous_line + 1
        if current and (heading or peer or paragraph):
            groups.append(current)
            current = []
        if heading:
            headers.append(ref)
        else:
            current.append(ref)
        previous_line = number
    if current:
        groups.append(current)
    for group in groups:
        if any(r in requirements for r in group):
            active = [r for r in (*headers, *group) if r in requirements]
            return {"requirements": {r: requirements[r] for r in dict.fromkeys(active)},
                    "context": dict(context)}
    first = next(iter(requirements))
    return {"requirements": {first: requirements[first]}, "context": dict(context)}


def member_error(declarations: list[str]) -> str:
    """Reject pseudo-Java before freezing a contract (not a Java type checker)."""
    for text in declarations:
        if re.match(r"\s*(?:(?:public|protected|private|static|final|abstract)\s+)*(?:class|interface|enum|record)\b", text):
            return "Supply member declarations only; class/enum/type definitions are host-owned."
        if '{' in text or '}' in text or re.search(r"=\s*;?\s*$", text):
            return "Supply complete member signatures without bodies or incomplete field initializers."
        if not re.search(r"\b[A-Za-z_$][\w$]*\s*(?:\([^{}]*\)(?:\s+throws\s+.+)?|(?:=.+)?)\s*;?$", text):
            return "Supply a Java field, constructor or method signature."
    return ""


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _decide(router: Any, stage: str, properties: dict[str, Any], context: dict[str, Any],
            ledger: dict[str, Any], save: Callable[[], None],
            validate: Callable[[dict[str, Any]], dict[str, str]] | None = None) -> dict[str, Any]:
    from .implementation_ir import ImplementationGraphError, digest
    from .root_cause_trace import emit_root_cause

    slot = ledger.setdefault(stage, {"values": {}, "failures": 0})
    if slot.get("terminal"):
        raise ImplementationGraphError(slot["terminal"])
    if slot.get("done"):
        return deepcopy(slot["values"])
    while True:
        missing = {key: value for key, value in properties.items() if key not in slot["values"]}
        schema = _schema(missing)
        request = {**context, "accepted_fields": slot["values"]}
        if slot.get("errors"):
            request["correct_only"] = slot["errors"]
        try:
            value = router._generate_tool_decision_impl(
                "planner",
                [{"role": "system", "content": (
                    "Resolve only this one implementation decision for the approved work packet. "
                    "Keep state ownership and dependency APIs consistent with the supplied contracts. "
                    "The host owns the mod entrypoint; generated Java owners are helper classes, "
                    "not additional Fabric entrypoints. "
                    "Return only the fields of the required native tool. Never output nodes, a graph, "
                    "requirement IDs, wrappers, or JSON embedded inside strings. Accepted fields are "
                    "host-owned. Do not restate them. Do not claim code or tests ran."
                )}, {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
                tool_name=stage, parameters=schema,
                description="One host-selected implementation decision; no source mutation.",
            )
        except NativeToolDecisionRejected as exc:
            matching = [r for r in exc.rejections if r.get("original_tool") == stage]
            value = None
            if len(matching) == 1:
                try:
                    value = json.loads(matching[0].get("raw_arguments") or "")
                except (TypeError, ValueError):
                    pass
            # Only schema-invalid arguments may enter field repair. Never turn an
            # unrelated native rejection into a successful decision.
            if value is not None and not list(Draft202012Validator(schema).iter_errors(value)):
                raise
        errors: dict[str, str] = {}
        before = len(slot["values"])
        if isinstance(value, dict):
            for key, field in missing.items():
                if key not in value:
                    errors[key] = "Required field missing."
                elif list(Draft202012Validator(field).iter_errors(value[key])):
                    errors[key] = "Expected " + json.dumps(field, ensure_ascii=False)
                else:
                    slot["values"][key] = value[key]
            if set(value) - set(properties):
                errors["response"] = "Unexpected fields; return only the requested fields."
        else:
            errors["response"] = "Expected the tool's argument object, not a string or graph array."
        if validate and all(key in slot["values"] for key in properties):
            semantic = validate(slot["values"])
            for key, message in semantic.items():
                slot["values"].pop(key, None)
                errors[key] = message
        if not errors and all(key in slot["values"] for key in properties):
            slot["done"] = True
            save()
            emit_root_cause("implementation_decision_accepted", stage="production", result="PASS",
                            operation=stage, details={"decision": slot["values"], "work_packet": context.get("work_packet")})
            return deepcopy(slot["values"])
        # Progress is accepted, independently valid fields, never fewer error
        # messages. One corrective request is allowed; then progress is mandatory.
        slot["failures"] += 1
        slot["errors"] = errors
        if slot["failures"] > 1 and len(slot["values"]) <= before:
            slot["terminal"] = f"IMPLEMENTATION_DECISION_NO_PROGRESS: {stage}: {json.dumps(errors)}"
        save()
        emit_root_cause("implementation_decision_rejected", stage="production", result="REJECTED",
                        operation=stage, details={"errors": errors, "accepted_fields": sorted(slot["values"]),
                                                 "response_sha256": digest(value)})
        if slot.get("terminal"):
            raise ImplementationGraphError(slot["terminal"])


def compile_contribution(router: Any, name: str, payload: dict[str, Any],
                         state: dict[str, Any], checkpoint: Callable[[], None]) -> dict[str, Any]:
    """Native small-model frontend; output is the private host IR, not model JSON."""
    from .implementation_ir import ImplementationGraphError

    if name == "decompose_implementation_node":
        return decompose_contribution(router, payload, state, checkpoint)
    owners = {n["symbol"]: n for n in payload.get("accepted_nodes", [])}
    unresolved = payload.get("unresolved_dependencies", [])
    resolving = not payload.get("unit_ids") and bool(unresolved)
    packet = work_packet(payload["requirements"], payload.get("unit_context") or payload["requirements"])
    context = {"work_packet": packet, "platform": payload["platform"], "package": payload["package"],
               "mod_id": payload["mod_id"], "project_context": payload.get("project_context", ""),
               "owner_catalog": [{"symbol": n["symbol"], "kind": n["kind"],
                                  "responsibility": n["responsibility"]} for n in owners.values()]}
    if resolving:
        identity = {"symbol": unresolved[0], "kind": "java"}
        context["consumers"] = [n for n in owners.values() if unresolved[0] in n["depends_on"]]
    else:
        identity = _decide(router, "select_implementation_owner", {
            "symbol": {**_SYMBOL, "description": "Reuse the existing state owner when appropriate; otherwise name one new owner."},
            "kind": {"type": "string", "enum": ["java", "resource"]},
        }, context, state, checkpoint,
            lambda v: {"kind": "Existing owner kind cannot change."}
            if v["symbol"] in owners and owners[v["symbol"]]["kind"] != v["kind"] else {})
    owner = owners.get(identity["symbol"])
    context.pop("owner_catalog", None)
    context["owner"] = owner or identity
    behavior = _decide(router, "define_implementation_behavior", {
        "state_transition": {**_TEXT, "description": "One responsibility: named state owner, input/trigger and resulting state/effect; delegate separate state to dependencies."},
        "success_condition": {**_TEXT, "description": "Observable success for this work packet."},
        "failure_condition": {**_TEXT, "description": "Rejection/failure behavior and what state must remain unchanged; say explicitly if no failure applies."},
    }, context, state, checkpoint)
    context["behavior"] = behavior
    dependency_schema = {"type": "array", "uniqueItems": True, "items": _SYMBOL,
                         "description": "Actual state/API owners consumed by this work. No artificial ordering edges."}
    if resolving:
        dependency_schema["items"] = {"type": "string", "enum": list(owners)}
    dependency_choice = _decide(router, "select_implementation_dependencies", {
        "depends_on": dependency_schema,
    }, {**context, "owner_catalog": [{"symbol": n["symbol"], "responsibility": n["responsibility"]}
                                      for n in owners.values()]}, state, checkpoint,
        lambda v: _dependency_errors(v["depends_on"], identity["symbol"], owners))
    context["dependencies"] = [owners[s] for s in dependency_choice["depends_on"] if s in owners]
    context["unresolved_dependencies"] = [s for s in dependency_choice["depends_on"] if s not in owners]
    properties = {
        "public_api": {**_LIST, "description": "Only new Java member signatures, no type declarations or bodies. Existing signatures need not be repeated. Empty for resources."},
        "activation": {"type": "boolean", "description": "Does this owner need runtime event registration? Host supplies initialize()."},
        "estimated_tokens": {"type": "integer", "minimum": 1, "description": "Complete serialized source estimate for this owner after this contribution."},
    }
    if identity["kind"] == "resource":
        properties["public_api"] = {"type": "array", "maxItems": 0}
        properties["activation"] = {"const": False}
    interface = _decide(router, "define_implementation_interface", properties, context, state, checkpoint,
                        lambda v: _interface_errors(v, identity, owner))
    path = owner["resource_path"] if owner else ""
    if identity["kind"] == "resource" and not owner:
        path = _decide(router, "locate_implementation_resource", {
            "resource_path": {"type": "string", "pattern": rf"^src/main/resources/(assets|data)/{re.escape(payload['mod_id'])}/(?!.*(?:\.\.|\\)).+\.json$"},
        }, context, state, checkpoint)["resource_path"]
    node = {**identity, **interface, **dependency_choice, "resource_path": path,
            "responsibility": owner["responsibility"] if owner else behavior["state_transition"],
            "requirements": list(packet["requirements"]),
            "obligations": [json.dumps(behavior, ensure_ascii=False)]}
    if resolving and node["symbol"] in node["depends_on"]:
        raise ImplementationGraphError("IMPLEMENTATION_IR_DEPENDENCY_CYCLE")
    return {"nodes": [node]}


def _interface_errors(value: dict[str, Any], identity: dict[str, Any], owner: dict[str, Any] | None) -> dict[str, str]:
    errors = {}
    if identity["kind"] == "java":
        error = member_error(value["public_api"])
        if error:
            errors["public_api"] = error
        elif not value["public_api"] and not value["activation"] and not owner:
            errors["public_api"] = "New inactive Java owner needs at least one concrete member API."
    return errors


def _dependency_errors(dependencies: list[str], symbol: str, owners: dict[str, Any]) -> dict[str, str]:
    pending = list(dependencies)
    visited: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency == symbol:
            return {"depends_on": "This dependency creates a cycle back to the selected owner; consume shared state without reverse ownership."}
        if dependency not in visited:
            visited.add(dependency)
            pending.extend(owners.get(dependency, {}).get("depends_on", []))
    return {}


def decompose_contribution(router: Any, payload: dict[str, Any], state: dict[str, Any],
                           checkpoint: Callable[[], None]) -> dict[str, Any]:
    """Extract one helper while the host retains the original facade and APIs."""
    original = payload["rejected_node"]
    from .implementation_ir import ImplementationGraphError
    if original["kind"] != "java":
        raise ImplementationGraphError("IMPLEMENTATION_IR_RESOURCE_SPLIT_REQUIRES_RESOURCE_COMPOSITION")
    context = {"work_packet": {"requirements": payload["requirements"]}, "owner": original,
               "dependencies": payload["existing_contracts"], "reason": payload["reason"],
               "instruction": "Extract one coherent state/responsibility into one helper. The host retains the original facade and all of its APIs."}
    identity = _decide(router, "select_implementation_helper", {
        "symbol": {"type": "string", "pattern": "^" + re.escape(original["symbol"]) + r"Part[A-Za-z0-9_]+$"},
    }, context, state, checkpoint)
    behavior = _decide(router, "define_helper_behavior", {
        "state_transition": _TEXT, "success_condition": _TEXT, "failure_condition": _TEXT,
        "facade_work": {**_TEXT, "description": "Work retained by the facade, including delegation through helper APIs."},
    }, context, state, checkpoint)
    limit = original["estimated_tokens"] - 1
    if payload.get("admission_tokens"):
        limit = min(limit, payload["admission_tokens"])
    if limit < 1:
        raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_NO_PROGRESS")
    allowed = [n["symbol"] for n in payload["existing_contracts"] if n["symbol"] != original["symbol"]]
    dependency_schema = {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": allowed}} if allowed else {"type": "array", "maxItems": 0}
    interface = _decide(router, "define_helper_interface", {
        "public_api": {**_LIST, "minItems": 1},
        "depends_on": dependency_schema,
        "estimated_tokens": {"type": "integer", "minimum": 1, "maximum": limit},
        "facade_estimated_tokens": {"type": "integer", "minimum": 1, "maximum": limit},
    }, {**context, "helper": identity, "behavior": behavior}, state, checkpoint,
        lambda v: {"public_api": member_error(v["public_api"])} if member_error(v["public_api"]) else {})
    facade = deepcopy(original)
    facade.pop("path", None)
    facade["depends_on"] = list(dict.fromkeys([*original["depends_on"], identity["symbol"]]))
    facade["obligations"] = [behavior["facade_work"]]
    facade["estimated_tokens"] = interface.pop("facade_estimated_tokens")
    helper = {**identity, **interface, "kind": "java", "resource_path": "", "activation": False,
              "requirements": original["requirements"], "responsibility": behavior["state_transition"],
              "obligations": [json.dumps({k: v for k, v in behavior.items() if k != "facade_work"}, ensure_ascii=False)]}
    return {"nodes": [facade, helper]}
