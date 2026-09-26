"""Host-owned lowering of approved authored work into bounded implementation units.

Production does not ask the local text model to design owners, dependencies, behavior
summaries, APIs, resource paths, or helper graphs. The model is reserved for later,
bounded source-generation tasks after the host has fixed the work unit and contract.
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


from .authored_execution_schema import concern_contracts, section_spec


def _normalized_unit_role(value: Any) -> str:
    text = re.sub(r"\s*\([^)]*\)\s*$", "", str(value or "").strip()).casefold()
    return re.sub(r"[\s-]+", "_", text).strip("_")


def _unit_role(payload: Mapping[str, Any]) -> str:
    units = payload.get("current_units") or ()
    if isinstance(units, (list, tuple)) and units:
        return _normalized_unit_role(units[0])
    return ""


def _planned_roles(payload: Mapping[str, Any]) -> set[str]:
    raw = payload.get("planned_units") or ()
    if not isinstance(raw, (list, tuple)):
        return set()
    return {_normalized_unit_role(value) for value in raw}


def _host_owner_symbol(payload: dict[str, Any], packet: Mapping[str, Any]) -> str:
    """Return a stable owner from the authored schema role, never from model naming."""
    role = _unit_role(payload)
    contract = section_spec(role)
    if contract:
        return str(contract["symbol"])
    unit_ids = [str(value).strip() for value in payload.get("unit_ids", []) if str(value).strip()]
    seed = unit_ids[0] if unit_ids else next(iter(packet.get("requirements", {})), "work")
    parts = re.findall(r"[A-Za-z0-9]+", seed)
    suffix = "".join(part[:1].upper() + part[1:] for part in parts) or "Work"
    if suffix[0].isdigit():
        suffix = "Unit" + suffix
    return "Authored" + suffix


def _role_dependencies(payload: Mapping[str, Any]) -> list[str]:
    role = _unit_role(payload)
    contract = section_spec(role)
    if not contract:
        return []
    planned = _planned_roles(payload)
    result: list[str] = []
    for dependency_role in contract["depends_on"]:
        if dependency_role not in planned:
            continue
        dependency = section_spec(dependency_role)
        if dependency:
            result.append(str(dependency["symbol"]))
    return result


def _host_responsibility(payload: Mapping[str, Any]) -> str:
    role = _unit_role(payload)
    contract = section_spec(role)
    if contract:
        return str(contract["responsibility"])
    units = [str(value).strip() for value in payload.get("current_units", []) if str(value).strip()]
    return (
        f"Implement the approved authored unit: {units[0]}"
        if units
        else "Implement the approved authored behavior."
    )


def _host_instruction(payload: Mapping[str, Any]) -> str:
    role = _unit_role(payload)
    contract = section_spec(role)
    if contract:
        return str(contract["instruction"])
    return (
        "Implement these approved requirements exactly in this bounded Java unit. "
        "The host owns lifecycle wiring and graph structure; do not invent sibling owners."
    )
def _host_obligation(requirements: Mapping[str, str], *, instruction: str) -> str:
    return json.dumps(
        {"source_requirements": dict(requirements), "instruction": instruction},
        ensure_ascii=False,
        sort_keys=True,
    )


def _host_estimated_tokens(requirements: Mapping[str, str]) -> int:
    """Conservative deterministic source allowance; never a model-authored estimate."""
    encoded = "\n".join(str(value) for value in requirements.values()).encode("utf-8")
    return max(768, 512 + len(encoded))


def _unique_helper_symbol(original: Mapping[str, Any], contracts: list[dict[str, Any]]) -> str:
    occupied = {str(item.get("symbol", "")) for item in contracts}
    index = 1
    while True:
        candidate = f"{original['symbol']}Part{index}"
        if candidate not in occupied:
            return candidate
        index += 1


def _split_requirement_ids(requirements: list[str]) -> tuple[list[str], list[str]]:
    if not requirements:
        return [], []
    if len(requirements) == 1:
        # Provenance IDs are indivisible. The strictly decreasing source budget is
        # the well-founded measure, so repeated output pressure still terminates.
        return list(requirements), list(requirements)
    midpoint = max(1, len(requirements) // 2)
    return requirements[:midpoint], requirements[midpoint:]


def _requirements_for(ids: list[str], source: Mapping[str, str]) -> dict[str, str]:
    return {req_id: source[req_id] for req_id in ids if req_id in source}


def compile_contribution(router: Any, name: str, payload: dict[str, Any],
                         state: dict[str, Any], checkpoint: Callable[[], None]) -> dict[str, Any]:
    """Lower authored work deterministically; no implementation-planning LLM call exists here.

    The text model is reserved for the later bounded source-generation task. The host
    owns owner naming, lifecycle, graph shape, requirement provenance and dependency
    topology so a malformed planning/tool response cannot block production.
    """
    if name == "decompose_implementation_node":
        return decompose_contribution(router, payload, state, checkpoint)

    packet = work_packet(
        payload["requirements"], payload.get("unit_context") or payload["requirements"]
    )
    owners = {node["symbol"]: node for node in payload.get("accepted_nodes", [])}
    symbol = _host_owner_symbol(payload, packet)
    owner = owners.get(symbol)
    unit_requirements = payload.get("unit_context") or packet["requirements"]
    responsibility = owner["responsibility"] if owner else _host_responsibility(payload)
    role = _unit_role(payload)
    concerns = concern_contracts(role)
    obligations = (
        [
            _host_obligation(
                packet["requirements"],
                instruction=json.dumps(
                    {
                        "section": role,
                        "concern": concern["concern"],
                        "concern_template": concern["identifier"],
                        "task": concern["task"],
                        "rules": concern["rules"],
                        "section_instruction": _host_instruction(payload),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            for concern in concerns
        ]
        if concerns
        else [
            _host_obligation(
                packet["requirements"],
                instruction=_host_instruction(payload),
            )
        ]
    )
    node = {
        "symbol": symbol,
        "kind": "java",
        "resource_path": "",
        "responsibility": responsibility,
        "requirements": list(packet["requirements"]),
        "obligations": obligations,
        "public_api": [],
        "depends_on": _role_dependencies(payload),
        "activation": True,
        "estimated_tokens": (\n            max(768, (_host_estimated_tokens(unit_requirements) + len(concerns) - 1) // len(concerns))\n            if concerns\n            else _host_estimated_tokens(unit_requirements)\n        ),
    }
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
    """Deterministically split one host-owned Java unit after observed output pressure."""
    from .implementation_ir import ImplementationGraphError

    original = payload["rejected_node"]
    if original["kind"] != "java":
        raise ImplementationGraphError(
            "IMPLEMENTATION_IR_RESOURCE_SPLIT_REQUIRES_RESOURCE_COMPOSITION"
        )
    allowed_host_apis = {"public static void initialize()", "public static void run()"}
    if any(api not in allowed_host_apis for api in original["public_api"]):
        raise ImplementationGraphError("IMPLEMENTATION_IR_HOST_SPLIT_UNSUPPORTED_API")

    original_cost = int(original["estimated_tokens"])
    limit = original_cost - 1
    budget = payload.get("admission_tokens")
    if budget is not None:
        try:
            budget_value = int(budget)
        except (TypeError, ValueError):
            budget_value = 0
        if budget_value > 0:
            limit = min(limit, budget_value)
    if limit < 1:
        raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_NO_PROGRESS")

    helper_symbol = _unique_helper_symbol(
        original, list(payload.get("existing_contracts", []))
    )
    left_ids, right_ids = _split_requirement_ids(list(original["requirements"]))
    source_requirements = payload.get("requirements") or {}
    left_requirements = _requirements_for(left_ids, source_requirements)
    right_requirements = _requirements_for(right_ids, source_requirements)
    child_cost = max(1, min(limit, max(1, original_cost // 2)))

    facade = deepcopy(original)
    facade.pop("path", None)
    facade["requirements"] = left_ids
    facade["depends_on"] = list(
        dict.fromkeys([*original["depends_on"], helper_symbol])
    )
    facade["estimated_tokens"] = child_cost
    entry_method = "initialize" if original["activation"] else "run"
    facade["obligations"] = [
        _host_obligation(
            left_requirements,
            instruction=(
                f"Implement the retained requirements and invoke {helper_symbol}.run() from "
                f"{entry_method}() so the delegated requirements remain active."
            ),
        )
    ]

    helper = {
        "symbol": helper_symbol,
        "kind": "java",
        "resource_path": "",
        "responsibility": f"Delegated bounded part of {original['symbol']}",
        "requirements": right_ids,
        "obligations": [
            _host_obligation(
                right_requirements,
                instruction="Implement only this delegated requirement subset in run().",
            )
        ],
        "public_api": ["public static void run()"],
        "depends_on": list(original["depends_on"]),
        "activation": False,
        "estimated_tokens": child_cost,
    }
    return {"nodes": [facade, helper]}
