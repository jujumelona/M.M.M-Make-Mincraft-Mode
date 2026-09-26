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


def _required_section_contract(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    role = _unit_role(payload)
    contract = section_spec(role)
    if contract is None:
        from .implementation_ir import ImplementationGraphError

        raise ImplementationGraphError(
            f"IMPLEMENTATION_IR_NONCANONICAL_UNIT: {role or '<missing>'}"
        )
    return role, contract


def _host_owner_symbol(payload: Mapping[str, Any]) -> str:
    _, contract = _required_section_contract(payload)
    return str(contract["symbol"])


def _role_dependencies(payload: Mapping[str, Any]) -> list[str]:
    _, contract = _required_section_contract(payload)
    planned = _planned_roles(payload)
    result: list[str] = []
    for dependency_role in contract["depends_on"]:
        if dependency_role not in planned:
            continue
        dependency = section_spec(dependency_role)
        if dependency is None:
            from .implementation_ir import ImplementationGraphError

            raise ImplementationGraphError(
                f"IMPLEMENTATION_IR_CANONICAL_DEPENDENCY_MISSING: {dependency_role}"
            )
        result.append(str(dependency["symbol"]))
    return result


def _host_responsibility(payload: Mapping[str, Any]) -> str:
    _, contract = _required_section_contract(payload)
    return str(contract["responsibility"])


def _host_instruction(payload: Mapping[str, Any]) -> str:
    _, contract = _required_section_contract(payload)
    return str(contract["instruction"])

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


def _host_concern_work(
    payload: Mapping[str, Any],
    packet: Mapping[str, Any],
    unit_requirements: Mapping[str, str],
) -> tuple[list[str], int]:
    role, _contract = _required_section_contract(payload)
    concerns = concern_contracts(role)
    if not concerns:
        from .implementation_ir import ImplementationGraphError

        raise ImplementationGraphError(
            f"IMPLEMENTATION_IR_CONCERN_CONTRACT_MISSING: {role}"
        )
    section_instruction = _host_instruction(payload)
    obligations: list[str] = []
    for concern in concerns:
        instruction = json.dumps(
            {
                "section": role,
                "concern": concern["concern"],
                "concern_template": concern["identifier"],
                "task": concern["task"],
                "rules": concern["rules"],
                "section_instruction": section_instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        obligations.append(
            _host_obligation(packet["requirements"], instruction=instruction)
        )
    raw_budget = _host_estimated_tokens(unit_requirements)
    concern_budget = max(768, (raw_budget + len(concerns) - 1) // len(concerns))
    return obligations, concern_budget


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
    symbol = _host_owner_symbol(payload)
    owner = owners.get(symbol)
    unit_requirements = payload.get("unit_context") or packet["requirements"]
    responsibility = owner["responsibility"] if owner else _host_responsibility(payload)
    obligations, estimated_tokens = _host_concern_work(
        payload, packet, unit_requirements
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
        "estimated_tokens": estimated_tokens,
    }
    return {"nodes": [node]}

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
