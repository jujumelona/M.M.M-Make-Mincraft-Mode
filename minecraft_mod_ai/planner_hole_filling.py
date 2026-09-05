from __future__ import annotations

"""Bounded small-model filling for host-owned Minecraft implementation templates.

The planner model never invents modules, target coordinates, dependencies, artifacts, gates,
or hole identities. It receives the host sketch and fills only the declared holes. One
missing-hole repair pass is allowed; a second omission fails closed instead of silently
shipping an incomplete plan.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .planner_structured_router import structured_planner_router
from .planner_template_schema import merge_model_output_into_skeleton

_SCHEMA = "mmm/planner-hole-fill-packet-v1"


class PlanningHoleFillError(RuntimeError):
    """The bounded planner could not fill the complete host-owned implementation sketch."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(
        dict.fromkeys(
            text
            for item in value
            if (text := str(item or "").strip())
        )
    )


def _module_packet(module: Mapping[str, Any], only_holes: set[str] | None = None) -> dict[str, Any]:
    config = _mapping(module.get("config"))
    task = _mapping(config.get("evidence_task"))
    template = _mapping(config.get("implementation_template"))
    holes = [
        dict(hole)
        for hole in template.get("holes", [])
        if isinstance(hole, Mapping)
        and (
            only_holes is None
            or str(hole.get("hole_id") or "") in only_holes
        )
    ]
    return {
        "module_id": str(module.get("module_id") or ""),
        "scope": str(config.get("scope") or task.get("semantic_outcome") or ""),
        "request_context": _mapping(task.get("request_context")),
        "implementation_template": {
            "schema_version": template.get("schema_version"),
            "task_ref": template.get("task_ref"),
            "semantic_outcome": template.get("semantic_outcome"),
            "target_constraints": _mapping(template.get("target_constraints")),
            "minecraft_checklist": [
                dict(item)
                for item in template.get("minecraft_checklist", [])
                if isinstance(item, Mapping)
            ],
            "holes": holes,
            "completion_policy": _mapping(template.get("completion_policy")),
        },
    }


def _packet(
    skeleton: Mapping[str, Any],
    missing: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    modules = [
        _module_packet(
            module,
            None if missing is None else missing.get(str(module.get("module_id") or ""), set()),
        )
        for module in skeleton.get("modules", [])
        if isinstance(module, Mapping)
        and isinstance(_mapping(module.get("config")).get("implementation_template"), Mapping)
        and (
            missing is None
            or missing.get(str(module.get("module_id") or ""), set())
        )
    ]
    return {
        "schema_version": _SCHEMA,
        "phase": "fill_host_owned_minecraft_implementation_holes",
        "modules": modules,
        "rules": [
            "Return one fill for every supplied hole_id and do not invent hole_ids.",
            "Do not invent or change Minecraft version, loader, mappings, Java version, module identity, dependency edges, artifacts, gates, or target paths.",
            "Use implementation_decision for the concrete design choice inside the hole.",
            "Use local_steps for ordered implementation actions at method/class/resource granularity when known; do not use arbitrary fixed step counts.",
            "Use code_bindings only for symbols/resources implied by host-owned anchors or artifacts; never fabricate an API merely to make the plan look complete.",
            "Use reference_uses only for references actually present in the hole evidence/reference fields. If reference contents are unavailable, state that in uncertainties instead of guessing.",
            "Use verification_intent to bind the hole to compile/static/resource/GameTest/runtime evidence appropriate to the supplied gates and Minecraft checklist.",
            "If a detail cannot be grounded, record it in uncertainties while still giving the most concrete safe plan supported by the host contract.",
        ],
        "response_contract": {
            "modules": [
                {
                    "module_id": "<exact supplied module_id>",
                    "config": {
                        "implementation_notes": "<short cross-hole integration note>",
                        "hole_fills": [
                            {
                                "hole_id": "<exact supplied hole_id>",
                                "implementation_decision": "<concrete local choice>",
                                "local_steps": ["<ordered step>", "..."],
                                "code_bindings": [],
                                "reference_uses": [],
                                "verification_intent": "<how host will prove this hole>",
                                "uncertainties": [],
                            }
                        ],
                    },
                }
            ]
        },
    }


def _messages(packet: Mapping[str, Any], *, repair: bool) -> list[dict[str, str]]:
    system = (
        "You are the bounded implementation planner inside a Minecraft mod compiler. "
        "The host has already decided what must exist. Your job is not to redesign the mod; "
        "fill every supplied implementation hole with concrete, internally consistent coding "
        "steps that a small coding model can follow. Respect exact Minecraft target constraints, "
        "server/client authority, registration lifecycle, resources, persistence, networking, "
        "and executable validation. Output JSON only and exactly the response shape described "
        "by response_contract. Never add modules, holes, dependencies, gates, or paths."
    )
    user_prefix = (
        "The first pass omitted these holes. Fill every supplied missing hole now."
        if repair
        else "Fill every host-owned hole in this packet."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": user_prefix
            + "\n"
            + json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]


def _decode(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        raise PlanningHoleFillError("planner returned an empty hole-fill response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanningHoleFillError(
            "planner hole-fill response was not valid JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise PlanningHoleFillError("planner hole-fill response must be a JSON object")
    return dict(value)


def _expected_holes(page: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for module in page.get("modules", []):
        if not isinstance(module, Mapping):
            continue
        module_id = str(module.get("module_id") or "")
        config = _mapping(module.get("config"))
        template = _mapping(config.get("implementation_template"))
        policy = _mapping(template.get("completion_policy"))
        required = set(_strings(policy.get("required_hole_ids")))
        if required:
            result[module_id] = required
    return result


def _filled_holes(page: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for module in page.get("modules", []):
        if not isinstance(module, Mapping):
            continue
        module_id = str(module.get("module_id") or "")
        config = _mapping(module.get("config"))
        model_fill = _mapping(config.get("model_fill"))
        fills = model_fill.get("hole_fills")
        result[module_id] = {
            str(fill.get("hole_id") or "")
            for fill in fills if isinstance(fill, Mapping) and len(fill) > 1
        } if isinstance(fills, list) else set()
    return result


def _missing_holes(page: Mapping[str, Any]) -> dict[str, set[str]]:
    expected = _expected_holes(page)
    filled = _filled_holes(page)
    return {
        module_id: required - filled.get(module_id, set())
        for module_id, required in expected.items()
        if required - filled.get(module_id, set())
    }


def _raw_fills(value: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    modules = value.get("modules")
    for module in modules if isinstance(modules, list) else []:
        if not isinstance(module, Mapping):
            continue
        module_id = str(module.get("module_id") or "").strip()
        config = _mapping(module.get("config"))
        fills = config.get("hole_fills")
        if not module_id or not isinstance(fills, list):
            continue
        result[module_id] = [dict(item) for item in fills if isinstance(item, Mapping)]
    return result


def _combined_output(
    first_page: Mapping[str, Any],
    repair_output: Mapping[str, Any],
) -> dict[str, Any]:
    repair = _raw_fills(repair_output)
    modules: list[dict[str, Any]] = []
    for module in first_page.get("modules", []):
        if not isinstance(module, Mapping):
            continue
        module_id = str(module.get("module_id") or "")
        config = _mapping(module.get("config"))
        model_fill = _mapping(config.get("model_fill"))
        existing = model_fill.get("hole_fills")
        existing_fills = [
            dict(item) for item in existing if isinstance(item, Mapping)
        ] if isinstance(existing, list) else []
        modules.append(
            {
                "module_id": module_id,
                "config": {
                    "implementation_notes": str(
                        model_fill.get("implementation_notes") or ""
                    ),
                    "hole_fills": [*existing_fills, *repair.get(module_id, [])],
                },
            }
        )
    return {"modules": modules}


def fill_evidence_page(
    router: Any,
    skeleton: Mapping[str, Any],
    *,
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    """Fill every host hole with at most one missing-hole repair pass."""
    expected = _expected_holes(skeleton)
    if not expected:
        return dict(skeleton)

    bounded_router = structured_planner_router(router)
    first_raw = bounded_router.generate_text(
        "planner",
        _messages(_packet(skeleton), repair=False),
        response_format="json",
    )
    first_output = _decode(first_raw)
    page = merge_model_output_into_skeleton(
        skeleton,
        first_output,
        valid_module_catalog,
    )
    missing = _missing_holes(page)
    if not missing:
        return page

    repair_raw = bounded_router.generate_text(
        "planner",
        _messages(_packet(skeleton, missing), repair=True),
        response_format="json",
    )
    repair_output = _decode(repair_raw)
    page = merge_model_output_into_skeleton(
        skeleton,
        _combined_output(page, repair_output),
        valid_module_catalog,
    )
    missing = _missing_holes(page)
    if missing:
        compact = {key: sorted(value) for key, value in missing.items()}
        raise PlanningHoleFillError(
            "planner omitted mandatory host-owned implementation holes after bounded repair: "
            + json.dumps(compact, sort_keys=True, separators=(",", ":"))
        )
    return page


__all__ = ["PlanningHoleFillError", "fill_evidence_page"]
