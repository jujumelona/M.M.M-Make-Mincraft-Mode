from __future__ import annotations

"""Fill immutable implementation holes using bounded, host-numbered text pages.

The host owns JSON assembly and identities. Valid blocks survive a local repair; a
malformed or missing block never becomes an accepted implementation by fallback.
"""

import json
from collections.abc import Mapping
from typing import Any

from .planner_hole_text import parse_hole_text
from .planner_operation import planner_operation
from .planner_stage_trace import PlannerStageTrace
from .planner_template_schema import merge_model_output_into_skeleton

_MAX_PAGE_HOLES = 16


class PlanningHoleFillError(RuntimeError):
    """A mandatory host-owned implementation hole remains unresolved."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _module_packet(
    module: Mapping[str, Any], only_holes: set[str] | None = None
) -> dict[str, Any]:
    config = _mapping(module.get("config"))
    task = _mapping(config.get("evidence_task"))
    template = _mapping(config.get("implementation_template"))
    holes = [
        dict(hole)
        for hole in template.get("holes", [])
        if isinstance(hole, Mapping)
        and (only_holes is None or str(hole.get("hole_id") or "") in only_holes)
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


def _messages(module: Mapping[str, Any], *, error: str = "") -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Fill only the supplied Minecraft implementation holes. Do not redesign gameplay "
                "or change target coordinates, module identities, dependencies, gates or paths. "
                "Use only supplied evidence references; record unavailable API facts as uncertainties. "
                "Return plain text, no JSON and no code fences. Number supplied holes in order, "
                "starting at 1. Each block must have this exact form:\n"
                "### Hole 1\nDecision: concrete local choice\nSteps:\n- ordered coding action\n"
                "Bindings:\n- supplied symbol or resource (or none)\n"
                "References:\n- supplied evidence reference (or none)\n"
                "Verification: executable proof of this hole\nUncertainties:\n- unresolved fact (or none)\n"
                "Use short complete blocks. Quotes and code fragments need no JSON escaping."
            ),
        },
        {
            "role": "user",
            "content": (
                (
                    "Repair only these unresolved holes. " + error
                    if error
                    else "Fill these host-owned holes."
                )
                + "\n"
                + json.dumps(
                    {"modules": [module]}, ensure_ascii=False, separators=(",", ":")
                )
            ),
        },
    ]


def _fill_page(
    router: Any, module: dict[str, Any], trace: PlannerStageTrace
) -> list[dict[str, Any]]:
    holes = module["implementation_template"]["holes"]
    remaining = list(holes)
    accepted: dict[str, dict[str, Any]] = {}
    error = ""
    for attempt in (1, 2):
        packet = {
            **module,
            "implementation_template": {
                **module["implementation_template"],
                "holes": remaining,
            },
        }
        raw = ""
        try:
            with planner_operation(
                "implementation_holes", output_tokens=128 + 256 * len(remaining)
            ):
                raw = router.generate_text(
                    "planner",
                    _messages(packet, error=error),
                    response_format="text",
                    enable_tools=False,
                )
            fills = parse_hole_text(str(raw or ""), remaining)
            accepted.update((fill["hole_id"], fill) for fill in fills)
            remaining = [hole for hole in holes if hole["hole_id"] not in accepted]
            error = "Missing complete Decision, Steps or Verification blocks."
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            trace.record_attempt(
                raw_output=str(raw), validation_error=f"{type(exc).__name__}: {exc}"
            )
            raise
        trace.record_attempt(
            raw_output=str(raw),
            validation_error=error if remaining else None,
            accepted={"hole_fills": list(accepted.values())},
            context={
                "attempt": attempt,
                "module_id": module["module_id"],
                "remaining_hole_ids": [hole["hole_id"] for hole in remaining],
            },
        )
        if not remaining:
            return [accepted[hole["hole_id"]] for hole in holes]
    raise PlanningHoleFillError(
        "Unresolved implementation holes after bounded repair: "
        + ", ".join(str(hole["hole_id"]) for hole in remaining)
        + "; "
        + error
    )


def fill_evidence_page(
    router: Any,
    skeleton: Mapping[str, Any],
    *,
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    modules = []
    for raw_module in skeleton.get("modules", []):
        if not isinstance(raw_module, Mapping):
            continue
        module = _module_packet(raw_module)
        template = module["implementation_template"]
        holes = template["holes"]
        if not holes:
            continue
        trace = PlannerStageTrace(
            stage="implementation_holes",
            prompt=module["scope"],
            metadata={"module_id": module["module_id"]},
        )
        fills = []
        for offset in range(0, len(holes), _MAX_PAGE_HOLES):
            page = {
                **module,
                "implementation_template": {
                    **template,
                    "holes": holes[offset : offset + _MAX_PAGE_HOLES],
                },
            }
            fills.extend(_fill_page(router, page, trace))
        required = set(template["completion_policy"].get("required_hole_ids", []))
        if not required <= {fill["hole_id"] for fill in fills}:
            raise PlanningHoleFillError(
                "Host template references unknown mandatory holes"
            )
        modules.append(
            {"module_id": module["module_id"], "config": {"hole_fills": fills}}
        )
    return merge_model_output_into_skeleton(
        skeleton, {"modules": modules}, valid_module_catalog
    )


__all__ = ["PlanningHoleFillError", "fill_evidence_page"]
