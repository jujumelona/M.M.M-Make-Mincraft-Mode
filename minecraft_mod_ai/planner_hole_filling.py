from __future__ import annotations

"""Fill immutable implementation holes using bounded, host-numbered text pages.

The host owns JSON assembly and identities. Valid blocks survive a local repair; a
malformed or missing block never becomes an accepted implementation by fallback.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .planner_hole_text import parse_hole_text, parse_multi_page_hole_text
from .planner_operation import planner_operation
from .planner_stage_trace import PlannerStageTrace
from .planner_template_schema import merge_model_output_into_skeleton

# At 256 output tokens per hole plus a 128-token envelope, twelve holes remain
# comfortably below the 4096-token operation ceiling. Character budgeting is enforced
# independently because evidence-rich holes can vary greatly in prompt size.
_MAX_PAGE_HOLES = 12
_MAX_BUNDLE_CHARS = 24_000
_MAX_OUTPUT_TOKENS = 4_096


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


def _page_prompt_packet(
    page: Mapping[str, Any], holes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Expose one page once: shared module context plus its host-owned holes."""
    base_module = dict(page.get("module") or {})
    template = dict(base_module.get("implementation_template") or {})
    template.pop("holes", None)
    base_module["module_id"] = str(
        page.get("module_id") or base_module.get("module_id") or ""
    )
    base_module["implementation_template"] = template
    return {
        "page_id": str(page["page_id"]),
        "holes": [dict(hole) for hole in holes],
        "module": base_module,
    }


def _multi_page_messages(
    pages: Sequence[Mapping[str, Any]], *, error: str = ""
) -> list[dict[str, str]]:
    system = (
        "Fill only the supplied Minecraft implementation holes across the supplied pages. "
        "Do not redesign gameplay or change target coordinates, module identities, dependencies, "
        "gates or paths. Use only supplied evidence references; record unavailable API facts as "
        "uncertainties. Return plain text bounded blocks using host-owned IDs only. No JSON and "
        "no code fences.\n"
        "Wrap each page with:\n"
        "BEGIN PAGE <host_page_id>\n"
        "...\n"
        "END PAGE <host_page_id>\n\n"
        "Within each page, wrap each hole with:\n"
        "BEGIN <hole_id>\n"
        "Decision: concrete local choice\n"
        "Steps:\n"
        "- ordered coding action\n"
        "Bindings:\n"
        "- supplied symbol or resource (or none)\n"
        "References:\n"
        "- supplied evidence reference (or none)\n"
        "Verification: executable proof of this hole\n"
        "Uncertainties:\n"
        "- unresolved fact (or none)\n"
        "END <hole_id>\n\n"
        "Use short complete blocks. Quotes and code fragments need no JSON escaping."
    )
    user_header = (
        f"Repair only these unresolved holes. {error}"
        if error
        else "Fill these host-owned pages and holes."
    )
    payload = {
        "pages": [
            _page_prompt_packet(page, page.get("holes") or ())
            for page in pages
        ]
    }
    user_content = user_header + "\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _bundle_prompt_size(pages: Sequence[Mapping[str, Any]]) -> int:
    payload = {
        "pages": [
            _page_prompt_packet(page, page.get("holes") or ())
            for page in pages
        ]
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _single_prompt_size(module: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            {"modules": [module]}, ensure_ascii=False, separators=(",", ":")
        )
    )


def _page_entry(
    module: Mapping[str, Any],
    *,
    skeleton_index: int,
    page_id: str,
    holes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    template = _mapping(module.get("implementation_template"))
    return {
        "page_id": page_id,
        "module_id": str(module.get("module_id") or ""),
        "scope": str(module.get("scope") or ""),
        "skeleton_index": skeleton_index,
        "holes": [dict(hole) for hole in holes],
        "completion_policy": _mapping(template.get("completion_policy")),
        "module": dict(module),
    }


def _split_module_pages(
    module: Mapping[str, Any], *, skeleton_index: int
) -> list[dict[str, Any]]:
    """Greedily split a module by both output-hole and serialized-prompt budgets."""
    template = _mapping(module.get("implementation_template"))
    holes = [dict(hole) for hole in template.get("holes", []) if isinstance(hole, Mapping)]
    if not holes:
        return []
    module_id = str(module.get("module_id") or "")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for hole in holes:
        candidate = [*current, hole]
        provisional = _page_entry(
            module,
            skeleton_index=skeleton_index,
            page_id=f"{module_id}_p{len(chunks)}",
            holes=candidate,
        )
        exceeds = (
            len(candidate) > _MAX_PAGE_HOLES
            or _bundle_prompt_size([provisional]) > _MAX_BUNDLE_CHARS
        )
        if current and exceeds:
            chunks.append(current)
            current = [hole]
            single = _page_entry(
                module,
                skeleton_index=skeleton_index,
                page_id=f"{module_id}_p{len(chunks)}",
                holes=current,
            )
            if _bundle_prompt_size([single]) > _MAX_BUNDLE_CHARS:
                raise PlanningHoleFillError(
                    f"implementation hole exceeds bounded prompt budget for {module_id}: "
                    f"{hole.get('hole_id', '<unknown>')}"
                )
        else:
            current = candidate
            if not chunks and len(current) == 1 and exceeds:
                raise PlanningHoleFillError(
                    f"implementation hole exceeds bounded prompt budget for {module_id}: "
                    f"{hole.get('hole_id', '<unknown>')}"
                )

    if current:
        chunks.append(current)

    multiple = len(chunks) > 1
    pages = [
        _page_entry(
            module,
            skeleton_index=skeleton_index,
            page_id=f"{module_id}_p{index}" if multiple else module_id,
            holes=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]
    for page in pages:
        if _bundle_prompt_size([page]) > _MAX_BUNDLE_CHARS:
            raise PlanningHoleFillError(
                f"implementation page exceeds bounded prompt budget: {page['page_id']}"
            )
    return pages


def _fill_page(
    router: Any, module: dict[str, Any], trace: PlannerStageTrace
) -> list[dict[str, Any]]:
    holes = module["implementation_template"]["holes"]
    if len(holes) > _MAX_PAGE_HOLES:
        raise PlanningHoleFillError(
            f"single implementation page exceeds bounded hole limit: {len(holes)} > {_MAX_PAGE_HOLES}"
        )
    if _single_prompt_size(module) > _MAX_BUNDLE_CHARS:
        raise PlanningHoleFillError("single implementation page exceeds bounded prompt budget")

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
            output_tokens = min(
                _MAX_OUTPUT_TOKENS,
                max(128, 128 + 256 * len(remaining)),
            )
            with planner_operation("implementation_holes", output_tokens=output_tokens):
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


def _fill_bundle(
    router: Any,
    bundle_pages: Sequence[Mapping[str, Any]],
    traces_by_module: Mapping[str, PlannerStageTrace],
) -> dict[str, list[dict[str, Any]]]:
    """Fill a budgeted bundle of multi-page implementation holes with isolated repair."""
    pages_holes: dict[str, list[dict[str, Any]]] = {
        str(page["page_id"]): list(page["holes"]) for page in bundle_pages
    }
    pages_by_id = {str(page["page_id"]): page for page in bundle_pages}
    accepted: dict[str, dict[str, dict[str, Any]]] = {
        page_id: {} for page_id in pages_holes
    }
    remaining_by_page = {
        page_id: list(holes) for page_id, holes in pages_holes.items()
    }
    error = ""

    for attempt in (1, 2):
        active_pages = []
        for page_id, remaining in remaining_by_page.items():
            if not remaining:
                continue
            original = pages_by_id[page_id]
            active = dict(original)
            active["holes"] = remaining
            active_pages.append(active)
        if not active_pages:
            break

        total_holes = sum(len(page["holes"]) for page in active_pages)
        if total_holes > _MAX_PAGE_HOLES:
            raise PlanningHoleFillError(
                f"implementation bundle exceeds bounded hole limit: {total_holes} > {_MAX_PAGE_HOLES}"
            )
        if _bundle_prompt_size(active_pages) > _MAX_BUNDLE_CHARS:
            raise PlanningHoleFillError("implementation bundle exceeds bounded prompt budget")
        output_tokens = min(
            _MAX_OUTPUT_TOKENS,
            max(128, 128 + 256 * total_holes),
        )

        messages = _multi_page_messages(active_pages, error=error)
        raw = ""
        try:
            with planner_operation("implementation_holes", output_tokens=output_tokens):
                raw = router.generate_text(
                    "planner",
                    messages,
                    response_format="text",
                    enable_tools=False,
                )

            if (
                len(active_pages) > 1
                and not str(raw or "").strip().startswith("{")
                and not re.search(
                    r"BEGIN\s+PAGE|###\s*Page", str(raw or ""), re.IGNORECASE
                )
            ):
                first_page = active_pages[0]
                first_page_id = str(first_page["page_id"])
                try:
                    first_fills = parse_hole_text(
                        str(raw or ""), first_page["holes"]
                    )
                    for fill in first_fills:
                        hole_id = str(fill.get("hole_id") or "")
                        if hole_id:
                            accepted[first_page_id][hole_id] = fill
                except Exception:
                    pass
            else:
                parsed = parse_multi_page_hole_text(
                    str(raw or ""),
                    {page["page_id"]: page["holes"] for page in active_pages},
                )
                for page_id, fills in parsed.items():
                    for fill in fills:
                        hole_id = str(fill.get("hole_id") or "")
                        if hole_id:
                            accepted[page_id][hole_id] = fill

            new_remaining = {}
            for page_id, expected_holes in pages_holes.items():
                unfilled = [
                    hole
                    for hole in expected_holes
                    if str(hole.get("hole_id") or "") not in accepted[page_id]
                ]
                if unfilled:
                    new_remaining[page_id] = unfilled
            remaining_by_page = new_remaining
            error = "Missing complete Decision, Steps or Verification blocks."
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            for page in active_pages:
                module_id = str(page.get("module_id") or "")
                if module_id in traces_by_module:
                    traces_by_module[module_id].record_attempt(
                        raw_output=str(raw),
                        validation_error=f"{type(exc).__name__}: {exc}",
                    )
            raise

        for page in active_pages:
            module_id = str(page.get("module_id") or "")
            page_id = str(page["page_id"])
            if module_id in traces_by_module:
                traces_by_module[module_id].record_attempt(
                    raw_output=str(raw),
                    validation_error=error if remaining_by_page.get(page_id) else None,
                    accepted={"hole_fills": list(accepted[page_id].values())},
                    context={
                        "attempt": attempt,
                        "page_id": page_id,
                        "remaining_hole_ids": [
                            hole["hole_id"]
                            for hole in remaining_by_page.get(page_id, [])
                        ],
                    },
                )

        if not remaining_by_page:
            break

    result: dict[str, list[dict[str, Any]]] = {}
    for page_id, expected_holes in pages_holes.items():
        result[page_id] = [
            accepted[page_id][hole["hole_id"]]
            for hole in expected_holes
            if hole["hole_id"] in accepted[page_id]
        ]
    return result


def _pack_pages(pages: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Deterministic first-fit packing reduces model calls without exceeding either budget."""
    bundles: list[list[Mapping[str, Any]]] = []
    for page in pages:
        placed = False
        for bundle in bundles:
            candidate = [*bundle, page]
            if (
                sum(len(item["holes"]) for item in candidate) <= _MAX_PAGE_HOLES
                and _bundle_prompt_size(candidate) <= _MAX_BUNDLE_CHARS
            ):
                bundle.append(page)
                placed = True
                break
        if not placed:
            if _bundle_prompt_size([page]) > _MAX_BUNDLE_CHARS:
                raise PlanningHoleFillError(
                    f"implementation page exceeds bounded prompt budget: {page['page_id']}"
                )
            bundles.append([page])
    return bundles


def fill_evidence_pages(
    router: Any,
    skeletons: Sequence[Mapping[str, Any]],
    *,
    valid_module_catalog: set[str],
) -> list[dict[str, Any]]:
    """Fill bounded implementation-hole pages across dependency-ready skeletons."""
    all_pages: list[dict[str, Any]] = []
    page_ids_by_module: dict[str, list[str]] = {}
    traces_by_module: dict[str, PlannerStageTrace] = {}

    for skeleton_index, skeleton in enumerate(skeletons):
        for raw_module in skeleton.get("modules", []):
            if not isinstance(raw_module, Mapping):
                continue
            module = _module_packet(raw_module)
            template = module["implementation_template"]
            if not template["holes"]:
                continue
            module_id = module["module_id"]
            if module_id not in traces_by_module:
                traces_by_module[module_id] = PlannerStageTrace(
                    stage="implementation_holes",
                    prompt=module["scope"],
                    metadata={"module_id": module_id},
                )
            pages = _split_module_pages(module, skeleton_index=skeleton_index)
            all_pages.extend(pages)
            page_ids_by_module[module_id] = [str(page["page_id"]) for page in pages]

    if not all_pages:
        return [
            merge_model_output_into_skeleton(
                skeleton, {"modules": []}, valid_module_catalog
            )
            for skeleton in skeletons
        ]

    if len(all_pages) == 1:
        single_page = all_pages[0]
        module_id = str(single_page["module_id"])
        trace = traces_by_module[module_id]
        page_module = dict(single_page["module"])
        template = dict(page_module["implementation_template"])
        template["holes"] = list(single_page["holes"])
        page_module["implementation_template"] = template
        fills = _fill_page(router, page_module, trace)
        all_fills_by_page = {str(single_page["page_id"]): fills}
    else:
        all_fills_by_page: dict[str, list[dict[str, Any]]] = {}
        for bundle in _pack_pages(all_pages):
            bundle_fills = _fill_bundle(router, bundle, traces_by_module)
            all_fills_by_page.update(bundle_fills)

    filled_skeletons: list[dict[str, Any]] = []
    for skeleton in skeletons:
        modules = []
        for raw_module in skeleton.get("modules", []):
            if not isinstance(raw_module, Mapping):
                continue
            module = _module_packet(raw_module)
            template = module["implementation_template"]
            holes = template["holes"]
            if not holes:
                continue
            module_id = module["module_id"]
            module_fills = [
                fill
                for page_id in page_ids_by_module.get(module_id, ())
                for fill in all_fills_by_page.get(page_id, ())
            ]

            required = set(template["completion_policy"].get("required_hole_ids", []))
            filled_ids = {fill["hole_id"] for fill in module_fills}
            if not required <= filled_ids:
                missing = sorted(required - filled_ids)
                raise PlanningHoleFillError(
                    f"Host template references unknown or unfilled mandatory holes for {module_id}: "
                    + ", ".join(missing)
                )
            modules.append(
                {"module_id": module_id, "config": {"hole_fills": module_fills}}
            )

        filled_skeletons.append(
            merge_model_output_into_skeleton(
                skeleton, {"modules": modules}, valid_module_catalog
            )
        )

    return filled_skeletons


def fill_evidence_page(
    router: Any,
    skeleton: Mapping[str, Any],
    *,
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    return fill_evidence_pages(
        router,
        [skeleton],
        valid_module_catalog=valid_module_catalog,
    )[0]


__all__ = [
    "PlanningHoleFillError",
    "_fill_page",
    "fill_evidence_page",
    "fill_evidence_pages",
]
