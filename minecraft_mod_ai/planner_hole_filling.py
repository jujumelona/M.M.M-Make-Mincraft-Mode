from __future__ import annotations

"""Fill immutable implementation holes using bounded, host-numbered text pages.

The host owns JSON assembly and identities. Valid blocks survive a local repair; a
malformed or missing block never becomes an accepted implementation by fallback.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .planner_hole_text import parse_hole_text, parse_multi_page_hole_text
from .planner_operation import planner_operation
from .planner_stage_trace import PlannerStageTrace
from .planner_template_schema import merge_model_output_into_skeleton

_MAX_PAGE_HOLES = 32
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
    modules_payload = []
    for p in pages:
        base_mod = dict(p.get("module") or {})
        tmpl = dict(base_mod.get("implementation_template") or {})
        tmpl["holes"] = list(p.get("holes") or [])
        base_mod["module_id"] = str(
            p.get("module_id") or base_mod.get("module_id") or ""
        )
        base_mod["implementation_template"] = tmpl
        modules_payload.append(base_mod)
    payload = {
        "modules": modules_payload,
        "pages": list(pages),
    }
    user_content = user_header + "\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
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


def _fill_bundle(
    router: Any,
    bundle_pages: Sequence[Mapping[str, Any]],
    traces_by_module: Mapping[str, PlannerStageTrace],
) -> dict[str, list[dict[str, Any]]]:
    """Fill a budgeted bundle of multi-page implementation holes with isolated failure retry."""
    pages_holes: dict[str, list[dict[str, Any]]] = {
        str(page["page_id"]): list(page["holes"]) for page in bundle_pages
    }
    accepted: dict[str, dict[str, dict[str, Any]]] = {
        page_id: {} for page_id in pages_holes
    }
    remaining_by_page = {
        page_id: list(holes) for page_id, holes in pages_holes.items()
    }
    error = ""

    for attempt in (1, 2):
        active_pages = [
            {
                "page_id": page_id,
                "module_id": str(p["module_id"]),
                "scope": str(p.get("scope") or ""),
                "holes": remaining_by_page[page_id],
            }
            for p in bundle_pages
            if (page_id := str(p["page_id"])) in remaining_by_page and remaining_by_page[page_id]
        ]
        if not active_pages:
            break

        total_holes = sum(len(p["holes"]) for p in active_pages)
        output_tokens = min(_MAX_OUTPUT_TOKENS, max(128, 128 + 256 * total_holes))

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
            import re

            if (
                len(active_pages) > 1
                and not str(raw or "").strip().startswith("{")
                and not re.search(r"BEGIN\s+PAGE|###\s*Page", str(raw or ""), re.IGNORECASE)
            ):
                first_page = active_pages[0]
                first_page_id = str(first_page["page_id"])
                try:
                    first_fills = parse_hole_text(str(raw or ""), first_page["holes"])
                    for fill in first_fills:
                        hole_id = str(fill.get("hole_id") or "")
                        if hole_id:
                            accepted[first_page_id][hole_id] = fill
                except Exception:
                    pass
            else:
                parsed = parse_multi_page_hole_text(
                    str(raw or ""),
                    {p["page_id"]: p["holes"] for p in active_pages},
                )
                for page_id, fills in parsed.items():
                    for fill in fills:
                        hole_id = str(fill.get("hole_id") or "")
                        if hole_id:
                            accepted[page_id][hole_id] = fill

            # Isolated failure isolation: sibling pages survive!
            new_remaining = {}
            for page_id, expected_holes in pages_holes.items():
                unfilled = [
                    h for h in expected_holes if str(h.get("hole_id") or "") not in accepted[page_id]
                ]
                if unfilled:
                    new_remaining[page_id] = unfilled
            remaining_by_page = new_remaining
            error = "Missing complete Decision, Steps or Verification blocks."
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            for p in active_pages:
                mod_id = str(p.get("module_id") or "")
                if mod_id in traces_by_module:
                    traces_by_module[mod_id].record_attempt(
                        raw_output=str(raw),
                        validation_error=f"{type(exc).__name__}: {exc}",
                    )
            raise

        for p in active_pages:
            mod_id = str(p.get("module_id") or "")
            page_id = str(p["page_id"])
            if mod_id in traces_by_module:
                traces_by_module[mod_id].record_attempt(
                    raw_output=str(raw),
                    validation_error=error if remaining_by_page.get(page_id) else None,
                    accepted={"hole_fills": list(accepted[page_id].values())},
                    context={
                        "attempt": attempt,
                        "page_id": page_id,
                        "remaining_hole_ids": [
                            h["hole_id"] for h in remaining_by_page.get(page_id, [])
                        ],
                    },
                )

        if not remaining_by_page:
            break

    result: dict[str, list[dict[str, Any]]] = {}
    for page_id, expected_holes in pages_holes.items():
        page_fills = [
            accepted[page_id][h["hole_id"]]
            for h in expected_holes
            if h["hole_id"] in accepted[page_id]
        ]
        result[page_id] = page_fills

    return result


def fill_evidence_pages(
    router: Any,
    skeletons: Sequence[Mapping[str, Any]],
    *,
    valid_module_catalog: set[str],
) -> list[dict[str, Any]]:
    """Fill multi-page bounded implementation holes across ready skeletons."""
    all_pages: list[dict[str, Any]] = []
    page_registry: dict[str, dict[str, Any]] = {}
    traces_by_module: dict[str, PlannerStageTrace] = {}

    for skel_index, skeleton in enumerate(skeletons):
        for raw_module in skeleton.get("modules", []):
            if not isinstance(raw_module, Mapping):
                continue
            module = _module_packet(raw_module)
            template = module["implementation_template"]
            holes = template["holes"]
            if not holes:
                continue
            mod_id = module["module_id"]
            if mod_id not in traces_by_module:
                traces_by_module[mod_id] = PlannerStageTrace(
                    stage="implementation_holes",
                    prompt=module["scope"],
                    metadata={"module_id": mod_id},
                )
            for offset in range(0, len(holes), _MAX_PAGE_HOLES):
                page_slice = holes[offset : offset + _MAX_PAGE_HOLES]
                page_idx = offset // _MAX_PAGE_HOLES
                page_id = f"{mod_id}_p{page_idx}" if len(holes) > _MAX_PAGE_HOLES else mod_id
                page_entry = {
                    "page_id": page_id,
                    "module_id": mod_id,
                    "scope": module["scope"],
                    "skeleton_index": skel_index,
                    "holes": page_slice,
                    "completion_policy": template["completion_policy"],
                    "module": module,
                }
                all_pages.append(page_entry)
                page_registry[page_id] = page_entry

    if not all_pages:
        return [
            merge_model_output_into_skeleton(skeleton, {"modules": []}, valid_module_catalog)
            for skeleton in skeletons
        ]

    # If single page with single module, use _fill_page directly for maximum compatibility
    if len(all_pages) == 1:
        single_page = all_pages[0]
        mod_id = single_page["module_id"]
        trace = traces_by_module[mod_id]
        page_mod = {
            **single_page["module"],
            "implementation_template": {
                **single_page["module"]["implementation_template"],
                "holes": single_page["holes"],
            },
        }
        fills = _fill_page(router, page_mod, trace)
        all_fills_by_page = {single_page["page_id"]: fills}
    else:
        # Budget into multi-page bundles
        bundles: list[list[dict[str, Any]]] = []
        current_bundle: list[dict[str, Any]] = []
        current_chars = 0
        current_holes = 0

        for page in all_pages:
            page_json = json.dumps(page, ensure_ascii=False)
            p_chars = len(page_json)
            p_holes = len(page["holes"])
            if (
                current_bundle
                and (current_chars + p_chars > _MAX_BUNDLE_CHARS or current_holes + p_holes > _MAX_PAGE_HOLES)
            ):
                bundles.append(current_bundle)
                current_bundle = [page]
                current_chars = p_chars
                current_holes = p_holes
            else:
                current_bundle.append(page)
                current_chars += p_chars
                current_holes += p_holes

        if current_bundle:
            bundles.append(current_bundle)

        # Fill each bundle with isolated failure handling
        all_fills_by_page = {}
        for bundle in bundles:
            bundle_fills = _fill_bundle(router, bundle, traces_by_module)
            all_fills_by_page.update(bundle_fills)

    # Deterministic merge into skeleton order
    filled_skeletons: list[dict[str, Any]] = []
    for skel_index, skeleton in enumerate(skeletons):
        modules = []
        for raw_module in skeleton.get("modules", []):
            if not isinstance(raw_module, Mapping):
                continue
            module = _module_packet(raw_module)
            template = module["implementation_template"]
            holes = template["holes"]
            if not holes:
                continue
            mod_id = module["module_id"]
            # Collect fills across all pages for this module
            mod_fills = []
            for offset in range(0, len(holes), _MAX_PAGE_HOLES):
                page_idx = offset // _MAX_PAGE_HOLES
                page_id = f"{mod_id}_p{page_idx}" if len(holes) > _MAX_PAGE_HOLES else mod_id
                mod_fills.extend(all_fills_by_page.get(page_id, []))

            required = set(template["completion_policy"].get("required_hole_ids", []))
            if not required <= {fill["hole_id"] for fill in mod_fills}:
                missing = sorted(required - {fill["hole_id"] for fill in mod_fills})
                raise PlanningHoleFillError(
                    f"Host template references unknown or unfilled mandatory holes for {mod_id}: "
                    + ", ".join(missing)
                )
            modules.append(
                {"module_id": mod_id, "config": {"hole_fills": mod_fills}}
            )

        filled_skel = merge_model_output_into_skeleton(
            skeleton, {"modules": modules}, valid_module_catalog
        )
        filled_skeletons.append(filled_skel)

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
