from __future__ import annotations

"""Refine host-owned implementation templates without making model output authoritative.

Every implementation hole has a deterministic host fill before the model is called. The
model gets one bounded opportunity to improve those fills. Missing blocks, malformed text,
prompt-budget overflow, or a model exception never make the planner fail; the host fill
remains in place. Only contradictions inside the host-owned template itself are fatal.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .planner_hole_text import parse_hole_text, parse_multi_page_hole_text
from .planner_operation import planner_operation
from .planner_stage_trace import PlannerStageTrace
from .planner_template_schema import merge_model_output_into_skeleton

_MAX_PAGE_HOLES = 12
_MAX_BUNDLE_CHARS = 24_000
_MAX_OUTPUT_TOKENS = 4_096


class PlanningHoleFillError(RuntimeError):
    """The host-owned implementation template is internally contradictory."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        raw: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        return []
    return list(
        dict.fromkeys(
            text
            for item in raw
            if (text := str(item or "").strip())
        )
    )


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


def _host_default_fill(hole: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one executable planning fill entirely from host-owned template facts."""
    hole_id = str(hole.get("hole_id") or "").strip()
    if not hole_id:
        raise PlanningHoleFillError("host implementation template contains a hole without hole_id")

    kind = str(hole.get("kind") or "implementation").strip()
    subject = str(hole.get("subject") or kind).strip()
    target = _mapping(hole.get("target_constraints"))
    artifact_refs = _strings(hole.get("artifact_refs"))
    consumes = _strings(hole.get("consumes"))
    provides = _strings(hole.get("provides"))
    acceptance_refs = _strings(hole.get("acceptance_refs"))
    evidence_refs = list(
        dict.fromkeys(
            [
                *_strings(hole.get("evidence_refs")),
                *_strings(hole.get("reference_slice_refs")),
            ]
        )
    )

    target_bits = [
        str(target.get(key) or "").strip()
        for key in ("minecraft_version", "loader", "mappings", "java_version")
        if str(target.get(key) or "").strip()
    ]
    target_note = ", ".join(target_bits) if target_bits else "the host-selected target"

    steps = [
        f"Implement {subject} inside the host-owned task and dependency boundaries.",
        f"Keep the implementation compatible with {target_note} and preserve all host-owned identifiers and gates.",
    ]
    if consumes:
        steps.append("Read only the declared inputs: " + ", ".join(consumes) + ".")
    if provides:
        steps.append("Produce the declared outputs: " + ", ".join(provides) + ".")
    if artifact_refs:
        steps.append("Realize the declared artifacts: " + ", ".join(artifact_refs) + ".")

    verification = (
        "Verify the observable acceptance contract: " + "; ".join(acceptance_refs)
        if acceptance_refs
        else f"Verify {subject} with the host-required compile/static/resource/GameTest/runtime gates that apply to this task."
    )

    uncertainties: list[str] = []
    if not evidence_refs:
        uncertainties.append(
            "Exact target API symbols are intentionally left to code-time evidence resolution; the host task contract remains authoritative."
        )

    bindings = list(dict.fromkeys([*artifact_refs, *consumes, *provides]))
    return {
        "hole_id": hole_id,
        "implementation_decision": (
            f"Implement {subject} as the local realization of the host-owned {kind} obligation without changing task scope."
        ),
        "local_steps": steps,
        "code_bindings": bindings,
        "reference_uses": evidence_refs,
        "verification_intent": verification,
        "uncertainties": uncertainties,
        "fill_source": "host_template_default",
    }


def _merge_refinement(
    host_fill: Mapping[str, Any], model_fill: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Overlay only useful model detail; host default remains complete underneath."""
    result = dict(host_fill)
    if not isinstance(model_fill, Mapping):
        return result
    scalar_fields = ("implementation_decision", "verification_intent")
    list_fields = ("local_steps", "code_bindings", "reference_uses", "uncertainties")
    changed = False
    for field in scalar_fields:
        value = str(model_fill.get(field) or "").strip()
        if value:
            result[field] = value
            changed = True
    for field in list_fields:
        values = _strings(model_fill.get(field))
        if values:
            result[field] = values
            changed = True
    if changed:
        result["fill_source"] = "model_refined_host_template"
    return result


def _messages(module: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Refine only the supplied host-owned Minecraft implementation holes. The host "
                "already has a complete fallback plan, so do not redesign gameplay or change target "
                "coordinates, identities, dependencies, gates, paths, or hole count. Return plain "
                "text only. Number holes in supplied order. Each useful block may use:\n"
                "### Hole 1\nDecision: concrete local choice\nSteps:\n- ordered coding action\n"
                "Bindings:\n- supplied symbol/resource\nReferences:\n- supplied evidence\n"
                "Verification: executable proof\nUncertainties:\n- unresolved fact\n"
                "If a detail is unknown, omit it rather than inventing it."
            ),
        },
        {
            "role": "user",
            "content": "Refine these host-owned holes.\n"
            + json.dumps(
                {"modules": [module]}, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def _page_prompt_packet(
    page: Mapping[str, Any], holes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
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


def _multi_page_messages(pages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    system = (
        "Refine only the supplied host-owned implementation holes. The host already has complete "
        "fallback fills. Do not redesign or change host-owned identities, dependencies, gates, "
        "paths, targets, or hole count. Return plain text using host page/hole IDs.\n"
        "BEGIN PAGE <page_id>\nBEGIN <hole_id>\nDecision: ...\nSteps:\n- ...\n"
        "Bindings:\n- ...\nReferences:\n- ...\nVerification: ...\nUncertainties:\n- ...\n"
        "END <hole_id>\nEND PAGE <page_id>\n"
        "Omit unknown detail rather than inventing it."
    )
    payload = {
        "pages": [
            _page_prompt_packet(page, page.get("holes") or ()) for page in pages
        ]
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Refine these host-owned pages.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _bundle_prompt_size(pages: Sequence[Mapping[str, Any]]) -> int:
    payload = {
        "pages": [
            _page_prompt_packet(page, page.get("holes") or ()) for page in pages
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
    """Split only for optional model refinement; host fills do not depend on these budgets."""
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
        if current and (
            len(candidate) > _MAX_PAGE_HOLES
            or _bundle_prompt_size([provisional]) > _MAX_BUNDLE_CHARS
        ):
            chunks.append(current)
            current = [hole]
        else:
            current = candidate
    if current:
        chunks.append(current)
    multiple = len(chunks) > 1
    return [
        _page_entry(
            module,
            skeleton_index=skeleton_index,
            page_id=f"{module_id}_p{index}" if multiple else module_id,
            holes=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def _fill_page(
    router: Any, module: dict[str, Any], trace: PlannerStageTrace
) -> list[dict[str, Any]]:
    """Return complete host fills, optionally refined by one model call."""
    holes = [
        dict(hole)
        for hole in module["implementation_template"].get("holes", [])
        if isinstance(hole, Mapping)
    ]
    defaults = {str(hole["hole_id"]): _host_default_fill(hole) for hole in holes}
    if not holes:
        return []

    skip_reason = ""
    if len(holes) > _MAX_PAGE_HOLES:
        skip_reason = "model refinement skipped: page exceeds bounded hole limit"
    elif _single_prompt_size(module) > _MAX_BUNDLE_CHARS:
        skip_reason = "model refinement skipped: page exceeds bounded prompt budget"

    raw = ""
    parsed: list[dict[str, Any]] = []
    if not skip_reason:
        try:
            output_tokens = min(
                _MAX_OUTPUT_TOKENS,
                max(128, 128 + 256 * len(holes)),
            )
            with planner_operation("implementation_holes", output_tokens=output_tokens):
                raw = router.generate_text(
                    "planner",
                    _messages(module),
                    response_format="text",
                    enable_tools=False,
                )
            parsed = parse_hole_text(str(raw or ""), holes)
        except Exception as exc:
            skip_reason = f"model refinement ignored: {type(exc).__name__}: {exc}"

    by_id = {
        str(fill.get("hole_id") or ""): fill
        for fill in parsed
        if isinstance(fill, Mapping) and str(fill.get("hole_id") or "") in defaults
    }
    result = [
        _merge_refinement(defaults[str(hole["hole_id"])], by_id.get(str(hole["hole_id"])))
        for hole in holes
    ]
    trace.record_attempt(
        raw_output=str(raw),
        validation_error=skip_reason or None,
        accepted={"hole_fills": result},
        context={
            "attempt": 1,
            "module_id": module["module_id"],
            "host_default_count": len(defaults),
            "model_refined_count": sum(
                1 for fill in result if fill.get("fill_source") == "model_refined_host_template"
            ),
        },
    )
    return result


def _fill_bundle(
    router: Any,
    bundle_pages: Sequence[Mapping[str, Any]],
    traces_by_module: Mapping[str, PlannerStageTrace],
) -> dict[str, list[dict[str, Any]]]:
    """Return complete host fills for all pages and optionally refine them once."""
    defaults: dict[str, dict[str, dict[str, Any]]] = {}
    pages_holes: dict[str, list[dict[str, Any]]] = {}
    for page in bundle_pages:
        page_id = str(page["page_id"])
        holes = [dict(hole) for hole in page.get("holes", ()) if isinstance(hole, Mapping)]
        pages_holes[page_id] = holes
        defaults[page_id] = {
            str(hole["hole_id"]): _host_default_fill(hole) for hole in holes
        }

    total_holes = sum(len(holes) for holes in pages_holes.values())
    skip_reason = ""
    if total_holes > _MAX_PAGE_HOLES:
        skip_reason = "model refinement skipped: bundle exceeds bounded hole limit"
    elif _bundle_prompt_size(bundle_pages) > _MAX_BUNDLE_CHARS:
        skip_reason = "model refinement skipped: bundle exceeds bounded prompt budget"

    raw = ""
    parsed: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in pages_holes}
    if not skip_reason and total_holes:
        try:
            output_tokens = min(
                _MAX_OUTPUT_TOKENS,
                max(128, 128 + 256 * total_holes),
            )
            with planner_operation("implementation_holes", output_tokens=output_tokens):
                raw = router.generate_text(
                    "planner",
                    _multi_page_messages(bundle_pages),
                    response_format="text",
                    enable_tools=False,
                )
            parsed = parse_multi_page_hole_text(str(raw or ""), pages_holes)
        except Exception as exc:
            skip_reason = f"model refinement ignored: {type(exc).__name__}: {exc}"

    result: dict[str, list[dict[str, Any]]] = {}
    for page in bundle_pages:
        page_id = str(page["page_id"])
        by_id = {
            str(fill.get("hole_id") or ""): fill
            for fill in parsed.get(page_id, ())
            if isinstance(fill, Mapping)
            and str(fill.get("hole_id") or "") in defaults[page_id]
        }
        page_result = [
            _merge_refinement(defaults[page_id][str(hole["hole_id"])], by_id.get(str(hole["hole_id"])))
            for hole in pages_holes[page_id]
        ]
        result[page_id] = page_result
        module_id = str(page.get("module_id") or "")
        trace = traces_by_module.get(module_id)
        if trace is not None:
            trace.record_attempt(
                raw_output=str(raw),
                validation_error=skip_reason or None,
                accepted={"hole_fills": page_result},
                context={
                    "attempt": 1,
                    "page_id": page_id,
                    "host_default_count": len(page_result),
                    "model_refined_count": sum(
                        1
                        for fill in page_result
                        if fill.get("fill_source") == "model_refined_host_template"
                    ),
                },
            )
    return result


def _pack_pages(pages: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Pack only optional refinement calls; oversize pages simply receive host fills."""
    bundles: list[list[Mapping[str, Any]]] = []
    for page in pages:
        if _bundle_prompt_size([page]) > _MAX_BUNDLE_CHARS:
            bundles.append([page])
            continue
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
            bundles.append([page])
    return bundles


def fill_evidence_pages(
    router: Any,
    skeletons: Sequence[Mapping[str, Any]],
    *,
    valid_module_catalog: set[str],
) -> list[dict[str, Any]]:
    """Complete every host template first, then optionally refine model-visible holes."""
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

    all_fills_by_page: dict[str, list[dict[str, Any]]] = {}
    if len(all_pages) == 1:
        single_page = all_pages[0]
        module_id = str(single_page["module_id"])
        page_module = dict(single_page["module"])
        template = dict(page_module["implementation_template"])
        template["holes"] = list(single_page["holes"])
        page_module["implementation_template"] = template
        all_fills_by_page[str(single_page["page_id"])] = _fill_page(
            router, page_module, traces_by_module[module_id]
        )
    else:
        for bundle in _pack_pages(all_pages):
            all_fills_by_page.update(_fill_bundle(router, bundle, traces_by_module))

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
            known_hole_ids = {str(hole.get("hole_id") or "") for hole in holes}
            required = set(_strings(template["completion_policy"].get("required_hole_ids")))
            unknown_required = sorted(required - known_hole_ids)
            if unknown_required:
                raise PlanningHoleFillError(
                    f"host template invariant is invalid for {module_id}; unknown required holes: "
                    + ", ".join(unknown_required)
                )

            module_fills = [
                fill
                for page_id in page_ids_by_module.get(module_id, ())
                for fill in all_fills_by_page.get(page_id, ())
            ]
            filled_ids = {str(fill.get("hole_id") or "") for fill in module_fills}
            missing = sorted(known_hole_ids - filled_ids)
            if missing:
                raise PlanningHoleFillError(
                    f"host template completion invariant failed for {module_id}: "
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
