from __future__ import annotations

"""Make design -> retrieval/task ownership traceable to authored requirements.

``planner_graph_integrity_contract`` expands game-design details into retrieval facets.
Historically, a facet with no lexical evidence was assigned to a requirement by narrative
position.  That preserved determinism but silently invented ownership, which can make the
design summary and downstream task graph disagree.

This late contract keeps explicit module requirement refs authoritative.  For other design
facets it uses only positive lexical evidence; when no such evidence exists it preserves the
facet conservatively under every authored requirement instead of pretending one arbitrary
requirement owns it.  The shared capability node is still unique, while planned-work
ownership and graph parent edges remain complete and auditable.
"""

import re
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .spec import SpecValidationError

_INSTALLED = False
_MARKER = "__mmm_requirement_traceable_design_facets_v1__"
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(str(value or "")) if len(item) >= 2}


def _work_text(work: Mapping[str, Any]) -> str:
    return " ".join(
        (
            str(work.get("objective") or ""),
            " ".join(str(item) for item in work.get("capabilities", ()) if str(item)),
            " ".join(str(item) for item in work.get("acceptance", ()) if str(item)),
        )
    )


def _explicit_module_refs(
    design: Mapping[str, Any],
    known_requirement_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    modules = design.get("modules")
    if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes, bytearray)):
        return result
    for index, raw in enumerate(modules):
        if not isinstance(raw, Mapping):
            continue
        raw_refs = raw.get("requirement_refs")
        if not isinstance(raw_refs, Sequence) or isinstance(
            raw_refs, (str, bytes, bytearray)
        ):
            continue
        refs = tuple(dict.fromkeys(str(item).strip() for item in raw_refs if str(item).strip()))
        if not refs:
            continue
        unknown = sorted(set(refs) - known_requirement_ids)
        if unknown:
            raise SpecValidationError(
                "design retrieval traceability cites unknown requirement ids: "
                + ", ".join(unknown)
            )
        result[f"game_design.modules[{index}]"] = refs
    return result


def _desired_requirement_refs(
    facet: Mapping[str, Any],
    work: Sequence[Mapping[str, Any]],
    explicit_by_source: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], str]:
    source = str(facet.get("source") or "")
    explicit = explicit_by_source.get(source)
    if explicit:
        return explicit, "explicit_module_requirement_refs"

    facet_tokens = _tokens(f"{facet.get('label', '')} {facet.get('detail', '')}")
    scores: list[tuple[str, int]] = []
    for item in work:
        requirement_ref = str(item.get("requirement_ref") or "").strip()
        if not requirement_ref:
            continue
        overlap = len(facet_tokens & _tokens(_work_text(item)))
        scores.append((requirement_ref, overlap))
    if not scores:
        raise SpecValidationError("design retrieval traceability has no authored requirement work")

    best = max(score for _requirement_ref, score in scores)
    if best > 0:
        refs = tuple(requirement_ref for requirement_ref, score in scores if score == best)
        return refs, "positive_lexical_evidence"

    # No semantic ownership evidence is available at this boundary.  Binding to all
    # authored requirements is deliberately conservative: it preserves the design detail
    # without fabricating a single owner and lets later verified evidence narrow reuse.
    return tuple(requirement_ref for requirement_ref, _score in scores), "conservative_all_requirements"


def _first_parent_capability(work: Mapping[str, Any], facet_capability: str) -> str:
    capabilities = [
        str(item)
        for item in work.get("capabilities", ())
        if str(item) and str(item) != facet_capability
    ]
    return next(
        (item for item in capabilities if not item.startswith("design.")),
        capabilities[0] if capabilities else "",
    )


def enforce_requirement_design_traceability(
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind design facets without arbitrary positional ownership."""

    raw_facets = plan.get("design_retrieval_facets")
    raw_work = plan.get("planned_work")
    raw_graph = plan.get("capability_graph")
    if not isinstance(raw_facets, list) or not raw_facets:
        return dict(plan)
    if not isinstance(raw_work, list) or not isinstance(raw_graph, Mapping):
        raise SpecValidationError("design retrieval facets exist without a valid work graph")

    work = [dict(item) for item in raw_work if isinstance(item, Mapping)]
    if len(work) != len(raw_work) or not work:
        raise SpecValidationError("design retrieval traceability requires object planned_work rows")
    work_by_ref: dict[str, dict[str, Any]] = {}
    for item in work:
        requirement_ref = str(item.get("requirement_ref") or "").strip()
        if not requirement_ref or requirement_ref in work_by_ref:
            raise SpecValidationError("planned work has missing or duplicate requirement refs")
        work_by_ref[requirement_ref] = item

    explicit_by_source = _explicit_module_refs(design, set(work_by_ref))
    graph = dict(raw_graph)
    edges = [dict(item) for item in graph.get("edges", ()) if isinstance(item, Mapping)]
    rebuilt_bindings: list[dict[str, Any]] = []

    for raw_facet in raw_facets:
        if not isinstance(raw_facet, Mapping):
            raise SpecValidationError("design_retrieval_facets must contain objects")
        facet = dict(raw_facet)
        capability = str(facet.get("capability") or "").strip()
        if not capability:
            raise SpecValidationError("design retrieval facet has no capability id")
        desired_refs, basis = _desired_requirement_refs(facet, work, explicit_by_source)
        unknown = sorted(set(desired_refs) - set(work_by_ref))
        if unknown:
            raise SpecValidationError(
                "design retrieval facet resolved to unknown requirements: " + ", ".join(unknown)
            )

        # Remove the old single-owner guess before applying evidence-backed ownership.
        for item in work:
            raw_capabilities = item.get("capabilities", ())
            capabilities = [
                str(value)
                for value in raw_capabilities
                if str(value) and str(value) != capability
            ]
            item["capabilities"] = list(dict.fromkeys(capabilities))

        # Parent edges into this design leaf must agree with its requirement owners too.
        edges = [edge for edge in edges if str(edge.get("to") or "") != capability]
        for requirement_ref in desired_refs:
            owner = work_by_ref[requirement_ref]
            capabilities = [str(value) for value in owner.get("capabilities", ()) if str(value)]
            if capability not in capabilities:
                capabilities.append(capability)
            owner["capabilities"] = capabilities
            parent = _first_parent_capability(owner, capability)
            if parent and parent != capability:
                edge = {"from": parent, "to": capability}
                if edge not in edges:
                    edges.append(edge)
            row = dict(facet)
            row["work_id"] = str(owner.get("work_id") or "")
            row["requirement_ref"] = requirement_ref
            row["binding_basis"] = basis
            rebuilt_bindings.append(row)

    graph["edges"] = edges
    result = dict(plan)
    result["planned_work"] = work
    result["capability_graph"] = graph
    result["design_retrieval_facets"] = rebuilt_bindings
    return result


def install() -> None:
    """Install after planner_graph_integrity_contract so this is the final binding owner."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import reuse_planner as reuse

    original = reuse.compile_pre_retrieval_plan
    if getattr(original, _MARKER, False):
        _INSTALLED = True
        return

    @wraps(original)
    def guarded(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
        plan = original(prompt, design)
        traced = enforce_requirement_design_traceability(plan, design)
        if traced == plan:
            return plan
        traced["plan_sha256"] = ""
        traced["plan_sha256"] = reuse._plan_hash(traced)
        reuse.validate_pre_retrieval_plan(traced, prompt=prompt, design=design)
        return traced

    setattr(guarded, _MARKER, True)
    reuse.compile_pre_retrieval_plan = guarded
    _INSTALLED = True


__all__ = ["enforce_requirement_design_traceability", "install"]
