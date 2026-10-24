"""Preserve request scope by authoring one bounded worksheet DAG per feature."""
from collections.abc import Mapping
from copy import deepcopy
import json

from . import authored_structured_design as design
from .planning_detail_slots import DETAIL_RECORDS
from .state_definition_merge import merge_state_definitions
from .request_record_ownership import bind_request_records


def _unique_rows(rows):
    result = {}
    for row in rows:
        identity = json.dumps(row, ensure_ascii=False, sort_keys=True)
        result.setdefault(identity, deepcopy(row))
    return list(result.values())


def _merge_section(section, prior, addition):
    specs = [row.get("specification", {}) for row in (prior, addition)]
    specification = {}
    for concern in DETAIL_RECORDS[section]:
        before, after = (spec.get(concern, []) for spec in specs)
        specification[concern] = (
            merge_state_definitions(before, after)
            if (section, concern) == ("state_model", "variables")
            else _unique_rows([*before, *after])
        )
    reasons = {
        row["concern"]: deepcopy(row)
        for spec in specs for row in spec.get("inapplicable_concerns", ())
        if not specification[row["concern"]]
    }
    specification["inapplicable_concerns"] = list(reasons.values())
    return {
        "specification": specification,
        "constraint_evidence_refs": list(dict.fromkeys(
            ref for row in (prior, addition)
            for ref in row.get("constraint_evidence_refs", ())
        )),
    }


def merge_request_sections(prior, addition):
    """Accumulate validated records, preserving shared state and source effects."""
    result = design.normalize_structured_sections(prior)
    for section, row in design.normalize_structured_sections(addition).items():
        result[section] = _merge_section(section, result.get(section, {}), row)
    return design.normalize_structured_sections(result)


def _feature_prompt(original, feature, shared_variables):
    return (
        "Original request (context; do not author neighboring features):\n" + original
        + "\n\nACTIVE REQUEST FEATURE:\n"
        + json.dumps({
            "requirement_id": feature["requirement_id"],
            "statement": feature["statement"],
            "source_text": feature.get("source_span", {}).get("text", ""),
        }, ensure_ascii=False, separators=(",", ":"))
        + "\nAuthor this feature's concrete activation, result and implementation only. "
        "Keep its state changes, UI interactions and verification in the same scope. "
        "Preserve every distinct action within this feature. Opening a screen alone "
        "does not implement the requested gameplay."
        + "\n\nPreviously accepted shared state declarations (read-only):\n"
        + json.dumps(shared_variables, ensure_ascii=False, separators=(",", ":"))
        + "\nThese symbols are available to state expressions and mutations. Declare "
        "only additional state, or repeat a shared declaration exactly. Do not change "
        "its type, default, owner, unit or domain to fit this feature."
    )


def author_request_structured_sections(
    router, prompt, request_scope, *, media_paths=(), budget=None,
    checkpoint=None,
):
    """The host iterates source requirements; the model cannot omit a neighbor."""
    completed = {}
    requirements = request_scope.get("requirements", ())
    if not requirements:
        return design.author_structured_sections(
            router, prompt, media_paths=media_paths, budget=budget,
        )
    original = str(request_scope.get("requested_prompt") or prompt)
    for feature in requirements:
        if not isinstance(feature, Mapping):
            raise ValueError("REQUEST_DESIGN_FEATURE_OBJECT_REQUIRED")
        variables = (
            completed.get("state_model", {}).get("specification", {})
            .get("variables", [])
        )
        sections = checkpoint.load_request_design(feature, variables) if checkpoint else None
        if sections is not None:
            try:
                sections = design.normalize_structured_sections(sections)
            except ValueError:
                sections = None
        if sections is None:
            focused = _feature_prompt(original, feature, variables)
            sections = design.author_structured_sections(
                router, focused, media_paths=media_paths, budget=budget,
                shared_state_variables=variables,
            )
        sections = bind_request_records(sections, feature["requirement_id"])
        completed = merge_request_sections(completed, sections)
        if checkpoint:
            checkpoint.save_request_design(feature, variables, sections)
        print("REQUEST_DESIGN: accepted " + str(feature["requirement_id"]), flush=True)
    return completed
