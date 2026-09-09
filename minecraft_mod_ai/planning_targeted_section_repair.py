from __future__ import annotations

"""Target one missing detailed-planning section without weakening plan invariants."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .fixed_template_generation import generate_fixed_template_value
from .planning_criterion_fragments import (
    criterion_fragment_messages,
    criterion_fragment_schema,
    validate_criterion_fragment,
)
from .planning_detail_template import normalize_required_sections


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def generate_targeted_section_fragment(
    router: Any,
    *,
    requirement: Mapping[str, Any],
    criterion: str,
    selected_sections: Iterable[str],
    target_section: str,
    evidence: list[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    """Fill exactly one missing section while validating the full host plan selection.

    ``selected_sections`` remains the canonical whole-plan selection and therefore keeps
    the mandatory-core-section invariant intact. ``target_section`` only narrows the model
    transport: the copied tool schema exposes a one-value enum and maxItems=1, so a small
    model cannot satisfy the repair call by authoring some other valid worksheet section.
    """

    selected = normalize_required_sections(selected_sections)
    target = _text(target_section)
    if target not in selected:
        raise ValueError(
            "DETAILED_PLAN_TARGET_SECTION: repair target is not part of the host-selected plan: "
            + target
        )

    schema = deepcopy(criterion_fragment_schema(selected))
    updates = schema["properties"]["section_updates"]
    updates["minItems"] = 1
    updates["maxItems"] = 1
    updates["items"]["properties"]["section"]["enum"] = [target]

    messages = criterion_fragment_messages(
        requirement,
        criterion,
        selected,
        evidence,
        repair_no_progress=True,
    )
    messages.append(
        {
            "role": "system",
            "content": (
                "This is a host-targeted missing-section repair. Fill exactly one section_updates row. "
                f"Its section is fixed to {target!r}; do not select, discuss, or return any other section."
            ),
        }
    )

    value = generate_fixed_template_value(
        router,
        "planner",
        messages,
        response_schema=schema,
        enable_tools=False,
        tool_name=f"submit_missing_{target}_fragment",
        description=(
            f"Fill exactly the missing {target} worksheet section. The section field is host-fixed."
        ),
    )
    validated = validate_criterion_fragment(
        value,
        selected_sections=selected,
        allowed_refs=allowed_refs,
    )
    targeted = [
        row
        for row in validated["section_updates"]
        if row["section"] == target
        and (_text(row.get("implementation")) or _text(row.get("constraint")))
    ]
    if not targeted:
        raise ValueError(
            "DETAILED_PLAN_TARGET_SECTION: fixed repair returned no concrete target-section content"
        )
    return {"section_updates": targeted}


__all__ = ["generate_targeted_section_fragment"]
