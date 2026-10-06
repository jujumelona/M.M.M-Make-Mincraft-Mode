from __future__ import annotations

"""Shared ownership contract for authored resource/UI content lowering.

Structured resource/UI concerns are engineering context, not independent root
content requirements. The content graph receives one coherent requirement bundle
when at least one executable content-driving concern is active. Path rows are
constraints on deterministic artifact placement and never force the model to invent
an otherwise nonexistent Minecraft content entity.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .authored_structured_design import active_concern_records


CONTENT_GRAPH_DRIVER_CONCERNS = (
    "registries",
    "data_resources",
    "assets",
    "interactions",
    "displayed_state",
)

CONTENT_GRAPH_CONTEXT_CONCERNS = (
    *CONTENT_GRAPH_DRIVER_CONCERNS,
    "paths",
)

CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS = frozenset({"paths"})

# Cross-stage ownership for resources_and_ui.  These sets are the authority used by
# planning coverage, Typed PlatformIR and production lowering.  Do not duplicate
# concern-name lists in downstream stages.
RESOURCE_POLICY_CONCERNS = frozenset({
    "missing_resources",
    "accessibility",
})

RESOURCES_AND_UI_OWNED_CONCERNS = frozenset({
    *CONTENT_GRAPH_CONTEXT_CONCERNS,
    *RESOURCE_POLICY_CONCERNS,
})


def content_request_catalog(
    structured_sections: Mapping[str, Any],
    *,
    requested_prompt: str = "",
) -> dict[str, Any]:
    """Bind resource/UI constraints to the gameplay requirement they implement."""

    records = active_concern_records(
        structured_sections,
        "resources_and_ui",
    )
    if not any(records.get(concern) for concern in CONTENT_GRAPH_DRIVER_CONCERNS):
        return {"requirements": []}

    payload: list[dict[str, Any]] = []
    statements: list[str] = []
    for concern in CONTENT_GRAPH_CONTEXT_CONCERNS:
        for index, record in enumerate(records.get(concern, ()), start=1):
            normalized = dict(record)
            payload.append({
                "concern": concern,
                "record": normalized,
            })
            fields = "; ".join(
                f"{key}={value}"
                for key, value in normalized.items()
                if str(value).strip()
            )
            statements.append(
                f"resources_and_ui.{concern}[{index}]: {fields}"
            )

    from .planning_section_dependencies import SECTION_DEPENDENCIES

    # Resource/UI rows describe engineering constraints. They are not a substitute
    # for the gameplay that required those resources (a code registry alone, for
    # example, says nothing about the concrete items or screens it serves).
    # Preserve the same authored prerequisites used to write this worksheet slice.
    prerequisites = {
        section: active
        for section in SECTION_DEPENDENCIES["resources_and_ui"]
        if (active := active_concern_records(structured_sections, section))
    }
    source = {
        "requested_prompt": requested_prompt,
        "authored_prerequisites": prerequisites,
        "resource_constraints": payload,
    }
    encoded = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    statement_parts = []
    if requested_prompt.strip():
        statement_parts.append("Original game request:\n" + requested_prompt)
    if prerequisites:
        statement_parts.append(
            "Canonical gameplay and integration requirements:\n"
            + json.dumps(prerequisites, ensure_ascii=False, sort_keys=True)
        )
    statement_parts.append("Resource/UI constraints:\n" + "\n".join(statements))
    statement = "\n\n".join(statement_parts)
    requirement_id = f"content_resources_and_ui_{digest}"
    return {
        "requirements": [{
            "requirement_id": requirement_id,
            "statement": statement,
            "source_span": {"text": statement},
        }]
    }


def content_owned_refs(
    structured_sections: Mapping[str, Any],
    content_design: Mapping[str, Any] | None,
) -> frozenset[str]:
    """Return resource/UI refs owned outside Typed PlanIR platform modules.

    Path binding is always host-owned: artifact implementations choose concrete
    output paths deterministically. Other content concerns are externally covered
    only after the persisted content graph contains implementation facts.
    """

    records = active_concern_records(
        structured_sections,
        "resources_and_ui",
    )
    owned = {
        f"resources_and_ui.{concern}"
        for concern in CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS
        if records.get(concern)
    }

    design = content_design if isinstance(content_design, Mapping) else {}
    if design.get("_implementation_facts"):
        owned.update(
            f"resources_and_ui.{concern}"
            for concern in CONTENT_GRAPH_CONTEXT_CONCERNS
            if records.get(concern)
        )
    return frozenset(owned)


__all__ = [
    "CONTENT_GRAPH_CONTEXT_CONCERNS",
    "CONTENT_GRAPH_DRIVER_CONCERNS",
    "CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS",
    "RESOURCE_POLICY_CONCERNS",
    "RESOURCES_AND_UI_OWNED_CONCERNS",
    "content_owned_refs",
    "content_request_catalog",
]
