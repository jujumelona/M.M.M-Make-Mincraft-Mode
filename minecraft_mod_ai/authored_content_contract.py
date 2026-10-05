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


def content_request_catalog(
    structured_sections: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one coherent content requirement for the active resource/UI slice."""

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

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    statement = "\n".join(statements)
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
    "content_owned_refs",
    "content_request_catalog",
]
