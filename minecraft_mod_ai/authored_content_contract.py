from __future__ import annotations

"""Shared ownership contract for authored resource/UI content lowering.

Each active resource/UI concern remains an owned coverage obligation with its original
gameplay context. Concrete-content concerns may require an entity binding, while
engineering-only registry/data-resource concerns may contribute zero content identities
and remain host/content-graph constraints rather than forcing a synthetic Typed PlatformIR
module. Path rows likewise constrain deterministic artifact placement and never force the
model to invent an otherwise nonexistent Minecraft content entity.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .authored_structured_design import active_concern_records
from .content_design_contract import CONTENT_CONCERN_KINDS, CONTENT_KIND_TO_FACT_TYPE


CONTENT_GRAPH_DRIVER_CONCERNS = (
    # Discover concrete owners before registry binding. Registry rows describe
    # engineering identifiers/bindings and are the weakest source of content
    # identity; running them first forced the small model to invent pseudo-content
    # from catalog/registry names instead of reusing already discovered owners.
    "data_resources",
    "assets",
    "interactions",
    "displayed_state",
    "registries",
)

CONTENT_GRAPH_CONTEXT_CONCERNS = (
    *CONTENT_GRAPH_DRIVER_CONCERNS,
    "paths",
)

CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS = frozenset({
    "paths",
    "registries",
    "data_resources",
})

# Registry/data-resource rows are engineering constraints. They may describe a
# binding or data file without introducing a new Minecraft content identity of
# their own. Keep those coverage units active, but do not force the small model
# to invent a pseudo item/recipe/tag merely to satisfy cardinality.
CONTENT_CONCERN_MINIMUM_ENTITY_COUNT = {
    "registries": 0,
    "data_resources": 0,
    "assets": 1,
    "interactions": 1,
    "displayed_state": 1,
}


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
    for concern in CONTENT_GRAPH_CONTEXT_CONCERNS:
        for record in records.get(concern, ()):
            normalized = dict(record)
            payload.append({
                "concern": concern,
                "record": normalized,
            })

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
    requirements = []
    for concern in CONTENT_GRAPH_DRIVER_CONCERNS:
        rows = records.get(concern)
        if not rows:
            continue
        coverage_ref = f"resources_and_ui.{concern}"
        digest = hashlib.sha256(
            (encoded + "\n" + coverage_ref).encode("utf-8")
        ).hexdigest()[:16]
        requirement_id = f"content_{concern}_{digest}"
        focused_statement = (
            f"Implement the declared obligation {coverage_ref}:\n"
            + json.dumps(rows, ensure_ascii=False, sort_keys=True)
        )
        requirements.append({
            "requirement_id": requirement_id,
            "statement": focused_statement,
            "source_span": {"text": focused_statement},
            "design_context": source,
            "coverage_ref": coverage_ref,
            "source_records": rows,
            "allowed_content_kinds": list(CONTENT_CONCERN_KINDS[concern]),
            "minimum_entity_count": CONTENT_CONCERN_MINIMUM_ENTITY_COUNT[concern],
        })
    return {"requirements": requirements}


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
    if "_content_requirements" in design:
        requirements = design["_content_requirements"]
        if not isinstance(requirements, list):
            raise ValueError("CONTENT_REQUIREMENT_BINDINGS_INVALID")
        modules = {
            row.get("module_id")
            for row in design.get("modules", ())
            if isinstance(row, Mapping)
        }
        facts = {
            (row.get("subject"), row.get("fact_type"))
            for row in design.get("_implementation_facts", ())
            if isinstance(row, Mapping)
        }
        seen = set()
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                raise ValueError("CONTENT_REQUIREMENT_BINDINGS_INVALID")
            ref = requirement.get("coverage_ref")
            if not isinstance(ref, str) or ref in seen:
                raise ValueError("CONTENT_REQUIREMENT_BINDINGS_INVALID")
            seen.add(ref)
            section, _, concern = ref.partition(".")
            if (
                section != "resources_and_ui"
                or concern not in CONTENT_CONCERN_KINDS
                or requirement.get("source_records") != records.get(concern)
            ):
                raise ValueError(f"CONTENT_REQUIREMENT_SOURCE_MISMATCH: {ref}")
            rid = requirement.get("requirement_id")
            for entity in design.get("_content_entities", ()):
                if not isinstance(entity, Mapping):
                    continue
                kind = entity.get("kind")
                entity_id = entity.get("entity_id")
                if (
                    rid in entity.get("requirement_refs", ())
                    and kind in CONTENT_CONCERN_KINDS[concern]
                    and entity_id in modules
                    and (entity_id, CONTENT_KIND_TO_FACT_TYPE[kind].value) in facts
                ):
                    owned.add(ref)
                    break
        return frozenset(owned)
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
    "CONTENT_CONCERN_MINIMUM_ENTITY_COUNT",
    "RESOURCE_POLICY_CONCERNS",
    "RESOURCES_AND_UI_OWNED_CONCERNS",
    "content_owned_refs",
    "content_request_catalog",
]
