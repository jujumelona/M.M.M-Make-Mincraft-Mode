from __future__ import annotations

"""Keep pre-design research distinct from post-design evidence obligations.

The authored requirement graph is useful after game design is frozen, when concrete design
leaves can be checked for reuse, dependencies, licensing and validation. Expanding every
authored requirement into those implementation obligations before design multiplies model
calls without improving the design brief. This contract keeps the pre-design phase bounded
to design-critical Minecraft/Fabric and local-project evidence, while preserving the full
obligation DAG for the downstream evidence-first production planner.

The research-note JSON schema is also aligned with the host parser. The parser already owns
normalization of compact model variants, so the transport schema must not reject outputs the
host can deterministically canonicalize.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

_INSTALLED = False
_PRE_DESIGN_DOMAIN_ID = "pre_design_request"


def _pre_design_candidate(prompt: str) -> dict[str, Any]:
    return {
        "summary": (
            "Design-critical pre-design research only. Reusable donor selection, dependency "
            "closure and license validation are deferred until the detailed design is frozen."
        ),
        "domains": [
            {
                "domain_id": _PRE_DESIGN_DOMAIN_ID,
                "objective": (
                    "Resolve Minecraft/Fabric mechanics, platform constraints and existing "
                    "local-project capabilities needed to design the authored request."
                ),
                "requirements": [prompt],
                "evidence_kinds": [
                    "minecraft_api",
                    "compatibility",
                    "runtime_behavior",
                    "local_project",
                    "testing",
                ],
                "queries": [
                    prompt,
                    (
                        "Minecraft Fabric API registration items entities dimensions world "
                        "interaction networking persistence data components GameTest"
                    ),
                ],
                "providers": ["official_docs", "project_rag"],
                "depends_on": [],
            }
        ],
        "unresolved_questions": [],
    }


def _is_internal_pre_design_seed(game_design: Any) -> bool:
    return (
        isinstance(game_design, Mapping)
        and set(game_design) == {"title"}
        and game_design.get("title") == "pre-design research"
    )


def _is_pre_design_brief(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    domains = value.get("domains")
    return (
        isinstance(domains, list)
        and len(domains) == 1
        and isinstance(domains[0], Mapping)
        and domains[0].get("domain_id") == _PRE_DESIGN_DOMAIN_ID
    )


def _relax_transport_schema(agentic_module: Any) -> None:
    """Make transport validation no stricter than the canonical host parser."""

    schema = agentic_module._RESEARCH_NOTE_SCHEMA
    if not isinstance(schema, dict):
        return
    schema.pop("required", None)
    schema["additionalProperties"] = True
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    note = properties.get("research_note")
    if not isinstance(note, dict):
        return
    note.pop("required", None)
    note["additionalProperties"] = True
    note_properties = note.get("properties")
    if not isinstance(note_properties, dict):
        return

    domain_id = note_properties.get("domain_id")
    if isinstance(domain_id, dict):
        domain_id.pop("minLength", None)

    claims = note_properties.get("claims")
    if isinstance(claims, dict):
        claims.pop("maxItems", None)
        # _parse_research_note accepts either claim objects or compact strings and
        # canonicalizes both. An unconstrained item schema lets that host logic run.
        claims["items"] = {}

    for field in ("gaps", "next_queries"):
        value = note_properties.get(field)
        if not isinstance(value, dict):
            continue
        value.pop("maxItems", None)
        item = value.get("items")
        if isinstance(item, dict):
            item.pop("minLength", None)
            item.pop("maxLength", None)


def install(agentic_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    _relax_transport_schema(agentic_module)

    current_normalize = agentic_module.normalize_research_brief
    if not getattr(current_normalize, "_mmm_pre_design_phase_boundary_v1", False):

        @wraps(current_normalize)
        def normalize(
            prompt: str,
            game_design: dict[str, Any],
            candidate: Any | None = None,
        ) -> dict[str, Any]:
            if candidate is None and _is_internal_pre_design_seed(game_design):
                candidate = _pre_design_candidate(prompt)
            return current_normalize(prompt, game_design, candidate)

        normalize._mmm_pre_design_phase_boundary_v1 = True
        agentic_module.normalize_research_brief = normalize

    current_ecosystem = agentic_module.collect_ecosystem_seed_bundle
    if not getattr(current_ecosystem, "_mmm_post_design_donor_boundary_v1", False):

        @wraps(current_ecosystem)
        def ecosystem(*args: Any, **kwargs: Any) -> Any:
            research_brief = kwargs.get("research_brief")
            if _is_pre_design_brief(research_brief):
                return {
                    "schema_version": "mmm/deferred-ecosystem-discovery-v1",
                    "status": "deferred_until_design_freeze",
                    "candidate_count": 0,
                    "reason": (
                        "Donor/reuse discovery belongs to the post-design leaf evidence phase."
                    ),
                }
            return current_ecosystem(*args, **kwargs)

        ecosystem._mmm_post_design_donor_boundary_v1 = True
        agentic_module.collect_ecosystem_seed_bundle = ecosystem

    _INSTALLED = True


__all__ = ["install"]
