from __future__ import annotations

"""Deterministic host-owned game-design compiler.

The authoritative requirement catalog owns request semantics, capability IDs,
observable behavior, acceptance, dependencies, and implementation obligations. Game
design is only a validated projection of that frozen catalog. No language-model call,
model-generated JSON, retry loop, or model-owned identifier is permitted here.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .design_requirement_contract import (
    _active_requirement_ledger,
    _validate_requirement_coverage,
)
from .model_meta_output_contract import assert_design_field_clean
from .planner import HeuristicPlanner
from .spec import SpecValidationError

_GAME_DESIGN_FIELDS = (
    "title",
    "pitch",
    "core_loop",
    "progression",
    "combat",
    "mod_context",
    "modules",
    "assets",
    "acceptance_tests",
)
_OPTIONAL_GAME_DESIGN_FIELDS = ("art_direction",)


def supports_agentic_research_router(router: Any) -> bool:
    """Return whether the normal runtime router can use this host compiler."""
    from .model_router import ModelRouter

    return isinstance(router, ModelRouter)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _statement(requirement: Mapping[str, Any]) -> str:
    return (
        _text(requirement.get("semantic_statement"))
        or _text(requirement.get("authored_text"))
        or _text(requirement.get("capability"))
        or "Authored Minecraft behavior"
    )


def _observable_obligation(requirement: Mapping[str, Any]) -> str:
    behavior = requirement.get("observable_behavior")
    if not isinstance(behavior, Mapping):
        return _statement(requirement)
    given = _text(behavior.get("given"))
    when = _text(behavior.get("when"))
    then = _text(behavior.get("then"))
    parts = [part for part in (given, when, then) if part]
    return "; ".join(parts) if parts else _statement(requirement)


def _module(requirement: Mapping[str, Any]) -> dict[str, Any]:
    requirement_id = _text(requirement.get("requirement_id"))
    acceptance = requirement.get("acceptance")
    obligations = [_statement(requirement), _observable_obligation(requirement)]
    if isinstance(acceptance, Sequence) and not isinstance(
        acceptance, (str, bytes, bytearray)
    ):
        obligations.extend(_text(item) for item in acceptance if _text(item))
    obligations = list(dict.fromkeys(item for item in obligations if item))
    return {
        "plugin_id": f"design_{requirement_id}",
        "status": "custom_required",
        "capability": _text(requirement.get("capability")),
        "reason": _statement(requirement),
        "requirement_refs": [requirement_id],
        "implementation_obligations": obligations,
    }


def _acceptance_tests(ledger: Sequence[Mapping[str, Any]]) -> list[str]:
    tests: list[str] = []
    for requirement in ledger:
        raw = requirement.get("acceptance")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            tests.extend(_text(item) for item in raw if _text(item))
        if not raw:
            tests.append(_observable_obligation(requirement))
    return list(dict.fromkeys(item for item in tests if item))


def _combat_context(ledger: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    values = [
        _statement(requirement)
        for requirement in ledger
        if any(
            token in _text(requirement.get("capability")).casefold()
            for token in ("combat", "weapon", "alien", "damage", "boss")
        )
    ]
    return {"authored_combat": values} if values else {}


def _validate_design(design: Mapping[str, Any]) -> None:
    for field in ("title", "pitch"):
        value = design.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SpecValidationError(f"game_design.{field} must be a non-empty string")
    for field in ("core_loop", "progression", "acceptance_tests", "modules", "assets"):
        if not isinstance(design.get(field), list):
            raise SpecValidationError(f"game_design.{field} must be a list")
    for field in ("combat", "mod_context"):
        if not isinstance(design.get(field), dict):
            raise SpecValidationError(f"game_design.{field} must be an object")
    for field in (*_GAME_DESIGN_FIELDS, *_OPTIONAL_GAME_DESIGN_FIELDS):
        if field not in design:
            continue
        try:
            assert_design_field_clean(field, design[field])
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc


def canonical_game_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private/non-schema fields and validate the host-owned design payload."""
    result = {field: design[field] for field in _GAME_DESIGN_FIELDS if field in design}
    for field in _OPTIONAL_GAME_DESIGN_FIELDS:
        if field in design:
            result[field] = design[field]
    _validate_design(result)
    return result


def validate_ready_design(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed before retrieval when the host projection is incomplete."""
    result = canonical_game_design(design)
    for field in ("core_loop", "progression", "acceptance_tests"):
        value = result.get(field)
        if not isinstance(value, list) or not any(_text(item) for item in value):
            raise SpecValidationError(f"design readiness failed: {field} is empty")
    modules = result.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SpecValidationError("design readiness failed: modules are empty")

    ledger = _active_requirement_ledger(prompt)
    if ledger:
        result = _validate_requirement_coverage(result, ledger)
    return result


def deterministic_bootstrap(prompt: str, design: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Proposal bootstrap deterministically; no model planning is involved."""
    proposal = HeuristicPlanner().plan(prompt)
    spec = proposal.spec
    title = _text(design.get("title")) or spec.mod_name
    pitch = _text(design.get("pitch")) or spec.summary
    normalized = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in title.lower()
    )
    stem = "_".join(part for part in normalized.split("_") if part)
    if not stem:
        stem = f"mmm_{hashlib.sha256(title.encode('utf-8')).hexdigest()[:10]}"
    if not stem[0].isalpha():
        stem = f"mmm_{stem}"
    mod_id = f"{stem[:55].rstrip('_')}_mod"
    return {
        "mod_id": mod_id,
        "mod_name": title,
        "package_name": f"ai.minecraft.generated.{mod_id}",
        "summary": pitch,
        "contents": [
            {
                "content_id": content.content_id,
                "kind": content.kind.value,
                "display_name_en": content.display_name_en,
                "display_name_ko": content.display_name_ko,
                "color": content.color,
                "recipe": content.recipe,
            }
            for content in spec.contents
        ],
        "deferred_capabilities": [
            deferred.capability for deferred in proposal.deferred_requests
        ],
    }


def generate_sectioned_game_design(
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
    research: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the frozen requirement ledger into the complete design schema.

    The arguments carrying runtime/research context are accepted only because this
    compiler sits at that pipeline boundary. They cannot alter authored semantics.
    """
    del router, media_paths, research, trace_metadata
    ledger = _active_requirement_ledger(prompt)
    if not ledger:
        raise SpecValidationError("Host-owned game design requires an active requirement ledger.")

    statements = [_statement(requirement) for requirement in ledger]
    authored = [
        _text(requirement.get("authored_text"))
        for requirement in ledger
        if _text(requirement.get("authored_text"))
    ]
    design: dict[str, Any] = {
        "title": "Requested Minecraft Mod",
        "pitch": "Implement the authored Minecraft behaviors without expanding their scope.",
        "core_loop": list(dict.fromkeys(statements)),
        "progression": list(dict.fromkeys(statements)),
        "combat": _combat_context(ledger),
        "mod_context": {"authored_scope": list(dict.fromkeys(authored))} if authored else {},
        "modules": [_module(requirement) for requirement in ledger],
        "assets": [],
        "acceptance_tests": _acceptance_tests(ledger),
    }
    return validate_ready_design(prompt, design)


__all__ = [
    "canonical_game_design",
    "deterministic_bootstrap",
    "generate_sectioned_game_design",
    "supports_agentic_research_router",
    "validate_ready_design",
]
