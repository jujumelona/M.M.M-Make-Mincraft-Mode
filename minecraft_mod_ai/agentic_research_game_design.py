from __future__ import annotations

"""Deterministic host-owned game-design projection.

The authoritative requirement catalog already owns request semantics, capability IDs,
observable behavior, acceptance, dependencies, and implementation obligations.  Game
design is therefore a projection of that frozen catalog, not another model-generation
stage.  This keeps small-model meta output, retries, and per-field serial calls off the
critical path.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .design_requirement_contract import (
    _active_requirement_ledger,
    _validate_requirement_coverage,
)


def supports_agentic_research_router(router: Any) -> bool:
    """Use the host-owned design path for the normal ModelRouter runtime."""
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


def generate_sectioned_game_design(
    game_design_module: Any,
    router: Any,
    prompt: str,
    *,
    media_paths: Sequence[str | Path] = (),
    research: Mapping[str, Any],
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the frozen requirement ledger into the complete design schema.

    No model call occurs here.  Research stays available to downstream implementation
    planning but does not get a second chance to rewrite authored game design.
    """
    del router, media_paths, research, trace_metadata
    ledger = _active_requirement_ledger(prompt)
    if not ledger:
        raise ValueError("Host-owned game design requires an active requirement ledger.")

    statements = [_statement(requirement) for requirement in ledger]
    authored = [
        _text(requirement.get("authored_text"))
        for requirement in ledger
        if _text(requirement.get("authored_text"))
    ]
    acceptance_tests = _acceptance_tests(ledger)
    design: dict[str, Any] = {
        "title": "Requested Minecraft Mod",
        "pitch": "Implement the authored Minecraft behaviors without expanding their scope.",
        "core_loop": list(dict.fromkeys(statements)),
        "progression": list(dict.fromkeys(statements)),
        "combat": _combat_context(ledger),
        "mod_context": {"authored_scope": list(dict.fromkeys(authored))} if authored else {},
        "modules": [_module(requirement) for requirement in ledger],
        "assets": [],
        "acceptance_tests": acceptance_tests,
    }
    game_design_module._validate_design(design)
    return _validate_requirement_coverage(design, ledger)


__all__ = ["generate_sectioned_game_design", "supports_agentic_research_router"]
