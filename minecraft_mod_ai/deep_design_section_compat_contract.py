from __future__ import annotations

"""Preserve the prose-first game-design section API through late runtime composition.

Game-design drafting now emits Markdown/text and is parsed by the host.  The older
``deep_design_execution_contract`` wrapper predates that boundary and still requires the
retired model-repair arguments ``prior_error`` and ``prior_candidate``.  Runtime wrapper
integrity correctly rejects that API drift.  This contract arms the deep-design installer
before late finalization and, immediately after that installer runs, replaces only the
stale section-message wrapper with a four-argument wrapper that keeps the production-depth
guidance.  No JSON response schema or model repair loop is reintroduced.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_DEPTH_GUIDANCE = (
    "\n\nPRODUCTION DEPTH: finish the game/mod design before implementation search. "
    "Decompose every requested mechanic into the smallest meaningful subsystems "
    "that can be independently implemented, tested, and searched for reuse. Split "
    "different player verbs, resources, state transitions, purchase/assembly steps, "
    "upgrade gates, travel phases, encounters, combat outcomes, world interactions, "
    "persistence-visible state, networking/client surfaces, and integration rules when "
    "they can fail independently. The modules array is the implementation-leaf index: "
    "every implementation-bearing core-loop/progression/combat/mod-context behavior "
    "must have a concrete modules entry with a stable snake_case plugin_id and a reason "
    "that states its owned behavior. Do not collapse an epic such as planet interaction, "
    "ship construction, trading, or progression into one generic module. Use as many "
    "leaf modules as the authored design genuinely needs; never add unrelated features. "
    "Use the supplied research evidence for Minecraft/Fabric facts and unresolved "
    "assumptions, but do not claim a third-party donor was selected here: donor/reuse "
    "selection happens only after this design is frozen."
)

_INSTALLED = False
_ARMED = False
_INSTALLER_MARKER = "_mmm_prose_first_deep_design_compat_v1"


def _repair_for(agentic: Any, deep: Any) -> bool:
    """Replace exactly the stale six-argument wrapper; leave later owners untouched."""

    current = agentic._section_messages
    stale = deep._deep_section_messages
    if current is not stale:
        return False

    base = getattr(stale, "__wrapped__", None)
    if not callable(base):
        raise RuntimeError("deep-design section wrapper lost its base callable")

    @wraps(base)
    def compatible_section_messages(
        *,
        prompt: str,
        section_id: str,
        fields: Sequence[str],
        research: Mapping[str, Any],
    ) -> list[dict[str, str]]:
        messages = base(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
        )
        if not messages:
            return messages
        output = [dict(message) for message in messages]
        output[0]["content"] = str(output[0].get("content") or "") + _DEPTH_GUIDANCE
        return output

    setattr(compatible_section_messages, deep._SECTION_MARKER, True)
    deep._deep_section_messages = compatible_section_messages
    agentic._section_messages = compatible_section_messages
    return True


def _repair() -> bool:
    from . import agentic_research_game_design as agentic
    from . import deep_design_execution_contract as deep

    return _repair_for(agentic, deep)


def install() -> None:
    """Apply the post-deep-installer repair once in the current process."""

    global _INSTALLED
    if _INSTALLED:
        return
    _repair()
    _INSTALLED = True


def arm() -> None:
    """Wrap the late deep-design installer so compatibility is repaired before checks."""

    global _ARMED
    if _ARMED:
        return

    from . import deep_design_execution_contract as deep

    original_install = deep.install
    if getattr(original_install, _INSTALLER_MARKER, False):
        _ARMED = True
        return

    @wraps(original_install)
    def install_with_prose_compat() -> None:
        global _INSTALLED
        original_install()
        _INSTALLED = False
        install()

    setattr(install_with_prose_compat, _INSTALLER_MARKER, True)
    install_with_prose_compat.__wrapped__ = original_install  # type: ignore[attr-defined]
    deep.install = install_with_prose_compat
    _ARMED = True


__all__ = ["arm", "install"]
