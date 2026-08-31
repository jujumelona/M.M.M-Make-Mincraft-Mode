from __future__ import annotations

"""Preserve prose-first game-design planning through late runtime composition.

Game-design drafting emits Markdown/text and is parsed by the host. Older late-runtime
contracts still contain two JSON-era assumptions: the deep-design prompt wrapper expects
retired repair arguments, and the requirement-readiness prompt wrapper tries to decode the
Markdown user message as JSON. This compatibility owner removes both assumptions without
restoring a model JSON schema or a model repair loop.

Requirement traceability stays host-owned. When approved requirements are active, the
model receives a plain-text ledger and writes traceability columns in ``## modules`` rows.
A host parser converts those extra columns to ``requirement_refs`` and
``implementation_obligations`` after generation.
"""

import re
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

_TRACEABILITY_GUIDANCE = (
    " When approved requirements are present, every implementation-bearing requirement "
    "must be covered by at least one ## modules row. In that section use exactly five "
    "pipe-separated columns per non-empty module row: "
    "plugin_id | status | reason | requirement_refs | implementation_obligations. "
    "Write requirement_refs as comma-separated exact requirement IDs from the ledger. "
    "Write multiple implementation obligations separated by semicolons. Do not invent, "
    "rename, merge away, or summarize away requirement IDs."
)

_INSTALLED = False
_ARMED = False
_INSTALLER_MARKER = "_mmm_prose_first_deep_design_compat_v2"
_MODULE_ROWS_MARKER = "_mmm_prose_first_traceable_module_rows_v1"


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split()).replace("|", "/").strip()


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    from . import planner_design_readiness_contract as readiness

    return readiness._active_requirement_ledger(prompt)


def _render_requirement_ledger(ledger: Sequence[Mapping[str, Any]]) -> str:
    lines = ["APPROVED REQUIREMENTS (HOST AUTHORITY)"]
    for item in ledger:
        requirement_id = _one_line(item.get("requirement_id"))
        lines.append(f"- requirement_id: {requirement_id}")
        capability = _one_line(item.get("capability"))
        if capability:
            lines.append(f"  capability: {capability}")
        authored = _one_line(item.get("authored_text"))
        if authored:
            lines.append(f"  authored_text: {authored}")
        semantic = _one_line(item.get("semantic_statement"))
        if semantic:
            lines.append(f"  semantic_statement: {semantic}")
        acceptance = item.get("acceptance")
        if isinstance(acceptance, list):
            rendered = "; ".join(_one_line(value) for value in acceptance if _one_line(value))
            if rendered:
                lines.append(f"  acceptance: {rendered}")
    return "\n".join(lines)


def _prose_base(base: Any) -> Any:
    """Bypass only the retired JSON-mutating requirement-message wrapper."""

    if getattr(base, "__mmm_requirement_design_messages__", False):
        candidate = getattr(base, "__wrapped__", None)
        if callable(candidate):
            return candidate
    return base


def _repair_for(agentic: Any, deep: Any) -> bool:
    """Replace exactly the stale deep wrapper while preserving wrapper metadata."""

    current = agentic._section_messages
    stale = deep._deep_section_messages
    if current is not stale:
        return False

    base = getattr(stale, "__wrapped__", None)
    if not callable(base):
        raise RuntimeError("deep-design section wrapper lost its base callable")
    prose_base = _prose_base(base)

    @wraps(base)
    def compatible_section_messages(
        *,
        prompt: str,
        section_id: str,
        fields: Sequence[str],
        research: Mapping[str, Any],
    ) -> list[dict[str, str]]:
        messages = prose_base(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
        )
        if not messages:
            return messages
        output = [dict(message) for message in messages]
        output[0]["content"] = str(output[0].get("content") or "") + _DEPTH_GUIDANCE

        ledger = _active_requirement_ledger(prompt)
        if ledger:
            output[0]["content"] += _TRACEABILITY_GUIDANCE
            output[-1]["content"] = (
                str(output[-1].get("content") or "")
                + "\n\n"
                + _render_requirement_ledger(ledger)
            )
        return output

    setattr(compatible_section_messages, deep._SECTION_MARKER, True)
    deep._deep_section_messages = compatible_section_messages
    agentic._section_messages = compatible_section_messages
    return True


def _install_traceable_module_rows() -> None:
    from . import agentic_research_game_design as agentic
    from .spec import SpecValidationError

    original = agentic._module_rows
    if getattr(original, _MODULE_ROWS_MARKER, False):
        return

    @wraps(original)
    def traceable_module_rows(body: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in body.splitlines():
            value = agentic._strip_list_marker(line)
            if not value or value.casefold() in {"none", "n/a", "없음"}:
                continue
            parts = [part.strip() for part in value.split("|")]
            if len(parts) < 3 or not all(parts[:3]):
                raise SpecValidationError(
                    "Each ## modules row must be: plugin_id | status | reason"
                )
            row: dict[str, Any] = {
                "plugin_id": parts[0],
                "status": parts[1],
                "reason": parts[2] if len(parts) >= 5 else " | ".join(parts[2:]),
            }
            if len(parts) >= 5:
                refs = [ref.strip() for ref in parts[3].split(",") if ref.strip()]
                obligations = [
                    item.strip()
                    for item in re.split(r"\s*;\s*", " | ".join(parts[4:]))
                    if item.strip()
                ]
                if not refs:
                    raise SpecValidationError(
                        "Traceable ## modules rows require at least one requirement_ref"
                    )
                if not obligations:
                    raise SpecValidationError(
                        "Traceable ## modules rows require implementation_obligations"
                    )
                row["requirement_refs"] = refs
                row["implementation_obligations"] = obligations
            rows.append(row)
        return rows

    setattr(traceable_module_rows, _MODULE_ROWS_MARKER, True)
    agentic._module_rows = traceable_module_rows


def _repair() -> bool:
    from . import agentic_research_game_design as agentic
    from . import deep_design_execution_contract as deep

    return _repair_for(agentic, deep)


def install() -> None:
    """Apply prose compatibility after the late deep-design installer."""

    global _INSTALLED
    if _INSTALLED:
        return
    _repair()
    _install_traceable_module_rows()
    _INSTALLED = True


def arm() -> None:
    """Wrap the late deep-design installer so compatibility precedes integrity checks."""

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
