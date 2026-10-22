"""Reject invented physical-input triggers before turning prose into a game hook.

This is an input-*modality* provenance check, not a special case for any
gameplay theme or particular key. An unsourced input claim is not a runnable
contract; preserve the feature and leave its trigger unspecified instead of
manufacturing a click/keyboard requirement.
"""
from __future__ import annotations

import re
import unicodedata


# Device and action vocabulary, not concrete key names or gameplay concepts.
# A bare 'space' is intentionally not evidence of a keyboard instruction.
_PHYSICAL_INPUT = re.compile(
    r"\b(?:key(?:board|bind|binding|stroke)?|key(?:press|down|up)|"
    r"mouse|click(?:ed|ing)?|press(?:ed|ing)?|tap(?:ped|ping)?|"
    r"button|hotkey|shortcut|glfw|inputconstants)\b"
    r"|(?:키보드|단축키|키\s*(?:입력|누르|눌|보드)|마우스|"
    r"클릭|버튼|누르|눌러|눌렀|키입력)",
    re.IGNORECASE,
)


def claim_is_physical_input(trigger: str) -> bool:
    return bool(_PHYSICAL_INPUT.search(unicodedata.normalize("NFKC", trigger)))


def preserve_grounded_trigger(
    trigger: str, *, source_quote: str, source_text: str
) -> tuple[str, dict | None]:
    """Only a user-grounded input-control request can authorize key/mouse hooks.

    For other trigger categories the host cannot prove semantic equivalence
    with this lexical gate, so they remain the planner's unverified proposal.
    An omitted physical trigger is an unresolved implementation choice, never
    permission to assume a user pressed a particular key.
    """
    raw = str(trigger or "").strip()
    if not raw or not claim_is_physical_input(raw):
        return raw, None
    evidence = unicodedata.normalize("NFKC", source_quote + "\n" + source_text)
    if _PHYSICAL_INPUT.search(evidence):
        return raw, None
    return "", {
        "code": "CONSTRUCTION_TRIGGER_INPUT_NOT_REQUESTED",
        "claimed_trigger": raw,
        "source_quote": source_quote,
        "resolution": "unspecified_trigger_requires_real_execution_design",
    }


__all__ = ["claim_is_physical_input", "preserve_grounded_trigger"]
