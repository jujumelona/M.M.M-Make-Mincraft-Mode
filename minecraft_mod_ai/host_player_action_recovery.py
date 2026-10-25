"""Deterministic host evidence for an action a small planner wrongly declines.

This does not infer new game mechanics or convert passive world events to
commands. It only recovers a *quoted original* player-owned action whose verb
is independently present in the behavior intent. The original behavioral
preconditions and modality validator remain authoritative.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .player_action_modality import assess_player_action_modality
from .source_occurrence_error import SourceHookUnavailable

_WORD = re.compile(r"[A-Za-z]+")
_PLAYER_ACTION = re.compile(
    r"^\s*(?:the\s+)?(?:player|user)\s+(?P<verb>[A-Za-z]+)\b", re.IGNORECASE,
)
_NON_ACTION = frozenset({
    "be", "is", "are", "was", "were", "been", "being",
    "has", "have", "had", "having", "want", "wants", "wanted",
    "need", "needs", "needed", "does", "do", "did", "not",
    "can", "may", "must", "should", "will", "would",
    "get", "gets", "got", "gain", "gains", "gained",
    "receive", "receives", "received", "lose", "loses", "lost",
    "die", "dies", "died", "spawn", "spawns", "respawn", "respawns",
    "enter", "enters", "reach", "reaches", "arrive", "arrives",
    "travel", "travels", "move", "moves", "teleport", "teleports",
    "leave", "leaves", "change", "changes",
    "damage", "damages", "attack", "attacks", "kill", "kills",
})


def _forms(word: str) -> set[str]:
    """Small spelling variants for *matching*, never semantic substitution."""
    word = word.casefold()
    forms = {word}
    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            forms.add(stem)
            if suffix in ("ing", "ed"):
                forms.add(stem + "e")
    return forms


def recover_player_action_quote(behavior: Mapping) -> tuple[str, str] | None:
    """Return an existing (field, quote), only for an unambiguous manual action.

    An actor in a *desired gameplay effect* plus its same verb in the original
    intent is positive source evidence. Eligibility/status verbs, movement and
    unsolicited automatic events are not manual action evidence. This path
    cannot produce a Fabric API or certify gameplay; it only permits the host
    to plan a real typed action producer and listener.
    """
    intent = str(behavior.get("intent") or "")
    intent_forms = set().union(*(_forms(word) for word in _WORD.findall(intent))) if intent else set()
    if not intent_forms:
        return None
    for field in ("effects", "trigger", "source_quote"):
        original = behavior.get(field)
        if not isinstance(original, str):
            continue
        for clause in re.split(r"[.;!?\n]+", original):
            match = _PLAYER_ACTION.match(clause)
            if match is None:
                continue
            forms = _forms(match.group("verb"))
            if forms & _NON_ACTION or not forms & intent_forms:
                continue
            quote = clause.strip()
            if not quote or len(quote) > 512:
                continue
            try:
                assess_player_action_modality(behavior, quote)
            except SourceHookUnavailable:
                continue
            return field, quote
    return None
