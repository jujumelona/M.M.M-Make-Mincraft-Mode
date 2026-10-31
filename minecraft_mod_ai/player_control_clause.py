"""Find an explicit control under its original player subject.

These words describe input controls, not gameplay features. A prerequisite
such as arrival/ownership does not consume a later coordinated control, and
an intervening subject or subordinate clause cannot donate its control to
the player. The caller still validates modality across the complete contract.
"""
from __future__ import annotations

import re

PASSIVE_PLAYER_HEADS = frozenset({
    "has", "have", "had", "wants", "want", "needs", "need",
    "is", "are", "was", "were", "can", "could", "should", "must",
    "does", "do", "did", "gets", "get", "receives", "receive",
    "reaches", "reach", "arrives", "arrive", "enters", "enter",
    "travels", "travel", "dies", "die", "takes", "take",
    "falls", "fall", "suffers", "suffer", "spawns", "spawn",
    "owns", "own", "holds", "hold", "possesses", "possess",
    "accumulates", "accumulate", "becomes", "become",
})
_CONTROL = re.compile(
    r"\b(?:selects?|chooses?|clicks?|confirms?|initiates?|requests?|submits?|"
    r"press(?:es)?|interacts?|activates?)\b|선택|클릭|확인|요청|누르",
    re.IGNORECASE,
)
_PLAYER = re.compile(r"\b(?:player|user)\b|플레이어|사용자", re.IGNORECASE)
_PREFIX = re.compile(
    r"\s*(?:(?:manually|explicitly|actively|deliberately|then|automatically|"
    r"autonomously|randomly|must|can|may|will|should|does|do|did|not|never|"
    r"cannot|can't|don't|doesn't|won't|mustn't|shouldn't)\s+|needs?\s+to\s+)*",
    re.IGNORECASE,
)
_COORDINATE = re.compile(r"\band\s+|,\s*then\s+", re.IGNORECASE)
_SCOPE_CHANGE = re.compile(
    r"\b(?:without|while|unless|until|that|which|who|because)\b",
    re.IGNORECASE,
)
_WITHOUT_INPUT = re.compile(
    r"\bwithout\s+(?:(?:any|manual)\s+)*(?:player|user)\s+input\b",
    re.IGNORECASE,
)
_RESTRICTED = re.compile(
    r"\b(?:automatically|autonomously|randomly|without|not|never|cannot|"
    r"can't|don't|doesn't|won't|mustn't|shouldn't)\b", re.IGNORECASE,
)


def explicit_player_control(text: str, *, include_restricted: bool = False) -> str:
    """Return the original control verb, or empty when no owned control exists.

    Modality validation needs the lexical control even when it is prohibited
    or automatic; occurrence generation only accepts the manual form.
    """
    for actor in _PLAYER.finditer(text):
        if re.search(r"\b(?:no|not|without)(?:\s+\w+){0,2}\s*$",
                     text[:actor.start()], re.IGNORECASE):
            continue
        tail = re.split(r"[.;!?\n]", text[actor.end():], maxsplit=1)[0]
        if actor.group() in {"플레이어", "사용자"}:
            action = _CONTROL.search(tail)
            if action and not re.search(r"자동|서버|시스템|없이|않", tail):
                return action.group().casefold()
            continue
        for index, segment in enumerate(_COORDINATE.split(tail)):
            prefix = _PREFIX.match(segment)
            action = _CONTROL.match(segment, prefix.end())
            head = re.match(r"[a-z]+\b", segment[prefix.end():], re.IGNORECASE)
            # The first predicate has the explicit player subject. Later
            # predicates inherit it only through a recognized control or
            # eligibility head. An unknown head may introduce ANY other
            # subject (a name, pronoun or noun); stop, rather than keeping
            # a name denylist and stealing that subject's subsequent verbs.
            if index and action is None and (
                    head is None or head.group().casefold() not in PASSIVE_PLAYER_HEADS):
                break
            if action is not None:
                rest = segment[action.end():]
                if include_restricted or not _RESTRICTED.search(prefix.group() + rest):
                    return action.group().casefold()
            # A restriction attached to the preceding prerequisite does
            # not alter the subject of a later independent control.
            if _SCOPE_CHANGE.search(_WITHOUT_INPUT.sub("", segment)):
                break
    return ""
