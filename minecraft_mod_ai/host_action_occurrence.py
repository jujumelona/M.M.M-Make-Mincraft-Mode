"""Compile explicitly requested player actions into a host-owned occurrence contract.

This is source planning, not a recovery of a failed model decision. The event
owner is determined from the ORIGINAL behavior before choosing an unrelated
Fabric callback or asking a 9B model whether an unseen Java method exists.
No command may be created from a world condition or an implicit player wish.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .player_action_modality import assess_player_action_modality
from .source_occurrence_error import SourceHookUnavailable

# This grammar admits a *stated actor and verb*, not a synthesized request.
_ACTOR = re.compile(
    r"(?<![a-z])(?:the\\s+)?(?:player|user)\\s+"
    r"(?:(?:explicitly|manually|actively|deliberately)\\s+)*"
    r"(?:(?:can|may|must|will|should)\\s+)?"
    r"(?P<verb>[a-z]+)\\b", re.IGNORECASE,
)
_KOREAN_ACTOR = re.compile(r"(?:플레이어|사용자)(?:가|는)\\s*[^.;!?\\n]{1,180}")
_KOREAN_MANUAL = re.compile(r"구매|거래|판매|선택|확인|요청|제작|건설|강화|설치|장착|교환|사용")
_NON_ACTION = frozenset({
    "has", "have", "had", "wants", "want", "needs", "need",
    "is", "are", "was", "were", "can", "could", "should", "must",
    "does", "do", "did", "gets", "get", "receives", "receive",
    "reaches", "reach", "arrives", "arrive", "enters", "enter",
    "travels", "travel", "dies", "die", "takes", "take",
    "falls", "fall", "suffers", "suffer", "spawns", "spawn",
    "owns", "own", "holds", "hold", "possesses", "possess",
    "accumulates", "accumulate", "becomes", "become",
})
_NO_MANUAL = re.compile(
    r"\\b(?:automatically|autonomously|without\\s+(?:any\\s+)?(?:player|user)\\s+input)\\b"
    r"|자동으로|자동적|입력\\s*없이|않(?:는|다)|안\\s+", re.IGNORECASE,
)


def _clauses(text):
    return (item.strip() for item in re.split(r"[.;!?\\n]+", text) if item.strip())


def select_original_player_action(behavior: Mapping) -> dict | None:
    """Identify an actual player command, not a model-invented UI trigger.

    The selection binds to the exact original field/substring, which is
    revalidated by the existing activation-scope verifier at materialization.
    Any explicit source site/event binding takes precedence elsewhere.
    """
    for field in ("trigger", "effects"):
        for clause in _clauses(str(behavior.get(field) or "")):
            for actor in _ACTOR.finditer(clause):
                verb = actor["verb"].casefold()
                if verb in _NON_ACTION or _NO_MANUAL.search(clause):
                    continue
                original = clause[actor.start():].strip()
                # Directly bind a short original action clause, never compose
                # a new sentence, even if the user's request is in Korean.
                if len(original) > 512:
                    continue
                try:
                    scope = assess_player_action_modality(behavior, original)
                except SourceHookUnavailable:
                    continue
                return {"request_field": field, "request_quote": original,
                        "activation_scope": scope}
            # Korean requires the original explicit action marker; unbound
            # nouns/prerequisites are not silently converted into commands.
            for actor in _KOREAN_ACTOR.finditer(clause):
                original = actor.group().strip()
                if (len(original) > 512 or _NO_MANUAL.search(original)
                        or not _KOREAN_MANUAL.search(original)):
                    continue
                try:
                    scope = assess_player_action_modality(behavior, original)
                except SourceHookUnavailable:
                    continue
                return {"request_field": field, "request_quote": original,
                        "activation_scope": scope}
    return None


def author_host_owned_player_occurrence(router, behavior, *, symbols,
                                        package, mod_id, minecraft_version):
    """Emit the host's typed event; MiMo may define bounded INPUTS, not its owner."""
    selected = select_original_player_action(behavior)
    if selected is None:
        return None
    from .player_action_evidence import _behavior_hash
    from .source_owned_ui_event import plan_typed_ui_occurrence
    from .typed_action_authoring import author_action_metadata

    evidence = {"schema": "mmm/player-action-evidence-v1",
                "behavior_sha256": _behavior_hash(behavior), **selected}
    record = {"trigger": str(behavior.get("trigger") or ""),
              "server_action": str(behavior.get("intent") or ""),
              "original_behavior": dict(behavior), "activation_evidence": evidence}
    metadata = author_action_metadata(
        router, record, requested_prompt=str(behavior.get("source_quote") or ""))
    return plan_typed_ui_occurrence(
        behavior=behavior, action_metadata=metadata, activation_evidence=evidence,
        package=package, mod_id=mod_id, minecraft_version=minecraft_version,
        symbols=symbols,
    )
