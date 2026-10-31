"""Replayable activation scope over the original contract, not verb allowlists.

The selected action supplies its lexical anchor. Modality in a prerequisite
or a different operation is retained without being mistaken for activation.
These structural checks supplement, rather than certify, semantic planning.
"""
from __future__ import annotations

import re

from .player_control_clause import PASSIVE_PLAYER_HEADS, explicit_player_control
from .source_occurrence_error import SourceHookUnavailable

ACTION_FIELDS = ("trigger", "intent", "effects", "source_quote")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_SUBJECT = re.compile(
    r"^\s*(?:the\s+)?(?:player|user|server|system)\b\s*|^\s*(?:플레이어|사용자)(?:가|는)?\s*",
    re.IGNORECASE)
_ADVERBS = frozenset({"explicitly", "manually", "actively", "deliberately", "immediately",
                      "automatically", "autonomously", "randomly"})
_SKIP = _ADVERBS | {"must", "should", "can", "may", "will", "would", "does", "do", "did", "not", "never",
                    "cannot", "can't", "don't", "doesn't", "won't", "mustn't", "shouldn't"}
_AUTO = re.compile(r"\b(?:automatic(?:ally)?|autonomous(?:ly)?)\b|자동으로|자동적", re.IGNORECASE)
_AFTER_AUTO = frozenset({"without", "with", "every", "each", "per", "on", "at", "over",
                         "for", "when", "while", "after", "before", "once", "and", "then"})
_NEGATED = re.compile(
    r"\b(?:(?:must|should|would|will|may|can|do|does|did)\s+not|not|never|cannot|"
    r"can't|don't|doesn't|won't|mustn't|shouldn't)\s+(?:(?:"
    + "|".join(sorted(_ADVERBS)) + r")\s+)*(?P<verb>[^\W\d_]+)",
    re.IGNORECASE)
_WITHOUT = re.compile(r"\bwithout\s+(?:(?:having|needing)\s+to\s+)?(?P<action>[^.;!?]+)", re.IGNORECASE)
_NO_INPUT = re.compile(
    r"\bno\s+(?:(?:manual|player|user)\s+)*input\s+(?:is\s+)?required\b|"
    r"(?:플레이어|사용자)\s*입력\s*없이", re.IGNORECASE)
_WITHOUT_INPUT = re.compile(
    r"\bwithout\s+(?:(?:any|manual)\s+)*(?:player|user)\s+input\b",
    re.IGNORECASE)
_BOUNDARY = re.compile(
    r"[.;!?\n]+|\b(?:and|but|while|when|then)\s+(?=(?:the\s+)?"
    r"(?:player|user|server|system)\b|the\s+)", re.IGNORECASE)


def _words(text):
    return [m.group().casefold() for m in _WORD.finditer(text)]


def _coordinated_control_anchor(text, first, *, legacy):
    if legacy or first not in PASSIVE_PLAYER_HEADS:
        return first
    # Include prohibited controls so a passive prerequisite cannot hide
    # the selected control's negation/automation in another original field.
    return explicit_player_control(text, include_restricted=True) or first


def _action_word(text, *, legacy=False):
    words = _words(_SUBJECT.sub("", text, count=1))
    first = next((word for word in words if word not in _SKIP), "")
    return _coordinated_control_anchor(text, first, legacy=legacy)


def _forms(word):
    result = {word}
    for ending in ("s", "es", "ed", "ing"):
        if len(word) > len(ending) + 2 and word.endswith(ending):
            stem = word[:-len(ending)]
            result.update((stem, stem + "e"))
    if word.endswith("ies"):
        result.add(word[:-3] + "y")
    return result


def _clauses(text):
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        yield start, boundary.start(), text[start:boundary.start()]
        start = boundary.end()
    yield start, len(text), text[start:]


def _automatic_operations(text, *, legacy=False):
    operations = []
    for marker in _AUTO.finditer(text):
        following = next((w for w in _words(text[marker.end():]) if w not in _ADVERBS), "")
        operation = following
        if not following or following in _AFTER_AUTO:
            operation = _action_word(text[:marker.start()], legacy=legacy)
        operations.append(operation)
    return operations


def _negated_operations(text):
    operations = []
    for marker in _NEGATED.finditer(text):
        # A historical eligibility fact does not prohibit a new action.
        if marker.group().casefold().startswith("never") and re.search(
                r"\b(?:has|have|had)\s*$", text[:marker.start()], re.IGNORECASE):
            continue
        operations.append(marker["verb"].casefold())
    return operations


def _input_absence_scope(text, anchor, *, legacy):
    without = [word for m in _WITHOUT.finditer(text)
               if legacy or not _WITHOUT_INPUT.match(text, m.start())
               for word in _words(m["action"])]
    standalone_no_input = bool(_NO_INPUT.search(text))
    input_markers = list(_WITHOUT_INPUT.finditer(text)) if not legacy else []
    # "without player input" describes the operation in this clause. It
    # prohibits converting THAT occurrence into a manual command; the same
    # phrase on an independent server-side consequence does not prohibit
    # the preceding player action.
    related_input = any(
        _forms(_action_word(text[:marker.start()])) & anchor
        for marker in input_markers)
    return without, standalone_no_input or bool(input_markers), standalone_no_input or related_input


def _mode_row(field, start, end, text, anchor, *, legacy=False):
    automatic, negated = _automatic_operations(text, legacy=legacy), _negated_operations(text)
    without, no_input, input_restricted = _input_absence_scope(text, anchor, legacy=legacy)
    if not (automatic or negated or without or no_input):
        return None
    related = any(_forms(word) & anchor for word in (*automatic, *negated, *without))
    return {"field": field, "start": start, "end": end, "text": text,
            "automatic_operations": automatic, "negated_operations": negated,
            "without_operations": without, "no_player_input": no_input,
            "restricts_activation": input_restricted or related}


def assess_player_action_modality(behavior, request_quote, *, legacy=False):
    """Bind scoped modality from every original field for deterministic replay."""
    action = _action_word(request_quote, legacy=legacy)
    anchor = _forms(action)
    rows = []
    for field in ACTION_FIELDS:
        for start, end, text in _clauses(str(behavior.get(field) or "")):
            row = _mode_row(field, start, end, text, anchor, legacy=legacy)
            if row is not None:
                rows.append(row)
    if any(row["restricts_activation"] for row in rows):
        raise SourceHookUnavailable(str(behavior.get("behavior_id") or ""),
                                    reason="original_occurrence_is_not_manual_activation", inspected=0)
    return {"action_anchor": action, "clauses": rows}
