from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentEvent:
    start: int
    end: int
    term: str
    requested: bool


@dataclass(frozen=True)
class CountIntent:
    count: int
    explicit: bool
    overflow: int | None = None


_ENGLISH_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "hundred": 100,
}
_KOREAN_NUMBER_WORDS = {
    "한": 1,
    "하나": 1,
    "두": 2,
    "둘": 2,
    "세": 3,
    "셋": 3,
    "네": 4,
    "넷": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}


def _ascii_term_body(term: str) -> str:
    if not re.fullmatch(r"[a-z]+", term):
        return re.escape(term)
    if term.endswith("y") and len(term) > 1:
        return rf"{re.escape(term[:-1])}(?:y|ies)"
    if term.endswith(("s", "x", "z", "ch", "sh")):
        return rf"{re.escape(term)}(?:es)?"
    return rf"{re.escape(term)}s?"


def _term_pattern(term: str) -> re.Pattern[str]:
    normalized = term.strip().lower()
    if not normalized:
        raise ValueError("Intent terms must not be empty.")
    if re.fullmatch(r"[a-z]+", normalized):
        body = _ascii_term_body(normalized)
        return re.compile(
            rf"(?<![a-z0-9_])(?:{body})(?![a-z0-9_])",
            re.IGNORECASE,
        )
    if re.fullmatch(r"[a-z0-9]+(?:[- ][a-z0-9]+)+", normalized):
        return re.compile(
            rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])",
            re.IGNORECASE,
        )
    return re.compile(re.escape(normalized), re.IGNORECASE)


def term_occurrences(text: str, terms: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]:
    matches = {
        (match.start(), match.end(), term.lower())
        for term in terms
        for match in _term_pattern(term).finditer(text)
    }
    return tuple(sorted(matches, key=lambda item: (item[0], item[1], item[2])))


def _is_negative_context(text: str, start: int, end: int) -> bool:
    lowered = text.lower()
    clause_start = max(
        lowered.rfind(separator, 0, start)
        for separator in ("\n", ".", "!", "?", ";")
    )
    clause_end_candidates = [
        position
        for separator in ("\n", ".", "!", "?", ";")
        if (position := lowered.find(separator, end)) >= 0
    ]
    clause_end = min(clause_end_candidates, default=len(lowered))
    before = lowered[max(clause_start + 1, start - 96) : start]
    after = lowered[end : min(clause_end, end + 96)]

    english_prefix = re.search(
        r"(?:"
        r"\b(?:remove|delete|exclude|omit)\s+"
        r"(?:(?!(?:and|but|then|add|include|create|make)\b)[a-z0-9_-]+\s+){0,4}"
        r"|\bwithout\s+(?:(?:any|all|the|an?)\s+){0,2}"
        r"|\bno\s+(?:(?:other|more|additional)\s+){0,2}"
        r"|\b(?:do\s+not|don't|dont)\s+"
        r"(?:add|include|create|make|use|want|need)\s+"
        r"(?:(?:any|all|the|an?|more)\s+){0,3}"
        r")$",
        before,
    )
    english_suffix = re.match(
        r"^\s*(?:,?\s*)?(?:"
        r"(?:(?:should|must|will|is|are|was|were)\s+)?"
        r"(?:not|never)\s+(?:be\s+)?"
        r"(?:included|added|created|made|used|needed|present|wanted)"
        r"|(?:is|are)\s+(?:not\s+needed|unneeded|off)"
        r"|(?:please\s+)?(?:remove|delete|exclude|omit)\b"
        r")",
        after,
    )
    korean_suffix = re.match(
        r"^[\s,·]*(?:은|는|이|가|을|를|도|만)?[\s,·]*"
        r"(?:(?:전부|모두|전체)\s*)?"
        r"(?:"
        r"빼|빼고|빼줘|제외|제거|없애|삭제|"
        r"필요\s*없|"
        r"(?:넣|만들)지\s*마|"
        r"(?:추가|포함|생성)하지\s*마|"
        r"하지\s*마|"
        r"안\s*(?:넣|만들|추가|포함|생성)|"
        r"없이"
        r")",
        after,
    )
    return bool(english_prefix or english_suffix or korean_suffix)


def intent_events(text: str, terms: tuple[str, ...]) -> tuple[IntentEvent, ...]:
    return tuple(
        IntentEvent(
            start=start,
            end=end,
            term=term,
            requested=not _is_negative_context(text, start, end),
        )
        for start, end, term in term_occurrences(text, terms)
    )


def latest_intent_event(
    text: str,
    terms: tuple[str, ...],
    *,
    cascade_removals: tuple[str, ...] = (),
) -> IntentEvent | None:
    events = list(intent_events(text, terms))
    for event in intent_events(text, cascade_removals):
        if not event.requested:
            events.append(event)
    if not events:
        return None
    return max(events, key=lambda event: (event.start, event.end))


def is_requested(
    text: str,
    terms: tuple[str, ...],
    *,
    cascade_removals: tuple[str, ...] = (),
) -> bool:
    event = latest_intent_event(
        text,
        terms,
        cascade_removals=cascade_removals,
    )
    return event is not None and event.requested


def _number_matches(text: str, offset: int) -> list[tuple[int, int]]:
    scrubbed = re.sub(r"(?<!\d)\d+\s*[x×]\s*\d+(?!\d)", " ", text)
    matches: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<!\d)\d+(?!\d)", scrubbed):
        matches.append((offset + match.start(), int(match.group(0))))
    for word, value in _ENGLISH_NUMBER_WORDS.items():
        for match in re.finditer(
            rf"(?<![a-z0-9_]){word}(?![a-z0-9_])",
            scrubbed,
            re.IGNORECASE,
        ):
            matches.append((offset + match.start(), value))
    for word, value in _KOREAN_NUMBER_WORDS.items():
        for match in re.finditer(
            rf"(?<![가-힣]){word}"
            rf"(?=(?:\s*개)?(?:은|는|이|가|을|를|의|도|만)?(?![가-힣]))",
            scrubbed,
        ):
            matches.append((offset + match.start(), value))
    return matches


def _nearest_clause_start(text: str, position: int) -> int:
    start = max(
        text.rfind(separator, 0, position)
        for separator in ("\n", ".", "!", "?", ";", ",")
    )
    segment_start = max(start + 1, position - 96)
    segment = text[segment_start:position]
    conjunctions = list(
        re.finditer(
            r"(?i)(?<![a-z0-9_])(?:and|but|then|also)(?![a-z0-9_])",
            segment,
        )
    )
    if conjunctions:
        segment_start += conjunctions[-1].end()
    return segment_start


def _nearest_clause_end(text: str, position: int) -> int:
    ends = [
        found
        for separator in ("\n", ".", "!", "?", ";", ",")
        if (found := text.find(separator, position)) >= 0
    ]
    end = min(ends, default=min(len(text), position + 96))
    segment = text[position:end]
    conjunction = re.search(
        r"(?i)(?<![a-z0-9_])(?:and|but|then|also|with)(?![a-z0-9_])",
        segment,
    )
    if conjunction:
        end = position + conjunction.start()
    return min(end, position + 96)


def requested_count(
    text: str,
    *,
    terms: tuple[str, ...],
    default: int,
    maximum: int | None = None,
) -> CountIntent:
    event = latest_intent_event(text, terms)
    if event is None or not event.requested:
        return CountIntent(count=0, explicit=False)

    before_start = _nearest_clause_start(text.lower(), event.start)
    after_end = _nearest_clause_end(text.lower(), event.end)
    candidates = _number_matches(text[before_start:event.start], before_start)
    candidates.extend(_number_matches(text[event.end:after_end], event.end))
    if not candidates:
        return CountIntent(count=default, explicit=False)

    _, count = max(candidates, key=lambda item: item[0])
    if maximum is not None and count > maximum:
        return CountIntent(count=0, explicit=True, overflow=count)
    return CountIntent(count=max(0, count), explicit=True)
