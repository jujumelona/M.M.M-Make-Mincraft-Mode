from __future__ import annotations

import re
from collections.abc import Iterable

_REDACTED = "<redacted>"
_VALUE_DELIMITERS = frozenset("\r\n,;\"'")
_LINE_VALUE_DELIMITERS = frozenset("\r\n\"'")
_SENSITIVE_KEYS = (
    "authorization",
    "api_key",
    "api-key",
    "apikey",
    "password",
    "passwd",
    "cookie",
    "secret",
    "token",
)
_LINE_VALUE_KEYS = frozenset({"authorization", "cookie"})
_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(re.escape(key) for key in _SENSITIVE_KEYS)
    + r")(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_MAX_KEY_LENGTH = max(map(len, _SENSITIVE_KEYS))


class StreamingRedactor:
    """Incrementally redact sensitive audit output with bounded memory.

    Exact configured secrets and labelled values are redacted across arbitrary
    chunk boundaries. Label parsing understands quoted mapping keys/values and
    balanced structured values without buffering the sensitive payload.

    ``replacement`` lets callers choose a context-safe marker without duplicating
    the redaction algorithm (for example an XML-escaped marker for JUnit files).
    """

    __slots__ = (
        "_active_key",
        "_escape_next",
        "_exact_pending",
        "_label_pending",
        "_max_secret_length",
        "_replacement",
        "_secrets",
        "_state",
        "_structured_escape_next",
        "_structured_quote",
        "_structured_stack",
        "_value_quote",
    )

    def __init__(self, secrets: Iterable[str] = (), *, replacement: str = _REDACTED) -> None:
        if not isinstance(replacement, str) or not replacement:
            raise ValueError("replacement must be a non-empty string")
        self._replacement = replacement
        self._secrets = tuple(
            sorted(
                {str(secret) for secret in secrets if str(secret)},
                key=lambda item: (-len(item), item),
            )
        )
        self._max_secret_length = max(
            (len(secret) for secret in self._secrets),
            default=1,
        )
        self._exact_pending = ""
        self._label_pending = ""
        self._state = "normal"
        self._active_key: str | None = None
        self._value_quote: str | None = None
        self._escape_next = False
        self._structured_stack: list[str] = []
        self._structured_quote: str | None = None
        self._structured_escape_next = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        labelled = self._redact_labels(text, final=False)
        return self._redact_exact(labelled, final=False)

    def finish(self) -> str:
        labelled = self._redact_labels("", final=True)
        return self._redact_exact(labelled, final=True)

    def _redact_labels(self, text: str, *, final: bool) -> str:
        data = self._label_pending + text
        self._label_pending = ""
        output: list[str] = []
        position = 0

        while position < len(data):
            if self._state == "mask_quoted":
                position = self._consume_quoted_value(data, position, output)
                if self._state == "mask_quoted":
                    break
                continue

            if self._state == "mask_structured":
                position = self._consume_structured_value(data, position)
                if self._state == "mask_structured":
                    break
                continue

            if self._state == "mask_value":
                delimiters = (
                    _LINE_VALUE_DELIMITERS
                    if self._active_key in _LINE_VALUE_KEYS
                    else _VALUE_DELIMITERS
                )
                delimiter = _first_delimiter(data, position, delimiters)
                if delimiter is None:
                    position = len(data)
                    break
                output.append(data[delimiter])
                position = delimiter + 1
                self._reset_value_state()
                continue

            if self._state == "after_key":
                if data[position] in "\"'":
                    output.append(data[position])
                    position += 1
                    if position == len(data):
                        break
                whitespace_end = position
                while whitespace_end < len(data) and data[whitespace_end].isspace():
                    whitespace_end += 1
                if whitespace_end > position:
                    output.append(data[position:whitespace_end])
                    position = whitespace_end
                if position == len(data):
                    break
                if data[position] in ":=":
                    output.append(data[position])
                    position += 1
                    self._state = "before_value"
                    continue
                self._reset_value_state()
                continue

            if self._state == "before_value":
                whitespace_end = position
                while whitespace_end < len(data) and data[whitespace_end].isspace():
                    whitespace_end += 1
                if whitespace_end > position:
                    output.append(data[position:whitespace_end])
                    position = whitespace_end
                if position == len(data):
                    break
                if data[position] in ",;\r\n":
                    output.append(data[position])
                    position += 1
                    self._reset_value_state()
                    continue
                if data[position] in "\"'":
                    self._value_quote = data[position]
                    self._escape_next = False
                    output.append(data[position])
                    output.append(self._replacement)
                    position += 1
                    self._state = "mask_quoted"
                    continue
                if data[position] in "{[":
                    self._structured_stack = [_matching_bracket(data[position])]
                    self._structured_quote = None
                    self._structured_escape_next = False
                    output.append(self._replacement)
                    position += 1
                    self._state = "mask_structured"
                    continue
                output.append(self._replacement)
                self._state = "mask_value"
                continue

            match = _KEY_RE.search(data, position)
            if match is not None:
                output.append(data[position : match.end()])
                self._active_key = match.group(0).lower()
                position = match.end()
                self._state = "after_key"
                continue

            if final:
                output.append(data[position:])
                position = len(data)
                break

            keep = _normal_pending_length(data[position:])
            cutoff = len(data) - keep
            if cutoff > position:
                output.append(data[position:cutoff])
            self._label_pending = data[cutoff:]
            position = len(data)

        if final:
            if self._state not in {"mask_value", "mask_quoted", "mask_structured"}:
                output.append(self._label_pending)
            self._label_pending = ""
            self._reset_value_state()
        return "".join(output)

    def _consume_quoted_value(
        self,
        data: str,
        position: int,
        output: list[str],
    ) -> int:
        quote = self._value_quote
        while position < len(data):
            character = data[position]
            if self._escape_next:
                self._escape_next = False
                position += 1
                continue
            if character == "\\":
                self._escape_next = True
                position += 1
                continue
            if character == quote:
                output.append(character)
                position += 1
                self._reset_value_state()
                break
            position += 1
        return position

    def _consume_structured_value(self, data: str, position: int) -> int:
        while position < len(data):
            character = data[position]
            if self._structured_quote is not None:
                if self._structured_escape_next:
                    self._structured_escape_next = False
                elif character == "\\":
                    self._structured_escape_next = True
                elif character == self._structured_quote:
                    self._structured_quote = None
                position += 1
                continue

            if character in "\"'":
                self._structured_quote = character
                position += 1
                continue
            if character in "{[":
                self._structured_stack.append(_matching_bracket(character))
                position += 1
                continue
            if self._structured_stack and character == self._structured_stack[-1]:
                self._structured_stack.pop()
                position += 1
                if not self._structured_stack:
                    self._reset_value_state()
                    break
                continue
            position += 1
        return position

    def _reset_value_state(self) -> None:
        self._state = "normal"
        self._active_key = None
        self._value_quote = None
        self._escape_next = False
        self._structured_stack.clear()
        self._structured_quote = None
        self._structured_escape_next = False

    def _redact_exact(self, text: str, *, final: bool) -> str:
        if not self._secrets:
            return text

        data = self._exact_pending + text
        safe_cutoff = (
            len(data)
            if final
            else max(0, len(data) - (self._max_secret_length - 1))
        )
        if safe_cutoff == 0 and not final:
            self._exact_pending = data
            return ""

        output: list[str] = []
        cursor = 0
        while cursor < safe_cutoff:
            start, secret = _next_secret(data, cursor, self._secrets)
            if start is None or secret is None or start >= safe_cutoff:
                output.append(data[cursor:safe_cutoff])
                cursor = safe_cutoff
                break
            output.append(data[cursor:start])
            output.append(self._replacement)
            cursor = start + len(secret)

        if final and cursor < len(data):
            remainder = data[cursor:]
            for secret in self._secrets:
                remainder = remainder.replace(secret, self._replacement)
            output.append(remainder)
            cursor = len(data)

        self._exact_pending = data[cursor:]
        return "".join(output)


def _matching_bracket(character: str) -> str:
    return "}" if character == "{" else "]"


def _first_delimiter(
    text: str,
    start: int,
    delimiters: frozenset[str],
) -> int | None:
    for index in range(start, len(text)):
        if text[index] in delimiters:
            return index
    return None


def _partial_key_suffix_length(text: str) -> int:
    lowered = text.lower()
    limit = min(len(lowered), _MAX_KEY_LENGTH - 1)
    for length in range(limit, 0, -1):
        suffix = lowered[-length:]
        if any(key.startswith(suffix) for key in _SENSITIVE_KEYS):
            return length
    return 0


def _normal_pending_length(text: str) -> int:
    if not text:
        return 0
    partial = _partial_key_suffix_length(text)
    if partial:
        # Keep one preceding character so a key split across chunks still has
        # enough context for the identifier-boundary check on the next feed.
        return min(len(text), partial + 1)
    return 1


def _next_secret(
    text: str,
    start: int,
    secrets: tuple[str, ...],
) -> tuple[int | None, str | None]:
    best_start: int | None = None
    best_secret: str | None = None
    for secret in secrets:
        index = text.find(secret, start)
        if index < 0:
            continue
        if best_start is None or index < best_start or (
            index == best_start
            and best_secret is not None
            and len(secret) > len(best_secret)
        ):
            best_start = index
            best_secret = secret
    return best_start, best_secret
