from __future__ import annotations

import re
from collections.abc import Iterable

_REDACTED = "<redacted>"
_VALUE_DELIMITERS = frozenset(" \t\r\n,;")
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
_KEY_RE = re.compile("|".join(re.escape(key) for key in _SENSITIVE_KEYS), re.IGNORECASE)
_MAX_KEY_LENGTH = max(map(len, _SENSITIVE_KEYS))


class StreamingRedactor:
    """Incrementally redact audit output without retaining unbounded process text.

    Memory use is bounded by the caller's chunk size plus the longest configured
    exact secret. Labelled values such as ``token=...`` are masked with a small
    state machine, so an arbitrarily long value never has to be buffered.
    """

    __slots__ = (
        "_exact_pending",
        "_label_pending",
        "_max_secret_length",
        "_secrets",
        "_state",
    )

    def __init__(self, secrets: Iterable[str] = ()) -> None:
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
            if self._state == "mask_value":
                delimiter = _first_delimiter(data, position)
                if delimiter is None:
                    return "".join(output)
                output.append(data[delimiter])
                position = delimiter + 1
                self._state = "normal"
                continue

            if self._state == "after_key":
                whitespace_end = position
                while whitespace_end < len(data) and data[whitespace_end].isspace():
                    whitespace_end += 1
                if whitespace_end > position:
                    output.append(data[position:whitespace_end])
                    position = whitespace_end
                if position == len(data):
                    return "".join(output)
                if data[position] in ":=":
                    output.append(data[position])
                    position += 1
                    self._state = "before_value"
                    continue
                self._state = "normal"
                continue

            if self._state == "before_value":
                whitespace_end = position
                while whitespace_end < len(data) and data[whitespace_end].isspace():
                    whitespace_end += 1
                if whitespace_end > position:
                    output.append(data[position:whitespace_end])
                    position = whitespace_end
                if position == len(data):
                    return "".join(output)
                if data[position] in ",;":
                    output.append(data[position])
                    position += 1
                    self._state = "normal"
                    continue
                output.append(_REDACTED)
                position += 1
                self._state = "mask_value"
                continue

            match = _KEY_RE.search(data, position)
            if match is not None:
                output.append(data[position:match.end()])
                position = match.end()
                self._state = "after_key"
                continue

            if final:
                output.append(data[position:])
                position = len(data)
                break

            suffix_length = _partial_key_suffix_length(data[position:])
            if suffix_length:
                cutoff = len(data) - suffix_length
                output.append(data[position:cutoff])
                self._label_pending = data[cutoff:]
            else:
                output.append(data[position:])
            position = len(data)

        if final and self._label_pending:
            output.append(self._label_pending)
            self._label_pending = ""
        return "".join(output)

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
            output.append(_REDACTED)
            cursor = start + len(secret)

        if final and cursor < len(data):
            remainder = data[cursor:]
            for secret in self._secrets:
                remainder = remainder.replace(secret, _REDACTED)
            output.append(remainder)
            cursor = len(data)

        self._exact_pending = data[cursor:]
        return "".join(output)


def _first_delimiter(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if text[index] in _VALUE_DELIMITERS:
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
