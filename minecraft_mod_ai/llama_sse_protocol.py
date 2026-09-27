from __future__ import annotations

"""Pure llama.cpp SSE protocol parsing shared by transport and liveness owners."""

import json
from collections.abc import Mapping
from typing import Any


class LlamaSseServerError(RuntimeError):
    """An explicit server-side error delivered inside an SSE stream."""

    def __init__(self, status_code: int, error: Mapping[str, Any]) -> None:
        self.status_code = max(400, int(status_code))
        self.error = dict(error)
        super().__init__(str(self.error.get("message", "llama-server stream error")))


class LlamaNativeResponseFormatError(RuntimeError):
    """llama.cpp generated a turn but its native chat parser rejected that turn."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = max(400, int(status_code))
        self.detail = str(detail or "").strip()
        super().__init__(
            f"llama server returned HTTP {self.status_code}"
            + (f": {self.detail}" if self.detail else "")
        )


_NATIVE_FORMAT_ERROR_MARKERS = (
    "peg-native format",
    "common_chat_peg_parse",
)


def is_recoverable_native_format_error(value: Any) -> bool:
    """Recognize server-side native chat parsing failures through wrapper chains."""

    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        if isinstance(current, BaseException):
            text = str(current).casefold()
            for nested in (
                getattr(current, "cause", None),
                current.__cause__,
                current.__context__,
            ):
                if nested is not None and nested is not current:
                    pending.append(nested)
        elif isinstance(current, Mapping):
            try:
                text = json.dumps(dict(current), ensure_ascii=False).casefold()
            except (TypeError, ValueError):
                text = str(current).casefold()
        else:
            text = str(current or "").casefold()

        if any(token in text for token in _NATIVE_FORMAT_ERROR_MARKERS):
            return True
    return False


def _error_status(value: Any) -> int:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return 500
    return status if 400 <= status <= 599 else 500


def _normalize_error(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        error = dict(value)
        error.setdefault("message", "llama-server stream error")
        return error
    if isinstance(value, str) and value.strip():
        return {"code": 500, "message": value.strip(), "type": "server_error"}
    return None


def sse_error_from_line(raw_line: Any) -> tuple[int, dict[str, Any]] | None:
    """Parse current ``data: {error: ...}`` and legacy ``error: ...`` records."""

    if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace").strip()
    else:
        line = str(raw_line or "").strip()
    if not line:
        return None

    if line.startswith("data:"):
        payload_text = line[5:].strip()
        legacy = False
        if not payload_text or payload_text == "[DONE]":
            return None
    elif line.startswith("error:"):
        payload_text = line[6:].strip()
        legacy = True
    else:
        return None

    try:
        decoded = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        if legacy and payload_text:
            return 500, {
                "code": 500,
                "message": payload_text,
                "type": "server_error",
            }
        return None

    if legacy:
        error = _normalize_error(decoded)
    elif isinstance(decoded, Mapping):
        error = _normalize_error(decoded.get("error"))
    else:
        error = None
    if error is None:
        return None
    status = _error_status(error.get("code"))
    error["code"] = status
    return status, error


__all__ = [
    "LlamaNativeResponseFormatError",
    "LlamaSseServerError",
    "is_recoverable_native_format_error",
    "sse_error_from_line",
]
