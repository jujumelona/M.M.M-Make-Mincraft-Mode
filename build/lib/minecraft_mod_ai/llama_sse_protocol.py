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


__all__ = ["LlamaSseServerError", "sse_error_from_line"]
