from __future__ import annotations

"""Explicit opt-in gate for remote trajectory and temporary-skill persistence.

Remote persistence is never implied by model/tool configuration. Colab owns the
user-facing choice and writes the process flag. Every remote store backend must
call ``require_remote_write_consent`` immediately before a network write.
"""

import os
from collections.abc import Mapping
from typing import Any

from .remote_store_defaults import apply_remote_store_defaults

CONSENT_ENV = "MMM_REMOTE_TRAJECTORY_STORE_CONSENT"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_REMOTE_PRIVATE_KEYS = frozenset(
    {
        "node_id",
        "member_ids",
        "error_signature",
        "source_body",
        "source_code",
        "source_path",
        "uri",
        "path",
        "content",
        "prompt",
    }
)

# Configure only the destination. This does not enable remote persistence: the
# consent environment variable below remains fail-closed and every write path
# re-checks it immediately before network I/O.
apply_remote_store_defaults()


def remote_write_allowed() -> bool:
    """Return True only for the explicit affirmative Colab/runtime choice."""

    return os.environ.get(CONSENT_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_remote_write_consent() -> None:
    """Fail closed before any GitHub/Hugging Face/network persistence write."""

    if not remote_write_allowed():
        raise PermissionError(
            "Remote trajectory/temporary-skill persistence is disabled. "
            "Enable ALLOW_REMOTE_TRAJECTORY_STORE explicitly in the Colab "
            "configuration cell before setup if you want remote persistence."
        )


def sanitize_remote_payload(value: Any) -> Any:
    """Project opted-in records onto reusable, non-source structural facts only."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            lowered = key.casefold()
            if lowered in _REMOTE_PRIVATE_KEYS:
                continue
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                continue
            sanitized[key] = sanitize_remote_payload(raw_value)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_remote_payload(item) for item in value]
    return value


__all__ = [
    "CONSENT_ENV",
    "remote_write_allowed",
    "require_remote_write_consent",
    "sanitize_remote_payload",
]
