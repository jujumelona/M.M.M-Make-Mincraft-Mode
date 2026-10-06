from __future__ import annotations

"""Canonical identifiers shared by proposal producers and validators."""

import hashlib
import re

SPEC_ID_MAX_LENGTH = 64
SPEC_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def canonical_spec_id(value: str, *, fallback: str = "id") -> str:
    """Return one deterministic CompleteProposal-safe identifier.

    Readable identifiers are preserved when already valid. Long identifiers retain
    a readable prefix plus a digest suffix so distinct semantic names never collide
    merely because they share the first 64 characters.
    """

    raw = str(value or "").strip().casefold()
    normalized = re.sub(r"[^a-z0-9_]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    fallback_value = re.sub(r"[^a-z0-9_]+", "_", str(fallback or "id").casefold())
    fallback_value = re.sub(r"_+", "_", fallback_value).strip("_") or "id"
    if not fallback_value[0].isalpha():
        fallback_value = "id_" + fallback_value

    if not normalized:
        normalized = fallback_value
    if not normalized[0].isalpha():
        normalized = "id_" + normalized

    if len(normalized) <= SPEC_ID_MAX_LENGTH and SPEC_ID_RE.fullmatch(normalized):
        return normalized

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    prefix_budget = SPEC_ID_MAX_LENGTH - len(digest) - 1
    prefix = normalized[:prefix_budget].rstrip("_") or fallback_value[:prefix_budget]
    if not prefix[0].isalpha():
        prefix = ("id_" + prefix)[:prefix_budget].rstrip("_")
    result = f"{prefix}_{digest}"
    if not SPEC_ID_RE.fullmatch(result):
        raise ValueError(f"SPEC_ID_NORMALIZATION_FAILED: {value!r}")
    return result


__all__ = [
    "SPEC_ID_MAX_LENGTH",
    "SPEC_ID_RE",
    "canonical_spec_id",
]
