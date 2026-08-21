from __future__ import annotations

"""Reject persistent procedural skills whose content no longer matches ``skill_id``.

Skill IDs are SHA-256 commitments created before the persistence-only schema_version
field is added. Persistent JSONL is mutable storage, so loading by the stored ID alone
would let a corrupted/tampered row keep an old trusted identity while changing its
procedure. This contract validates the commitment at the load boundary and deduplicates
only byte-equivalent committed identities before any composition dict can collapse them.
"""

import hashlib
import json
import re
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_persistent_skill_identity_v1"
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _committed_identity(row: Mapping[str, Any]) -> str:
    payload = dict(row)
    payload.pop("skill_id", None)
    # schema_version is a persistence envelope field and was not present when the
    # canonical procedural skill identity was originally computed.
    payload.pop("schema_version", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def verified_persistent_skills(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only rows whose declared identity commits to their exact content."""

    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        identity = str(row.get("skill_id", "")).strip()
        if not _SHA256_ID.fullmatch(identity):
            continue
        committed = _committed_identity(row)
        if committed != identity:
            continue
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = seen.get(identity)
        if previous is not None:
            # A valid SHA-256 commitment makes conflicting content for the same ID
            # impossible without a hash collision. Identical persisted duplicates are
            # harmless and should not become multiple composition candidates.
            if previous != canonical:  # pragma: no cover - cryptographic invariant
                continue
            continue
        seen[identity] = canonical
        result.append(row)
    return result


def install(skills_module: Any) -> None:
    current = skills_module._load_persistent_skills
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def load_persistent_skills(path: Any, *, limit: int = 256):
        rows = current(path, limit=limit)
        return verified_persistent_skills(rows)

    setattr(load_persistent_skills, _MARKER, True)
    load_persistent_skills.__wrapped__ = current  # type: ignore[attr-defined]
    skills_module._load_persistent_skills = load_persistent_skills


def assert_installed(skills_module: Any) -> None:
    if getattr(skills_module._load_persistent_skills, _MARKER, False) is not True:
        raise RuntimeError("persistent procedural skill identity validation is not installed")


__all__ = ["assert_installed", "install", "verified_persistent_skills"]
