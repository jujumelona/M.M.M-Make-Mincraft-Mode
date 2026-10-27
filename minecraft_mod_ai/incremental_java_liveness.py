"""Semantic-state liveness for structured authoring, independent of feature text.

Two structurally identical questions, with the same grounded evidence and
schema, mean no accepted progress was made between them. Preserve arbitrarily
large successful programs: there is no arbitrary total-call or operation cap.
"""
from __future__ import annotations

import hashlib
import json


def assert_forward_progress(seen: set[str], payload: dict, schema: dict) -> None:
    state = {"payload": payload, "schema": schema}
    canonical = json.dumps(state, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest in seen:
        member = (payload.get("current_member") or {}).get("id", "?")
        page = payload.get("requested_page", "?")
        raise ValueError(f"JAVA_INCREMENTAL_NO_PROGRESS_IDENTICAL_REQUEST: {member}:{page}")
    seen.add(digest)
