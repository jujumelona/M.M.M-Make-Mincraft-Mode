from __future__ import annotations

"""Re-derive trajectory trust metadata from stored verifier chains.

This is the read-side trust boundary for local/remote procedural memory. Stored
levels, confidence and eligibility booleans are cached claims and must agree with
the objective verifier chain before a record can influence a temporary skill.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_TRAJECTORY_SCHEMA = "mmm/verified-trajectory-v3"
_VERIFICATION_SCHEMA = "mmm/trajectory-verification-v1"
_CODE_TASKS = {"repair", "generation", "build", "runtime", "quality", "release"}
_TASK_CLASSES = _CODE_TASKS | {"research", "planning", "general"}
_FAILURE_LEVEL = {
    "static": 1,
    "build": 2,
    "synthetic_counterexample": 2,
    "test": 3,
    "gametest": 3,
    "reproduction": 3,
    "runtime": 4,
    "acceptance": 5,
}
_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _chain(row: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return None
    raw = verification.get("verifier_chain")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if len(raw) > 48:
        return None
    result: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        kind = str(item.get("kind", ""))
        status = str(item.get("status", ""))
        source = item.get("source")
        if not kind or status not in {"PASS", "FAIL"} or not isinstance(source, str) or len(source) > 240:
            return None
        result.append(item)
    return result


def _local_identity_valid(row: Mapping[str, Any]) -> bool:
    identity = str(row.get("trajectory_id", ""))
    if not _ID.fullmatch(identity):
        return False
    # Remote records are intentionally sanitized after the local trajectory hash
    # was created; their post-sanitization integrity is checked by remote_record_id.
    if row.get("remote_format_version") is not None or row.get("remote_record_id") is not None:
        return True
    body = dict(row)
    body.pop("trajectory_id", None)
    rendered = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return identity == expected


def derive_levels(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("schema_version") != _TRAJECTORY_SCHEMA:
        return None
    if row.get("record_type") != "verified_trajectory" or row.get("storage_format") != "jsonl":
        return None
    verification = row.get("verification")
    if not isinstance(verification, Mapping) or verification.get("schema_version") != _VERIFICATION_SCHEMA:
        return None
    outcome = str(row.get("outcome", ""))
    if outcome not in {"SUCCESS", "FAIL"}:
        return None
    task_class = str(row.get("task_class", ""))
    if task_class not in _TASK_CLASSES:
        return None
    chain = _chain(row)
    if chain is None:
        return None

    passed = {str(item.get("kind")) for item in chain if item.get("status") == "PASS"}
    failed = {str(item.get("kind")) for item in chain if item.get("status") == "FAIL"}
    static_pass = "static" in passed
    build_pass = "build" in passed
    test_pass = bool(passed & {"test", "gametest"})
    runtime_pass = "runtime" in passed
    acceptance_pass = "acceptance" in passed
    explicit_reproduction = "reproduction" in passed
    independent_replay = static_pass and "synthetic_counterexample" in passed
    reproduced = explicit_reproduction or independent_replay

    level = 0
    if static_pass:
        level = 1
    if build_pass:
        level = max(level, 2)
    if test_pass and (build_pass or task_class not in _CODE_TASKS):
        level = max(level, 3)
    if runtime_pass and (build_pass or task_class not in _CODE_TASKS):
        level = max(level, 4)
    if acceptance_pass and (test_pass or runtime_pass or task_class not in _CODE_TASKS):
        level = 5
    failure_level = max((_FAILURE_LEVEL[kind] for kind in failed if kind in _FAILURE_LEVEL), default=0)

    success = outcome == "SUCCESS"
    pass_diversity = len(passed)
    fail_diversity = len(failed)
    confidence = min(1.0, 0.12 * level + 0.05 * min(pass_diversity, 5) + (0.12 if reproduced else 0.0))
    if not success and failure_level >= 1:
        confidence = max(confidence, min(0.98, 0.20 + 0.12 * failure_level + 0.05 * min(fail_diversity, 4)))

    return {
        "level_index": level,
        "failure_level_index": failure_level,
        "confidence": round(confidence, 4),
        "memory_eligible": (success and level >= 2) or ((not success) and failure_level >= 1),
        "strong_skill_eligible": success and level >= 3,
        "remote_eligible": (success and level >= 3) or ((not success) and failure_level >= 1),
        "verified_failure": (not success) and failure_level >= 1,
        "reproduced": reproduced,
        "reproduction_basis": "explicit" if explicit_reproduction else ("independent-clean-snapshot" if independent_replay else "none"),
        "checks": {
            "static": static_pass,
            "build": build_pass,
            "tests": test_pass,
            "runtime": runtime_pass,
            "acceptance": acceptance_pass,
        },
    }


def validate_trajectory_record(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    if derived is None or not _local_identity_valid(row):
        return False
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return False
    try:
        confidence = float(verification.get("confidence", -1))
        stored_level = int(verification.get("level_index", -1))
        stored_failure = int(verification.get("failure_level_index", -1))
    except (TypeError, ValueError):
        return False
    if not 0.0 <= confidence <= 1.0:
        return False
    if str(verification.get("level")) != f"L{stored_level}":
        return False
    if str(verification.get("failure_level")) != f"L{stored_failure}":
        return False
    if stored_level != derived["level_index"] or stored_failure != derived["failure_level_index"]:
        return False
    if abs(confidence - float(derived["confidence"])) > 1e-9:
        return False
    for key in (
        "memory_eligible",
        "strong_skill_eligible",
        "remote_eligible",
        "verified_failure",
        "reproduced",
    ):
        if verification.get(key) is not derived[key]:
            return False
    if str(verification.get("reproduction_basis", "none")) != derived["reproduction_basis"]:
        return False
    checks = verification.get("checks")
    if not isinstance(checks, Mapping):
        return False
    for key, value in derived["checks"].items():
        if checks.get(key) is not value:
            return False
    return True


def record_memory_eligible(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    return bool(derived and validate_trajectory_record(row) and derived["memory_eligible"])


def record_strong_skill_eligible(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    return bool(derived and validate_trajectory_record(row) and derived["strong_skill_eligible"])


def record_remote_eligible(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    return bool(derived and validate_trajectory_record(row) and derived["remote_eligible"])


__all__ = [
    "derive_levels",
    "record_memory_eligible",
    "record_remote_eligible",
    "record_strong_skill_eligible",
    "validate_trajectory_record",
]
