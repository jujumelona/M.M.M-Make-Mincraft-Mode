from __future__ import annotations

"""Re-derive trajectory eligibility from stored verifier chains.

This is the read-side trust boundary for local/remote procedural memory. Stored
eligibility booleans are treated as cached claims and must agree with the evidence
chain before a record can influence a temporary skill.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

_TRAJECTORY_SCHEMA = "mmm/verified-trajectory-v3"
_VERIFICATION_SCHEMA = "mmm/trajectory-verification-v1"
_CODE_TASKS = {"repair", "generation", "build", "runtime", "quality", "release"}
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


def _chain(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return []
    raw = verification.get("verifier_chain")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in raw[:48]:
        if not isinstance(item, Mapping):
            return []
        kind = str(item.get("kind", ""))
        status = str(item.get("status", ""))
        source = item.get("source")
        if not kind or status not in {"PASS", "FAIL"} or not isinstance(source, str):
            return []
        result.append(item)
    return result


def derive_levels(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("schema_version") != _TRAJECTORY_SCHEMA:
        return None
    verification = row.get("verification")
    if not isinstance(verification, Mapping) or verification.get("schema_version") != _VERIFICATION_SCHEMA:
        return None
    outcome = str(row.get("outcome", ""))
    if outcome not in {"SUCCESS", "FAIL"}:
        return None
    task_class = str(row.get("task_class", ""))
    chain = _chain(row)
    if not chain:
        level = 0
        failure_level = 0
    else:
        passed = {str(item.get("kind")) for item in chain if item.get("status") == "PASS"}
        failed = {str(item.get("kind")) for item in chain if item.get("status") == "FAIL"}
        static_pass = "static" in passed
        build_pass = "build" in passed
        test_pass = bool(passed & {"test", "gametest"})
        runtime_pass = "runtime" in passed
        acceptance_pass = "acceptance" in passed
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
    return {
        "level_index": level,
        "failure_level_index": failure_level,
        "memory_eligible": (success and level >= 2) or ((not success) and failure_level >= 1),
        "strong_skill_eligible": success and level >= 3,
        "remote_eligible": (success and level >= 3) or ((not success) and failure_level >= 1),
        "verified_failure": (not success) and failure_level >= 1,
    }


def validate_trajectory_record(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    if derived is None:
        return False
    identity = str(row.get("trajectory_id", ""))
    if not _ID.fullmatch(identity):
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
    for key in ("memory_eligible", "strong_skill_eligible", "remote_eligible", "verified_failure"):
        if verification.get(key) is not derived[key]:
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
