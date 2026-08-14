from __future__ import annotations

"""Qualify trajectory records using objective verifier evidence only.

The model's own completion claim never raises a verification level. Levels are
portable across local/remote stores and deliberately conservative:

L0 unverified/model claim only
L1 static or compiler verification
L2 clean/build verification
L3 executable tests or GameTest verification
L4 runtime/playtest behavioral verification
L5 acceptance/quality-contract verification
"""

from collections.abc import Mapping, Sequence
from typing import Any

TRAJECTORY_SCHEMA_VERSION = "mmm/verified-trajectory-v3"
VERIFICATION_SCHEMA_VERSION = "mmm/trajectory-verification-v1"
REMOTE_FORMAT_VERSION = "v3"

_PASS = {"PASS", "PASSED", "SUCCESS", "SUCCEEDED", "OK"}
_FAIL = {"FAIL", "FAILED", "ERROR"}
_CODE_TASKS = {"repair", "generation", "build", "runtime", "quality", "release"}


def _status(value: Any) -> str:
    return str(value or "").strip().upper()


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _walk(value: Any, *, path: str = "", depth: int = 0):
    if depth > 7:
        return
    if isinstance(value, Mapping):
        yield path, value
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk(item, path=child, depth=depth + 1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value[:128]):
            yield from _walk(item, path=f"{path}[{index}]", depth=depth + 1)


def _commands(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = value.get("commands")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _command_pass(command: Mapping[str, Any]) -> bool:
    exit_code = _int(command.get("exit_code"))
    return exit_code == 0 and command.get("timed_out") is not True


def _command_fail(command: Mapping[str, Any]) -> bool:
    exit_code = _int(command.get("exit_code"))
    return command.get("timed_out") is True or (exit_code is not None and exit_code != 0)


def _kind_for_command(name: str) -> str:
    value = name.casefold()
    if "gametest" in value or "game_test" in value:
        return "gametest"
    if "test" in value:
        return "test"
    if "build" in value or "gradle" in value:
        return "build"
    return "command"


def _append(chain: list[dict[str, Any]], *, kind: str, status: str, source: str, details: Mapping[str, Any] | None = None) -> None:
    record = {
        "kind": kind,
        "status": status,
        "source": source[:240],
    }
    if details:
        record["details"] = dict(details)
    identity = (record["kind"], record["status"], record["source"], repr(record.get("details", {})))
    if any((item["kind"], item["status"], item["source"], repr(item.get("details", {}))) == identity for item in chain):
        return
    chain.append(record)


def _collect_chain(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    for path, node in _walk(receipt):
        lower_path = path.casefold()
        status = _status(node.get("status"))

        jdt_status = _status(node.get("jdt_status"))
        jdt_errors = _int(node.get("jdt_error_count"))
        if jdt_status in _PASS and jdt_errors == 0:
            _append(chain, kind="static", status="PASS", source=path or "receipt", details={"engine": "jdt", "errors": 0})
        elif jdt_errors is not None and jdt_errors > 0:
            _append(chain, kind="static", status="FAIL", source=path or "receipt", details={"engine": "jdt", "errors": jdt_errors})

        active_build = _status(node.get("active_build_status"))
        build_status = _status(node.get("build_status"))
        for value, source_key in ((active_build, "active_build_status"), (build_status, "build_status")):
            if value in _PASS:
                _append(chain, kind="build", status="PASS", source=f"{path}.{source_key}".strip("."))
            elif value in _FAIL:
                _append(chain, kind="build", status="FAIL", source=f"{path}.{source_key}".strip("."))

        if ("build" in lower_path or "gradle" in lower_path) and status in _PASS | _FAIL:
            _append(chain, kind="build", status="PASS" if status in _PASS else "FAIL", source=path or "receipt")

        for command in _commands(node):
            name = str(command.get("name", "")).strip()
            if not name:
                continue
            kind = _kind_for_command(name)
            if _command_pass(command):
                _append(chain, kind=kind, status="PASS", source=f"{path}.commands:{name}", details={"exit_code": 0})
            elif _command_fail(command):
                _append(chain, kind=kind, status="FAIL", source=f"{path}.commands:{name}", details={"exit_code": command.get("exit_code"), "timed_out": bool(command.get("timed_out"))})

        assertion_count = _int(node.get("assertion_count"))
        interaction_count = _int(node.get("interaction_count"))
        if (assertion_count or 0) > 0:
            kind = "runtime" if any(marker in lower_path for marker in ("runtime", "playtest")) else "test"
            if status in _PASS:
                _append(chain, kind=kind, status="PASS", source=path or "receipt", details={"assertions": assertion_count, "interactions": interaction_count or 0})
            elif status in _FAIL:
                _append(chain, kind=kind, status="FAIL", source=path or "receipt", details={"assertions": assertion_count})

        if node.get("server_running") is True and ("runtime" in lower_path or "server" in lower_path):
            _append(chain, kind="runtime", status="PASS", source=path or "receipt", details={"server_running": True})

        if any(marker in lower_path for marker in ("acceptance", "quality_contract", "quality_gate", "quality_evidence")) and status in _PASS | _FAIL:
            evidence_count = len(node.get("evidence", ())) if isinstance(node.get("evidence"), Sequence) and not isinstance(node.get("evidence"), (str, bytes)) else 0
            _append(chain, kind="acceptance", status="PASS" if status in _PASS else "FAIL", source=path or "receipt", details={"evidence_count": evidence_count})

        reproduction = node.get("reproduction") or node.get("replay") or node.get("reproducibility")
        if isinstance(reproduction, Mapping):
            repro_status = _status(reproduction.get("status"))
            if repro_status in _PASS | _FAIL:
                _append(chain, kind="reproduction", status="PASS" if repro_status in _PASS else "FAIL", source=f"{path}.reproduction".strip("."))

        counterexample = node.get("counterexample_result")
        if isinstance(counterexample, Mapping):
            counter_status = _status(counterexample.get("status"))
            details = {
                "gametest_requested": bool(counterexample.get("gametest_requested")),
                "json_ok": counterexample.get("json_ok"),
            }
            if counter_status in _PASS | _FAIL:
                _append(chain, kind="synthetic_counterexample", status="PASS" if counter_status in _PASS else "FAIL", source=f"{path}.counterexample_result".strip("."), details=details)

        synthetic = node.get("synthetic_verification")
        if isinstance(synthetic, Mapping):
            synthetic_status = _status(synthetic.get("status"))
            cases = synthetic.get("scenarios") or synthetic.get("cases") or ()
            case_count = len(cases) if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)) else 0
            if synthetic_status in _PASS | _FAIL and case_count > 0:
                _append(chain, kind="synthetic_scenarios", status="PASS" if synthetic_status in _PASS else "FAIL", source=f"{path}.synthetic_verification".strip("."), details={"case_count": case_count})

    return chain


def _has(chain: Sequence[Mapping[str, Any]], kinds: set[str], status: str = "PASS") -> bool:
    return any(str(item.get("kind")) in kinds and str(item.get("status")) == status for item in chain)


def classify_verification(*, task_class: str, outcome: str, receipt: Mapping[str, Any] | None, error: str = "") -> dict[str, Any]:
    chain = _collect_chain(receipt or {})
    static_pass = _has(chain, {"static"})
    build_pass = _has(chain, {"build"})
    test_pass = _has(chain, {"test", "gametest", "synthetic_scenarios"})
    runtime_pass = _has(chain, {"runtime"})
    acceptance_pass = _has(chain, {"acceptance"})
    reproduced = _has(chain, {"reproduction"})
    verified_failure = any(str(item.get("status")) == "FAIL" and str(item.get("kind")) in {"static", "build", "test", "gametest", "runtime", "acceptance", "synthetic_scenarios", "synthetic_counterexample", "reproduction"} for item in chain)

    level = 0
    if static_pass:
        level = max(level, 1)
    if build_pass:
        level = max(level, 2)
    if test_pass and (build_pass or task_class not in _CODE_TASKS):
        level = max(level, 3)
    if runtime_pass and (build_pass or task_class not in _CODE_TASKS):
        level = max(level, 4)
    if acceptance_pass and (test_pass or runtime_pass or task_class not in _CODE_TASKS):
        level = 5

    successful = str(outcome).upper() == "SUCCESS"
    memory_eligible = (successful and level >= 2) or ((not successful) and verified_failure)
    strong_skill_eligible = successful and level >= 3
    remote_eligible = (successful and level >= 3) or ((not successful) and verified_failure and level >= 1)

    diversity = len({str(item.get("kind")) for item in chain if item.get("status") == "PASS"})
    confidence = min(1.0, 0.12 * level + 0.05 * min(diversity, 5) + (0.12 if reproduced else 0.0))
    if verified_failure and not successful:
        confidence = max(confidence, min(0.95, 0.35 + 0.08 * diversity))

    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "level": f"L{level}",
        "level_index": level,
        "confidence": round(confidence, 4),
        "memory_eligible": memory_eligible,
        "strong_skill_eligible": strong_skill_eligible,
        "remote_eligible": remote_eligible,
        "verified_failure": verified_failure,
        "reproduced": reproduced,
        "checks": {
            "static": static_pass,
            "build": build_pass,
            "tests": test_pass,
            "runtime": runtime_pass,
            "acceptance": acceptance_pass,
        },
        "verifier_chain": chain[:48],
        "error_present": bool(str(error).strip()),
        "policy": "model completion claims never count as verification evidence",
    }


def record_memory_eligible(row: Mapping[str, Any]) -> bool:
    verification = row.get("verification")
    return isinstance(verification, Mapping) and verification.get("memory_eligible") is True


def record_strong_skill_eligible(row: Mapping[str, Any]) -> bool:
    verification = row.get("verification")
    return isinstance(verification, Mapping) and verification.get("strong_skill_eligible") is True


def record_remote_eligible(row: Mapping[str, Any]) -> bool:
    verification = row.get("verification")
    return isinstance(verification, Mapping) and verification.get("remote_eligible") is True


__all__ = [
    "REMOTE_FORMAT_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "classify_verification",
    "record_memory_eligible",
    "record_remote_eligible",
    "record_strong_skill_eligible",
]
