from __future__ import annotations

"""Bind execution observations to validation feedback and re-run only impacted work.

The durable work graph already knows how to invalidate a node plus its dependents.  The
missing link was semantic ownership: validation failures were not mapped back to the
smallest generation shard that produced the failing path, so callers either repaired
files out-of-band or had to replay a broad plan.

This contract makes that link explicit:
* every generation receipt yields an observation, even when it is not LLM-generated;
* batched receipts retain *all* module owners instead of the historical zip(first N)
  projection;
* host diagnostics are matched only against observed touched paths (fail closed when
  ownership is unknown);
* the ledger invalidates the owning generation shard and its graph dependents only;
* execution may resume in the same approved plan while unaffected succeeded receipts
  remain valid.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

from .compiler_diagnostics import (
    compiler_log_diagnostics as _compiler_log_diagnostics,
)
from .root_cause_trace import emit_root_cause, trace_scope

_SCHEMA = "mmm/execution-feedback-replan-v1"
_PATH_TOKEN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\s:'\"<>|]*?(?:src[/\\][^\s:'\"<>|]+|[A-Za-z0-9_.-]+\.(?:java|json|kt|kts|gradle|mcmeta|png|ogg)))"
)
_JAVAC_DIAGNOSTIC = re.compile(
    r"^[ \t]*(?P<path>(?:[A-Za-z]:)?[^:\r\n]+\.java):(?P<line>\d+):\s*"
    r"(?P<kind>error|warning):\s*(?P<message>[^\r\n]*)$",
    re.MULTILINE,
)
_BUILD_LOG_WINDOW_BYTES = 256 * 1024
_BASE_EXACT_SUFFIXES = (
    "build.gradle",
    "build.gradle.kts",
    "gradle.properties",
    "settings.gradle",
    "settings.gradle.kts",
    "src/main/resources/fabric.mod.json",
    "src/main/resources/pack.mcmeta",
)
_VERIFIER_INFRASTRUCTURE_CODES = frozenset(
    {"JDT_DIAGNOSTICS_UNAVAILABLE", "VERIFIER_UNAVAILABLE"}
)
_RELEASE_NOT_FOUND = re.compile(
    r"\brelease\s+(?P<major>\d+)\s+is\s+not\s+found\s+in\s+the\s+system\b",
    re.IGNORECASE,
)
_VERIFIER_UNAVAILABLE_TEXT = re.compile(
    r"\b(?:jdt(?:\s+diagnostics)?|verifier)\b[^\n]{0,160}\bunavailable\b",
    re.IGNORECASE,
)


def _sha(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _norm_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    if text.startswith("file://"):
        text = text[7:]
    return text.rstrip("/")


def _path_equivalent(left: str, right: str) -> bool:
    a = _norm_path(left).casefold()
    b = _norm_path(right).casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    # Absolute/relative representations of the same project path are common in JDT
    # and Gradle.  Suffix matching is allowed only across a path separator; basename-
    # only matching would incorrectly merge same-named files from different modules.
    return a.endswith("/" + b) or b.endswith("/" + a)


def _host_base_owned_path(value: Any) -> bool:
    path = _norm_path(value).casefold()
    if not path:
        return False
    for suffix in _BASE_EXACT_SUFFIXES:
        folded = suffix.casefold()
        if path == folded or path.endswith("/" + folded):
            return True
    marker = "/src/main/resources/assets/"
    candidate = "/" + path.lstrip("/")
    if marker not in candidate:
        return False
    tail = candidate.split(marker, 1)[1]
    parts = [part for part in tail.split("/") if part]
    return (
        len(parts) == 3
        and parts[1] == "lang"
        and parts[2] in {"en_us.json", "ko_kr.json"}
    )


def _failure_scalars(value: Any, *, depth: int = 0) -> list[tuple[str, str]]:
    if depth > 12:
        return []
    if isinstance(value, Mapping):
        result: list[tuple[str, str]] = []
        for key, child in value.items():
            key_text = str(key)
            if isinstance(child, Mapping) or (
                isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
            ):
                result.extend(_failure_scalars(child, depth=depth + 1))
            elif child is not None:
                result.append((key_text, str(child)))
        return result[:4096]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[tuple[str, str]] = []
        for child in value:
            result.extend(_failure_scalars(child, depth=depth + 1))
            if len(result) >= 4096:
                break
        return result[:4096]
    if value is None:
        return []
    return [("", str(value))]


def _verifier_infrastructure_failure(feedback: Any) -> dict[str, Any] | None:
    scalars = _failure_scalars(feedback)
    matched_code = ""
    required_java: int | None = None
    matched_text = ""
    for key, text in scalars:
        normalized_key = key.casefold().replace("-", "_")
        normalized_text = text.strip()
        upper = normalized_text.upper()
        if (
            normalized_key in {"code", "error_code", "failure_code", "reason_code"}
            and upper in _VERIFIER_INFRASTRUCTURE_CODES
        ):
            matched_code = upper
            matched_text = normalized_text
        release_match = _RELEASE_NOT_FOUND.search(normalized_text)
        if release_match is not None:
            required_java = int(release_match.group("major"))
            matched_text = normalized_text
        if not matched_code and _VERIFIER_UNAVAILABLE_TEXT.search(normalized_text):
            matched_code = "VERIFIER_UNAVAILABLE"
            matched_text = normalized_text
    if not matched_code and required_java is None:
        return None
    if not matched_code:
        matched_code = "JDT_DIAGNOSTICS_UNAVAILABLE"
    canonical = {
        "code": matched_code,
        "required_java": required_java,
        "message": matched_text,
    }
    canonical["fingerprint"] = _sha(canonical)
    return canonical


def _abort_verifier_infrastructure_retry(
    feedback: Mapping[str, Any],
    seen: set[str],
    exc: BaseException,
) -> None:
    infrastructure_failure = _verifier_infrastructure_failure(feedback)
    if infrastructure_failure is None:
        return
    fingerprint = str(infrastructure_failure["fingerprint"])
    seen.add(fingerprint)
    emit_root_cause(
        "execution_feedback_abort",
        stage="generation",
        operation="execute_with_feedback",
        gate="retry_eligibility",
        result="FAIL",
        reason=(
            "verifier infrastructure failure is not source-repairable; "
            "generation replay is forbidden"
        ),
        details={
            "feedback": feedback,
            "infrastructure_failure": infrastructure_failure,
            "seen_fingerprints": sorted(seen),
        },
    )
    raise exc


def _diagnostic_file_path(value: Any) -> str:
    """Reject verifier source labels masquerading as filesystem paths."""

    path = _norm_path(value)
    if not path:
        return ""
    folded = path.casefold()
    if "/" in path or re.search(
        r"\.(?:java|kt|kts|json|gradle|mcmeta|png|ogg)$",
        folded,
    ):
        return path
    return ""


def _collect_paths(value: Any, *, limit: int = 4096) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        if len(result) >= limit or not isinstance(raw, (str, Path)):
            return
        path = _norm_path(raw)
        if not path or path in seen:
            return
        seen.add(path)
        result.append(path)

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 12 or len(result) >= limit:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                normalized = str(key).casefold()
                if normalized in {
                    "path",
                    "uri",
                    "file",
                    "target",
                    "target_path",
                    "artifact_path",
                }:
                    add(child)
                elif normalized in {
                    "files",
                    "generated_files",
                    "written_files",
                    "touched_paths",
                } and isinstance(child, Sequence) and not isinstance(
                    child, (str, bytes, bytearray)
                ):
                    for item in child:
                        add(item)
                if isinstance(child, (Mapping, list, tuple)):
                    walk(child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, depth + 1)

    walk(value)
    return result


def _receipt_owner_ids(module: Any, receipt: Mapping[str, Any]) -> list[str]:
    """Use receipt-declared semantic ownership before positional module fallback."""

    owners: set[str] = set()
    for key in ("module_id", "entity_id"):
        raw = receipt.get(key)
        if isinstance(raw, str) and raw.strip():
            owners.add(raw.strip())
    raw_modules = receipt.get("modules")
    if isinstance(raw_modules, Sequence) and not isinstance(
        raw_modules, (str, bytes, bytearray)
    ):
        for raw in raw_modules:
            if isinstance(raw, str) and raw.strip():
                owners.add(raw.strip())
            elif isinstance(raw, Mapping):
                raw_id = str(raw.get("module_id") or "").strip()
                if raw_id:
                    owners.add(raw_id)
    if not owners:
        module_id = str(getattr(module, "module_id", "") or "").strip()
        if module_id:
            owners.add(module_id)
    return sorted(owners)


def _semantic_observation(
    module: Any,
    receipt: Mapping[str, Any],
    *,
    dependent_ids: Iterable[str] = (),
) -> dict[str, Any]:
    config = getattr(module, "config", None)
    config = config if isinstance(config, Mapping) else {}
    task = config.get("evidence_task")
    task = task if isinstance(task, Mapping) else {}
    owners = _receipt_owner_ids(module, receipt)
    touched = sorted(_collect_paths(receipt))
    core: dict[str, Any] = {
        "schema_version": "mmm/semantic-task-observation-v2",
        # task_id remains for old readers; task_ids is the authoritative multi-owner
        # field for batched deterministic generators.
        "task_id": owners[0] if owners else str(getattr(module, "module_id", "") or ""),
        "task_ids": owners,
        "task_sha256": str(task.get("task_sha256") or ""),
        "requirement_refs": sorted(
            {
                str(value)
                for value in task.get("requirement_refs", ())
                if isinstance(value, str) and value.strip()
            }
        ),
        "gap_refs": sorted(
            {
                str(value)
                for value in task.get("gap_refs", ())
                if isinstance(value, str) and value.strip()
            }
        ),
        "applied_action_count": int(receipt.get("operation_count") or 0),
        "touched_paths": touched,
        "touched_paths_sha256": _sha(touched),
        "patch_receipt": receipt.get("patch_receipt"),
        "source_observation_receipt": receipt.get("source_observation_receipt"),
        "impact_probes": list(task.get("impact_probes") or ()),
        "affected_downstream_task_ids": sorted(
            {str(value) for value in dependent_ids if str(value).strip()}
        ),
        "status": "OBSERVED",
    }
    core["observation_sha256"] = _sha(core)
    return core


def semantic_execution_observation(
    module: Any,
    receipt: dict[str, Any],
    *,
    dependent_ids: Iterable[str],
) -> dict[str, Any] | None:
    """Canonical source-owned semantic observation for generation receipts."""

    if not isinstance(receipt, Mapping):
        return None
    return _semantic_observation(module, receipt, dependent_ids=dependent_ids)


semantic_execution_observation._mmm_multi_owner_observation = True  # type: ignore[attr-defined]


def _diagnostic_severity_is_error(item: Mapping[str, Any]) -> bool:
    raw = item.get("severity")
    if raw is None:
        return True
    try:
        return int(raw) <= 2
    except (TypeError, ValueError):
        text = str(raw).strip().casefold()
        return text in {"error", "fatal", "1", "2", ""}


def _diagnostics_from_value(value: Any, *, limit: int = 256) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(item: Mapping[str, Any], inherited_path: str = "") -> None:
        if len(diagnostics) >= limit or not _diagnostic_severity_is_error(item):
            return
        path = _diagnostic_file_path(
            item.get("path")
            or item.get("uri")
            or item.get("file")
            or inherited_path
        )
        message = " ".join(
            str(item.get("message") or item.get("error") or item.get("reason") or "").split()
        )[:2000]
        code = str(item.get("code") or item.get("rule") or "").strip()[:200]
        if not path and message:
            match = _PATH_TOKEN.search(message)
            if match:
                path = _norm_path(match.group("path"))
        if not path and not message and not code:
            return
        body = {"path": path, "message": message, "code": code}
        fingerprint = _sha(body)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        diagnostics.append({**body, "diagnostic_sha256": fingerprint})

    def walk(node: Any, inherited_path: str = "", depth: int = 0) -> None:
        if depth > 14 or len(diagnostics) >= limit:
            return
        if isinstance(node, Mapping):
            local_path = _diagnostic_file_path(
                node.get("path") or node.get("uri") or node.get("file") or inherited_path
            )
            diagnostic_like = any(
                key in node
                for key in ("message", "error", "reason", "code", "severity")
            )
            if diagnostic_like:
                append(node, local_path)
            for key, child in node.items():
                if str(key).casefold() in {
                    "diagnostics",
                    "errors",
                    "issues",
                    "failures",
                    "commands",
                    "build",
                    "validation",
                    "checks",
                    "problems",
                } or isinstance(child, (Mapping, list, tuple)):
                    walk(child, local_path, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, inherited_path, depth + 1)
        elif isinstance(node, str):
            for match in _PATH_TOKEN.finditer(node[:16000]):
                append({"path": match.group("path"), "message": node[:2000]}, inherited_path)

    walk(value)
    return diagnostics


def _merge_diagnostics(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            item = dict(raw)
            fingerprint = str(item.get("diagnostic_sha256") or _sha(item))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            item["diagnostic_sha256"] = fingerprint
            merged.append(item)
    return merged


def _validation_failed(checkpoint_id: str, receipt: Mapping[str, Any]) -> bool:
    status = str(receipt.get("status") or "").strip().casefold()
    if status in {"fail", "failed", "error", "invalid", "rejected"}:
        return True
    if checkpoint_id == "validate-jdt":
        diagnostics = receipt.get("diagnostics")
        if isinstance(diagnostics, Sequence) and not isinstance(
            diagnostics, (str, bytes, bytearray)
        ):
            return any(
                isinstance(item, Mapping) and _diagnostic_severity_is_error(item)
                for item in diagnostics
            )
    if checkpoint_id == "gradle-build":
        build = receipt.get("build")
        return isinstance(build, Mapping) and str(build.get("status", "")).casefold() != "pass"
    return False


def _latest_failed_feedback(ledger: Any) -> dict[str, Any] | None:
    """Return failed validation evidence scoped to the exception being adjudicated."""

    from .execution_feedback_exception_scope_contract import (
        _active_exception,
        _checkpoint_for_exception,
    )

    checkpoint_id = _checkpoint_for_exception(_active_exception())
    if checkpoint_id is None:
        return None
    with ledger._connect() as connection:
        row = connection.execute(
            """
            SELECT checkpoint_id, input_hash, state, updated_at
            FROM checkpoints
            WHERE checkpoint_id = ? AND receipt_json IS NOT NULL
            LIMIT 1
            """,
            (checkpoint_id,),
        ).fetchone()
    if row is None:
        return None
    raw_id, input_hash, state, updated_at = row
    receipt = ledger.cached_checkpoint(
        str(raw_id),
        input_hash=str(input_hash),
    )
    if not isinstance(receipt, Mapping):
        return None
    if not _validation_failed(str(raw_id), receipt):
        return None
    diagnostics = _merge_diagnostics(
        _diagnostics_from_value(receipt),
        _compiler_log_diagnostics(receipt),
    )
    return {
        "schema_version": "mmm/execution-validation-feedback-v1",
        "checkpoint_id": str(raw_id),
        "checkpoint_state": str(state),
        "checkpoint_updated_at": float(updated_at),
        "diagnostics": diagnostics,
        "diagnostic_fingerprint": _sha(diagnostics),
        "failure_scope": "current_exception",
    }


def _generation_rows(ledger: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = ""
    while True:
        page = ledger.tasks(cursor=cursor, limit=1000)
        for task in page.get("tasks", ()):  # pragma: no branch - host-controlled page
            if not isinstance(task, Mapping):
                continue
            if not str(task.get("stage", "")).startswith("generate:"):
                continue
            rows.append(dict(task))
        cursor = str(page.get("next_cursor") or "")
        if not cursor:
            break
    return rows


def _observations(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipt = task.get("receipt")
    if not isinstance(receipt, Mapping):
        return []
    raw = receipt.get("semantic_observations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _member_ids(task: Mapping[str, Any]) -> set[str]:
    payload = task.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    members = payload.get("members")
    result: set[str] = set()
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes, bytearray)):
        for item in members:
            if not isinstance(item, Mapping):
                continue
            for key in ("module_id", "asset_id"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    result.add(raw.strip())
    return result


def _derive_impacted_seeds(
    ledger: Any, feedback: Mapping[str, Any]
) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    diagnostics = feedback.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Sequence) else ()
    generation = _generation_rows(ledger)
    seed_nodes: set[str] = set()
    owner_ids: set[str] = set()
    requirement_ids: set[str] = set()
    matched: list[dict[str, Any]] = []

    explicit_owners = {
        str(value)
        for key in ("module_ids", "task_ids")
        for value in (
            feedback.get(key)
            if isinstance(feedback.get(key), Sequence)
            and not isinstance(feedback.get(key), (str, bytes, bytearray))
            else ()
        )
        if isinstance(value, str) and value.strip()
    }
    explicit_requirements = {
        str(value)
        for value in (
            feedback.get("requirement_ids")
            if isinstance(feedback.get("requirement_ids"), Sequence)
            and not isinstance(feedback.get("requirement_ids"), (str, bytes, bytearray))
            else ()
        )
        if isinstance(value, str) and value.strip()
    }
    requirement_ids.update(explicit_requirements)

    for task in generation:
        node_id = str(task.get("node_id") or "")
        member_ids = _member_ids(task)
        task_observations = _observations(task)
        observation_owners: set[str] = set(member_ids)
        observation_requirements: set[str] = set()
        observed_paths: set[str] = set()
        for observation in task_observations:
            raw_ids = observation.get("task_ids")
            if isinstance(raw_ids, Sequence) and not isinstance(
                raw_ids, (str, bytes, bytearray)
            ):
                observation_owners.update(
                    str(value)
                    for value in raw_ids
                    if isinstance(value, str) and value.strip()
                )
            single = observation.get("task_id")
            if isinstance(single, str) and single.strip():
                observation_owners.add(single.strip())
            refs = observation.get("requirement_refs")
            if isinstance(refs, Sequence) and not isinstance(
                refs, (str, bytes, bytearray)
            ):
                observation_requirements.update(
                    str(value)
                    for value in refs
                    if isinstance(value, str) and value.strip()
                )
            paths = observation.get("touched_paths")
            if isinstance(paths, Sequence) and not isinstance(
                paths, (str, bytes, bytearray)
            ):
                observed_paths.update(
                    _norm_path(value)
                    for value in paths
                    if isinstance(value, str) and _norm_path(value)
                )

        explicit_match = bool(observation_owners & explicit_owners)
        requirement_match = bool(observation_requirements & explicit_requirements)
        path_matches: list[str] = []
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            diagnostic_path = _norm_path(diagnostic.get("path"))
            if not diagnostic_path:
                continue
            if any(_path_equivalent(diagnostic_path, path) for path in observed_paths):
                path_matches.append(diagnostic_path)
        if explicit_match or requirement_match or path_matches:
            seed_nodes.add(node_id)
            owner_ids.update(observation_owners)
            requirement_ids.update(observation_requirements)
            matched.append(
                {
                    "node_id": node_id,
                    "owner_ids": sorted(observation_owners),
                    "requirement_ids": sorted(observation_requirements),
                    "diagnostic_paths": sorted(set(path_matches)),
                    "match": {
                        "explicit_owner": explicit_match,
                        "explicit_requirement": requirement_match,
                        "observed_path": bool(path_matches),
                    },
                }
            )
    already_matched: list[str] = []
    for item in matched:
        paths = item.get("diagnostic_paths")
        if isinstance(paths, Sequence) and not isinstance(
            paths, (str, bytes, bytearray)
        ):
            already_matched.extend(
                _norm_path(path) for path in paths if _norm_path(path)
            )

    base_paths: set[str] = set()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        path = _norm_path(diagnostic.get("path"))
        if not path or not _host_base_owned_path(path):
            continue
        if any(_path_equivalent(path, observed) for observed in already_matched):
            continue
        base_paths.add(path)

    if base_paths:
        seed_nodes.add("prepare-project")
        owner_ids.add("prepare-project")
        matched.append(
            {
                "node_id": "prepare-project",
                "owner_ids": ["prepare-project"],
                "requirement_ids": [],
                "diagnostic_paths": sorted(base_paths),
                "match": {
                    "explicit_owner": False,
                    "explicit_requirement": False,
                    "observed_path": False,
                    "host_base_project": True,
                },
            }
        )
    return seed_nodes, owner_ids, requirement_ids, matched


def invalidate_execution_feedback(
    ledger: Any, feedback: Mapping[str, Any]
) -> dict[str, Any]:
    seed_nodes, owner_ids, requirement_ids, matched = _derive_impacted_seeds(
        ledger, feedback
    )
    before = {
        str(task.get("node_id")): {
            "state": task.get("state"),
            "output_hash": task.get("output_hash"),
        }
        for task in _generation_rows(ledger)
    }
    diagnostics = feedback.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Sequence) else ()
    feedback_fingerprint = _sha(
        {
            "checkpoint_id": feedback.get("checkpoint_id"),
            "diagnostics": list(diagnostics),
            "seed_nodes": sorted(seed_nodes),
        }
    )

    if not seed_nodes:
        receipt = {
            "schema_version": _SCHEMA,
            "status": "GLOBAL_REPLAN_REQUIRED",
            "global_replan_required": True,
            "reason": "validation feedback could not be bound to an observed generation owner",
            "feedback_fingerprint": feedback_fingerprint,
            "diagnostic_paths": sorted(
                {
                    _norm_path(item.get("path"))
                    for item in diagnostics
                    if isinstance(item, Mapping) and _norm_path(item.get("path"))
                }
            ),
            "seed_node_ids": [],
            "impacted_node_ids": [],
            "preserved_generation_node_ids": sorted(before),
            "owner_ids": [],
            "requirement_ids": sorted(requirement_ids),
            "matches": [],
        }
        _persist_feedback_receipt(ledger, receipt)
        return receipt

    with ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        impacted = ledger._invalidate_many(connection, sorted(seed_nodes))
        connection.commit()
    impacted_set = set(impacted)
    all_rows: list[dict[str, Any]] = []
    cursor = ""
    while True:
        page = ledger.tasks(cursor=cursor, limit=1000)
        all_rows.extend(
            dict(item) for item in page.get("tasks", ()) if isinstance(item, Mapping)
        )
        cursor = str(page.get("next_cursor") or "")
        if not cursor:
            break
    preserved = sorted(
        str(item.get("node_id"))
        for item in all_rows
        if str(item.get("node_id")) not in impacted_set
        and str(item.get("state")) == "succeeded"
    )
    impacted_generation = sorted(
        node_id for node_id in impacted if node_id in before
    )
    receipt = {
        "schema_version": _SCHEMA,
        "status": "IMPACTED_SUBGRAPH_INVALIDATED",
        "global_replan_required": False,
        "feedback_fingerprint": feedback_fingerprint,
        "seed_node_ids": sorted(seed_nodes),
        "impacted_node_ids": list(impacted),
        "impacted_generation_node_ids": impacted_generation,
        "preserved_succeeded_node_ids": preserved,
        "owner_ids": sorted(owner_ids),
        "requirement_ids": sorted(requirement_ids),
        "matches": matched,
        "previous_generation_state": before,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    _persist_feedback_receipt(ledger, receipt)
    return receipt


def feedback_run_context(current_open: Any) -> Any:
    """Bind the durable ledger to the orchestrator without late method rebinding."""

    if getattr(current_open, "_mmm_feedback_context", False):
        return current_open

    @wraps(current_open)
    def open_run(self: Any, run_name: str, plan: Any, *, resume: bool):
        root, ledger, resumed = current_open(self, run_name, plan, resume=resume)
        self._mmm_feedback_run_root = root
        self._mmm_feedback_ledger = ledger
        self._mmm_feedback_plan = plan
        return root, ledger, resumed

    open_run._mmm_feedback_context = True  # type: ignore[attr-defined]
    return open_run


def execution_feedback_scoped(current_execute: Any) -> Any:
    """Run one production execution with bounded semantic feedback re-entry."""

    if getattr(current_execute, "_mmm_impacted_feedback_loop", False):
        return current_execute

    @wraps(current_execute)
    def execute_with_feedback(self: Any, *args: Any, **kwargs: Any):
        from .complete_orchestrator import CompleteExecutionOptions
        from .complete_orchestrator_support import CompleteProductionError

        seen: set[str] = set()
        call_kwargs = dict(kwargs)
        with trace_scope("complete_production"):
            emit_root_cause(
                "pipeline_boundary_start",
                stage="runtime",
                operation="complete_production",
                gate="end_to_end_execution",
                result="START",
                details={"args": args, "kwargs": kwargs},
            )
            try:
                while True:
                    try:
                        result = current_execute(self, *args, **call_kwargs)
                        break
                    except CompleteProductionError as exc:
                        emit_root_cause(
                            "execution_feedback_failure_observed",
                            stage="generation",
                            operation="execute_with_feedback",
                            gate="adjudication",
                            result="FAIL",
                            reason=f"{type(exc).__name__}: {exc}",
                            details={"seen_fingerprints": sorted(seen)},
                            exc=exc,
                        )
                        ledger = getattr(self, "_mmm_feedback_ledger", None)
                        if ledger is None or not hasattr(
                            ledger, "invalidate_execution_feedback"
                        ):
                            raise
                        feedback = _latest_failed_feedback(ledger)
                        if not isinstance(feedback, Mapping):
                            raise

                        _abort_verifier_infrastructure_retry(feedback, seen, exc)
                        receipt = ledger.invalidate_execution_feedback(feedback)
                        fingerprint = str(receipt.get("feedback_fingerprint") or "")
                        emit_root_cause(
                            "execution_feedback_adjudicated",
                            stage="generation",
                            operation="execute_with_feedback",
                            gate="impact_analysis",
                            result="PASS",
                            details={
                                "feedback": feedback,
                                "invalidation_receipt": receipt,
                                "fingerprint": fingerprint,
                            },
                        )
                        if (
                            receipt.get("global_replan_required") is True
                            or not receipt.get("impacted_generation_node_ids")
                            or not fingerprint
                            or fingerprint in seen
                        ):
                            emit_root_cause(
                                "execution_feedback_abort",
                                stage="generation",
                                operation="execute_with_feedback",
                                gate="retry_eligibility",
                                result="FAIL",
                                reason="feedback cannot produce a novel owner-bound retry",
                                details={
                                    "receipt": receipt,
                                    "seen_fingerprints": sorted(seen),
                                },
                            )
                            raise
                        seen.add(fingerprint)
                        options = call_kwargs.get("options")
                        if options is None:
                            options = CompleteExecutionOptions(resume=True)
                        else:
                            options = replace(options, resume=True)
                        call_kwargs["options"] = options
                        emit_root_cause(
                            "execution_feedback_retry",
                            stage="generation",
                            operation="execute_with_feedback",
                            gate="retry_eligibility",
                            result="START",
                            reason="novel impacted nodes invalidated",
                            details={
                                "fingerprint": fingerprint,
                                "options": options,
                                "seen_fingerprint_count": len(seen),
                            },
                        )
            except BaseException as exc:
                emit_root_cause(
                    "pipeline_boundary_failure",
                    stage="runtime",
                    operation="complete_production",
                    gate="end_to_end_execution",
                    result="FAIL",
                    reason=f"{type(exc).__name__}: {exc}",
                    exc=exc,
                )
                raise
            emit_root_cause(
                "pipeline_boundary_result",
                stage="runtime",
                operation="complete_production",
                gate="end_to_end_execution",
                result="PASS",
                details={"result": result},
            )
            return result

    execute_with_feedback._mmm_impacted_feedback_loop = True  # type: ignore[attr-defined]
    execute_with_feedback._mmm_semantic_convergence = True  # type: ignore[attr-defined]
    return execute_with_feedback


def _persist_feedback_receipt(ledger: Any, receipt: Mapping[str, Any]) -> None:
    target = Path(ledger.path).resolve().parent / "execution-feedback-replan.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                dict(receipt),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        )


__all__ = [
    "_derive_impacted_seeds",
    "_diagnostics_from_value",
    "_path_equivalent",
    "_verifier_infrastructure_failure",
    "execution_feedback_scoped",
    "feedback_run_context",
    "invalidate_execution_feedback",
    "semantic_execution_observation",
]
