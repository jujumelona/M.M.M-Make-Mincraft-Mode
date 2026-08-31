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
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

_SCHEMA = "mmm/execution-feedback-replan-v1"
_PATH_TOKEN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\s:'\"<>|]*?(?:src[/\\][^\s:'\"<>|]+|[A-Za-z0-9_.-]+\.(?:java|json|kt|kts|gradle|mcmeta|png|ogg)))"
)
_INSTALLED = False


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
    owners: set[str] = set()
    module_id = str(getattr(module, "module_id", "") or "").strip()
    if module_id:
        owners.add(module_id)
    for key in ("module_id", "entity_id", "pack_id"):
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
        path = _norm_path(
            item.get("path")
            or item.get("uri")
            or item.get("file")
            or item.get("source")
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
            local_path = _norm_path(
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
    with ledger._connect() as connection:
        rows = connection.execute(
            """
            SELECT checkpoint_id, receipt_json, state, updated_at
            FROM checkpoints
            WHERE receipt_json IS NOT NULL
            ORDER BY updated_at DESC, checkpoint_id
            """
        ).fetchall()
    for checkpoint_id, receipt_json, state, updated_at in rows:
        try:
            receipt = json.loads(receipt_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(receipt, Mapping) or not _validation_failed(
            str(checkpoint_id), receipt
        ):
            continue
        diagnostics = _diagnostics_from_value(receipt)
        return {
            "schema_version": "mmm/execution-validation-feedback-v1",
            "checkpoint_id": str(checkpoint_id),
            "checkpoint_state": str(state),
            "checkpoint_updated_at": float(updated_at),
            "diagnostics": diagnostics,
            "diagnostic_fingerprint": _sha(diagnostics),
        }
    return None


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
    return seed_nodes, owner_ids, requirement_ids, matched


def _install_ledger_feedback(work_graph_module: Any) -> None:
    cls = work_graph_module.DurableWorkLedger
    if hasattr(cls, "invalidate_execution_feedback"):
        return

    def invalidate_execution_feedback(
        self: Any, feedback: Mapping[str, Any]
    ) -> dict[str, Any]:
        seed_nodes, owner_ids, requirement_ids, matched = _derive_impacted_seeds(
            self, feedback
        )
        before = {
            str(task.get("node_id")): {
                "state": task.get("state"),
                "output_hash": task.get("output_hash"),
            }
            for task in _generation_rows(self)
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
            _persist_feedback_receipt(self, receipt)
            return receipt

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            impacted = self._invalidate_many(connection, sorted(seed_nodes))
            connection.commit()
        impacted_set = set(impacted)
        all_rows: list[dict[str, Any]] = []
        cursor = ""
        while True:
            page = self.tasks(cursor=cursor, limit=1000)
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
        _persist_feedback_receipt(self, receipt)
        return receipt

    cls.invalidate_execution_feedback = invalidate_execution_feedback


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


def _install_observation_owner(orchestrator_module: Any) -> None:
    current = orchestrator_module._semantic_execution_observation
    if getattr(current, "_mmm_multi_owner_observation", False):
        return

    def semantic_execution_observation(
        module: Any,
        receipt: dict[str, Any],
        *,
        dependent_ids: Iterable[str],
    ) -> dict[str, Any] | None:
        if not isinstance(receipt, Mapping):
            return None
        return _semantic_observation(
            module, receipt, dependent_ids=dependent_ids
        )

    semantic_execution_observation._mmm_multi_owner_observation = True
    semantic_execution_observation.__wrapped__ = current
    orchestrator_module._semantic_execution_observation = semantic_execution_observation


def _install_run_context(orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator
    current_open = cls._open_run
    if not getattr(current_open, "_mmm_feedback_context", False):
        @wraps(current_open)
        def open_run(self: Any, run_name: str, plan: Any, *, resume: bool):
            root, ledger, resumed = current_open(self, run_name, plan, resume=resume)
            self._mmm_feedback_run_root = root
            self._mmm_feedback_ledger = ledger
            self._mmm_feedback_plan = plan
            return root, ledger, resumed

        open_run._mmm_feedback_context = True
        open_run.__wrapped__ = current_open
        cls._open_run = open_run

    current_execute = cls.execute
    if getattr(current_execute, "_mmm_impacted_feedback_loop", False):
        return

    @wraps(current_execute)
    def execute_with_feedback(self: Any, *args: Any, **kwargs: Any):
        # Termination is bounded by both semantic progress and an absolute host cap:
        # duplicate validation/ownership fingerprints never replay, and even distinct
        # new evidence gets at most two repair re-entries for one execute() call.
        seen: set[str] = set()
        repair_attempts = 0
        call_kwargs = dict(kwargs)
        while True:
            try:
                return current_execute(self, *args, **call_kwargs)
            except orchestrator_module.CompleteProductionError:
                if repair_attempts >= 2:
                    raise
                ledger = getattr(self, "_mmm_feedback_ledger", None)
                if ledger is None or not hasattr(ledger, "invalidate_execution_feedback"):
                    raise
                feedback = _latest_failed_feedback(ledger)
                if not isinstance(feedback, Mapping):
                    raise
                receipt = ledger.invalidate_execution_feedback(feedback)
                fingerprint = str(receipt.get("feedback_fingerprint") or "")
                if (
                    receipt.get("global_replan_required") is True
                    or not receipt.get("impacted_generation_node_ids")
                    or not fingerprint
                    or fingerprint in seen
                ):
                    raise
                seen.add(fingerprint)
                repair_attempts += 1
                options = call_kwargs.get("options")
                if options is None:
                    options = orchestrator_module.CompleteExecutionOptions(resume=True)
                else:
                    try:
                        options = replace(options, resume=True)
                    except TypeError:
                        raise
                call_kwargs["options"] = options
                # Re-enter the approved execution on the same durable run.  The work
                # ledger preserves unaffected succeeded nodes and exposes only the
                # invalidated generation shard plus its dependents as pending.
                continue

    execute_with_feedback._mmm_impacted_feedback_loop = True
    execute_with_feedback.__wrapped__ = current_execute
    cls.execute = execute_with_feedback


def install(*, orchestrator_module: Any, work_graph_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_ledger_feedback(work_graph_module)
    _install_observation_owner(orchestrator_module)
    _install_run_context(orchestrator_module)

    # Repair wrappers imported class methods before this late contract in some test
    # processes.  Update only direct aliases that still point at the old class method;
    # do not overwrite independently wrapped callables.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("minecraft_mod_ai.") or module is None:
            continue
        if getattr(module, "DurableWorkLedger", None) is work_graph_module.DurableWorkLedger:
            setattr(module, "DurableWorkLedger", work_graph_module.DurableWorkLedger)
    _INSTALLED = True


__all__ = [
    "_derive_impacted_seeds",
    "_diagnostics_from_value",
    "_path_equivalent",
    "install",
]
