"""Requirement-by-requirement construction receipts.

Source publication, build success and runtime behavior are different evidence
levels. Never mark a feature complete from a class name, JAR or build alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .model_concurrency import model_execution_publication


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _coverage_path(project_root, requirement_id):
    root = Path(project_root).resolve()
    return root / ".mmm" / "construction_coverage" / (
        _digest(requirement_id.encode()) + ".json")


def _authored_source_path(project_root, target_path):
    root = Path(project_root).resolve()
    return root / ".mmm" / "construction_authorship" / (
        _digest(target_path.encode()) + ".json")


def record_authored_source_calls(project_root, target_path, source, *,
                                 source_kind="typed"):
    """Record actual source authorship without claiming target build/runtime success."""
    if source_kind not in {"typed", "direct"}:
        raise ValueError("CONSTRUCTION_SOURCE_KIND_INVALID")
    output = {
        "schema_version": "mmm/construction-authorship-v2",
        "target_path": target_path,
        "source_sha256": _digest(source.encode("utf-8")),
        "status": ("DIRECT_SOURCE_RENDERED_RUNTIME_UNVERIFIED" if source_kind == "direct"
                   else "TYPED_SOURCE_RENDERED_RUNTIME_UNVERIFIED"),
    }
    path = _authored_source_path(project_root, target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) == output:
        return str(path)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(output, ensure_ascii=False,
                             sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with model_execution_publication():
            temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(path)


def record_constructed_requirement(project_root, spec, source, *, materialized_sha256):
    bindings = spec.get("bindings") or {}
    raw = bindings.get("construction_behaviors_json")
    if not raw:
        return None
    behaviors = json.loads(raw)
    parent = str(bindings.get("parent_requirement_id") or "")
    if not parent or not behaviors:
        raise ValueError("CONSTRUCTION_COVERAGE_REQUIREMENT_MISSING")
    source_sha = _digest(source.encode("utf-8"))
    if materialized_sha256 != source_sha:
        raise ValueError("CONSTRUCTION_COVERAGE_OUTPUT_HASH_MISMATCH")
    found_ids = set()
    children = []
    for behavior in behaviors:
        bid = str(behavior.get("behavior_id") or "")
        if not bid or bid in found_ids or behavior.get("requirement_id") != parent:
            raise ValueError("CONSTRUCTION_COVERAGE_BEHAVIOR_MISMATCH")
        found_ids.add(bid)
        registration = "registerBehavior" + hashlib.sha256(bid.encode()).hexdigest()[:20]
        if (registration + "()") not in source or ("this." + registration + "();") not in source:
            raise ValueError("CONSTRUCTION_COVERAGE_REGISTRATION_MISSING: " + bid)
        children.append({"behavior_id": bid, "intent": behavior["intent"],
                         "trigger": behavior.get("trigger", ""),
                         "effects": behavior.get("effects", ""),
                         "source_status": behavior.get("source_status", "grounded"),
                         "registration": registration, "stage": "SOURCE_MATERIALIZED",
                         "runtime_status": "NOT_OBSERVED", "verification_ids": [
                             bid + ".compile", bid + ".behavior"]})
    task_rows = json.loads(bindings.get("construction_tasks_json") or "[]")
    if not isinstance(task_rows, list) or not task_rows:
        raise ValueError("CONSTRUCTION_WORK_GRAPH_MISSING")
    task_ids = [row["task_id"] for row in task_rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("CONSTRUCTION_WORK_GRAPH_DUPLICATE_TASK")
    if {child["behavior_id"] + ".logic" for child in children} - set(task_ids):
        raise ValueError("CONSTRUCTION_WORK_GRAPH_BEHAVIOR_GAP")
    if {child["behavior_id"] + ".verify" for child in children} - set(task_ids):
        raise ValueError("CONSTRUCTION_WORK_GRAPH_VERIFICATION_GAP")
    task_set = set(task_ids)
    if any(set(row["depends_on"]) - task_set for row in task_rows):
        raise ValueError("CONSTRUCTION_WORK_GRAPH_UNBOUND_DEPENDENCY")
    task_receipts = []
    for task in task_rows:
        kind = task["kind"]
        record = {"task_id": task["task_id"], "kind": kind,
                  "behavior_id": task.get("behavior_id", ""),
                  "depends_on": task["depends_on"],
                  "verification_ids": task["verification_ids"]}
        if kind in {"logic", "glue"}:
            record["stage"] = "SOURCE_MATERIALIZED_RUNTIME_UNVERIFIED"
        elif kind == "verify":
            record["stage"] = "RUNTIME_UNVERIFIED"
        else:
            record["stage"] = "PLANNED"
        task_receipts.append(record)
    authorship_file = _authored_source_path(project_root, spec["target_path"])
    authorship = None
    if authorship_file.is_file():
        candidate = json.loads(authorship_file.read_text(encoding="utf-8"))
        if (candidate.get("schema_version") == "mmm/construction-authorship-v2"
            and candidate.get("source_sha256") == source_sha
            and candidate.get("target_path") == spec["target_path"]):
            authorship = candidate
    receipt = {"schema_version": "mmm/construction-coverage-v1",
               "requirement_id": parent, "target_path": spec["target_path"],
               "source_sha256": source_sha,
               "status": "RUNTIME_UNVERIFIED", "behaviors": children,
               "tasks": task_receipts,
               "authorship": authorship}
    path = _coverage_path(project_root, parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old == receipt:
            return str(path)
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(receipt, ensure_ascii=False,
                        sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with model_execution_publication():
            temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return str(path)


def requirement_completion(receipt, runtime_evidence):
    """Evaluate observed trigger/effect/state attestations for every child.

    This reads external evidence; it never manufactures a successful playtest.
    Each observation must identify the exact source and child behavior.
    """
    source_sha = receipt["source_sha256"]
    expected = {row["behavior_id"] for row in receipt["behaviors"]}
    successes = set()
    source_unresolved = sorted(
        row["behavior_id"] for row in receipt["behaviors"]
        if row.get("source_status", "grounded") != "grounded"
    )
    for evidence in runtime_evidence:
        if evidence.get("behavior_id") not in expected:
            continue
        if evidence.get("source_sha256") != source_sha:
            continue
        if (evidence.get("status") == "PASS"
            and evidence.get("trigger_observed") is True
            and evidence.get("effect_observed") is True
            and evidence.get("state_observed") is True):
            successes.add(evidence["behavior_id"])
    return {"requirement_id": receipt["requirement_id"],
            "status": ("SOURCE_REVIEW_REQUIRED" if source_unresolved else
                       "VERIFIED" if expected == successes and expected else
                       "RUNTIME_UNVERIFIED"),
            "verified_behavior_ids": sorted(successes),
            "source_review_required_ids": source_unresolved,
            "missing_behavior_ids": sorted(expected - successes)}


def summarize_construction_coverage(project_root, request_scope, runtime_receipts=()):
    """Aggregate exact per-child evidence from immutable source hashes.

    No aggregate build/JAR/runtime PASS is converted into a feature-specific
    assertion. Only explicit child trigger/effect/state observations qualify.
    """
    requirements = request_scope.get("requirements", ())
    behaviors = request_scope.get("behavior_requirements", ())
    expected = {}
    for behavior in behaviors:
        parent, bid = behavior["requirement_id"], behavior["behavior_id"]
        expected.setdefault(parent, set())
        if bid in expected[parent]:
            raise ValueError("CONSTRUCTION_DUPLICATE_BEHAVIOR: " + bid)
        expected[parent].add(bid)
    if not requirements:
        return {"schema_version": "mmm/construction-coverage-summary-v1",
                "status": "NOT_APPLICABLE", "requirements": []}
    observations = []
    for receipt in runtime_receipts:
        if not isinstance(receipt, dict):
            continue
        for row in receipt.get("construction_behavior_observations", ()):
            if isinstance(row, dict):
                observations.append(row)
    summaries = []
    delegated_ui = []
    for requirement in requirements:
        rid = requirement["requirement_id"]
        ids = expected.get(rid, set())
        path = _coverage_path(project_root, rid)
        # Interface-only features can be owned by the existing Typed/UI
        # graph. The general authored coverage gate checks those separately.
        if requirement.get("implementation_role") == "interface" and not path.exists():
            delegated_ui.append(rid)
            continue
        if not path.is_file():
            summaries.append({"requirement_id": rid,
                              "status": "SOURCE_NOT_MATERIALIZED",
                              "missing_behavior_ids": sorted(ids)})
            continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("schema_version") != "mmm/construction-coverage-v1":
            raise ValueError("CONSTRUCTION_COVERAGE_SCHEMA_MISMATCH")
        if recorded.get("requirement_id") != rid:
            raise ValueError("CONSTRUCTION_COVERAGE_PARENT_MISMATCH")
        child_ids = {x["behavior_id"] for x in recorded["behaviors"]}
        if child_ids != ids or not ids:
            raise ValueError("CONSTRUCTION_COVERAGE_SCOPE_MISMATCH: " + rid)
        target = (Path(project_root).resolve() / recorded["target_path"]).resolve()
        if not target.is_relative_to(Path(project_root).resolve()) or not target.is_file():
            summaries.append({"requirement_id": rid, "status": "SOURCE_MISSING",
                              "missing_behavior_ids": sorted(ids)})
            continue
        if _digest(target.read_bytes()) != recorded["source_sha256"]:
            summaries.append({"requirement_id": rid, "status": "SOURCE_HASH_DRIFT",
                              "missing_behavior_ids": sorted(ids)})
            continue
        matching = [row for row in observations
                    if row.get("requirement_id") == rid]
        summaries.append(requirement_completion(recorded, matching))
    success = bool(summaries) and all(row["status"] == "VERIFIED" for row in summaries)
    return {"schema_version": "mmm/construction-coverage-summary-v1",
            "status": ("NOT_APPLICABLE" if not summaries else
                       "VERIFIED" if success else "RUNTIME_UNVERIFIED"),
            "requirements": summaries,
            "delegated_ui_requirements": delegated_ui,
            "total_requirements": len(summaries),
            "verified_requirements": sum(x["status"] == "VERIFIED" for x in summaries)}
