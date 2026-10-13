"""Requirement-by-requirement construction receipts.

Source publication, build success and runtime behavior are different evidence
levels. Never mark a feature complete from a class name, JAR or build alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _coverage_path(project_root, requirement_id):
    root = Path(project_root).resolve()
    return root / ".mmm" / "construction_coverage" / (
        _digest(requirement_id.encode()) + ".json")


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
                         "registration": registration, "stage": "SOURCE_MATERIALIZED",
                         "runtime_status": "NOT_OBSERVED", "verification_ids": [
                             bid + ".compile", bid + ".behavior"]})
    receipt = {"schema_version": "mmm/construction-coverage-v1",
               "requirement_id": parent, "target_path": spec["target_path"],
               "source_sha256": source_sha,
               "status": "RUNTIME_UNVERIFIED", "behaviors": children}
    path = _coverage_path(project_root, parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("source_sha256") == source_sha and old.get("behaviors") == children:
            return str(path)
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(receipt, ensure_ascii=False,
                        sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists(): temp.unlink()
    return str(path)


def requirement_completion(receipt, runtime_evidence):
    """Evaluate observed trigger/effect/state attestations for every child.

    This reads external evidence; it never manufactures a successful playtest.
    Each observation must identify the exact source and child behavior.
    """
    source_sha = receipt["source_sha256"]
    expected = {row["behavior_id"] for row in receipt["behaviors"]}
    successes = set()
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
            "status": "VERIFIED" if expected == successes and expected else "RUNTIME_UNVERIFIED",
            "verified_behavior_ids": sorted(successes),
            "missing_behavior_ids": sorted(expected - successes)}
