from __future__ import annotations

"""Host-compiled execution contract for a small coding agent.

The planner must finish evidence-backed implementation design before coding starts. This
module lowers one validated execution task into exact ownership, obligations, target
constraints and verification gates. Semantic task labels are never implementation steps.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .target_contract import TargetContractError, target_coordinates_from_mapping

SCHEMA = "mmm/coder-execution-contract-v2"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    data = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = ""
    return _sha(payload)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        return ()
    return tuple(dict.fromkeys(text for item in raw if (text := str(item or "").strip())))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _artifact_records(task: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = task.get("artifact_obligations")
    if not isinstance(raw, list):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _target_constraints(task: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(task.get("target_cell"))
    try:
        coordinates = target_coordinates_from_mapping(target)
    except TargetContractError as exc:
        raise ValueError(f"coder execution target contract is invalid: {exc}") from exc
    java_version = str(target.get("java_version") or target.get("java") or "").strip()
    if not java_version and coordinates.minimum_java_major is not None:
        java_version = str(coordinates.minimum_java_major)
    return {
        "minecraft_version": coordinates.minecraft_version,
        "loader": coordinates.loader,
        "mappings": coordinates.mappings,
        "mappings_applicable": coordinates.mappings_applicable,
        "naming_regime": coordinates.naming_regime,
        "java_version": java_version,
        "policy": "Use only the immutable host-selected target and compatible evidence.",
    }


def _owned_anchors(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("owned_anchors")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _anchor_target(anchor: Mapping[str, Any]) -> dict[str, str]:
    locator = str(anchor.get("locator") or "").strip().replace("\\", "/")
    path, separator, symbol = locator.partition("#")
    kind = str(anchor.get("kind") or "").strip()
    status = str(anchor.get("status") or "").strip().casefold()
    operation = "modify" if status in {"existing", "reuse", "modify", "host_existing"} else "create_or_modify"
    return {
        "kind": kind,
        "locator": locator,
        "path": path,
        "symbol": symbol if separator else "",
        "operation": operation,
        "module_id": str(anchor.get("module_id") or "").strip(),
        "source_set": str(anchor.get("source_set") or "").strip(),
    }


def _verification_plan(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    gates = _strings(task.get("required_gates"))
    public = _strings(task.get("public_acceptance"))
    runtime = _strings(task.get("runtime_acceptance"))
    acceptance = _strings(task.get("acceptance"))
    plan: list[dict[str, Any]] = [
        {
            "sequence": sequence,
            "gate": gate,
            "executor": "host_gate_runner",
            "pass_condition": "The named host gate returns PASS for this task and immutable target.",
        }
        for sequence, gate in enumerate(gates)
    ]
    if public or runtime or acceptance:
        plan.append(
            {
                "sequence": len(plan),
                "gate": "observable_acceptance",
                "executor": "host_acceptance_runner",
                "public_acceptance": list(public),
                "runtime_acceptance": list(runtime),
                "acceptance": list(acceptance),
                "pass_condition": "Every declared observable acceptance statement is proven; compile success alone is insufficient.",
            }
        )
    return plan


def _artifact_obligation_text(artifact: Mapping[str, Any]) -> str:
    return " | ".join(
        text
        for text in (
            str(artifact.get("kind") or "").strip(),
            str(artifact.get("locator") or "").strip(),
            str(artifact.get("purpose") or "").strip(),
        )
        if text
    )


def _implementation_steps(task: Mapping[str, Any], targets: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    obligations = list(_strings(task.get("implementation_obligations")))
    obligations.extend(_strings(task.get("design_resolution_obligations")))
    obligations.extend(_strings(task.get("implementation_capabilities")))
    obligations.extend(
        description
        for artifact in _artifact_records(task)
        if (description := _artifact_obligation_text(artifact))
    )
    obligations = list(dict.fromkeys(item for item in obligations if item))
    if not obligations:
        task_id = str(task.get("task_id") or "").strip()
        raise ValueError(
            f"coder execution contract {task_id!r} is semantic-only: concrete researched implementation obligations are required"
        )
    target_refs = [item["locator"] for item in targets if item.get("locator")]
    return [
        {
            "sequence": index,
            "obligation": obligation,
            "target_refs": target_refs,
            "consumes": list(_strings(task.get("consumes"))),
            "must_provide": list(_strings(task.get("provides"))),
        }
        for index, obligation in enumerate(obligations)
    ]


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("coder execution contract schema mismatch")
    task_ref = str(contract.get("task_ref") or "").strip()
    if not task_ref:
        raise ValueError("coder execution contract requires task_ref")
    target_constraints = contract.get("target_constraints")
    if not isinstance(target_constraints, Mapping):
        raise ValueError(f"coder execution contract {task_ref!r} has no target contract")
    try:
        target_coordinates_from_mapping(target_constraints)
    except TargetContractError as exc:
        raise ValueError(f"coder execution contract {task_ref!r} target is invalid: {exc}") from exc
    targets = contract.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"coder execution contract {task_ref!r} has no exact target")
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"coder target {index} is not an object")
        if not str(target.get("locator") or "").strip() or not str(target.get("path") or "").strip():
            raise ValueError(f"coder target {index} has no exact locator/path")
        if str(target.get("kind") or "") == "symbol" and not str(target.get("symbol") or "").strip():
            raise ValueError(f"coder symbol target {index} has no exact symbol")
    steps = contract.get("implementation_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"coder execution contract {task_ref!r} has no implementation steps")
    if [item.get("sequence") for item in steps if isinstance(item, Mapping)] != list(range(len(steps))):
        raise ValueError(f"coder execution contract {task_ref!r} has unstable step ordering")
    if contract.get("contract_sha256") != _hash_without(contract, "contract_sha256"):
        raise ValueError(f"coder execution contract {task_ref!r} hash mismatch")


def build_implementation_template(task: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one task into a complete, non-redesignable coder handoff."""
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("coder execution contract requires task_id")
    targets = [_anchor_target(anchor) for anchor in _owned_anchors(task)]
    targets = [target for target in targets if target["locator"] and target["path"]]
    if not targets:
        raise ValueError(f"coder execution contract {task_id!r} has no owned target anchor")

    target_paths = list(dict.fromkeys(target["path"] for target in targets))
    contract: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_ref": task_id,
        "task_sha256_input": str(task.get("task_sha256") or ""),
        "sequence": int(task.get("sequence") or 0),
        "execution_role": str(task.get("execution_role") or "").strip(),
        "semantic_outcome": str(task.get("semantic_outcome") or "").strip(),
        "requirement_refs": list(_strings(task.get("requirement_refs"))),
        "depends_on": list(_strings(task.get("depends_on"))),
        "target_constraints": _target_constraints(task),
        "targets": targets,
        "implementation_steps": _implementation_steps(task, targets),
        "dataflow": {
            "consumes": list(_strings(task.get("consumes"))),
            "provides": list(_strings(task.get("provides"))),
        },
        "artifacts": list(_artifact_records(task)),
        "reuse_refs": list(_strings(task.get("reuse_refs"))),
        "protected_boundaries": {
            "writable_paths": target_paths,
            "rule": "Do not create, edit, rename, or delete files outside writable_paths unless a later host task explicitly owns them.",
            "dependency_rule": "Do not implement this task before every depends_on task has completed and exported its declared provides.",
            "architecture_rule": "Do not change task IDs, dependency edges, target coordinates, public acceptance, or artifact ownership.",
        },
        "verification_plan": _verification_plan(task),
        "completion_predicate": {
            "operator": "all",
            "conditions": [
                "every implementation_step is realized in its owned target",
                "every declared provides value is produced",
                "every verification_plan entry passes",
                "no protected boundary is violated",
            ],
            "model_self_report_is_authoritative": False,
        },
        "contract_sha256": "",
    }
    contract["contract_sha256"] = _hash_without(contract, "contract_sha256")
    _validate_contract(contract)
    return contract


__all__ = ["SCHEMA", "build_implementation_template"]
