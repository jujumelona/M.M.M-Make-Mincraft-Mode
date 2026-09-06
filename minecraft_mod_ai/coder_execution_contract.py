from __future__ import annotations

"""Deterministic host projection from one execution task to the small coder contract."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

CODER_EXECUTION_CONTRACT_SCHEMA = "mmm/coder-execution-contract-v2"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(
        dict.fromkeys(str(item).strip() for item in values if str(item).strip())
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _anchors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def build_coder_execution_contract(task: Mapping[str, Any]) -> dict[str, Any]:
    """Project only task-local host authority required for implementation.

    No prompt, global proposal, complete planning state, or unrelated requirement data is
    accepted as an input. If detailed implementation obligations are absent, the already
    host-owned semantic outcome becomes the single execution step instead of asking the
    coder to reinterpret upstream prose.
    """

    task_ref = str(task.get("task_id") or "").strip()
    objective = str(task.get("semantic_outcome") or "").strip()
    if not task_ref:
        raise ValueError("CODER_EXECUTION_CONTRACT: task_id is required")
    if not objective:
        raise ValueError("CODER_EXECUTION_CONTRACT: semantic_outcome is required")

    obligations = _strings(task.get("implementation_obligations"))
    if not obligations:
        obligations = (objective,)

    implementation_steps = [
        {
            "step": index,
            "obligation": obligation,
        }
        for index, obligation in enumerate(obligations, start=1)
    ]
    acceptance = _strings(task.get("acceptance"))
    contract: dict[str, Any] = {
        "schema_version": CODER_EXECUTION_CONTRACT_SCHEMA,
        "task_ref": task_ref,
        "source_task_sha256": str(task.get("task_sha256") or ""),
        "objective": objective,
        "execution_role": str(task.get("execution_role") or "").strip(),
        "requirement_refs": list(_strings(task.get("requirement_refs"))),
        "target_constraints": _mapping(task.get("target_cell")),
        "owned_anchors": _anchors(task.get("owned_anchors")),
        "reuse_refs": list(_strings(task.get("reuse_refs"))),
        "depends_on": list(_strings(task.get("depends_on"))),
        "consumes": list(_strings(task.get("consumes"))),
        "provides": list(_strings(task.get("provides"))),
        "implementation_steps": implementation_steps,
        "acceptance_checks": list(acceptance),
        "required_gates": list(_strings(task.get("required_gates"))),
        "impact_probes": list(_strings(task.get("impact_probes"))),
        "scope_policy": "task_local_host_authority_only",
        "contract_sha256": "",
    }
    contract["contract_sha256"] = _sha(contract)
    return contract


def project_task_for_coder(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal evidence-task envelope consumed by the coder request."""

    contract = build_coder_execution_contract(task)
    result: dict[str, Any] = {
        "task_id": contract["task_ref"],
        "coder_execution_contract": contract,
    }
    source_sha = str(task.get("task_sha256") or "").strip()
    if source_sha:
        result["task_sha256"] = source_sha
    return result


__all__ = [
    "CODER_EXECUTION_CONTRACT_SCHEMA",
    "build_coder_execution_contract",
    "project_task_for_coder",
]
