from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


def task_local_module_contract(
    module: Any,
    *,
    error_type: type[Exception],
    project_task: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    config = module.config if isinstance(module.config, dict) else {}
    authored = config.get("authored_plan")
    if isinstance(authored, dict):
        return {
            "module_id": module.module_id,
            "kind": module.kind,
            "authored_plan": dict(authored),
        }
    evidence_task = config.get("evidence_task")
    if not isinstance(evidence_task, dict):
        raise error_type(
            "TASK_LOCAL_CONTRACT_REQUIRED: "
            f"module {module.module_id!r} is missing config.evidence_task"
        )
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "evidence_task": project_task(evidence_task),
    }


def implementation_phase(module_contract: Mapping[str, Any]) -> str:
    return "implement_authored_design" if "authored_plan" in module_contract else "implement_module"


def apply_authored_request(request: dict[str, Any], module_contract: Mapping[str, Any]) -> None:
    if "authored_plan" not in module_contract:
        return
    request["task"] = (
        "Implement the saved authored_plan in this project. The host has already localized "
        "and frozen the exact writable target set before this coder turn. Preserve the "
        "approved authored behavior and edit only those host-bound targets. Other repository "
        "files are read-only dependency context; do not choose, create, rename, or widen "
        "the writable file set. Do not request a new plan, requirement JSON, cardinality "
        "decision, or coverage approval."
    )


def output_exhaustion_continuation_messages(
    context: Mapping[str, Any],
    *,
    task_contract: Callable[[Any], dict[str, Any]],
    continuation_path_preview: int,
) -> list[dict[str, str]]:
    module = context["module"]
    minecraft_version = str(context["minecraft_version"])
    loader = str(context["loader"])
    mappings = str(context["mappings"])
    java_version = str(context["java_version"])
    continuation_index = int(context["continuation_index"])
    state_sha256 = str(context["state_sha256"])
    touched_paths = context["touched_paths"]
    discarded_paths = context["discarded_paths"]
    source_observation_receipt = context.get("source_observation_receipt")
    host_grounding = context.get("host_grounding")

    touched = sorted({str(path) for path in touched_paths})
    discarded = sorted({str(path) for path in discarded_paths})
    request: dict[str, Any] = {
        "phase": "implement_authored_design" if "authored_plan" in module.config else "implement_module",
        "task": "Continue the approved module from the preserved staged workspace; do not restart completed work.",
        "workspace_project_root": ".",
        "target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "java": java_version,
        },
        "module": task_contract(module),
        "continuation": {
            "reason": "previous_tool_enabled_page_exhausted_output",
            "continuation_index": continuation_index,
            "preserved_source_state_sha256": state_sha256,
            "preserved_path_count": len(touched),
            "preserved_paths_preview": touched[:continuation_path_preview],
            "discarded_out_of_scope_path_count": len(discarded),
        },
        "rules": [
            "Inspect the current staged workspace before editing; correct prior edits are already persisted.",
            "Retrieve source by path/symbol/RAG only as needed; never reconstruct the whole repository in context.",
            "Use bounded tool actions and continue across tool turns until the module is complete.",
            "Do not repeat an exhausted action and do not put source code in the final summary.",
        ],
    }
    if source_observation_receipt is not None and host_grounding is not None:
        receipt = dict(source_observation_receipt)
        request["source_observation_receipt"] = receipt
        request["initial_exact_source_context"] = {
            "schema_version": "mmm/source-observation-context-v1",
            "ledger_receipt": receipt,
            "global_anchors": [],
            "page_observations": [],
        }
        request["host_grounding"] = dict(host_grounding)
    return [
        {
            "role": "system",
            "content": (
                "Continue one interrupted Minecraft mod implementation. The host preserved "
                "and hash-checked the staged workspace; resume with normal source/RAG tools."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]
