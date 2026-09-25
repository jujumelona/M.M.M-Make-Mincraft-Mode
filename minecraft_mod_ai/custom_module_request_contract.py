from __future__ import annotations

"""Small request-building helpers for custom module generation."""

import json
from collections.abc import Mapping
from typing import Any, Callable


def build_task_local_module_contract(
    module: Any,
    *,
    project_task_for_coder: Callable[[Mapping[str, Any]], dict[str, Any]],
    error_type: type[Exception],
) -> dict[str, Any]:
    config = module.config if isinstance(module.config, dict) else {}
    authored = config.get("authored_plan")
    if isinstance(authored, dict):
        contract = {
            "module_id": module.module_id,
            "kind": module.kind,
            "authored_plan": dict(authored),
        }
        mode = str(config.get("authored_execution_mode") or "").strip()
        if mode:
            contract["authored_execution_mode"] = mode
        if config.get("authored_bounded_scope") is True:
            contract["authored_write_scope"] = {
                "java_package": str(config.get("authored_java_package") or "").strip(),
                "mod_id": str(config.get("authored_mod_id") or "").strip(),
                "policy": "package_and_mod_namespace_only",
            }
        return contract
    evidence_task = config.get("evidence_task")
    if not isinstance(evidence_task, dict):
        raise error_type(
            "TASK_LOCAL_CONTRACT_REQUIRED: "
            f"module {module.module_id!r} is missing config.evidence_task"
        )
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "evidence_task": project_task_for_coder(evidence_task),
    }


def build_output_exhaustion_continuation_messages(
    context: Mapping[str, Any],
) -> list[dict[str, str]]:
    touched = sorted({str(path) for path in context["touched_paths"]})
    discarded = sorted({str(path) for path in context["discarded_paths"]})
    preview = int(context["path_preview"])
    request = {
        "phase": context["phase"],
        "task": "Continue the approved module from the preserved staged workspace; do not restart completed work.",
        "workspace_project_root": ".",
        "target": {
            "minecraft_version": context["minecraft_version"],
            "loader": context["loader"],
            "mappings": context["mappings"],
            "java": context["java_version"],
        },
        "module": dict(context["module_contract"]),
        "continuation": {
            "reason": "previous_tool_enabled_page_exhausted_output",
            "continuation_index": context["continuation_index"],
            "preserved_source_state_sha256": context["state_sha256"],
            "preserved_path_count": len(touched),
            "preserved_paths_preview": touched[:preview],
            "discarded_out_of_scope_path_count": len(discarded),
        },
        "rules": [
            "Inspect the current staged workspace before editing; correct prior edits are already persisted.",
            "Retrieve source by path/symbol/RAG only as needed; never reconstruct the whole repository in context.",
            "Use bounded tool actions and continue across tool turns until the module is complete.",
            "Do not repeat an exhausted action and do not put source code in the final summary.",
        ],
    }
    source_receipt = context.get("source_observation_receipt")
    host_grounding = context.get("host_grounding")
    if source_receipt is not None and host_grounding is not None:
        receipt = dict(source_receipt)
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
            "content": json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    ]


__all__ = [
    "build_output_exhaustion_continuation_messages",
    "build_task_local_module_contract",
]
