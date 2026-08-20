from __future__ import annotations

"""Reviewed causal contracts for model-facing MMM tools.

Known MMM tools never derive causal semantics from their natural-language
description. Each reviewed name has host-owned preconditions/effects. Unknown
plugin tools are deliberately opaque and therefore cannot satisfy critical goals
until a reviewed transition is added.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionSpec:
    preconditions: frozenset[str]
    effects: frozenset[str]
    cost: int = 1


def _spec(pre: tuple[str, ...] = ("workspace_bound",), effects: tuple[str, ...] = ("generic_observation",), cost: int = 1) -> TransitionSpec:
    return TransitionSpec(frozenset(pre), frozenset(effects), cost)


TRANSITIONS: dict[str, TransitionSpec] = {
    "discover_mmm_capabilities": _spec(effects=("capabilities_observed",)),
    "inspect_existing_mod": _spec(effects=("project_observed",)),
    "work_status": _spec(effects=("work_observed",)),
    "work_tasks": _spec(effects=("work_observed",)),
    "quality_status": _spec(effects=("quality_observed",)),
    "inspect_modrinth_project": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "inspect_github_repository": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "inspect_huggingface_model": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "discover_ecosystem_resources": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "build_technology_radar": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "assess_technology_compatibility": _spec(effects=("ecosystem_evidence", "evidence_ready")),
    "search_project_rag": _spec(effects=("project_evidence", "evidence_ready")),
    "search_code_rag": _spec(effects=("project_observed", "code_evidence", "evidence_ready")),
    "index_project_rag": _spec(effects=("rag_index_ready",)),
    "java_workspace_symbols": _spec(effects=("project_observed", "code_evidence", "evidence_ready")),
    "plan_game": _spec(effects=("plan_ready",)),
    "plan_complete_game": _spec(effects=("plan_ready",)),
    "revise_complete_plan": _spec(pre=("plan_ready",), effects=("plan_ready",)),
    "revise_plan": _spec(pre=("plan_ready",), effects=("plan_ready",)),
    "approve_plan": _spec(pre=("plan_ready",), effects=("plan_approved",)),
    "approve_complete_plan": _spec(pre=("plan_ready",), effects=("plan_approved",)),
    "read_complete_plan_section": _spec(pre=("plan_ready",), effects=("plan_observed",)),
    "read_quality_contract": _spec(effects=("quality_contract",)),
    "execute_complete_project": _spec(pre=("plan_approved",), effects=("project_changed", "source_generated", "generated"), cost=2),
    "generate_fabric_project": _spec(pre=("evidence_ready",), effects=("project_changed", "source_generated", "generated"), cost=2),
    "generate_assets": _spec(pre=("project_observed",), effects=("project_changed", "assets_generated", "generated"), cost=2),
    "generate_geckolib_entity": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "source_generated", "generated"), cost=2),
    "generate_system_plugin": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "source_generated", "generated"), cost=2),
    "apply_minecraft_content_spec": _spec(pre=("project_observed",), effects=("project_changed", "source_generated", "generated"), cost=1),
    "apply_source_edit": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "repaired"), cost=1),
    "apply_source_patch": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "repaired"), cost=1),
    "apply_java_operations": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "repaired"), cost=2),
    "repair_project": _spec(pre=("project_observed", "evidence_ready"), effects=("project_changed", "repaired"), cost=2),
    "work_cancel_run": _spec(pre=("work_observed",), effects=("work_changed",), cost=2),
    "work_resume_run": _spec(pre=("work_observed",), effects=("work_changed",), cost=2),
    "java_diagnostics": _spec(pre=("project_observed",), effects=("static_verified", "verified"), cost=1),
    "jdt_diagnostics": _spec(pre=("project_observed",), effects=("static_verified", "verified"), cost=1),
    "run_static_validation": _spec(pre=("project_observed",), effects=("static_verified", "verified"), cost=1),
    "run_gradle_build": _spec(pre=("project_observed",), effects=("build_verified", "verified"), cost=2),
    "gradle_build": _spec(pre=("project_observed",), effects=("build_verified", "verified"), cost=2),
    "run_gametest": _spec(pre=("build_verified",), effects=("gametest_verified", "verified"), cost=2),
    "smoke_test": _spec(pre=("build_verified",), effects=("test_verified", "verified"), cost=2),
    "validate_geometry": _spec(pre=("project_observed",), effects=("geometry_verified", "verified"), cost=1),
    "blockbench_list_tools": _spec(effects=("blockbench_capabilities",)),
    "blockbench_execute": _spec(pre=("blockbench_capabilities",), effects=("asset_observation",)),
    "validate_quality_evidence": _spec(pre=("quality_contract",), effects=("quality_verified", "verified"), cost=2),
    "quality_contract": _spec(effects=("quality_contract",)),
    "runtime_assertions": _spec(pre=("runtime_observed",), effects=("runtime_verified", "verified"), cost=2),
    "runtime_quality_mcp_probe": _spec(pre=("runtime_observed",), effects=("runtime_verified", "verified"), cost=2),
    "benchmark_resolution_probe": _spec(pre=("project_observed",), effects=("benchmark_verified", "verified"), cost=2),
    "runtime_prepare_instance": _spec(pre=("build_verified",), effects=("runtime_prepared",), cost=2),
    "runtime_start_server": _spec(pre=("runtime_prepared",), effects=("server_started",), cost=2),
    "gradle_run_server": _spec(pre=("build_verified",), effects=("server_started",), cost=2),
    "runtime_start_client": _spec(pre=("runtime_prepared",), effects=("client_started",), cost=2),
    "gradle_run_client": _spec(pre=("build_verified",), effects=("client_started",), cost=2),
    "runtime_send_command": _spec(pre=("server_started",), effects=("runtime_observed",), cost=1),
    "runtime_logs": _spec(pre=("server_started",), effects=("runtime_observed",), cost=1),
    "runtime_register_screenshot": _spec(pre=("client_started",), effects=("runtime_observed",), cost=1),
    "runtime_status": _spec(pre=("runtime_prepared",), effects=("runtime_observed",), cost=1),
    "runtime_playtest": _spec(pre=("server_started",), effects=("runtime_observed", "runtime_verified", "verified"), cost=2),
    "mineflayer_connect": _spec(pre=("server_started",), effects=("mineflayer_connected",), cost=1),
    "mineflayer_status": _spec(pre=("mineflayer_connected",), effects=("runtime_observed",), cost=1),
    "mineflayer_walk_to": _spec(pre=("mineflayer_connected",), effects=("runtime_observed",), cost=1),
    "mineflayer_interact_block": _spec(pre=("mineflayer_connected",), effects=("runtime_observed",), cost=1),
    "mineflayer_inventory": _spec(pre=("mineflayer_connected",), effects=("runtime_observed",), cost=1),
    "mineflayer_playtest": _spec(pre=("mineflayer_connected",), effects=("runtime_observed", "runtime_verified", "verified"), cost=2),
    "mineflayer_disconnect": _spec(pre=("mineflayer_connected",), effects=("runtime_stopped",), cost=1),
    "runtime_stop": _spec(pre=("runtime_prepared",), effects=("runtime_stopped",), cost=1),
    "inspect_jar": _spec(pre=("build_verified",), effects=("artifact_observed",)),
    "package_release": _spec(pre=("build_verified",), effects=("packaged",), cost=2),
    "package_mod": _spec(pre=("build_verified",), effects=("packaged",), cost=2),
    "external_mcp_capabilities": _spec(effects=("external_capabilities",)),
    "external_mcp_schema": _spec(effects=("external_schema",)),
    "external_mcp_call": _spec(effects=("external_observation", "evidence_ready"), cost=2),
    "run_model_smoke": _spec(effects=("model_smoke_observed",)),
    "record_training_trace": _spec(effects=("training_trace_recorded",), cost=2),
    "export_training_dataset": _spec(pre=("training_trace_recorded",), effects=("training_dataset_exported",), cost=2),
    "model_info": _spec(effects=("model_observed",)),
    "router_status": _spec(effects=("model_observed",)),
    "download_model": _spec(effects=("model_ready",), cost=2),
}


CRITICAL_EFFECTS = frozenset({
    "project_changed",
    "repaired",
    "generated",
    "verified",
    "runtime_verified",
    "quality_verified",
    "packaged",
    "external_observation",
})


def reviewed_transition(name: str) -> TransitionSpec | None:
    return TRANSITIONS.get(name)


def opaque_transition(name: str) -> TransitionSpec:
    return TransitionSpec(
        preconditions=frozenset({"workspace_bound"}),
        effects=frozenset({f"opaque:{name}" if name else "opaque:unknown"}),
        cost=4,
    )


__all__ = ["CRITICAL_EFFECTS", "TRANSITIONS", "TransitionSpec", "opaque_transition", "reviewed_transition"]
