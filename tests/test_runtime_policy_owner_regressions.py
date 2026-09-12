from __future__ import annotations

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.llama_parallel_runtime_contract as parallel_contract
import minecraft_mod_ai.llama_server_runtime_tuning as runtime_tuning
import minecraft_mod_ai.llama_vram_parallel_policy as vram_policy
import minecraft_mod_ai.scheduler_parallel_safety_contract as scheduler_contract
import minecraft_mod_ai.work_graph as work_graph_module


def test_parallel_integration_preserves_canonical_runtime_policy_owners() -> None:
    owners = (
        runtime_tuning._explicit_parallel,
        runtime_tuning._parallel_candidates,
        runtime_tuning._parallel_resource_feasible,
        vram_policy._recommended_parallel,
        vram_policy.validated_active_parallelism,
    )

    parallel_contract._install_dynamic_runtime_parallelism(
        runtime_tuning,
        vram_policy,
    )

    assert (
        runtime_tuning._explicit_parallel,
        runtime_tuning._parallel_candidates,
        runtime_tuning._parallel_resource_feasible,
        vram_policy._recommended_parallel,
        vram_policy.validated_active_parallelism,
    ) == owners


def test_scheduler_installer_preserves_work_graph_claim_owner() -> None:
    claimant = work_graph_module.DurableWorkLedger.claim_ready

    scheduler_contract.install(
        work_graph_module=work_graph_module,
        orchestrator_module=orchestrator_module,
    )

    assert work_graph_module.DurableWorkLedger.claim_ready is claimant
    assert not hasattr(claimant, "__wrapped__")
    assert not getattr(claimant, "_mmm_parallel_lane_claim", False)
    assert callable(scheduler_contract.claim_orchestrator_ready)
