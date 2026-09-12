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


def test_scheduler_installer_replaces_pre_versioned_claimant() -> None:
    original = work_graph_module.DurableWorkLedger.claim_ready

    def legacy_claim(self, worker_id, *, stages=(), lease_seconds=900):
        return original(
            self,
            worker_id,
            stages=stages,
            lease_seconds=lease_seconds,
        )

    legacy_claim._mmm_parallel_lane_claim = True
    work_graph_module.DurableWorkLedger.claim_ready = legacy_claim
    try:
        scheduler_contract.install(
            work_graph_module=work_graph_module,
            orchestrator_module=orchestrator_module,
        )
        claimant = work_graph_module.DurableWorkLedger.claim_ready
        assert claimant is not legacy_claim
        assert getattr(claimant, "_mmm_parallel_lane_claim_version", 0) >= 2
        assert getattr(claimant, "_mmm_stage_lock_admission", False)
    finally:
        # Reinstall the canonical claimant rather than restoring the intentionally stale
        # fixture so this test cannot poison later tests in the same pytest process.
        scheduler_contract.install(
            work_graph_module=work_graph_module,
            orchestrator_module=orchestrator_module,
        )
