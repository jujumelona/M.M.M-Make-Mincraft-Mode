import inspect

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator


def test_generation_ownership_lease_does_not_cap_node_execution_lifetime() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)
    assert 'node_deadlines' not in source
    assert 'Pipeline generation lease deadline exceeded' not in source
    assert 'args = (process_node, node)' in source
    assert 'timeout=heartbeat_seconds' in source


def test_generation_scheduler_still_uses_renewable_ownership_lease() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)
    assert "lease_seconds = 900" in source
    assert "ledger.claim_ready(worker_id='mmm-orchestrator'" in source
    assert 'lease_seconds=lease_seconds' in source
