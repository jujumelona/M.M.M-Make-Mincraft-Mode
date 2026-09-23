from types import SimpleNamespace

import pytest

from minecraft_mod_ai import (
    complete_orchestrator,
    complete_orchestrator_services,
    llama_server_autotune,
    resource_asset_production,
    runtime_bootstrap,
)


@pytest.mark.parametrize("provider,expected_stops", [("local", 1), ("remote", 0)])
def test_canonical_asset_rebinding_preserves_exclusive_gpu_handoff(
    monkeypatch, provider, expected_stops
):
    stops = []

    class Process:
        def poll(self):
            return 0 if stops else None

        def terminate(self):
            stops.append("terminated")

        def wait(self, **_kwargs):
            return 0

    process = Process()
    monkeypatch.setattr(llama_server_autotune, "_MANAGED_PROCESS", process)
    monkeypatch.setattr(
        llama_server_autotune, "_MANAGED_URL", "http://localhost:8910/v1"
    )
    monkeypatch.setattr(llama_server_autotune, "_MANAGED_KEY", "selected")
    monkeypatch.setattr(llama_server_autotune, "_ATTEMPTED_KEYS", {"selected"})
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://localhost:8910/v1")
    config = SimpleNamespace(
        provider=provider, adapter="image_diffusion", exclusive_gpu=True
    )
    router = SimpleNamespace(
        profile="test", registry=SimpleNamespace(role=lambda *_: config)
    )

    def produce(_router, *_args, **_kwargs):
        assert len(stops) == expected_stops
        return {"status": "generated"}

    monkeypatch.setattr(resource_asset_production, "generate_assets", produce)
    # Register the alias assignments with monkeypatch so this bootstrap replay is isolated.
    monkeypatch.setattr(complete_orchestrator_services, "generate_assets", produce)
    monkeypatch.setattr(complete_orchestrator, "generate_assets", produce)
    runtime_bootstrap._install_planner_contracts()
    assert (
        complete_orchestrator_services.generate_assets
        is resource_asset_production.generate_assets
    )
    assert (
        complete_orchestrator.generate_assets
        is resource_asset_production.generate_assets
    )
    assert complete_orchestrator_services.generate_assets(router) == {
        "status": "generated"
    }
