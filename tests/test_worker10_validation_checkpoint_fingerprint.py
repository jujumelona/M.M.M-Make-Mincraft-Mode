from __future__ import annotations

from minecraft_mod_ai import validation_checkpoint_policy as policy


def _fingerprinted_module_names(monkeypatch, checkpoint_id: str) -> set[str]:
    names: set[str] = set()

    def record(module):
        names.add(str(getattr(module, "__name__", "")))
        return "0" * 64

    monkeypatch.setattr(policy, "_file_digest", record)
    fingerprint = policy.validation_implementation_fingerprint(checkpoint_id)
    assert fingerprint.startswith("sha256:")
    return names


def test_source_validation_fingerprint_includes_active_validators(
    monkeypatch,
) -> None:
    names = _fingerprinted_module_names(monkeypatch, "validate-source")
    assert "minecraft_mod_ai.validation_checkpoint_policy" in names
    assert "minecraft_mod_ai.runtime_bootstrap" in names
    assert "minecraft_mod_ai.complete_orchestrator" in names
    assert "minecraft_mod_ai.scalable_validator" in names
    assert "minecraft_mod_ai.validator" in names
    assert "minecraft_mod_ai.scale_policy" in names
    assert "minecraft_mod_ai.platform_validation_contract" in names
    assert "minecraft_mod_ai.validator_boss_contract" not in names


def test_jdt_validation_fingerprint_includes_active_jdt_gates(
    monkeypatch,
) -> None:
    names = _fingerprinted_module_names(monkeypatch, "validate-jdt")
    assert "minecraft_mod_ai.validation_checkpoint_policy" in names
    assert "minecraft_mod_ai.runtime_bootstrap" in names
    assert "minecraft_mod_ai.complete_orchestrator" in names
    assert "minecraft_mod_ai.java_lsp" in names
    assert "minecraft_mod_ai.java_lsp_process_safety_contract" in names
    assert "minecraft_mod_ai.validation_diagnostic_contract" in names
    assert "minecraft_mod_ai.validation_execution_contract" in names
    assert "minecraft_mod_ai.research_validation_fingerprint_performance" in names
    assert "minecraft_mod_ai.orchestrator_jdt_gate_contract" not in names
