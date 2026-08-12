from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"


def _text(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_package_init_has_one_bootstrap_and_no_contract_patch_chain() -> None:
    source = _text("__init__.py")
    assert source.count("initialize_runtime()") == 1
    assert "_install_" not in source
    assert "_contract import" not in source
    assert "integrated_contract_bootstrap" not in source
    assert "platform_mcp_compatibility_contract" not in source


def test_runtime_bootstrap_is_flat_not_nested() -> None:
    source = _text("runtime_bootstrap.py")
    assert "integrated_contract_bootstrap" not in source
    assert "final_architecture_contract" not in source
    assert "platform_mcp_compatibility_contract" not in source
    assert source.count("install_scheduler_parallel_safety(") == 1
    assert source.count("install_proposal_deserialization(") == 1


def test_specialized_installers_do_not_reenter_global_bootstrap() -> None:
    tuning = _text("performance_final_tuning.py")
    assert "llama_server_efficiency_contract" not in tuning
    assert "project_manifest_hash_efficiency_contract" not in tuning
    assert "final_architecture_contract" not in tuning

    platform_mcp = _text("platform_mcp_contract.py")
    assert "def _call_supported(" in platform_mcp
    assert "platform_release_contract" in platform_mcp

    assert not (PACKAGE / "integrated_contract_bootstrap.py").exists()
    assert not (PACKAGE / "platform_mcp_compatibility_contract.py").exists()
