from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai import minecraft_domain_correctness_contract
from minecraft_mod_ai import platform_specialized_generator_contract as specialized


def _adapter(*kinds: str) -> Any:
    return SimpleNamespace(
        adapter_id="fabric_live_test",
        loader="fabric",
        minecraft_version="26.2",
        deterministic_module_kinds=frozenset(kinds),
    )


def _fake_modules(calls: list[str]):
    def contract_files(*args: Any, **kwargs: Any):
        return {}, 0

    def generate_system_pack(*args: Any, **kwargs: Any):
        calls.append("system")
        project_root = Path(kwargs["project_root"])
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "system-mutated.txt").write_text("mutated", encoding="utf-8")
        return {"status": "GENERATED"}

    def generate_geckolib_entity_assets(*args: Any, **kwargs: Any):
        calls.append("gecko")
        project_root = Path(kwargs["project_root"])
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "gecko-mutated.txt").write_text("mutated", encoding="utf-8")
        return {"status": "GENERATED"}

    system = SimpleNamespace(
        _DIRECTORY_SCHEMA=1,
        _system_contract_files=contract_files,
        generate_system_pack=generate_system_pack,
    )
    gecko = SimpleNamespace(
        generate_geckolib_entity_assets=generate_geckolib_entity_assets,
    )
    orchestrator = SimpleNamespace(
        generate_system_pack=generate_system_pack,
        generate_geckolib_entity_assets=generate_geckolib_entity_assets,
    )
    return system, gecko, orchestrator


def _install_fake_runtime(
    calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
):
    system, gecko, orchestrator = _fake_modules(calls)
    monkeypatch.setattr(specialized, "adapter_from_project", lambda _root: adapter)
    # The legacy Worker-8 boundary is independently tested in
    # test_minecraft_domain_correctness_contract.py. Avoid coupling these specialized
    # unit tests to the package-global wrapper graph while still exercising this
    # contract's installation and alias-binding behavior.
    monkeypatch.setattr(minecraft_domain_correctness_contract, "install", lambda: None)
    specialized.install(
        system_module=system,
        geckolib_module=gecko,
        orchestrator_module=orchestrator,
    )
    return system, gecko, orchestrator


def test_system_pack_guard_fails_before_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    system, _gecko, orchestrator = _install_fake_runtime(
        calls,
        monkeypatch,
        _adapter(),
    )
    root = tmp_path / "system"

    assert orchestrator.generate_system_pack is system.generate_system_pack
    with pytest.raises(ValueError, match="deterministic templates are not declared"):
        orchestrator.generate_system_pack(
            project_root=root,
            pack_id="quest-system",
            mod_id="demo",
            package_name="demo.mod",
            config={},
        )

    assert calls == []
    assert not root.exists()


def test_geckolib_guard_fails_before_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _system, gecko, orchestrator = _install_fake_runtime(
        calls,
        monkeypatch,
        _adapter(),
    )
    root = tmp_path / "gecko"

    assert orchestrator.generate_geckolib_entity_assets is gecko.generate_geckolib_entity_assets
    with pytest.raises(ValueError, match="deterministic templates are not declared"):
        orchestrator.generate_geckolib_entity_assets(
            project_root=root,
            mod_id="demo",
            package_name="demo.mod",
            module_id="boss",
            config={},
        )

    assert calls == []
    assert not root.exists()


def test_unknown_system_pack_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    system, _gecko, _orchestrator = _install_fake_runtime(
        calls,
        monkeypatch,
        _adapter("quest", "entity"),
    )
    root = tmp_path / "unknown"

    with pytest.raises(ValueError, match="deterministic templates are not declared"):
        system.generate_system_pack(
            project_root=root,
            pack_id="unknown-system",
            mod_id="demo",
            package_name="demo.mod",
            config={},
        )

    assert calls == []
    assert not root.exists()


def test_reviewed_specialized_capabilities_execute_original_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    system, gecko, _orchestrator = _install_fake_runtime(
        calls,
        monkeypatch,
        _adapter("quest", "entity"),
    )

    system_root = tmp_path / "system"
    gecko_root = tmp_path / "gecko"
    assert system.generate_system_pack(
        project_root=system_root,
        pack_id="quest-system",
        mod_id="demo",
        package_name="demo.mod",
        config={},
    ) == {"status": "GENERATED"}
    assert gecko.generate_geckolib_entity_assets(
        project_root=gecko_root,
        mod_id="demo",
        package_name="demo.mod",
        module_id="boss",
        config={},
    ) == {"status": "GENERATED"}

    assert calls == ["system", "gecko"]
    assert (system_root / "system-mutated.txt").is_file()
    assert (gecko_root / "gecko-mutated.txt").is_file()
