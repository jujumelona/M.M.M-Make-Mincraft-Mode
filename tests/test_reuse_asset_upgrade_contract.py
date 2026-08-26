from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai import reuse_asset_upgrade_contract as contract


def test_joint_optimizer_propagates_evidence_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_calls: list[dict[str, Any]] = []

    def legacy_optimizer(prompt: str, **kwargs: Any) -> str:
        legacy_calls.append({"prompt": prompt, **kwargs})
        return "legacy-target"

    resolver = SimpleNamespace(_optimize=legacy_optimizer)
    contract._install_joint_platform_optimizer(resolver)
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "auto")

    def fail_joint(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("reuse evidence is incomplete")

    monkeypatch.setattr(contract, "optimize_platform_and_reuse", fail_joint)

    with pytest.raises(ValueError, match="reuse evidence is incomplete"):
        resolver._optimize(
            "add a machine",
            design={},
            module_kinds=("machine",),
        )

    assert legacy_calls == []


def test_explicit_discovery_off_keeps_host_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_calls: list[dict[str, Any]] = []

    def legacy_optimizer(prompt: str, **kwargs: Any) -> str:
        legacy_calls.append({"prompt": prompt, **kwargs})
        return "legacy-target"

    resolver = SimpleNamespace(_optimize=legacy_optimizer)
    contract._install_joint_platform_optimizer(resolver)
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")

    result = resolver._optimize(
        "add a machine",
        design={},
        module_kinds=("machine",),
        version_constraint="1.21.1",
    )

    assert result == "legacy-target"
    assert len(legacy_calls) == 1
    assert legacy_calls[0]["version_constraint"] is None


@dataclass(frozen=True)
class _Module:
    config: Mapping[str, Any]


def test_source_transplant_requires_project_local_adaptation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: list[_Module] = []

    class Generator:
        def generate(self, project_root, *, module: _Module, **kwargs: Any) -> _Module:
            captured.append(module)
            return module

    namespace = SimpleNamespace(CustomModuleGenerator=Generator)
    monkeypatch.setattr(contract, "_materialize_once", lambda *_args, **_kwargs: {"donors": []})
    contract._install_reuse_materialization(namespace)

    module = _Module(
        config={
            "_owned_reuse_plan": {
                "capabilities": [
                    {
                        "capability": "machine.menu",
                        "mode": "source_transplant",
                        "source_id": "donor:menu",
                    },
                    {
                        "capability": "machine.logic",
                        "mode": "same_project",
                        "source_id": "project:machine",
                    },
                    {
                        "capability": "machine.recipe",
                        "mode": "fresh",
                        "source_id": "fresh:machine.recipe",
                    },
                ]
            }
        }
    )

    result = Generator().generate(tmp_path, module=module)
    config = result.config

    assert config["_fresh_only_capabilities"] == ["machine.recipe"]
    assert config["_adapter_capabilities"] == ["machine.menu"]
    assert "source_transplant decision is pinned donor evidence" in config["_generation_rule"]
    assert "not a completed project implementation" in config["_generation_rule"]
    assert captured == [result]
