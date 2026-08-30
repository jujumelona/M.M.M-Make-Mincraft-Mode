from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import minecraft_domain_correctness_contract as contract


class _Runtime:
    class AgentToolRuntimeError(RuntimeError):
        pass

    @staticmethod
    def _discover_model_project_root(workspace_root):
        return Path(workspace_root) / "demo", "demo"


class _Extended:
    _SUPPORTED = frozenset({"item", "block", "recipe"})


def _adapter(*kinds: str):
    return SimpleNamespace(
        adapter_id="fabric_live_test",
        loader="fabric",
        minecraft_version="26.2",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(kinds),
    )


def test_live_target_without_reviewed_templates_fails_before_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "_adapter_from_project", lambda _root: _adapter())

    with pytest.raises(
        _Runtime.AgentToolRuntimeError,
        match="no reviewed deterministic module templates",
    ) as raised:
        contract._guard_target_capability(
            _Runtime,
            _Extended,
            tmp_path,
            {"modules": [{"id": "copper_hammer", "kind": "item"}]},
        )

    assert "No files were changed" in str(raised.value)
    assert "fabric_live_ai" in str(raised.value)


def test_target_specific_module_allowlist_rejects_unreviewed_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "_adapter_from_project",
        lambda _root: _adapter("item"),
    )

    with pytest.raises(
        _Runtime.AgentToolRuntimeError,
        match="requested module kinds are not reviewed for this target: block",
    ):
        contract._guard_target_capability(
            _Runtime,
            _Extended,
            tmp_path,
            {
                "modules": [
                    {"id": "copper_hammer", "kind": "item"},
                    {"id": "copper_block", "kind": "block"},
                ]
            },
        )


def test_target_specific_module_allowlist_accepts_reviewed_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "_adapter_from_project",
        lambda _root: _adapter("item", "recipe"),
    )

    contract._guard_target_capability(
        _Runtime,
        _Extended,
        tmp_path,
        {
            "modules": [
                {"id": "copper_hammer", "kind": "item"},
                {"id": "copper_recipe", "kind": "recipe"},
            ]
        },
    )


def test_install_guards_existing_execute_without_replacing_its_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class _Contract:
        @staticmethod
        def _execute(runtime_module, extended_module, workspace_root, payload):
            calls.append(dict(payload))
            return {"status": "GENERATED"}

    original = _Contract._execute
    contract.install(_Contract)
    guarded = _Contract._execute
    assert guarded.__wrapped__ is original

    monkeypatch.setattr(contract, "_adapter_from_project", lambda _root: _adapter())
    with pytest.raises(_Runtime.AgentToolRuntimeError):
        guarded(
            _Runtime,
            _Extended,
            tmp_path,
            {"modules": [{"id": "copper_hammer", "kind": "item"}]},
        )
    assert calls == []

    monkeypatch.setattr(
        contract,
        "_adapter_from_project",
        lambda _root: _adapter("item"),
    )
    result = guarded(
        _Runtime,
        _Extended,
        tmp_path,
        {"modules": [{"id": "copper_hammer", "kind": "item"}]},
    )
    assert result == {"status": "GENERATED"}
    assert calls == [{"modules": [{"id": "copper_hammer", "kind": "item"}]}]
