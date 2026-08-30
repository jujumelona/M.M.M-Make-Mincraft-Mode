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


def _spec(*kinds: str, boss: bool = False):
    return SimpleNamespace(
        platform=object(),
        contents=tuple(
            SimpleNamespace(kind=SimpleNamespace(value=kind)) for kind in kinds
        ),
        boss=object() if boss else None,
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


def test_project_generator_guard_rejects_live_target_before_root_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "_adapter_for_spec", lambda _spec: _adapter())
    root = tmp_path / "generated-project"
    calls: list[Path] = []

    class _GenerationError(RuntimeError):
        pass

    class _Generator:
        def generate(self, spec, project_root):
            calls.append(project_root)
            project_root.mkdir(parents=True)
            return "generated"

    original = _Generator.generate
    contract._install_generate_guard(_Generator, error_type=_GenerationError)

    with pytest.raises(
        _GenerationError,
        match="no reviewed deterministic module templates",
    ) as raised:
        _Generator().generate(_spec("item"), root)

    assert _Generator.generate.__wrapped__ is original
    assert "No project files were generated" in str(raised.value)
    assert calls == []
    assert not root.exists()


def test_project_generator_guard_requires_every_requested_domain_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "_adapter_for_spec",
        lambda _spec: _adapter("item", "block"),
    )

    class _GenerationError(RuntimeError):
        pass

    class _Generator:
        def generate(self, spec, project_root):
            return project_root

    contract._install_generate_guard(_Generator, error_type=_GenerationError)

    with pytest.raises(
        _GenerationError,
        match="requested module kinds are not reviewed for this target: boss",
    ):
        _Generator().generate(_spec("item", boss=True), tmp_path / "project")


def test_project_generator_guard_allows_reviewed_target_and_empty_inner_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "_adapter_for_spec",
        lambda _spec: _adapter("item", "block"),
    )
    calls: list[Path] = []

    class _GenerationError(RuntimeError):
        pass

    class _Generator:
        def generate(self, spec, project_root):
            calls.append(project_root)
            return "generated"

    contract._install_generate_guard(_Generator, error_type=_GenerationError)
    root = tmp_path / "project"

    assert _Generator().generate(_spec("item"), root) == "generated"
    assert _Generator().generate(_spec(), root) == "generated"
    assert calls == [root, root]
