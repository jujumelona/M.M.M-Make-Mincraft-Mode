from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai import minecraft_domain_correctness_contract as contract


class _ExtendedContentError(RuntimeError):
    pass


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


def _module(kind: str, module_id: str = "example") -> Any:
    return SimpleNamespace(kind=kind, module_id=module_id)


def _extended(original: Any) -> Any:
    return SimpleNamespace(
        _SUPPORTED=frozenset({"item", "block", "recipe"}),
        ExtendedContentError=_ExtendedContentError,
        generate_extended_content=original,
    )


def test_advertised_kinds_are_normalized_before_authorization() -> None:
    assert contract._raw_advertised_kinds(_adapter(" item ", "", "block")) == frozenset(
        {"item", "block"}
    )


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


def test_shared_extended_primitive_rejects_unreviewed_target_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "existing-project"
    calls: list[tuple[Any, ...]] = []

    def original(**kwargs: Any) -> dict[str, Any]:
        calls.append(tuple(kwargs["modules"]))
        root.mkdir(parents=True)
        return {"status": "GENERATED"}

    extended = _extended(original)
    monkeypatch.setattr(contract, "_adapter_from_project", lambda _root: _adapter())
    guarded = contract._install_extended_content_guard(extended)

    with pytest.raises(
        _ExtendedContentError,
        match="no reviewed deterministic module templates",
    ) as raised:
        guarded(
            project_root=root,
            mod_id="demo",
            package_name="demo.mod",
            modules=(_module("item"),),
        )

    assert "No files were changed" in str(raised.value)
    assert calls == []
    assert not root.exists()


def test_shared_extended_primitive_requires_every_requested_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def original(**kwargs: Any) -> dict[str, Any]:
        calls.append(tuple(kwargs["modules"]))
        return {"status": "GENERATED"}

    extended = _extended(original)
    monkeypatch.setattr(contract, "_adapter_from_project", lambda _root: _adapter("item"))
    guarded = contract._install_extended_content_guard(extended)

    with pytest.raises(
        _ExtendedContentError,
        match="requested module kinds are not reviewed for this target: block",
    ):
        guarded(
            project_root=tmp_path,
            mod_id="demo",
            package_name="demo.mod",
            modules=(_module("item", "one"), _module("block", "two")),
        )

    assert calls == []


def test_shared_extended_primitive_materializes_once_and_allows_reviewed_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[Any, ...]] = []

    def original(**kwargs: Any) -> dict[str, Any]:
        modules = kwargs["modules"]
        assert isinstance(modules, tuple)
        observed.append(modules)
        return {"status": "GENERATED"}

    extended = _extended(original)
    monkeypatch.setattr(
        contract,
        "_adapter_from_project",
        lambda _root: _adapter("item", "recipe"),
    )
    guarded = contract._install_extended_content_guard(extended)
    source = iter((_module("item", "one"), _module("recipe", "two")))

    result = guarded(
        project_root=tmp_path,
        mod_id="demo",
        package_name="demo.mod",
        modules=source,
    )

    assert result == {"status": "GENERATED"}
    assert [module.module_id for module in observed[0]] == ["one", "two"]
    assert tuple(source) == ()


def test_unsupported_only_batch_does_not_require_deterministic_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_lookups: list[Path] = []

    def original(**kwargs: Any) -> dict[str, Any]:
        return {"status": "SKIPPED", "modules": []}

    extended = _extended(original)

    def lookup(root: Path) -> Any:
        adapter_lookups.append(Path(root))
        return _adapter()

    monkeypatch.setattr(contract, "_adapter_from_project", lookup)
    guarded = contract._install_extended_content_guard(extended)

    assert guarded(
        project_root=tmp_path,
        mod_id="demo",
        package_name="demo.mod",
        modules=(_module("custom"),),
    ) == {"status": "SKIPPED", "modules": []}
    assert adapter_lookups == []


def test_stale_import_by_value_aliases_are_retargeted_across_wrapper_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raw(**kwargs: Any) -> dict[str, Any]:
        return {"status": "GENERATED", "count": len(tuple(kwargs["modules"]))}

    @wraps(raw)
    def registration_wrapper(**kwargs: Any) -> dict[str, Any]:
        return raw(**kwargs)

    holder_raw = ModuleType("minecraft_mod_ai._worker8_raw_alias")
    holder_raw.generate_extended_content = raw
    holder_middle = ModuleType("minecraft_mod_ai._worker8_middle_alias")
    holder_middle.generate_extended_content = registration_wrapper
    monkeypatch.setitem(sys.modules, holder_raw.__name__, holder_raw)
    monkeypatch.setitem(sys.modules, holder_middle.__name__, holder_middle)

    extended = _extended(registration_wrapper)
    monkeypatch.setattr(contract, "_adapter_from_project", lambda _root: _adapter("item"))
    guarded = contract._install_extended_content_guard(extended)

    assert holder_raw.generate_extended_content is guarded
    assert holder_middle.generate_extended_content is guarded
    assert guarded(
        project_root=tmp_path,
        mod_id="demo",
        package_name="demo.mod",
        modules=(_module("item"),),
    ) == {"status": "GENERATED", "count": 1}


def test_runtime_install_covers_central_and_project_boundaries_idempotently() -> None:
    from minecraft_mod_ai import complete_orchestrator, extended_content_generator
    from minecraft_mod_ai import scalable_generator
    from minecraft_mod_ai import deterministic_minecraft_content_contract
    from minecraft_mod_ai.generator import FabricProjectGenerator
    from minecraft_mod_ai.scalable_generator import ScalableFabricProjectGenerator

    content_generate = extended_content_generator.generate_extended_content
    base_generate = FabricProjectGenerator.generate
    scalable_generate = ScalableFabricProjectGenerator.generate
    execute = deterministic_minecraft_content_contract._execute

    assert getattr(content_generate, contract._MARKER, False)
    assert getattr(base_generate, contract._MARKER, False)
    assert getattr(scalable_generate, contract._MARKER, False)
    assert not getattr(execute, contract._MARKER, False)
    assert complete_orchestrator.generate_extended_content is content_generate
    assert scalable_generator.generate_extended_content is content_generate

    contract.install()

    assert extended_content_generator.generate_extended_content is content_generate
    assert FabricProjectGenerator.generate is base_generate
    assert ScalableFabricProjectGenerator.generate is scalable_generate
    assert deterministic_minecraft_content_contract._execute is execute
