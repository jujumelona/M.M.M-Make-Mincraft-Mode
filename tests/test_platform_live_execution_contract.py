import inspect
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import complete_orchestrator
from minecraft_mod_ai import platform_catalog
from minecraft_mod_ai import platform_live_execution_contract as contract
from minecraft_mod_ai import platform_validation_contract as validation_contract


def _adapter(*, loader="fabric", source_api_family="mojang", deterministic_module_kinds=()):
    return SimpleNamespace(
        loader=loader,
        source_api_family=source_api_family,
        deterministic_module_kinds=deterministic_module_kinds,
    )


def test_host_authoritative_fabric_without_reviewed_deterministic_kinds_uses_official_scaffold(monkeypatch):
    monkeypatch.setattr(
        contract,
        "provider_for_loader",
        lambda loader: SimpleNamespace(host_authoritative=True),
    )

    assert contract._uses_official_scaffold(_adapter()) is True


def test_provider_scaffold_target_uses_same_live_validation_route(monkeypatch):
    monkeypatch.setattr(
        contract,
        "provider_for_loader",
        lambda loader: SimpleNamespace(host_authoritative=True),
    )

    adapter = _adapter(
        source_api_family="mojang",
        deterministic_module_kinds=(),
    )
    assert contract._uses_official_scaffold(adapter) is True
    assert validation_contract._uses_provider_owned_scaffold(adapter) is True


def test_reviewed_deterministic_target_keeps_deterministic_generation_path(monkeypatch):
    monkeypatch.setattr(
        contract,
        "provider_for_loader",
        lambda loader: SimpleNamespace(host_authoritative=True),
    )

    assert (
        contract._uses_official_scaffold(
            _adapter(deterministic_module_kinds=("item",))
        )
        is False
    )


def test_live_discovered_fabric_target_keeps_official_scaffold_path(monkeypatch):
    monkeypatch.setattr(
        contract,
        "provider_for_loader",
        lambda loader: SimpleNamespace(host_authoritative=True),
    )

    assert (
        contract._uses_official_scaffold(
            _adapter(
                source_api_family="fabric_live_ai",
                deterministic_module_kinds=("item",),
            )
        )
        is True
    )


def test_non_fabric_target_never_uses_fabric_official_scaffold(monkeypatch):
    called = False

    def provider_for_loader(loader):
        nonlocal called
        called = True
        return SimpleNamespace(host_authoritative=True)

    monkeypatch.setattr(contract, "provider_for_loader", provider_for_loader)

    assert contract._uses_official_scaffold(_adapter(loader="forge")) is False
    assert called is False


def test_non_authoritative_provider_does_not_bypass_deterministic_path(monkeypatch):
    monkeypatch.setattr(
        contract,
        "provider_for_loader",
        lambda loader: SimpleNamespace(host_authoritative=False),
    )

    assert contract._uses_official_scaffold(_adapter()) is False


def test_unknown_fabric_provider_fails_closed(monkeypatch):
    def missing_provider(loader):
        raise ValueError(loader)

    monkeypatch.setattr(contract, "provider_for_loader", missing_provider)

    assert contract._uses_official_scaffold(_adapter()) is False


def test_real_host_fabric_1_21_8_adapter_uses_official_scaffold() -> None:
    adapter = platform_catalog.adapter_for_target("1.21.8", "fabric")

    assert tuple(adapter.deterministic_module_kinds) == ()
    assert contract._uses_official_scaffold(adapter) is True


def test_source_owned_prepare_routes_fresh_unreviewed_target_before_legacy_generator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    adapter = SimpleNamespace(
        loader="fabric",
        source_api_family="mojang",
        deterministic_module_kinds=(),
    )
    platform_lock = object()
    base = SimpleNamespace(
        spec=SimpleNamespace(platform=platform_lock, mod_id="debug_fixture")
    )
    approved = SimpleNamespace(base_proposal=base)
    expected = (tmp_path / "official-project").resolve()
    calls: list[tuple[object, Path, object, type[Exception]]] = []

    monkeypatch.setattr(
        platform_catalog,
        "adapter_for_lock_values",
        lambda value: adapter if value is platform_lock else None,
    )
    monkeypatch.setattr(contract, "_uses_official_scaffold", lambda value: value is adapter)

    def prepare_official(
        owner,
        approved_value,
        *,
        run_root,
        adapter: object,
        error_type: type[Exception],
    ):
        calls.append((approved_value, Path(run_root), adapter, error_type))
        return expected

    monkeypatch.setattr(contract, "prepare_official_fabric_project", prepare_official)

    class ForbiddenLegacyGenerator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("legacy deterministic generator must not be constructed")

    monkeypatch.setattr(
        complete_orchestrator,
        "FabricProjectGenerator",
        ForbiddenLegacyGenerator,
    )

    orchestrator = complete_orchestrator.CompleteProductionOrchestrator(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: None,
    )
    source_prepare = inspect.unwrap(
        complete_orchestrator.CompleteProductionOrchestrator._prepare_project
    )
    result = source_prepare(
        orchestrator,
        approved,
        run_root=tmp_path / "run",
        existing_input=None,
    )

    assert result == expected
    assert calls == [
        (
            approved,
            tmp_path / "run",
            adapter,
            complete_orchestrator.CompleteProductionError,
        )
    ]
