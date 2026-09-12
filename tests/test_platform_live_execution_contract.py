from types import SimpleNamespace

from minecraft_mod_ai import platform_live_execution_contract as contract


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
