from __future__ import annotations

import minecraft_mod_ai.bottleneck_elimination_contract as contract


def test_retired_bottleneck_contract_exposes_only_noop_installer() -> None:
    assert set(contract.__all__) == {"install"}
    assert not hasattr(contract, "_JsonObjectTracker")
    assert not hasattr(contract, "_READ_CACHE")
    assert not hasattr(contract, "_READ_INFLIGHT")
    assert not hasattr(contract, "_external_worker")
    assert not hasattr(contract, "_first_party_worker")


def test_retired_bottleneck_install_is_a_noop() -> None:
    assert contract.install() is None
