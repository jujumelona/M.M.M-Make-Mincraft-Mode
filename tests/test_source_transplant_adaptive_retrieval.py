from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import capability_implementation_locator as locator


def _fake_locator_index(count: int):
    paths = [f"src/main/java/demo/TradeHandler{index}.java" for index in range(count)]
    return SimpleNamespace(
        registry_to_path={},
        fqcn_to_path={},
        symbol_to_paths={"TradeHandler": paths},
        method_to_paths={},
        api_call_to_paths={},
        resource_to_path={},
        files_by_path={path: {"path": path} for path in paths},
    )


def test_locator_does_not_drop_structural_seeds_after_eighth():
    found = locator.CapabilityImplementationLocator.locate_seeds(
        "trade",
        _fake_locator_index(17),
    )

    assert len(found) == 17
    assert {item.node_id for item in found} == set(_fake_locator_index(17).files_by_path)


def test_locator_explicit_operator_cap_remains_available():
    found = locator.CapabilityImplementationLocator.locate_seeds(
        "trade",
        _fake_locator_index(17),
        max_seeds=5,
    )
    assert len(found) == 5


