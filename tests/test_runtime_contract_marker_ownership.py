from __future__ import annotations

from functools import wraps

import minecraft_mod_ai  # noqa: F401  # runtime composition must be installed
from minecraft_mod_ai.runtime_contract_wrappers import (
    contract_markers,
    contract_wraps,
    copied_contract_markers,
    has_contract_marker,
    owns_contract_marker,
)
from minecraft_mod_ai.runtime_wrapper_integrity import (
    iter_installed_wrappers,
    wrapped_chain,
)


def test_contract_wraps_does_not_copy_inner_ownership_markers() -> None:
    def original(value: object) -> object:
        return value

    original._mmm_inner_owner = True  # type: ignore[attr-defined]

    @contract_wraps(original)
    def outer(value: object) -> object:
        return original(value)

    outer._mmm_outer_owner = True  # type: ignore[attr-defined]

    assert owns_contract_marker(original, "_mmm_inner_owner")
    assert not owns_contract_marker(outer, "_mmm_inner_owner")
    assert owns_contract_marker(outer, "_mmm_outer_owner")
    assert has_contract_marker(outer, "_mmm_inner_owner")
    assert contract_markers(outer) == frozenset({"_mmm_outer_owner"})
    assert copied_contract_markers(outer) == frozenset()
    assert outer.__wrapped__ is original


def test_legacy_wraps_copy_is_not_mistaken_for_exact_ownership() -> None:
    def original(value: object) -> object:
        return value

    original._mmm_inner_owner = True  # type: ignore[attr-defined]

    @wraps(original)
    def legacy_outer(value: object) -> object:
        return original(value)

    assert legacy_outer.__dict__["_mmm_inner_owner"] is True
    assert has_contract_marker(legacy_outer, "_mmm_inner_owner")
    assert not owns_contract_marker(legacy_outer, "_mmm_inner_owner")
    assert owns_contract_marker(original, "_mmm_inner_owner")
    assert copied_contract_markers(legacy_outer) == frozenset({"_mmm_inner_owner"})


def test_installed_wrapper_markers_have_single_effective_owner() -> None:
    duplicates: list[str] = []
    for binding, outer in iter_installed_wrappers():
        owners: dict[str, list[int]] = {}
        for layer_index, layer in enumerate(wrapped_chain(outer)):
            for marker in contract_markers(layer):
                owners.setdefault(marker, []).append(layer_index)
        for marker, layer_indexes in owners.items():
            if len(layer_indexes) > 1:
                duplicates.append(
                    f"{binding}: {marker} has effective owners {layer_indexes}"
                )

    assert not duplicates, (
        "runtime contract markers have multiple effective owners:\n"
        + "\n".join(sorted(set(duplicates)))
    )
