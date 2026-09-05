from __future__ import annotations

from functools import wraps

from minecraft_mod_ai.runtime_preflight import _resolves_to_canonical_callable


def _canonical(value: int) -> int:
    return value


def test_canonical_callable_is_accepted_directly() -> None:
    assert _resolves_to_canonical_callable(_canonical, _canonical)


def test_transparent_wraps_chain_preserves_canonical_authority() -> None:
    @wraps(_canonical)
    def wrapped(value: int) -> int:
        return _canonical(value)

    @wraps(wrapped)
    def twice_wrapped(value: int) -> int:
        return wrapped(value)

    assert _resolves_to_canonical_callable(twice_wrapped, _canonical)


def test_unrelated_callable_cannot_claim_canonical_authority() -> None:
    def unrelated(value: int) -> int:
        return value

    assert not _resolves_to_canonical_callable(unrelated, _canonical)


def test_malformed_wrapper_cycle_fails_closed() -> None:
    def cyclic(value: int) -> int:
        return value

    cyclic.__wrapped__ = cyclic  # type: ignore[attr-defined]
    assert not _resolves_to_canonical_callable(cyclic, _canonical)
