from __future__ import annotations

import minecraft_mod_ai.extended_content_generator as extended
import minecraft_mod_ai.extended_registration_contract as contract
from minecraft_mod_ai import generator


def test_texture_cache_preserves_exact_original_bytes() -> None:
    cached = generator.make_texture_png
    original = cached.__wrapped__
    with contract._TEXTURE_CACHE_LOCK:
        contract._TEXTURE_CACHE.clear()

    for kind in ("item", "block", "entity"):
        for size in (16, 64):
            for seed in ("a", "o", "module_001", "module_999"):
                expected = original("#74c7ec", seed, kind=kind, size=size)
                actual = cached("#74c7ec", seed, kind=kind, size=size)
                assert actual == expected


def test_many_seed_names_collapse_to_at_most_fourteen_exact_patterns() -> None:
    cached = generator.make_texture_png
    with contract._TEXTURE_CACHE_LOCK:
        contract._TEXTURE_CACHE.clear()

    outputs = [
        cached("#748cab", f"module_{index:04d}", kind="item", size=16)
        for index in range(100)
    ]

    with contract._TEXTURE_CACHE_LOCK:
        keys = list(contract._TEXTURE_CACHE)
    assert len(keys) <= 14
    assert len(set(outputs)) <= 14


def test_extended_generator_uses_the_single_cached_texture_owner() -> None:
    assert extended.make_texture_png is generator.make_texture_png
    assert getattr(generator.make_texture_png, "_mmm_texture_equivalence_cache", False) is True
