from __future__ import annotations

from minecraft_mod_ai import generator, texture_equivalence_cache as texture_cache


def test_texture_cache_preserves_exact_original_bytes() -> None:
    cached = generator.make_texture_png
    original = cached.__wrapped__
    with texture_cache._TEXTURE_CACHE_LOCK:
        texture_cache._TEXTURE_CACHE.clear()

    for kind in ("item", "block", "entity"):
        for size in (16, 64):
            for seed in ("a", "o", "module_001", "module_999"):
                expected = original("#74c7ec", seed, kind=kind, size=size)
                actual = cached("#74c7ec", seed, kind=kind, size=size)
                assert actual == expected


def test_many_seed_names_collapse_to_at_most_fourteen_exact_patterns() -> None:
    cached = generator.make_texture_png
    with texture_cache._TEXTURE_CACHE_LOCK:
        texture_cache._TEXTURE_CACHE.clear()

    outputs = [
        cached("#748cab", f"module_{index:04d}", kind="item", size=16)
        for index in range(100)
    ]

    with texture_cache._TEXTURE_CACHE_LOCK:
        keys = list(texture_cache._TEXTURE_CACHE)
    assert len(keys) <= 14
    assert len(set(outputs)) <= 14


def test_generator_source_owns_the_single_texture_cache() -> None:
    assert getattr(generator.make_texture_png, "_mmm_texture_equivalence_cache", False) is True
    assert callable(generator.make_texture_png.__wrapped__)
    assert generator.make_texture_png.__wrapped__.__module__ == "minecraft_mod_ai.generator"
