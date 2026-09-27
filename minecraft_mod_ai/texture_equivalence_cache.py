from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps

_TEXTURE_CACHE_LOCK = threading.RLock()
_TEXTURE_CACHE: OrderedDict[tuple[str, str, int, int], bytes] = OrderedDict()
_TEXTURE_CACHE_LIMIT = 512


def _texture_pattern_key(color: str, seed: str, kind: str, size: int) -> tuple[str, str, int, int]:
    residue = sum((index + 1) * ord(char) for index, char in enumerate(seed)) % 14
    return str(color), str(kind), int(size), residue


def cached_texture_renderer(renderer: Callable[..., bytes]) -> Callable[..., bytes]:
    @wraps(renderer)
    def cached(color: str, seed: str, *, kind: str, size: int = 16) -> bytes:
        key = _texture_pattern_key(color, seed, kind, size)
        with _TEXTURE_CACHE_LOCK:
            value = _TEXTURE_CACHE.get(key)
            if value is not None:
                _TEXTURE_CACHE.move_to_end(key)
                return value
        rendered = renderer(color, seed, kind=kind, size=size)
        with _TEXTURE_CACHE_LOCK:
            value = _TEXTURE_CACHE.get(key)
            if value is not None:
                _TEXTURE_CACHE.move_to_end(key)
                return value
            _TEXTURE_CACHE[key] = rendered
            while len(_TEXTURE_CACHE) > _TEXTURE_CACHE_LIMIT:
                _TEXTURE_CACHE.popitem(last=False)
        return rendered

    cached._mmm_texture_equivalence_cache = True  # type: ignore[attr-defined]
    cached._mmm_texture_cache = _TEXTURE_CACHE  # type: ignore[attr-defined]
    return cached


__all__ = ["_TEXTURE_CACHE", "_TEXTURE_CACHE_LOCK", "cached_texture_renderer"]
