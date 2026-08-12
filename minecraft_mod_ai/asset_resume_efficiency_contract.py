from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any


_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(*, role: str, prompt: str, width: int, height: int, seed: int) -> str:
    body = json.dumps(
        {
            "version": _VERSION,
            "role": role,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _meta_path(output: Path) -> Path:
    return output.with_name(output.name + ".mmm-image-source.json")


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _valid_cached(output: Path, *, cache_key: str) -> bool:
    meta = _meta_path(output)
    if not output.is_file() or output.is_symlink() or not meta.is_file() or meta.is_symlink():
        return False
    try:
        value = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(value, dict):
        return False
    if value.get("version") != _VERSION or value.get("key") != cache_key:
        return False
    expected = str(value.get("sha256", ""))
    return bool(expected) and _sha256(output) == expected


class _CachedImageRouter:
    """Reuse exact prompt/seed image sources inside a retried asset shard."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)

    def generate_image(
        self,
        role: str,
        *,
        prompt: str,
        output_path: str | Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        output = Path(output_path).expanduser().resolve()
        cache_key = _key(
            role=role,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
        )
        if _valid_cached(output, cache_key=cache_key):
            return output

        generated = Path(
            self._router.generate_image(
                role,
                prompt=prompt,
                output_path=output,
                width=width,
                height=height,
                seed=seed,
            )
        ).expanduser().resolve()
        if generated != output or not output.is_file() or output.is_symlink():
            raise RuntimeError(
                "Image backend did not create the exact requested resumable output path."
            )
        _write_meta(
            _meta_path(output),
            {
                "version": _VERSION,
                "key": cache_key,
                "sha256": _sha256(output),
                "width": int(width),
                "height": int(height),
                "seed": int(seed),
            },
        )
        return output


def install(services_module: Any) -> None:
    """Keep GPU-resident asset shards but checkpoint every expensive image source."""

    for name in ("_generate_single_asset_source", "_generate_tiled_asset_source"):
        current = getattr(services_module, name)
        if getattr(current, "_mmm_resumable_image_sources", False):
            continue

        @wraps(current)
        def resumable(router: Any, *args: Any, __current=current, **kwargs: Any):
            return __current(_CachedImageRouter(router), *args, **kwargs)

        resumable._mmm_resumable_image_sources = True  # type: ignore[attr-defined]
        setattr(services_module, name, resumable)


__all__ = ["install", "_CachedImageRouter"]
