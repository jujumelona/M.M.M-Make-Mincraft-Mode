from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any

_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _backend_identity(router: Any, role: str) -> str:
    config = router.registry.role(router.profile, role)
    payload = _jsonable(config)
    if isinstance(payload, dict):
        payload.pop("api_key", None)
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _key(
    *,
    role: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    backend_identity: str,
) -> str:
    body = json.dumps(
        {
            "version": _VERSION,
            "role": role,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "backend_identity": backend_identity,
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
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _valid_cached(output: Path, *, cache_key: str) -> bool:
    meta = _meta_path(output)
    if (
        not output.is_file()
        or output.is_symlink()
        or not meta.is_file()
        or meta.is_symlink()
    ):
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
    "Content-address expensive deterministic image candidates across retries."

    def __init__(self, router: Any) -> None:
        self._router = router
        self._backend_identities: dict[str, str] = {}

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
        if role not in self._backend_identities:
            self._backend_identities[role] = _backend_identity(self._router, role)
        identity = self._backend_identities[role]
        cache_key = _key(
            role=role,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            backend_identity=identity,
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
                "backend_identity": identity,
            },
        )
        return output


def install(resource_asset_module: Any) -> None:
    "Bind retry-safe image source caching to the single canonical asset producer."

    current = resource_asset_module.generate_assets
    if getattr(current, "_mmm_resumable_image_sources", False):
        return

    @wraps(current)
    def resumable(
        router: Any,
        *args: Any,
        __current=current,
        **kwargs: Any,
    ):
        return __current(_CachedImageRouter(router), *args, **kwargs)

    resumable._mmm_resumable_image_sources = True  # type: ignore[attr-defined]
    setattr(resource_asset_module, "generate_assets", resumable)


__all__ = ["_CachedImageRouter", "install"]
