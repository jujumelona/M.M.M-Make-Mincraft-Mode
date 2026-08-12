from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

from .project_write_lock import project_write_lock


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


def _project_root_for_target(request: Any, target: Path) -> Path:
    relative = Path(str(request.target_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Asset target path is not project-relative.")
    parts = tuple(part for part in relative.parts if part not in {"", "."})
    if not parts:
        raise RuntimeError("Asset target path is empty.")
    root = target.resolve()
    for _ in parts:
        root = root.parent
    root = root.resolve()
    if (root / relative).resolve() != target.resolve():
        raise RuntimeError("Asset target could not be bound to its project root safely.")
    return root


def _target_state(target: Path) -> tuple[str, str | None]:
    if target.is_symlink():
        raise RuntimeError("Asset target may not be a symlink.")
    if not target.exists():
        return "missing", None
    if not target.is_file():
        raise RuntimeError("Asset target must be a regular file.")
    return "file", _sha256(target)


def _atomic_commit(staged: Path, target: Path, project_root: Path) -> str:
    if not staged.is_file() or staged.is_symlink():
        raise RuntimeError("Asset staging did not produce a regular file.")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with staged.open("rb") as source, os.fdopen(fd, "wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return _sha256(target)


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
    """Cache expensive image sources and commit final assets without GPU lock time."""

    for name in ("_generate_single_asset_source", "_generate_tiled_asset_source"):
        current = getattr(services_module, name)
        if getattr(current, "_mmm_resumable_image_sources", False):
            continue

        @wraps(current)
        def resumable(router: Any, *args: Any, __current=current, **kwargs: Any):
            target = Path(kwargs["target"]).expanduser().resolve()
            request = kwargs["request"]
            concept_dir = Path(kwargs["concept_dir"]).expanduser().resolve()
            project_root = _project_root_for_target(request, target)

            # Snapshot the target under the same lock used by text/source patchers,
            # then release it before any expensive image inference/composition.
            with project_write_lock(project_root):
                before = _target_state(target)

            staging_dir = concept_dir / ".final-staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            suffix = target.suffix or ".asset"
            staging_key = hashlib.sha256(
                f"{request.asset_id}\0{target}".encode("utf-8")
            ).hexdigest()[:16]
            staged = staging_dir / f"{request.asset_id}-{staging_key}{suffix}"

            staged_kwargs = dict(kwargs)
            staged_kwargs["target"] = staged
            receipt = __current(
                _CachedImageRouter(router),
                *args,
                **staged_kwargs,
            )

            # Commit only after the expensive work has completed. Optimistic target
            # state checking prevents a concurrent custom patch from being silently
            # overwritten by an asset that started from older project state.
            with project_write_lock(project_root):
                if _target_state(target) != before:
                    raise RuntimeError(
                        "Asset target changed while generation was in flight; refusing stale overwrite."
                    )
                committed_sha256 = _atomic_commit(staged, target, project_root)

            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
            value = dict(receipt)
            value["asset_commit_sha256"] = committed_sha256
            value["asset_commit_mode"] = "staged_atomic_replace"
            return value

        resumable._mmm_resumable_image_sources = True  # type: ignore[attr-defined]
        resumable._mmm_staged_asset_commit = True  # type: ignore[attr-defined]
        setattr(services_module, name, resumable)


__all__ = [
    "install",
    "_CachedImageRouter",
    "_project_root_for_target",
    "_target_state",
]
