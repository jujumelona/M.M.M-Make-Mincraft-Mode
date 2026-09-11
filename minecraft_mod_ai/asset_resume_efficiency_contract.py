from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from .project_write_lock import project_path_write_locks

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
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    if registry is None or profile is None:
        router_type = type(router)
        body = f"{router_type.__module__}.{router_type.__qualname__}:{role}"
        return hashlib.sha256(body.encode("utf-8")).hexdigest()
    config = registry.role(profile, role)
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


def _target_state(target: Path) -> tuple[str, str | None]:
    if target.is_symlink():
        raise RuntimeError("Asset target may not be a symlink.")
    if not target.exists():
        return "missing", None
    if not target.is_file():
        raise RuntimeError("Asset target must be a regular file.")
    return "file", _sha256(target)


@dataclass
class _WriteGuard:
    project_root: Path
    relative_paths: dict[Path, str]
    expected_states: dict[Path, tuple[str, str | None]]


_WRITE_GUARD: ContextVar[_WriteGuard | None] = ContextVar(
    "mmm_asset_write_guard",
    default=None,
)


def _safe_relative_target(project_root: Path, relative: str) -> tuple[Path, str]:
    normalized = PurePosixPath(str(relative).replace("\\", "/"))
    if normalized.is_absolute() or any(
        part in {"", ".", ".."} for part in normalized.parts
    ):
        raise RuntimeError(f"Asset target path is not project-relative: {relative!r}")
    rendered = normalized.as_posix()
    target = (project_root / Path(*normalized.parts)).resolve()
    try:
        target.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError(f"Asset target escaped project root: {relative!r}") from exc
    return target, rendered


def _planned_write_guard(proposal: Any, project_root: str | Path) -> _WriteGuard | None:
    game_design = getattr(proposal, "game_design", None)
    if not isinstance(game_design, Mapping):
        return None
    plan = game_design.get("_asset_generation_plan")
    if not isinstance(plan, Mapping):
        return None
    rows = plan.get("assets")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None

    root = Path(project_root).expanduser().resolve()
    relative_paths: dict[Path, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("container") or "mod") != "mod":
            continue
        for collection in (row.get("textures", ()), row.get("documents", ())):
            if not isinstance(collection, Sequence) or isinstance(
                collection, (str, bytes, bytearray)
            ):
                continue
            for item in collection:
                if not isinstance(item, Mapping):
                    continue
                relative = item.get("target_path")
                if not isinstance(relative, str) or not relative:
                    continue
                target, rendered = _safe_relative_target(root, relative)
                relative_paths[target] = rendered

    if not relative_paths:
        return None

    expected_states: dict[Path, tuple[str, str | None]] = {}
    for target, relative in relative_paths.items():
        with project_path_write_locks(root, (relative,)):
            expected_states[target] = _target_state(target)
    return _WriteGuard(root, relative_paths, expected_states)


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


def _install_atomic_write_guard(resource_asset_module: Any) -> None:
    current = resource_asset_module._atomic_write_bytes
    if getattr(current, "_mmm_asset_write_guard", False):
        return

    @wraps(current)
    def guarded(target: Path, data: bytes, __current=current) -> None:
        resolved = Path(target).expanduser().resolve()
        state = _WRITE_GUARD.get()
        if state is None or resolved not in state.expected_states:
            __current(target, data)
            return
        relative = state.relative_paths[resolved]
        with project_path_write_locks(state.project_root, (relative,)):
            if _target_state(resolved) != state.expected_states[resolved]:
                raise RuntimeError(
                    "Asset target changed while generation was in flight; refusing stale overwrite."
                )
            __current(resolved, data)
            state.expected_states[resolved] = _target_state(resolved)

    guarded._mmm_asset_write_guard = True  # type: ignore[attr-defined]
    setattr(resource_asset_module, "_atomic_write_bytes", guarded)


def install(resource_asset_module: Any) -> None:
    """Bind retry caching and path-scoped optimistic commits to the canonical producer."""

    _install_atomic_write_guard(resource_asset_module)
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
        proposal = kwargs.get("proposal")
        project_root = kwargs.get("project_root")
        if proposal is None and args:
            proposal = args[0]
        if project_root is None and len(args) >= 2:
            project_root = args[1]
        guard = (
            _planned_write_guard(proposal, project_root)
            if proposal is not None and project_root is not None
            else None
        )
        token = _WRITE_GUARD.set(guard)
        try:
            return __current(_CachedImageRouter(router), *args, **kwargs)
        finally:
            _WRITE_GUARD.reset(token)

    resumable._mmm_resumable_image_sources = True  # type: ignore[attr-defined]
    resumable._mmm_path_scoped_asset_commit = True  # type: ignore[attr-defined]
    setattr(resource_asset_module, "generate_assets", resumable)


__all__ = ["_CachedImageRouter", "_target_state", "install"]
