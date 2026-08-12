from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Iterable


_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_key(request: Any, *, sample_rate: int, chunk_frames: int) -> str:
    value = {
        "version": _VERSION,
        "sound_id": request.sound_id,
        "kind": request.kind,
        "duration_seconds": float(request.duration_seconds),
        "frequency_hz": float(request.frequency_hz),
        "volume": float(request.volume),
        "loop": bool(request.loop),
        "subtitle_en": str(request.subtitle_en),
        "subtitle_ko": str(request.subtitle_ko),
        "sample_rate": int(sample_rate),
        "chunk_frames": int(chunk_frames),
    }
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _meta_path(target: Path) -> Path:
    return target.with_name(target.name + ".mmm-audio-source.json")


def _write_meta(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(_canonical(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cached_entry(target: Path, *, key: str) -> dict[str, Any] | None:
    meta = _meta_path(target)
    if not target.is_file() or target.is_symlink() or not meta.is_file() or meta.is_symlink():
        return None
    try:
        value = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(value, dict) or value.get("version") != _VERSION or value.get("key") != key:
        return None
    expected = str(value.get("sha256", ""))
    entry = value.get("entry")
    if not expected or not isinstance(entry, dict) or _sha256(target) != expected:
        return None
    if str(entry.get("path", "")) != str(target):
        return None
    if int(entry.get("size_bytes", -1)) != target.stat().st_size:
        return None
    return dict(entry)


def install(audio_module: Any) -> None:
    """Checkpoint each deterministic OGG so a failed shard resumes per sound."""

    current = audio_module.synthesize_audio_files
    if getattr(current, "_mmm_resumable_audio_sources", False):
        return

    @wraps(current)
    def resumable_synthesize_audio_files(
        *,
        project_root: str | Path,
        mod_id: str,
        package_name: str,
        requests: Iterable[Any],
        sample_rate: int = 44_100,
        chunk_frames: int = 65_536,
        policy: Any | None = None,
    ) -> dict[str, Any]:
        policy = policy or audio_module.ScalePolicy.from_environment()
        policy.validate()
        items = tuple(requests)
        if not items:
            return current(
                project_root=project_root,
                mod_id=mod_id,
                package_name=package_name,
                requests=items,
                sample_rate=sample_rate,
                chunk_frames=chunk_frames,
                policy=policy,
            )
        if type(sample_rate) is not int or not 8_000 <= sample_rate <= 192_000:
            raise audio_module.AudioGenerationError(
                "sample_rate must be an integer between 8000 and 192000."
            )
        if type(chunk_frames) is not int or chunk_frames < 1024:
            raise audio_module.AudioGenerationError(
                "chunk_frames must be an integer >= 1024."
            )
        info = audio_module.inspect_fabric_project(project_root)
        if info.mod_id != mod_id or info.package_name != package_name:
            raise audio_module.AudioGenerationError(
                "Audio target does not match fabric.mod.json."
            )
        for item in items:
            item.validate(policy=policy)
        if len({item.sound_id for item in items}) != len(items):
            raise audio_module.AudioGenerationError("Audio request IDs must be unique.")

        sound_dir = info.root / f"src/main/resources/assets/{mod_id}/sounds"
        sound_dir.mkdir(parents=True, exist_ok=True)
        cached: dict[str, dict[str, Any]] = {}
        missing: list[Any] = []
        keys: dict[str, str] = {}
        for item in items:
            key = _request_key(
                item,
                sample_rate=sample_rate,
                chunk_frames=chunk_frames,
            )
            keys[item.sound_id] = key
            target = sound_dir / f"{item.sound_id}.ogg"
            entry = _cached_entry(target, key=key)
            if entry is None:
                missing.append(item)
            else:
                cached[item.sound_id] = entry

        generated: dict[str, dict[str, Any]] = {}
        if missing:
            receipt = current(
                project_root=project_root,
                mod_id=mod_id,
                package_name=package_name,
                requests=tuple(missing),
                sample_rate=sample_rate,
                chunk_frames=chunk_frames,
                policy=policy,
            )
            for raw in receipt.get("synthesized", []):
                if not isinstance(raw, dict):
                    continue
                sound_id = str(raw.get("sound_id", ""))
                if sound_id not in keys:
                    continue
                target = Path(str(raw.get("path", ""))).expanduser().resolve()
                if not target.is_file() or target.is_symlink():
                    raise audio_module.AudioGenerationError(
                        f"Synthesized audio disappeared before checkpointing: {sound_id}"
                    )
                entry = dict(raw)
                generated[sound_id] = entry
                _write_meta(
                    _meta_path(target),
                    {
                        "version": _VERSION,
                        "key": keys[sound_id],
                        "sha256": _sha256(target),
                        "entry": entry,
                    },
                )

        ordered: list[dict[str, Any]] = []
        for item in items:
            entry = generated.get(item.sound_id) or cached.get(item.sound_id)
            if entry is None:
                raise audio_module.AudioGenerationError(
                    f"Audio synthesis omitted requested sound: {item.sound_id}"
                )
            ordered.append(entry)
        return {
            "status": "SYNTHESIZED",
            "synthesized": ordered,
            "touched_paths": [str(item["path"]) for item in ordered],
        }

    resumable_synthesize_audio_files._mmm_resumable_audio_sources = True  # type: ignore[attr-defined]
    resumable_synthesize_audio_files.__wrapped__ = current  # type: ignore[attr-defined]
    audio_module.synthesize_audio_files = resumable_synthesize_audio_files


__all__ = ["install", "_cached_entry", "_request_key"]
