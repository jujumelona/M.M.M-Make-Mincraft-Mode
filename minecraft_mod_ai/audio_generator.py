from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .complete_spec import AudioRequest
from .project_edit import ensure_main_initializer_call, inspect_fabric_project, write_text_files


class AudioGenerationError(RuntimeError):
    pass


def generate_audio_assets(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    requests: Iterable[AudioRequest],
    sample_rate: int = 44_100,
) -> dict[str, Any]:
    """Synthesize deterministic OGG assets and register Fabric SoundEvents."""

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise AudioGenerationError("Audio target does not match fabric.mod.json.")
    items = tuple(requests)
    if not items:
        return {"schema_version": "mmm/audio-generation-v1", "status": "SKIPPED", "sounds": []}
    for item in items:
        item.validate()
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise AudioGenerationError(
            "Audio synthesis requires numpy and soundfile. Install the production-audio extra."
        ) from exc

    sound_dir = info.root / "src/main/resources/assets" / mod_id / "sounds"
    sound_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    sounds_json: dict[str, Any] = {}
    english: dict[str, str] = {}
    korean: dict[str, str] = {}
    for request in items:
        duration = float(request.duration_seconds)
        frame_count = max(1, int(sample_rate * duration))
        time_axis = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
        envelope = _envelope(np, frame_count, sample_rate, request.kind)
        waveform = _waveform(np, time_axis, request.frequency_hz, request.kind, request.sound_id)
        waveform = np.asarray(waveform * envelope * request.volume, dtype=np.float32)
        target = sound_dir / f"{request.sound_id}.ogg"
        sf.write(str(target), waveform, sample_rate, format="OGG", subtype="VORBIS")
        if not target.is_file() or target.stat().st_size < 128:
            raise AudioGenerationError(f"OGG encoder did not produce a valid asset: {target}")
        subtitle_key = f"subtitles.{mod_id}.{request.sound_id}"
        sounds_json[request.sound_id] = {
            "subtitle": subtitle_key,
            "sounds": [
                {
                    "name": f"{mod_id}:{request.sound_id}",
                    "stream": request.kind in {"ambient", "music"},
                    "volume": request.volume,
                    "pitch": 1.0,
                    "preload": request.kind == "ui",
                    "weight": 1,
                }
            ],
        }
        english[subtitle_key] = request.subtitle_en or request.sound_id.replace("_", " ").title()
        korean[subtitle_key] = request.subtitle_ko or english[subtitle_key]
        generated.append(
            {
                "sound_id": request.sound_id,
                "path": str(target),
                "size_bytes": target.stat().st_size,
                "loop": request.loop,
                "kind": request.kind,
            }
        )

    assets_root = f"src/main/resources/assets/{mod_id}"
    files = {
        f"{assets_root}/sounds.json": json.dumps(sounds_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        _sound_java_path(package_name): _sound_java(package_name, mod_id, items),
    }
    _merge_lang(info.root / f"{assets_root}/lang/en_us.json", english)
    _merge_lang(info.root / f"{assets_root}/lang/ko_kr.json", korean)
    receipt = write_text_files(info, files, replace_existing=True)
    binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.sound.GeneratedSounds",
        call_line="GeneratedSounds.register()",
        marker="audio:generated-sounds",
    )
    return {
        "schema_version": "mmm/audio-generation-v1",
        "status": "GENERATED",
        "sounds": generated,
        "source_receipt": receipt,
        "binding_receipt": binding,
        "required_gates": ["Gradle", "client sound playback", "volume and loop review"],
    }


def register_existing_ogg(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    sound_id: str,
    ogg_path: str | Path,
    kind: str = "effect",
    subtitle_en: str = "",
    subtitle_ko: str = "",
) -> dict[str, Any]:
    request = AudioRequest(
        sound_id=sound_id,
        kind=kind,
        duration_seconds=1.0,
        subtitle_en=subtitle_en,
        subtitle_ko=subtitle_ko,
    )
    request.validate()
    info = inspect_fabric_project(project_root)
    source = Path(ogg_path).expanduser().resolve()
    if source.suffix.lower() != ".ogg" or not source.is_file() or source.is_symlink():
        raise AudioGenerationError("Existing audio must be a regular .ogg file.")
    destination = info.root / f"src/main/resources/assets/{mod_id}/sounds/{sound_id}.ogg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    sounds_path = info.root / f"src/main/resources/assets/{mod_id}/sounds.json"
    sounds = json.loads(sounds_path.read_text(encoding="utf-8")) if sounds_path.is_file() else {}
    subtitle_key = f"subtitles.{mod_id}.{sound_id}"
    sounds[sound_id] = {
        "subtitle": subtitle_key,
        "sounds": [{"name": f"{mod_id}:{sound_id}", "stream": kind in {"ambient", "music"}}],
    }
    sounds_path.write_text(json.dumps(sounds, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _merge_lang(info.root / f"src/main/resources/assets/{mod_id}/lang/en_us.json", {subtitle_key: subtitle_en or sound_id})
    _merge_lang(info.root / f"src/main/resources/assets/{mod_id}/lang/ko_kr.json", {subtitle_key: subtitle_ko or subtitle_en or sound_id})
    items = tuple(
        AudioRequest(
            sound_id=key,
            kind="effect",
            duration_seconds=1.0,
            subtitle_en="",
            subtitle_ko="",
        )
        for key in sorted(sounds)
    )
    write_text_files(info, {_sound_java_path(package_name): _sound_java(package_name, mod_id, items)}, replace_existing=True)
    binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.sound.GeneratedSounds",
        call_line="GeneratedSounds.register()",
        marker="audio:generated-sounds",
    )
    return {"status": "REGISTERED", "path": str(destination), "binding_receipt": binding}


def _waveform(np: Any, t: Any, frequency: float, kind: str, seed_text: str) -> Any:
    phase = (sum(seed_text.encode("utf-8")) % 360) * math.pi / 180.0
    base = np.sin(2.0 * math.pi * frequency * t + phase)
    if kind == "music":
        return 0.55 * base + 0.25 * np.sin(2.0 * math.pi * frequency * 1.25 * t) + 0.2 * np.sin(2.0 * math.pi * frequency * 1.5 * t)
    if kind == "ambient":
        return 0.7 * np.sin(2.0 * math.pi * frequency * 0.5 * t) + 0.3 * np.sin(2.0 * math.pi * frequency * 0.503 * t)
    if kind == "ui":
        return np.sin(2.0 * math.pi * (frequency + 180.0 * t) * t)
    return 0.75 * base + 0.25 * np.sin(2.0 * math.pi * frequency * 2.0 * t)


def _envelope(np: Any, frames: int, sample_rate: int, kind: str) -> Any:
    envelope = np.ones(frames, dtype=np.float32)
    attack = min(frames // 3, max(1, int(sample_rate * (0.005 if kind == "ui" else 0.03))))
    release = min(frames // 2, max(1, int(sample_rate * (0.04 if kind in {"effect", "ui"} else 0.3))))
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope


def _sound_java_path(package_name: str) -> str:
    return "src/main/java/" + package_name.replace(".", "/") + "/sound/GeneratedSounds.java"


def _sound_java(package_name: str, mod_id: str, requests: Iterable[AudioRequest]) -> str:
    declarations = []
    registrations = []
    for request in requests:
        constant = request.sound_id.upper()
        declarations.append(f"    public static SoundEvent {constant};")
        registrations.append(
            f'        {constant} = Registry.register(Registries.SOUND_EVENT, new Identifier(MOD_ID, "{request.sound_id}"), SoundEvent.of(new Identifier(MOD_ID, "{request.sound_id}")));'
        )
    return f'''package {package_name}.sound;

import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.sound.SoundEvent;
import net.minecraft.util.Identifier;

public final class GeneratedSounds {{
    private static final String MOD_ID = "{mod_id}";
{chr(10).join(declarations)}

    private GeneratedSounds() {{}}

    public static void register() {{
{chr(10).join(registrations)}
    }}
}}
'''


def _merge_lang(path: Path, additions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(current, dict):
        raise AudioGenerationError(f"Language file is not a JSON object: {path}")
    current.update(additions)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
