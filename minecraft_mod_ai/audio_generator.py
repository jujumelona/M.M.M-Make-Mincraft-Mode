from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from .complete_spec import AudioRequest
from .project_edit import (
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .scale_policy import ScalePolicy


class AudioGenerationError(RuntimeError):
    pass


def generate_audio_assets(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    requests: Iterable[AudioRequest],
    sample_rate: int = 44_100,
    chunk_frames: int = 65_536,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    """Stream deterministic OGG assets and shard Fabric SoundEvent registration.

    Audio duration and sound count no longer force one in-memory waveform or one giant
    Java class. Existing ``sounds.json`` entries are preserved and merged.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if type(sample_rate) is not int or not 8_000 <= sample_rate <= 192_000:
        raise AudioGenerationError("sample_rate must be an integer between 8000 and 192000.")
    if type(chunk_frames) is not int or chunk_frames < 1024:
        raise AudioGenerationError("chunk_frames must be an integer >= 1024.")

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise AudioGenerationError("Audio target does not match fabric.mod.json.")
    items = tuple(requests)
    if not items:
        return {
            "schema_version": "mmm/audio-generation-v2",
            "status": "SKIPPED",
            "sounds": [],
        }
    for item in items:
        item.validate(policy=policy)
    if len({item.sound_id for item in items}) != len(items):
        raise AudioGenerationError("Audio request IDs must be unique.")

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise AudioGenerationError(
            "Audio synthesis requires numpy and soundfile. Install the production-audio extra."
        ) from exc

    assets_root = f"src/main/resources/assets/{mod_id}"
    sound_dir = info.root / assets_root / "sounds"
    sound_dir.mkdir(parents=True, exist_ok=True)
    sounds_path = info.root / assets_root / "sounds.json"
    sounds_json = _load_object(sounds_path)
    english: dict[str, str] = {}
    korean: dict[str, str] = {}
    generated: list[dict[str, Any]] = []

    manifest_entries, directory_manifest = _load_audio_manifest(info.root)

    for request in items:
        target = sound_dir / f"{request.sound_id}.ogg"
        _write_ogg_stream(
            np=np,
            sf=sf,
            target=target,
            request=request,
            sample_rate=sample_rate,
            chunk_frames=chunk_frames,
        )
        if not target.is_file() or target.stat().st_size < 128:
            raise AudioGenerationError(
                f"OGG encoder did not produce a valid asset: {target}"
            )
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
        english[subtitle_key] = (
            request.subtitle_en
            or request.sound_id.replace("_", " ").title()
        )
        korean[subtitle_key] = request.subtitle_ko or english[subtitle_key]
        entry = {
            "sound_id": request.sound_id,
            "kind": request.kind,
            "loop": request.loop,
            "duration_seconds": float(request.duration_seconds),
            "sample_rate": sample_rate,
            "path": str(target),
            "size_bytes": target.stat().st_size,
        }
        manifest_entries[request.sound_id] = entry
        generated.append(entry)

    all_ids = sorted(sounds_json)
    manifest_records = (
        generated
        if directory_manifest
        else [manifest_entries[key] for key in sorted(manifest_entries)]
    )
    files: dict[str, str] = {
        f"{assets_root}/sounds.json": json.dumps(
            sounds_json,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        ".minecraft_ai/audio-assets.json": _json_text(
            {
                "schema_version": "mmm/audio-asset-directory-v1",
                "sound_count": len(manifest_entries),
                "directory": ".minecraft_ai/audio-asset-records",
            }
        ),
    }
    files.update(
        {
            f".minecraft_ai/audio-asset-records/{item['sound_id']}.json": (
                _json_text(item)
            )
            for item in manifest_records
        }
    )
    java_root = (
        info.root
        / "src/main/java"
        / Path(*package_name.split("."))
        / "sound/GeneratedSounds.java"
    )
    incremental_registrar = (
        java_root.is_file()
        and "GeneratedSoundUnit" in java_root.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )
    emitted_sound_ids = (
        [item.sound_id for item in items]
        if incremental_registrar
        else all_ids
    )
    files.update(
        _sound_java_files(
            package_name=package_name,
            mod_id=mod_id,
            sound_ids=emitted_sound_ids,
        )
    )
    _merge_lang(
        info.root / f"{assets_root}/lang/en_us.json",
        english,
    )
    _merge_lang(
        info.root / f"{assets_root}/lang/ko_kr.json",
        korean,
    )
    receipt = write_text_files(info, files, replace_existing=True)
    binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.sound.GeneratedSounds",
        call_line="GeneratedSounds.register()",
        marker="audio:generated-sounds",
    )
    return {
        "schema_version": "mmm/audio-generation-v2",
        "status": "GENERATED",
        "sounds": generated,
        "sound_count": len(all_ids),
        "registrar_shards": len(emitted_sound_ids),
        "source_receipt": receipt,
        "binding_receipt": binding,
        "required_gates": [
            "Gradle",
            "client sound playback",
            "volume and loop review",
        ],
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
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    request = AudioRequest(
        sound_id=sound_id,
        kind=kind,
        duration_seconds=1.0,
        subtitle_en=subtitle_en,
        subtitle_ko=subtitle_ko,
    )
    request.validate(policy=policy)
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise AudioGenerationError("Audio target does not match fabric.mod.json.")
    source = Path(ogg_path).expanduser().resolve()
    if (
        source.suffix.lower() != ".ogg"
        or not source.is_file()
        or source.is_symlink()
    ):
        raise AudioGenerationError("Existing audio must be a regular .ogg file.")
    if source.stat().st_size > policy.max_single_file_bytes:
        raise AudioGenerationError(
            "Existing audio exceeds MMM_MAX_SINGLE_FILE_BYTES host policy."
        )
    destination = (
        info.root
        / f"src/main/resources/assets/{mod_id}/sounds/{sound_id}.ogg"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

    assets_root = f"src/main/resources/assets/{mod_id}"
    sounds_path = info.root / f"{assets_root}/sounds.json"
    sounds = _load_object(sounds_path)
    subtitle_key = f"subtitles.{mod_id}.{sound_id}"
    sounds[sound_id] = {
        "subtitle": subtitle_key,
        "sounds": [
            {
                "name": f"{mod_id}:{sound_id}",
                "stream": kind in {"ambient", "music"},
            }
        ],
    }
    manifest_entries, directory_manifest = _load_audio_manifest(info.root)
    manifest_entry = {
        "sound_id": sound_id,
        "kind": kind,
        "loop": False,
        "duration_seconds": None,
        "sample_rate": None,
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
    }
    manifest_entries[sound_id] = manifest_entry
    manifest_records = (
        [manifest_entry]
        if directory_manifest
        else [manifest_entries[key] for key in sorted(manifest_entries)]
    )
    files = {
        f"{assets_root}/sounds.json": json.dumps(
            sounds,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        ".minecraft_ai/audio-assets.json": _json_text(
            {
                "schema_version": "mmm/audio-asset-directory-v1",
                "sound_count": len(manifest_entries),
                "directory": ".minecraft_ai/audio-asset-records",
            }
        ),
    }
    files.update(
        {
            f".minecraft_ai/audio-asset-records/{item['sound_id']}.json": (
                _json_text(item)
            )
            for item in manifest_records
        }
    )
    java_root = (
        info.root
        / "src/main/java"
        / Path(*package_name.split("."))
        / "sound/GeneratedSounds.java"
    )
    incremental_registrar = (
        java_root.is_file()
        and "GeneratedSoundUnit" in java_root.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )
    files.update(
        _sound_java_files(
            package_name=package_name,
            mod_id=mod_id,
            sound_ids=(
                [sound_id]
                if incremental_registrar
                else sorted(sounds)
            ),
        )
    )
    write_receipt = write_text_files(info, files, replace_existing=True)
    _merge_lang(
        info.root / f"{assets_root}/lang/en_us.json",
        {subtitle_key: subtitle_en or sound_id},
    )
    _merge_lang(
        info.root / f"{assets_root}/lang/ko_kr.json",
        {subtitle_key: subtitle_ko or subtitle_en or sound_id},
    )
    binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.sound.GeneratedSounds",
        call_line="GeneratedSounds.register()",
        marker="audio:generated-sounds",
    )
    return {
        "schema_version": "mmm/audio-registration-v2",
        "status": "REGISTERED",
        "path": str(destination),
        "source_receipt": write_receipt,
        "binding_receipt": binding,
    }


def _write_ogg_stream(
    *,
    np: Any,
    sf: Any,
    target: Path,
    request: AudioRequest,
    sample_rate: int,
    chunk_frames: int,
) -> None:
    total_frames = max(
        1,
        int(round(sample_rate * float(request.duration_seconds))),
    )
    temporary = target.with_suffix(".ogg.part")
    if temporary.exists():
        temporary.unlink()
    try:
        with sf.SoundFile(
            str(temporary),
            mode="w",
            samplerate=sample_rate,
            channels=1,
            format="OGG",
            subtype="VORBIS",
        ) as stream:
            for start in range(0, total_frames, chunk_frames):
                stop = min(total_frames, start + chunk_frames)
                indices = np.arange(start, stop, dtype=np.float64)
                time_axis = indices / float(sample_rate)
                waveform = _waveform(
                    np,
                    time_axis,
                    request.frequency_hz,
                    request.kind,
                    request.sound_id,
                )
                envelope = _envelope_chunk(
                    np,
                    indices,
                    total_frames,
                    sample_rate,
                    request.kind,
                )
                stream.write(
                    np.asarray(
                        waveform * envelope * request.volume,
                        dtype=np.float32,
                    )
                )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _waveform(
    np: Any,
    t: Any,
    frequency: float,
    kind: str,
    seed_text: str,
) -> Any:
    phase = (
        sum(seed_text.encode("utf-8")) % 360
    ) * math.pi / 180.0
    base = np.sin(2.0 * math.pi * frequency * t + phase)
    if kind == "music":
        return (
            0.55 * base
            + 0.25
            * np.sin(2.0 * math.pi * frequency * 1.25 * t)
            + 0.2
            * np.sin(2.0 * math.pi * frequency * 1.5 * t)
        )
    if kind == "ambient":
        return (
            0.7
            * np.sin(2.0 * math.pi * frequency * 0.5 * t)
            + 0.3
            * np.sin(2.0 * math.pi * frequency * 0.503 * t)
        )
    if kind == "ui":
        return np.sin(
            2.0 * math.pi * (frequency + 180.0 * t) * t
        )
    return (
        0.75 * base
        + 0.25
        * np.sin(2.0 * math.pi * frequency * 2.0 * t)
    )


def _envelope_chunk(
    np: Any,
    indices: Any,
    total_frames: int,
    sample_rate: int,
    kind: str,
) -> Any:
    attack = min(
        total_frames // 3,
        max(
            1,
            int(
                sample_rate
                * (0.005 if kind == "ui" else 0.03)
            ),
        ),
    )
    release = min(
        total_frames // 2,
        max(
            1,
            int(
                sample_rate
                * (
                    0.04
                    if kind in {"effect", "ui"}
                    else 0.3
                )
            ),
        ),
    )
    envelope = np.ones(indices.shape, dtype=np.float64)
    if attack:
        attack_mask = indices < attack
        envelope[attack_mask] *= indices[attack_mask] / float(attack)
    if release:
        release_start = total_frames - release
        release_mask = indices >= release_start
        envelope[release_mask] *= (
            total_frames - 1 - indices[release_mask]
        ) / float(release)
    return np.clip(envelope, 0.0, 1.0)


def _sound_java_files(
    *,
    package_name: str,
    mod_id: str,
    sound_ids: list[str],
) -> dict[str, str]:
    package_path = package_name.replace(".", "/")
    files: dict[str, str] = {}
    for sound_id in sound_ids:
        class_name = _sound_unit_class_name(sound_id)
        files[
            f"src/main/java/{package_path}/sound/{class_name}.java"
        ] = _sound_shard_java(
            package_name,
            mod_id,
            class_name,
            [sound_id],
        )
    files[
        f"src/main/java/{package_path}/sound/GeneratedSounds.java"
    ] = f'''package {package_name}.sound;

import net.fabricmc.loader.api.FabricLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.TreeSet;

public final class GeneratedSounds {{
    private static boolean registered;
    private GeneratedSounds() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        Set<String> classes = new TreeSet<>();
        String relative = "{package_path}/sound";
        FabricLoader.getInstance().getModContainer("{mod_id}").orElseThrow().getRootPaths().forEach(root -> {{
            Path directory = root.resolve(relative);
            if (!Files.isDirectory(directory)) return;
            try (var paths = Files.list(directory)) {{
                paths.filter(path -> {{
                    String name = path.getFileName().toString();
                    return name.startsWith("GeneratedSoundUnit") && name.endsWith(".class");
                }}).forEach(path -> {{
                    String name = path.getFileName().toString();
                    classes.add("{package_name}.sound." + name.substring(0, name.length() - 6));
                }});
            }} catch (java.io.IOException error) {{
                throw new IllegalStateException("Could not enumerate generated sound units", error);
            }}
        }});
        for (String className : classes) {{
            try {{
                Class<?> unit = Class.forName(
                    className,
                    true,
                    GeneratedSounds.class.getClassLoader()
                );
                unit.getMethod("register").invoke(null);
            }} catch (ReflectiveOperationException error) {{
                throw new IllegalStateException(
                    "Could not register generated sound unit " + className,
                    error
                );
            }}
        }}
    }}
}}
'''
    return files


def _sound_unit_class_name(sound_id: str) -> str:
    return (
        "GeneratedSoundUnit"
        + hashlib.sha256(sound_id.encode("utf-8")).hexdigest()[:20]
    )


def _sound_shard_java(
    package_name: str,
    mod_id: str,
    class_name: str,
    sound_ids: list[str],
) -> str:
    declarations = "\n".join(
        f"    public static SoundEvent {_constant(value)};"
        for value in sound_ids
    )
    registrations = "\n".join(
        f'''        {_constant(value)} = Registry.register(
            Registries.SOUND_EVENT,
            new Identifier(MOD_ID, "{value}"),
            SoundEvent.of(new Identifier(MOD_ID, "{value}"))
        );'''
        for value in sound_ids
    )
    return f'''package {package_name}.sound;

import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.sound.SoundEvent;
import net.minecraft.util.Identifier;

public final class {class_name} {{
    private static final String MOD_ID = "{mod_id}";
    private static boolean registered;
{declarations}

    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
{registrations}
    }}
}}
'''


def _constant(value: str) -> str:
    rendered = "".join(
        character.upper()
        if character.isalnum() or character == "_"
        else "_"
        for character in value
    )
    if not rendered or rendered[0].isdigit():
        rendered = "SOUND_" + rendered
    return rendered


def _load_audio_manifest(
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], bool]:
    manifest_path = project_root / ".minecraft_ai/audio-assets.json"
    if not manifest_path.is_file():
        return {}, False
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") == "mmm/audio-asset-directory-v1":
        raw_directory = manifest.get("directory")
        expected = manifest.get("sound_count")
        if not isinstance(raw_directory, str) or type(expected) is not int:
            raise AudioGenerationError("Audio asset directory index is invalid.")
        directory = (project_root / raw_directory).resolve()
        try:
            directory.relative_to(project_root)
        except ValueError as exc:
            raise AudioGenerationError(
                "Audio asset directory escaped the project."
            ) from exc
        if not directory.is_dir() or directory.is_symlink():
            raise AudioGenerationError(
                "Audio asset directory is missing or unsafe."
            )
        entries: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            if not path.is_file() or path.is_symlink():
                raise AudioGenerationError(
                    "Audio asset record is unsafe."
                )
            item = _load_object(path)
            sound_id = str(item.get("sound_id", ""))
            if not sound_id or path.stem != sound_id:
                raise AudioGenerationError(
                    "Audio asset record is invalid."
                )
            entries[sound_id] = item
        if len(entries) != expected:
            raise AudioGenerationError(
                "Audio asset directory count does not match."
            )
        return entries, True
    return (
        {
            str(item["sound_id"]): dict(item)
            for item in manifest.get("sounds", [])
            if isinstance(item, dict) and item.get("sound_id")
        },
        False,
    )


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AudioGenerationError(
            f"Expected a JSON object: {path}"
        )
    return value


def _merge_lang(path: Path, additions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_object(path)
    current.update(additions)
    path.write_text(
        json.dumps(
            current,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
