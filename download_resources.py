"""Inspect or download the exact local models declared by config/model_registry.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.runner import GRADLE_SHA256, GRADLE_URL, GRADLE_VERSION


def resource_manifest() -> dict[str, object]:
    return {
        "model_registry": ModelRegistry().to_public_dict(),
        "build_tool": {
            "name": "Gradle",
            "version": GRADLE_VERSION,
            "url": GRADLE_URL,
            "sha256": GRADLE_SHA256,
        },
        "minecraft_target": {
            "selection": "host-selected PlatformLock",
            "historical_default": None,
            "discovery": "platform provider receipt",
        },
    }


def local_model_specs(profile: str = "t4_local") -> tuple[tuple[str, str], ...]:
    """Return unique (repo_id, exact_file) specs; exact_file is empty for snapshots."""
    loaded = ModelRegistry().load_profile(profile)
    specs: set[tuple[str, str]] = set()
    for config in loaded.roles.values():
        if config.provider != "local" or not config.model_id:
            continue
        filename = str(config.extra.get("gguf_filename", "")).strip()
        specs.add((config.model_id, filename))
    return tuple(sorted(specs))


def local_model_ids(profile: str = "t4_local") -> tuple[str, ...]:
    """Backward-compatible repository ID view used by manifests/tests."""
    return tuple(sorted({repo_id for repo_id, _filename in local_model_specs(profile)}))


def generate_download_script(
    output_path: str | Path = "download_models.sh",
    *,
    profile: str = "t4_local",
) -> str:
    target = Path(output_path)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Downloads only the artifacts required by config/model_registry.yaml.",
        "# GGUF roles download one exact quantized file; other model roles keep snapshots.",
        "command -v hf >/dev/null || { echo 'Install huggingface_hub first.' >&2; exit 2; }",
    ]
    for model_id, filename in local_model_specs(profile):
        if filename:
            lines.append(f"hf download {model_id} {filename}")
        else:
            lines.append(f"hf download {model_id}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o755)
    return str(target)


def download_models(
    *,
    profile: str = "t4_local",
    cache_dir: str | Path | None = None,
) -> list[str]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the local-model extra before downloading models.") from exc

    cache = str(cache_dir) if cache_dir is not None else None
    downloaded: list[str] = []
    for model_id, filename in local_model_specs(profile):
        if filename:
            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=cache,
            )
        else:
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache,
            )
        downloaded.append(str(path))
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="t4_local")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--script", nargs="?", const="download_models.sh")
    args = parser.parse_args()
    if args.script:
        print(generate_download_script(args.script, profile=args.profile))
        return
    if args.download:
        print(json.dumps(download_models(profile=args.profile, cache_dir=args.cache_dir), indent=2))
        return
    print(json.dumps(resource_manifest(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
