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
            "minecraft": "1.20.1",
            "java": 17,
            "loader": "0.16.10",
            "fabric_api": "0.92.11+1.20.1",
            "yarn": "1.20.1+build.1",
            "loom": "1.5.4",
        },
    }


def local_model_ids(profile: str = "t4_local") -> tuple[str, ...]:
    loaded = ModelRegistry().load_profile(profile)
    return tuple(
        sorted(
            {
                config.model_id
                for config in loaded.roles.values()
                if config.provider == "local" and config.model_id
            }
        )
    )


def generate_download_script(
    output_path: str | Path = "download_models.sh",
    *,
    profile: str = "t4_local",
) -> str:
    target = Path(output_path)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Downloads the exact repositories in config/model_registry.yaml.",
        "# Gated repositories require HF_TOKEN and prior license acceptance.",
        "command -v huggingface-cli >/dev/null || { echo 'Install huggingface_hub first.' >&2; exit 2; }",
    ]
    for model_id in local_model_ids(profile):
        lines.append(f"huggingface-cli download {model_id} --resume-download")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o755)
    return str(target)


def download_models(
    *,
    profile: str = "t4_local",
    cache_dir: str | Path | None = None,
) -> list[str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the local-model extra before downloading models.") from exc
    downloaded: list[str] = []
    for model_id in local_model_ids(profile):
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=(str(cache_dir) if cache_dir is not None else None),
            resume_download=True,
        )
        downloaded.append(path)
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
