"""Show the optional local-model and verified build resources.

Core generation and deterministic validation have no model dependency.
`mmm ui --local-model` downloads the optional Hugging Face model
through Transformers on first use.  Gradle is fetched by the build runner and
checked against the published SHA-256 before execution.
"""

from __future__ import annotations

import json

from minecraft_mod_ai.runner import GRADLE_SHA256, GRADLE_URL, GRADLE_VERSION


def resource_manifest() -> dict[str, object]:
    return {
        "optional_planner_model": {
            "id": "Qwen/Qwen3-4B-Instruct-2507",
            "purpose": "optional constrained ModSpec candidate drafting",
            "required_for_core_pipeline": False,
            "authority": "never authorizes tools or changes the platform lock",
        },
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


def print_model_catalog() -> None:
    manifest = resource_manifest()
    print("=== PDF v6 Model & Resource Catalog ===")
    print(f"• Optional LLM Model : Qwen/Qwen3.5-9B-Instruct / google/gemma-4-12B-it")
    print(f"• Build Tool         : {manifest['build_tool']['name']} {manifest['build_tool']['version']}")
    print(f"• Target Platform    : Minecraft {manifest['minecraft_target']['minecraft']} (Fabric Loader {manifest['minecraft_target']['loader']})")


def generate_download_script(output_path: str = "download_models.sh") -> str:
    script_content = "#!/usr/bin/env bash\n# PDF v6 Resource Download Helper\necho 'Downloading PDF v6 models...'\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    return output_path


if __name__ == "__main__":
    print(json.dumps(resource_manifest(), ensure_ascii=False, indent=2))

