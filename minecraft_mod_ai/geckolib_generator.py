from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


class GeckoLibGenerationError(ValueError):
    pass


def generate_geckolib_entity_assets(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    entity_id: str,
    texture_width: int = 64,
    texture_height: int = 64,
    geckolib_version: str = "4.8.2",
) -> dict[str, Any]:
    """Generate GeckoLib 4.8.2 resources and typed Java binding points.

    The output is intentionally build-gated. It does not register a new entity by
    itself; the MinecraftCoder must bind these generated classes to an approved
    entity type, then Gradle and runtime validation must pass.
    """

    if not _ID.fullmatch(mod_id) or not _ID.fullmatch(entity_id):
        raise GeckoLibGenerationError("Invalid mod or entity id.")
    if not _PACKAGE.fullmatch(package_name):
        raise GeckoLibGenerationError("Invalid Java package.")
    if texture_width not in {16, 32, 64, 128, 256} or texture_height not in {
        16,
        32,
        64,
        128,
        256,
    }:
        raise GeckoLibGenerationError("Unsupported GeckoLib texture size.")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    class_name = "".join(part.capitalize() for part in entity_id.split("_"))
    assets = root / "src" / "main" / "resources" / "assets" / mod_id
    geo_path = assets / "geo" / f"{entity_id}.geo.json"
    animation_path = assets / "animations" / f"{entity_id}.animation.json"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    animation_path.parent.mkdir(parents=True, exist_ok=True)
    geo = {
        "format_version": "1.12.0",
        "minecraft:geometry": [
            {
                "description": {
                    "identifier": f"geometry.{mod_id}.{entity_id}",
                    "texture_width": texture_width,
                    "texture_height": texture_height,
                    "visible_bounds_width": 2.5,
                    "visible_bounds_height": 3.0,
                    "visible_bounds_offset": [0, 1.0, 0],
                },
                "bones": [
                    {
                        "name": "root",
                        "pivot": [0, 0, 0],
                        "cubes": [
                            {
                                "origin": [-4, 0, -4],
                                "size": [8, 12, 8],
                                "uv": [0, 0],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    animation = {
        "format_version": "1.8.0",
        "animations": {
            f"animation.{mod_id}.{entity_id}.idle": {
                "loop": True,
                "animation_length": 2.0,
                "bones": {
                    "root": {
                        "rotation": {
                            "0.0": [0, 0, 0],
                            "1.0": [0, 1.5, 0],
                            "2.0": [0, 0, 0],
                        }
                    }
                },
            }
        },
    }
    _write_json(geo_path, geo)
    _write_json(animation_path, animation)

    java_root = (
        root
        / "src"
        / "main"
        / "java"
        / Path(*package_name.split("."))
        / "client"
        / "geckolib"
    )
    java_root.mkdir(parents=True, exist_ok=True)
    model_path = java_root / f"{class_name}GeoModel.java"
    binding_path = java_root / f"{class_name}GeoBinding.java"
    model_path.write_text(
        f"""package {package_name}.client.geckolib;

public final class {class_name}GeoModel {{
    public static final String MODEL = \"{mod_id}:geo/{entity_id}.geo.json\";
    public static final String TEXTURE = \"{mod_id}:textures/entity/{entity_id}.png\";
    public static final String ANIMATION = \"{mod_id}:animations/{entity_id}.animation.json\";

    private {class_name}GeoModel() {{}}
}}
""",
        encoding="utf-8",
    )
    binding_path.write_text(
        f"""package {package_name}.client.geckolib;

/**
 * Build-gated binding contract. MinecraftCoder must connect this immutable
 * descriptor to a GeckoLib 4 GeoModel/renderer for the approved entity type.
 */
public record {class_name}GeoBinding(
        String model,
        String texture,
        String animation,
        String idleAnimationName
) {{
    public static {class_name}GeoBinding create() {{
        return new {class_name}GeoBinding(
                {class_name}GeoModel.MODEL,
                {class_name}GeoModel.TEXTURE,
                {class_name}GeoModel.ANIMATION,
                \"animation.{mod_id}.{entity_id}.idle\"
        );
    }}
}}
""",
        encoding="utf-8",
    )
    gradle_snippet = root / ".minecraft_ai" / "geckolib.gradle.snippet"
    gradle_snippet.parent.mkdir(parents=True, exist_ok=True)
    gradle_snippet.write_text(
        f"""// Review and merge into build.gradle, then run Gradle.
repositories {{
    maven {{
        name = 'GeckoLib'
        url 'https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/'
        content {{
            includeGroupByRegex(\"software\\\\.bernie.*\")
            includeGroup(\"com.eliotlash.mclib\")
        }}
    }}
}}
dependencies {{
    modImplementation(\"software.bernie.geckolib:geckolib-fabric-1.20.1:{geckolib_version}\")
    implementation(\"com.eliotlash.mclib:mclib:20\")
}}
""",
        encoding="utf-8",
    )
    return {
        "schema_version": "mmm/geckolib-generation-v1",
        "entity_id": entity_id,
        "geckolib_version": geckolib_version,
        "files": [
            str(path)
            for path in (geo_path, animation_path, model_path, binding_path, gradle_snippet)
        ],
        "status": "binding_and_build_required",
        "required_gates": [
            "merge reviewed Gradle dependency snippet",
            "bind approved entity type to GeoModel and GeoEntityRenderer",
            "validate UV and texture in Blockbench",
            "Gradle clean build",
            "GameTest",
            "runtime animation screenshot review",
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
