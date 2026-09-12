"""HOST resource rules and exact subject bindings, independent of image models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import gcd
from typing import Any

# Canonical geometry belongs here, never in prompt templates or provider settings.
NATIVE_SPRITE_SIZE = (16, 16)


class ResourceCatalogError(ValueError):
    """Malformed or unresolved HOST resource facts."""


def host_binding(context: Any, subject: str) -> dict[str, Any]:
    facts = getattr(context, "facts", {})
    catalog = (
        facts.get("resource_asset_bindings", {}) if isinstance(facts, Mapping) else {}
    )
    if not isinstance(catalog, Mapping):
        raise ResourceCatalogError("HOST resource_asset_bindings must be a mapping.")
    binding = catalog.get(subject, {})
    if not isinstance(binding, Mapping):
        raise ResourceCatalogError(f"HOST resource binding is invalid: {subject}")

    def thaw(value):
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [thaw(item) for item in value]
        return value

    return json.loads(json.dumps(thaw(binding), allow_nan=False))


def _positive(value: Any, label: str, maximum: int = 4096) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"HOST {label} must be a positive bounded integer.")
    return value


def _regions(
    value: Any, width: int, height: int, *, protected: bool = False
) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ValueError("HOST layout requires explicit regions.")
    result, names = [], set()
    for row in value:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str)
            or not row["name"]
        ):
            raise ValueError("HOST region name is required.")
        name, box = row["name"], row.get("box")
        if (
            name in names
            or not isinstance(box, list)
            or len(box) != 4
            or any(type(v) is not int for v in box)
        ):
            raise ValueError("HOST region names and boxes must be exact and unique.")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError("HOST region escapes canonical geometry.")
        for previous in result:
            px, py, pw, ph = previous["box"]
            if x < px + pw and px < x + w and y < py + ph and py < y + h:
                raise ValueError("HOST regions overlap.")
        names.add(name)
        normalized = {"name": name, "box": box}
        if protected:
            rgba = row.get("rgba")
            if (
                not isinstance(rgba, list)
                or len(rgba) != 4
                or any(type(v) is not int or not 0 <= v <= 255 for v in rgba)
            ):
                raise ValueError("HOST protected region requires deterministic RGBA.")
            normalized["rgba"] = rgba
        result.append(normalized)
    return result


def resource_geometry(binding: Mapping[str, Any], render_kind: str) -> tuple[int, int]:
    layout_required = render_kind in {"entity.fixed_uv", "gui.sprite"}
    geometry = binding.get("geometry")
    if geometry is None:
        if layout_required:
            raise ValueError(
                f"HOST geometry/layout binding required for {render_kind}."
            )
        return NATIVE_SPRITE_SIZE
    if not isinstance(geometry, Mapping):
        raise ResourceCatalogError("HOST geometry must be a mapping.")
    return (
        _positive(geometry.get("width"), "width"),
        _positive(geometry.get("height"), "height"),
    )


def texture_contract(
    *,
    binding: Mapping[str, Any],
    render_kind: str,
    namespace: str,
    minecraft_version: str,
    target_path: str,
    role: str,
    width: int,
    height: int,
    topology: str,
    alpha: str,
) -> dict[str, Any]:
    animation = binding.get("animation", {})
    if not isinstance(animation, Mapping) or set(animation) - {
        "frame_count",
        "frametime",
    }:
        raise ValueError("HOST animation requires frame_count and frametime only.")
    frames = _positive(animation.get("frame_count", 1), "frame_count", 256)
    if width * height * frames > 16_777_216:
        raise ValueError("HOST texture exceeds bounded compositor pixel capacity.")
    frametime = _positive(animation.get("frametime", 1), "frametime")
    uv, gui = None, None
    if topology == "fixed_uv":
        uv = binding.get("uv_schema")
        if (
            not isinstance(uv, Mapping)
            or not uv.get("id")
            or not binding.get("model_binding")
        ):
            raise ValueError(
                "HOST entity requires UV schema and renderer/model binding."
            )
        uv = {
            "id": uv["id"],
            "regions": _regions(uv.get("regions"), width, height),
            "lora_compatible": uv.get("lora_compatible") is True,
        }
    if topology == "gui_sprite":
        gui = binding.get("gui")
        if not isinstance(gui, Mapping) or not binding.get("model_binding"):
            raise ValueError(
                "HOST GUI requires deterministic layout and screen binding."
            )
        gui = {
            "protected_regions": _regions(
                gui.get("protected_regions"), width, height, protected=True
            ),
            "generated_regions": _regions(gui.get("generated_regions"), width, height),
        }
        allowed = {"background", "frame", "decorative_border", "ornament"}
        if any(row["name"] not in allowed for row in gui["generated_regions"]):
            raise ValueError("HOST GUI generation is limited to decorative regions.")
    if frames > 1 and (uv or gui):
        raise ValueError(
            "HOST animated UV/GUI layouts are not admitted by this compositor."
        )
    if uv or gui:
        consumer = binding["model_binding"]
        if not isinstance(consumer, Mapping) or set(consumer) != {
            "path",
            "sha256",
            "texture_reference",
        }:
            raise ValueError(
                "HOST model_binding requires exact consumer path, sha256 and texture_reference."
            )
        import re

        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(consumer["sha256"])):
            raise ValueError("HOST model consumer hash is invalid.")
        texture_ref = namespace + ":" + target_path.split(f"assets/{namespace}/", 1)[1]
        if consumer["texture_reference"] != texture_ref:
            raise ValueError("HOST model consumer references a different texture.")
    divisor = gcd(width, height * frames)
    return {
        "schema_version": "mmm/resource-contract-v1",
        "asset_kind": render_kind.split(".")[0] + "_texture",
        "target": {"minecraft_version": minecraft_version},
        "resource": {"namespace": namespace, "path": target_path},
        "geometry": {
            "canonical_width": width,
            "canonical_height": height * frames,
            "aspect_ratio": [width // divisor, height * frames // divisor],
            "layout": "vertical_frames" if frames > 1 else topology,
        },
        "rendering": {
            "alpha": alpha,
            "tileable": topology == "seamless_tile",
            "uv_schema": uv,
        },
        "animation": {
            "animated": frames > 1,
            "frame_width": width,
            "frame_height": height,
            "frame_count": frames,
            "frametime": frametime,
            "layout": "vertical",
            "mcmeta_required": frames > 1,
        },
        "model": {
            "family": render_kind,
            "slot": role,
            "binding": binding.get("model_binding", render_kind),
        },
        "gui": gui,
        "generation_profile_ref": "image_generator",
        "postprocess": {
            "target_width": width,
            "target_height": height * frames,
            "interpolation": "nearest",
            "palette_colors": 32,
            "alpha_threshold": 128,
            "pixel_grid_preservation": True,
            "alpha_cleanup": True,
        },
    }
