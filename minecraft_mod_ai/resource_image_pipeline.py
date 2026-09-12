"""Contract-driven pixel processing and composition; never generates layout."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _pixels(image: Any) -> Any:
    reader = getattr(image, "get_flattened_data", None)
    return reader() if callable(reader) else image.getdata()


def _contract(texture: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = texture.get("resource_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema_version") != "mmm/resource-contract-v1"
    ):
        raise ValueError("Resolved resource contract is required.")
    geometry, resource = contract["geometry"], contract["resource"]
    if (texture["width"], texture["height"]) != (
        geometry["canonical_width"],
        geometry["canonical_height"],
    ):
        raise ValueError("Texture geometry differs from resource contract.")
    if (
        texture["target_path"] != resource["path"]
        or texture["alpha_policy"] != contract["rendering"]["alpha"]
    ):
        raise ValueError("Texture path/alpha differs from resource contract.")
    return contract


def _clear_boundary_background(image: Any) -> None:
    """Remove only an unambiguous flat background connected to the canvas boundary."""
    width, height = image.size
    corners = [
        image.getpixel(point)
        for point in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    ]
    if (
        any(pixel[3] == 0 for pixel in corners)
        or len({pixel[:3] for pixel in corners}) != 1
    ):
        return
    background = corners[0][:3]
    pending = deque(
        [(x, y) for x in range(width) for y in (0, height - 1)]
        + [(x, y) for y in range(height) for x in (0, width - 1)]
    )
    seen = set()
    while pending:
        x, y = pending.popleft()
        if (x, y) in seen or not 0 <= x < width or not 0 <= y < height:
            continue
        seen.add((x, y))
        if image.getpixel((x, y))[:3] != background:
            continue
        image.putpixel((x, y), (0, 0, 0, 0))
        pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))


def postprocess_region(
    source: Any, contract: Mapping[str, Any], size: tuple[int, int]
) -> Any:
    from PIL import Image

    rules = contract["postprocess"]
    if rules["interpolation"] != "nearest" or not rules["pixel_grid_preservation"]:
        raise ValueError("Unsupported pixel-grid postprocess contract.")
    image = source.convert("RGBA").resize(size, Image.Resampling.NEAREST)
    policy = contract["rendering"]["alpha"]
    layout = contract["geometry"]["layout"]
    if policy in {"transparent", "cutout"} and layout not in {"fixed_uv", "gui_sprite"}:
        _clear_boundary_background(image)
    pixels = []
    for r, g, b, a in _pixels(image):
        a = 255 if policy == "opaque" or a >= rules["alpha_threshold"] else 0
        pixels.append((r, g, b, a) if a else (0, 0, 0, 0))
    image.putdata(pixels)
    alpha = image.getchannel("A")
    quantized = image.quantize(
        colors=rules["palette_colors"],
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    ).convert("RGBA")
    image.close()
    quantized.putalpha(alpha)
    alpha.close()
    if contract["rendering"]["tileable"]:
        w, h = quantized.size
        # Copy existing palette colors; averaging seams would create new colors.
        for y in range(h):
            quantized.putpixel((w - 1, y), quantized.getpixel((0, y)))
        for x in range(w):
            quantized.putpixel((x, h - 1), quantized.getpixel((x, 0)))
    return quantized


def generation_regions(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    uv = contract["rendering"]["uv_schema"]
    if uv:
        return uv["regions"]
    if contract["gui"]:
        return contract["gui"]["generated_regions"]
    animation = contract["animation"]
    width, height = animation["frame_width"], animation["frame_height"]
    return [
        {"name": f"frame_{index}", "box": [0, index * height, width, height]}
        for index in range(animation["frame_count"])
    ]


def generate_candidate(
    generate: Callable,
    texture: Mapping[str, Any],
    *,
    prompt: str,
    directory: Path,
    output: Path,
    resolution: tuple[int, int],
    fallback: tuple[int, int] | None,
    seed: int,
) -> dict[str, Any]:
    from PIL import Image

    from .model_adapters.base import ModelBackendError
    from .model_adapters.image_diffusion import _is_cuda_memory_pressure

    contract = _contract(texture)
    directory.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGBA", (texture["width"], texture["height"]), (0, 0, 0, 0))
    sources = []
    try:
        for index, region in enumerate(generation_regions(contract)):
            x, y, width, height = region["box"]
            source = directory / f"region-{index:03d}.png"
            region_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8], "big"
            ) & ((1 << 63) - 1)
            region_prompt = prompt + f", semantic region {region['name']}"
            selected_resolution = resolution
            kwargs = {
                "prompt": region_prompt,
                "output_path": source,
                "seed": region_seed,
            }
            try:
                generate(**kwargs, width=resolution[0], height=resolution[1])
            except ModelBackendError as exc:
                if (
                    fallback is None
                    or fallback == resolution
                    or not _is_cuda_memory_pressure(exc)
                ):
                    raise
                selected_resolution = fallback
                generate(**kwargs, width=fallback[0], height=fallback[1])
            if not source.is_file() or source.is_symlink():
                raise ValueError("Image backend produced no regular source PNG.")
            with Image.open(source) as raw:
                raw.load()
                if raw.format != "PNG" or raw.size != selected_resolution:
                    raise ValueError(
                        "Image backend output does not match generation profile geometry/PNG format."
                    )
                processed = postprocess_region(raw, contract, (width, height))
            canvas.paste(processed, (x, y))
            processed.close()
            sources.append(
                {
                    "region": region["name"],
                    "resolution": list(selected_resolution),
                    "sha256": "sha256:"
                    + hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        if contract["gui"]:
            for region in contract["gui"]["protected_regions"]:
                x, y, width, height = region["box"]
                canvas.paste(tuple(region["rgba"]), (x, y, x + width, y + height))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG", optimize=False)
    finally:
        canvas.close()
    validate_texture(output, texture)
    return {
        "sources": sources,
        "resource_contract_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_texture(
    path: Path, texture: Mapping[str, Any], *, check_metadata: bool = False
) -> dict[str, Any]:
    from PIL import Image

    contract = _contract(texture)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Required texture is missing or a symlink.")
    with Image.open(path) as raw:
        raw.load()
        if (
            raw.format != "PNG"
            or raw.mode != "RGBA"
            or raw.size != (texture["width"], texture["height"])
        ):
            raise ValueError("PNG format/geometry/alpha channel mismatch.")
        rendering, animation = contract["rendering"], contract["animation"]
        all_alpha = set(_pixels(raw.getchannel("A")))
        if not all_alpha <= {0, 255} or 255 not in all_alpha:
            raise ValueError("Invalid alpha/pixel-grid coverage.")
        if rendering["alpha"] == "opaque" and all_alpha != {255}:
            raise ValueError("Opaque texture has transparent alpha.")
        if (
            rendering["alpha"] in {"transparent", "cutout"}
            and not rendering["uv_schema"]
            and not contract["gui"]
            and 0 not in all_alpha
        ):
            raise ValueError(
                "Sprite requires transparent alpha outside its silhouette."
            )
        regions = generation_regions(contract)
        for region in regions:
            x, y, width, height = region["box"]
            with raw.crop((x, y, x + width, y + height)) as part:
                if not part.getchannel("A").getbbox():
                    raise ValueError("UV/animation region has no visible coverage.")
                protected_colors = (
                    len(contract["gui"]["protected_regions"]) if contract["gui"] else 0
                )
                if (
                    part.getcolors(
                        contract["postprocess"]["palette_colors"] + protected_colors
                    )
                    is None
                ):
                    raise ValueError("Pixel-grid palette exceeds resource contract.")
                if rendering["tileable"] and (
                    any(
                        part.getpixel((0, py)) != part.getpixel((width - 1, py))
                        for py in range(height)
                    )
                    or any(
                        part.getpixel((px, 0)) != part.getpixel((px, height - 1))
                        for px in range(width)
                    )
                ):
                    raise ValueError("Non-matching tile edges.")
        if rendering["uv_schema"] or contract["gui"]:
            mask = Image.new("L", raw.size, 0)
            covered_regions = regions + (
                contract["gui"]["protected_regions"] if contract["gui"] else []
            )
            for region in covered_regions:
                x, y, width, height = region["box"]
                mask.paste(255, (x, y, x + width, y + height))
            if any(
                a and not allowed
                for a, allowed in zip(_pixels(raw.getchannel("A")), _pixels(mask))
            ):
                raise ValueError("UV/GUI pixels escape canonical regions.")
            mask.close()
        if contract["gui"]:
            for region in contract["gui"]["protected_regions"]:
                x, y, width, height = region["box"]
                with raw.crop((x, y, x + width, y + height)) as part:
                    if any(pixel != tuple(region["rgba"]) for pixel in _pixels(part)):
                        raise ValueError("GUI protected region was modified.")
    if (
        check_metadata
        and not animation["mcmeta_required"]
        and Path(str(path) + ".mcmeta").exists()
    ):
        raise ValueError("Static texture conflicts with existing animation mcmeta.")
    if check_metadata and animation["mcmeta_required"]:
        metadata = Path(str(path) + ".mcmeta")
        expected = {
            "animation": {
                "width": animation["frame_width"],
                "height": animation["frame_height"],
                "frametime": animation["frametime"],
                "frames": list(range(animation["frame_count"])),
            }
        }
        if (
            not metadata.is_file()
            or metadata.is_symlink()
            or json.loads(metadata.read_text(encoding="utf-8")) != expected
        ):
            raise ValueError("Animation mcmeta missing or mismatched.")
    return {
        "status": "PASS",
        "path": str(path),
        "geometry": [texture["width"], texture["height"]],
        "checked_regions": len(regions),
        "animation_metadata_checked": check_metadata and animation["mcmeta_required"],
    }


def validate_model_consumers(root: Path, textures: list[Mapping[str, Any]]) -> None:
    from .resource_asset_production import _safe_target

    for texture in textures:
        contract = _contract(texture)
        if not contract["rendering"]["uv_schema"] and not contract["gui"]:
            continue
        consumer = contract["model"]["binding"]
        path = _safe_target(root, consumer["path"])
        if not path.is_file():
            raise ValueError("HOST model consumer artifact is missing.")
        data = path.read_bytes()
        if "sha256:" + hashlib.sha256(data).hexdigest() != consumer["sha256"]:
            raise ValueError("HOST model consumer artifact hash mismatch.")
        literal = json.dumps(consumer["texture_reference"]).encode()
        if literal not in data:
            raise ValueError(
                "HOST model consumer lacks its exact texture reference literal."
            )
