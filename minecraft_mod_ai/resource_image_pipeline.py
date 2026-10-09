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
    source: Any, contract: Mapping[str, Any], size: tuple[int, int],
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
    # Reserve one RGB palette slot for transparent black on *any* cutout,
    # including BiRefNet output. Otherwise 32 opaque colors + 1 transparent
    # entry would violate the host's exact PNG palette-size contract.
    alpha_bounds = image.getchannel("A").getextrema()
    reserve_alpha_color = (
        policy in {"transparent", "cutout"} and alpha_bounds[0] == 0
    )
    quantization_colors = rules["palette_colors"] - int(reserve_alpha_color)
    if quantization_colors < 1:
        raise ValueError("Resource palette cannot reserve alpha silhouette color.")
    alpha = image.getchannel("A")
    quantized = image.quantize(
        colors=quantization_colors,
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


def frame_segmented_foreground(source: Any, *, alpha_threshold: int = 128) -> Any:
    """Fit an ONNX-produced, visible subject to the native sprite canvas.

    The diffusion model renders a large RGB image, while Minecraft items are
    usually only 16x16. Resizing the entire image can sample *only background*
    at the 16x16 nearest-neighbor grid even if the segmentation mask was valid.
    Crop to the actual alpha silhouette before downscaling; never invent alpha
    or accept an empty/low-confidence segmentation mask.
    """
    from PIL import Image

    if source.mode != "RGBA":
        raise ValueError("ALPHA_SEGMENTER_MISSING_RGBA_ALPHA")
    alpha = source.getchannel("A")
    try:
        extrema = alpha.getextrema()
        if extrema[1] < alpha_threshold:
            raise ValueError(
                f"ALPHA_MASK_BELOW_NATIVE_THRESHOLD: max_alpha={extrema[1]} "
                f"threshold={alpha_threshold}"
            )
        silhouette = alpha.point(lambda value: 255 if value >= alpha_threshold else 0)
        try:
            bounds = silhouette.getbbox()
        finally:
            silhouette.close()
    finally:
        alpha.close()
    if bounds is None:
        raise ValueError("ALPHA_MASK_EMPTY_AT_SOURCE")
    # A model that marks nearly the entire image as foreground has not
    # isolated an object; scaling such a mask would preserve the defect.
    x0, y0, x1, y1 = bounds
    if (x1 - x0) * (y1 - y0) >= source.width * source.height * 0.98:
        raise ValueError(
            f"ALPHA_MASK_NOT_ISOLATED: bbox={bounds} size={source.size}"
        )
    with source.crop(bounds) as subject:
        target_width, target_height = source.size
        scale = min(
            0.82 * target_width / max(1, subject.width),
            0.82 * target_height / max(1, subject.height),
        )
        width = max(1, min(target_width, round(subject.width * scale)))
        height = max(1, min(target_height, round(subject.height * scale)))
        with subject.resize((width, height), Image.Resampling.LANCZOS) as fitted:
            canvas = Image.new("RGBA", source.size, (0, 0, 0, 0))
            canvas.paste(
                fitted, ((target_width - width) // 2, (target_height - height) // 2)
            )
    return canvas


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


def prepare_candidate_sources(
    generate: Callable,
    texture: Mapping[str, Any],
    *,
    prompt: str,
    directory: Path,
    resolution: tuple[int, int],
    seed: int,
) -> list[tuple[dict[str, Any], Path, bool]]:
    """Run only FLUX generation and persist its actual RGB/alpha source images.

    Separating generation from matting permits all assets in a work shard to
    share one FLUX model residency before its CPU-offloaded weights are freed.
    """
    from PIL import Image

    contract = _contract(texture)
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, region in enumerate(generation_regions(contract)):
        source = directory / f"region-{index:03d}.png"
        region_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        generate(
            prompt=prompt + f", semantic region {region['name']}",
            output_path=source,
            seed=region_seed,
            width=resolution[0],
            height=resolution[1],
        )
        if not source.is_file() or source.is_symlink():
            raise ValueError("Image backend produced no regular source PNG.")
        with Image.open(source) as raw:
            raw.load()
            if raw.format != "PNG" or raw.size != resolution:
                raise ValueError(
                    "Image backend output does not match generation profile geometry/PNG format."
                )
            needs_segmentation = (
                contract["rendering"]["alpha"] in {"transparent", "cutout"}
                and contract["geometry"]["layout"] in {"isolated_sprite", "cutout_sprite"}
                and (
                    raw.mode not in {"RGBA", "LA"}
                    or raw.getchannel("A").getextrema() == (255, 255)
                )
            )
        pending.append((region, source, needs_segmentation))
    return pending


def finalize_candidate_sources(
    texture: Mapping[str, Any],
    pending: list[tuple[dict[str, Any], Path, bool]],
    *,
    output: Path,
    resolution: tuple[int, int],
    segment_foreground_callback: Callable | None = None,
) -> dict[str, Any]:
    """Matte already-generated PNGs without launching or reloading FLUX."""
    from PIL import Image
    from .resource_alpha_segmentation import segment_foreground_isolated as segment_foreground

    contract = _contract(texture)
    canvas = Image.new("RGBA", (texture["width"], texture["height"]), (0, 0, 0, 0))
    sources = []
    try:
        for region, source, needs_segmentation in pending:
            x, y, width, height = region["box"]
            if not source.is_file() or source.is_symlink():
                raise ValueError("Generated image source is missing before matting.")
            with Image.open(source) as raw:
                raw.load()
                if needs_segmentation:
                    matting = (
                        segment_foreground if segment_foreground_callback is None
                        else segment_foreground_callback
                    )
                    extracted = matting(raw)
                    try:
                        matte_method = extracted.info.get(
                            "mmm_alpha_matte", "rembg:birefnet-general-lite"
                        )
                        # Segmenter masks can be valid at source resolution
                        # but disappear under nearest-neighbor 16x16 sampling.
                        # Frame only real mask pixels; fail if the model did
                        # not identify a foreground at all.
                        with frame_segmented_foreground(
                            extracted,
                            alpha_threshold=int(contract["postprocess"]["alpha_threshold"]),
                        ) as framed:
                            processed = postprocess_region(
                                framed, contract, (width, height),
                            )
                    finally:
                        extracted.close()
                else:
                    processed = postprocess_region(raw, contract, (width, height))
                    matte_method = "source_alpha_or_boundary_background"
            try:
                canvas.paste(processed, (x, y))
            finally:
                processed.close()
            sources.append({
                "region": region["name"],
                "resolution": list(resolution),
                "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                "alpha_matte": matte_method,
            })
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


def generate_candidate(
    generate: Callable,
    texture: Mapping[str, Any],
    *,
    prompt: str,
    directory: Path,
    output: Path,
    resolution: tuple[int, int],
    seed: int,
    segment_foreground_callback: Callable | None = None,
    before_segmentation: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Compatibility entry: FLUX once, release, then actual segmentation."""
    pending = prepare_candidate_sources(
        generate, texture, prompt=prompt, directory=directory,
        resolution=resolution, seed=seed,
    )
    if any(needs for _, _, needs in pending) and before_segmentation is not None:
        before_segmentation()
    return finalize_candidate_sources(
        texture, pending, output=output, resolution=resolution,
        segment_foreground_callback=segment_foreground_callback,
    )


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
