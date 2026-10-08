"""Commercial-permissive foreground alpha for opaque diffusion image assets.

FLUX.2 Klein 4B's local Diffusers output is RGB, not native transparent PNG.
The host requests a *specific* BiRefNet general segmentation checkpoint via
rembg (MIT). NEVER let rembg choose its default: its default Bria weights have
a separate non-commercial license.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

ALPHA_SEGMENTATION_MODEL = "birefnet-general"
ALPHA_SEGMENTATION_ENGINE = "rembg"
ALPHA_SEGMENTATION_PROVIDER = "CPUExecutionProvider"
ALPHA_SEGMENTATION_CONTRACT = "mmm/alpha-segmentation-birefnet-general-v1"


@lru_cache(maxsize=1)
def _get_session() -> Any:
    try:
        from rembg import new_session
    except ImportError as exc:
        raise ValueError(
            'ALPHA_SEGMENTER_UNAVAILABLE: install "rembg[cpu]" via the image extra; '
            "opaque diffusion sprites cannot be published without alpha segmentation."
        ) from exc
    # Explicit model ID is a licensing contract. rembg's DEFAULT is not allowed.
    return new_session(
        ALPHA_SEGMENTATION_MODEL, providers=[ALPHA_SEGMENTATION_PROVIDER],
    )


def segment_foreground(image: Any) -> Any:
    """Return a separate RGBA image with model-produced foreground alpha.

    Never fake transparency by cutting a geometry-shaped region, and never
    accept a grayscale/RGB image returned by a misconfigured backend.
    Run ONNX on CPU so a resident 4B diffusion model does not OOM Colab T4.
    """
    from PIL import Image

    try:
        from rembg import remove
    except ImportError as exc:
        raise ValueError(
            'ALPHA_SEGMENTER_UNAVAILABLE: install "rembg[cpu]" via the image extra.'
        ) from exc

    source = image.convert("RGB")
    try:
        result = remove(source, session=_get_session(), alpha_matting=False)
    finally:
        source.close()
    if not isinstance(result, Image.Image) or result.size != image.size:
        raise ValueError("ALPHA_SEGMENTER_INVALID_IMAGE_OR_GEOMETRY")
    if result.mode != "RGBA":
        result.close()
        raise ValueError("ALPHA_SEGMENTER_MISSING_RGBA_ALPHA")
    alpha = result.getchannel("A")
    try:
        low, high = alpha.getextrema()
        if low == high or high == 0:
            result.close()
            raise ValueError("ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT")
    finally:
        alpha.close()
    result.info["mmm_alpha_matte"] = ALPHA_SEGMENTATION_ENGINE + ":" + ALPHA_SEGMENTATION_MODEL
    return result


__all__ = [
    "ALPHA_SEGMENTATION_MODEL",
    "ALPHA_SEGMENTATION_ENGINE",
    "ALPHA_SEGMENTATION_PROVIDER",
    "ALPHA_SEGMENTATION_CONTRACT",
    "segment_foreground",
]
