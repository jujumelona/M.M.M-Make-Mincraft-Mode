from __future__ import annotations

import hashlib
import json
from typing import Any

from .model_adapters.image_diffusion import ImageGenerationConfig
from .resource_contracts import TextureSpec

_TOPOLOGY = {
    "seamless_tile": "seamless tile with opposite edges matching exactly",
    "single_face": "single square block face, orthographic, edge-to-edge material",
    "cutout_sprite": "flat cutout sprite with crisp alpha edges",
    "isolated_sprite": "isolated inventory sprite, centered, no background scene",
    "fixed_uv": "fixed UV texture atlas, preserve layout and coverage",
    "gui_sprite": "flat GUI sprite, front-facing, no perspective",
}
_ALPHA = {
    "opaque": "fully opaque",
    "transparent": "transparent background outside the subject",
    "cutout": "binary-style cutout transparency, no soft photographic halo",
}


def image_profile_fingerprint(config: Any) -> str:
    p = ImageGenerationConfig.from_adapter_config(config)
    payload = {
        "model_id": p.model_id, "quantization": p.quantization, "torch_dtype": p.torch_dtype,
        "lora_model_id": p.lora_model_id, "lora_weight_name": p.lora_weight_name,
        "lora_adapter_name": p.lora_adapter_name, "lora_scale": p.lora_scale,
        "lora_trigger": p.lora_trigger, "num_inference_steps": p.num_inference_steps,
        "guidance_scale": p.guidance_scale, "candidate_count": p.candidate_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_texture_prompt(*, visual_description: str, texture: TextureSpec, image_config: Any) -> str:
    p = ImageGenerationConfig.from_adapter_config(image_config)
    description = str(visual_description).strip()
    if not description:
        raise ValueError("visual_description must not be empty.")
    topology = _TOPOLOGY.get(texture.topology)
    alpha = _ALPHA.get(texture.alpha_policy)
    if topology is None or alpha is None:
        raise ValueError("Unsupported texture topology or alpha policy.")
    parts = [p.lora_trigger.strip(), "Minecraft Java resource texture", "pixel art", topology, alpha,
             f"texture role {texture.role}", description, "no text", "no watermark", "no screenshot", "no 3D scene"]
    return ", ".join(part for part in parts if part)


__all__ = ["compile_texture_prompt", "image_profile_fingerprint"]
