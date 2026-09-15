from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .model_adapters.image_diffusion import ImageGenerationConfig
from .resource_contracts import TextureSpec
from .resource_visual_spec import VisualSpec, resolve_visual_spec
from .task_template_catalog import load_template

_TOPOLOGY = {
    "seamless_tile": "seamless tile with opposite edges matching exactly",
    "single_face": "single square block face, orthographic, edge-to-edge material",
    "cutout_sprite": "flat cutout sprite with crisp alpha edges",
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
    payload = {"profile": asdict(p), "adapter": config.adapter, "provider": config.provider,
               "base_url": config.base_url}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_texture_prompt(*, visual_description: str = "", texture: TextureSpec, image_config: Any,
                           visual_spec: VisualSpec | None = None, visual_bible: str = "",
                           feature_purpose: str = "") -> str:
    p = ImageGenerationConfig.from_adapter_config(image_config)
    description = (visual_spec or resolve_visual_spec(None, visual_description)).prompt_fragment()
    contract = texture.resource_contract
    if not contract:
        raise ValueError("Resolved resource contract is required by PromptCompiler.")
    layout = "uv_region" if texture.topology == "fixed_uv" else "gui_decoration" if texture.topology == "gui_sprite" else texture.topology
    if p.lora_model_id:
        if layout not in p.lora_allowed_layouts:
            raise ValueError(f"Registry LoRA policy does not admit {layout}.")
        uv = contract["rendering"]["uv_schema"]
        if uv and not uv["lora_compatible"]:
            raise ValueError("HOST UV contract does not guarantee LoRA compatibility.")
    topology = (load_template("asset/item_sprite")["render"]["body"] if texture.topology == "isolated_sprite"
                else _TOPOLOGY.get(texture.topology))
    alpha = _ALPHA.get(texture.alpha_policy)
    if topology is None or alpha is None:
        raise ValueError("Unsupported texture topology or alpha policy.")
    if layout == "uv_region":
        topology = "one semantic surface region for deterministic UV composition, no atlas layout"
    elif layout == "gui_decoration":
        topology = "decorative GUI layer only, no slots, buttons, text boxes, progress bars or icons"
    parts = [p.lora_trigger.strip(), "Minecraft Java resource texture", "pixel art", topology, alpha,
             f"texture role {texture.role}", visual_bible, feature_purpose, description, *p.prompt_requirements,
             "simple native-grid silhouette, avoid subpixel detail", "no text", "no numbers", "no watermark", "no screenshot", "no 3D scene"]
    return ", ".join(part for part in parts if part)


__all__ = ["compile_texture_prompt", "image_profile_fingerprint"]
