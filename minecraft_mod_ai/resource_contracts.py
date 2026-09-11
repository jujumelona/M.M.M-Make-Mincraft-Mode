from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from packaging.version import Version

from .task_template_catalog import load_template

_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+$")
SUPPORTED_RENDER_KINDS = frozenset({
    "item.generated", "item.handheld", "item.block_model_reuse",
    "block.cube_all", "block.cube_column", "block.cube_bottom_top",
    "block.orientable", "block.cross", "block.crop",
    "entity.fixed_uv", "gui.sprite",
})
_DEFAULT_RENDER_KIND = {
    "item": "item.generated", "icon": "item.generated",
    "block": "block.cube_all", "environment": "block.cube_all",
    "entity": "entity.fixed_uv", "gui": "gui.sprite",
}
_MODULE_RENDER_KIND = {
    "item": "item.generated", "food": "item.generated", "armor": "item.generated",
    "tool": "item.handheld", "weapon": "item.handheld",
    "block": "block.cube_all", "machine": "block.orientable", "crop": "block.crop",
}
_RESOURCE_TEMPLATES = {
    "block.blockstate_simple": "minecraft/resource/block/blockstate_simple",
    "block.blockstate_crop": "minecraft/resource/block/blockstate_crop",
    "block.cube_all": "minecraft/resource/block/model_cube_all",
    "block.cube_column": "minecraft/resource/block/model_cube_column",
    "block.cube_bottom_top": "minecraft/resource/block/model_cube_bottom_top",
    "block.orientable": "minecraft/resource/block/model_orientable",
    "block.cross": "minecraft/resource/block/model_cross",
    "block.crop_stage": "minecraft/resource/block/model_crop_stage",
    "item.generated": "minecraft/resource/item/model_generated",
    "item.handheld": "minecraft/resource/item/model_handheld",
    "item.block_model_reuse": "minecraft/resource/item/model_block_reuse",
    "item.client_item": "minecraft/resource/item/client_item",
    "item.client_block_item": "minecraft/resource/item/client_block_item",
}


@dataclass(frozen=True)
class TextureSpec:
    role: str
    topology: str
    alpha_policy: str
    size_policy: str
    uv_policy: str
    animation_policy: str
    target_path: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceDocument:
    template_id: str
    target_path: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "target_path": self.target_path,
            "payload": json.loads(json.dumps(self.payload, sort_keys=True)),
        }


@dataclass(frozen=True)
class ResolvedResourceAsset:
    asset_id: str
    subject_id: str
    render_kind: str
    container: str
    textures: tuple[TextureSpec, ...]
    documents: tuple[ResourceDocument, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id, "subject_id": self.subject_id,
            "render_kind": self.render_kind, "container": self.container,
            "textures": [item.to_dict() for item in self.textures],
            "documents": [item.to_dict() for item in self.documents],
        }


def _normalize_subject(value: str, fallback: str) -> str:
    raw = str(value or fallback).strip().casefold().replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_./-]", "_", raw).strip("_/")
    if not raw or any(part in {"", ".", ".."} for part in PurePosixPath(raw).parts):
        raise ValueError(f"Unsafe resource subject: {value!r}")
    return raw


def _default_size(asset: Any) -> tuple[int, int]:
    width = getattr(asset, "requested_width", None)
    height = getattr(asset, "requested_height", None)
    if width is None and height is None:
        return 16, 16
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Explicit resource texture dimensions require two positive integers.")
    return width, height


def _prefix(container: str) -> str:
    if container == "mod":
        return "src/main/resources/"
    if container == "resource_pack":
        return ""
    raise ValueError(f"Unsupported resource container: {container!r}")


def _substitute(value: Any, values: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for key, replacement in values.items():
            out = out.replace("{{" + key + "}}", replacement)
        if "{{" in out or "}}" in out:
            raise ValueError(f"Unresolved resource template placeholder: {out!r}")
        return out
    if isinstance(value, list):
        return [_substitute(item, values) for item in value]
    if isinstance(value, Mapping):
        return {_substitute(k, values): _substitute(v, values) for k, v in value.items()}
    return value


def _render_document(template_id: str, values: Mapping[str, str], version_context: Any | None) -> ResourceDocument:
    template = load_template(template_id)
    if version_context is not None:
        version_context.admit_template(template)
    render, target = template.get("render"), template.get("target")
    if not isinstance(render, Mapping) or not isinstance(target, Mapping):
        raise ValueError(f"Resource template {template_id} is not renderable.")
    body, file_name = render.get("body"), target.get("file")
    if not isinstance(body, Mapping) or not isinstance(file_name, str):
        raise ValueError(f"Resource template {template_id} has an invalid render contract.")
    return ResourceDocument(template_id, str(_substitute(file_name, values)), _substitute(body, values))


def _slots(render_kind: str) -> tuple[tuple[str, str, str, str], ...]:
    table = {
        "block.cube_all": (("all", "seamless_tile", "opaque", "block"),),
        "block.cube_column": (("side", "seamless_tile", "opaque", "block"), ("end", "seamless_tile", "opaque", "block")),
        "block.cube_bottom_top": (("side", "seamless_tile", "opaque", "block"), ("bottom", "seamless_tile", "opaque", "block"), ("top", "seamless_tile", "opaque", "block")),
        "block.orientable": (("side", "seamless_tile", "opaque", "block"), ("front", "single_face", "opaque", "block"), ("top", "seamless_tile", "opaque", "block")),
        "block.cross": (("cross", "cutout_sprite", "cutout", "block"),),
        "block.crop": (("cross", "cutout_sprite", "cutout", "block"),),
        "item.generated": (("layer0", "isolated_sprite", "transparent", "item"),),
        "item.handheld": (("layer0", "isolated_sprite", "transparent", "item"),),
        "entity.fixed_uv": (("diffuse", "fixed_uv", "transparent", "entity"),),
        "gui.sprite": (("sprite", "gui_sprite", "transparent", "gui"),),
        "item.block_model_reuse": (),
    }
    try:
        return table[render_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported render kind: {render_kind!r}") from exc


def resolve_asset(asset: Any, *, namespace: str, minecraft_version: str, version_context: Any | None = None) -> ResolvedResourceAsset:
    namespace = str(namespace).strip()
    if not _RESOURCE_ID.fullmatch(namespace):
        raise ValueError(f"Invalid resource namespace: {namespace!r}")
    subject = _normalize_subject(getattr(asset, "subject_id", ""), getattr(asset, "asset_id", "asset"))
    render_kind = str(getattr(asset, "render_kind", "") or _DEFAULT_RENDER_KIND.get(str(asset.kind), "")).strip()
    if render_kind not in SUPPORTED_RENDER_KINDS:
        raise ValueError(f"Unsupported resource render kind: {render_kind!r}")
    container = str(getattr(asset, "container", "mod") or "mod")
    prefix = _prefix(container)
    width, height = _default_size(asset)
    variant_count = int(getattr(asset, "variant_count", 1) or 1)
    if variant_count < 1:
        raise ValueError("variant_count must be positive.")

    textures: list[TextureSpec] = []
    if render_kind == "block.crop":
        for index in range(variant_count):
            textures.append(TextureSpec(
                role=f"stage{index}", topology="cutout_sprite", alpha_policy="cutout",
                size_policy="minecraft_native_default" if (width, height) == (16, 16) else "explicit",
                uv_policy="cross", animation_policy="static",
                target_path=f"{prefix}assets/{namespace}/textures/block/{subject}_stage{index}.png",
                width=width, height=height,
            ))
    else:
        slots = _slots(render_kind)
        for role, topology, alpha, folder in slots:
            suffix = "" if len(slots) == 1 else f"_{role}"
            textures.append(TextureSpec(
                role=role, topology=topology, alpha_policy=alpha,
                size_policy="minecraft_native_default" if (width, height) == (16, 16) else "explicit",
                uv_policy="fixed" if topology == "fixed_uv" else "model_slot",
                animation_policy="static",
                target_path=f"{prefix}assets/{namespace}/textures/{folder}/{subject}{suffix}.png",
                width=width, height=height,
            ))

    values = {"mod_id": namespace, "registry_path": subject, "subject": subject}
    documents: list[ResourceDocument] = []
    modern_item = Version(minecraft_version) >= Version("1.21.4")
    if render_kind.startswith("block."):
        if render_kind == "block.crop":
            for index in range(variant_count):
                documents.append(_render_document(_RESOURCE_TEMPLATES["block.crop_stage"], {**values, "stage": str(index)}, version_context))
            crop_blockstate_template = load_template(_RESOURCE_TEMPLATES["block.blockstate_crop"])
            if version_context is not None:
                version_context.admit_template(crop_blockstate_template)
            documents.append(ResourceDocument(
                _RESOURCE_TEMPLATES["block.blockstate_crop"],
                f"{prefix}assets/{namespace}/blockstates/{subject}.json",
                {"variants": {f"age={index}": {"model": f"{namespace}:block/{subject}_stage{index}"} for index in range(variant_count)}},
            ))
        else:
            documents.append(_render_document(_RESOURCE_TEMPLATES[render_kind], values, version_context))
            documents.append(_render_document(_RESOURCE_TEMPLATES["block.blockstate_simple"], values, version_context))
        documents.append(_render_document(_RESOURCE_TEMPLATES["item.block_model_reuse"], values, version_context))
        if modern_item:
            documents.append(_render_document(_RESOURCE_TEMPLATES["item.client_block_item"], values, version_context))
    elif render_kind in {"item.generated", "item.handheld"}:
        documents.append(_render_document(_RESOURCE_TEMPLATES[render_kind], values, version_context))
        if modern_item:
            documents.append(_render_document(_RESOURCE_TEMPLATES["item.client_item"], values, version_context))
    if container == "resource_pack":
        rebased: list[ResourceDocument] = []
        source_prefix = "src/main/resources/"
        for document in documents:
            if not document.target_path.startswith(source_prefix):
                raise ValueError(
                    f"Standalone resource document is not rooted under {source_prefix!r}: {document.target_path!r}"
                )
            rebased.append(ResourceDocument(
                document.template_id, document.target_path[len(source_prefix):], document.payload
            ))
        documents = rebased
    return ResolvedResourceAsset(str(asset.asset_id), subject, render_kind, container, tuple(textures), tuple(documents))


def infer_render_kind(kind: str, *, target_path: str = "") -> str:
    normalized = str(target_path).replace("\\", "/")
    if kind == "item" and any(token in normalized for token in ("tool", "weapon")):
        return "item.handheld"
    return _DEFAULT_RENDER_KIND.get(str(kind), "")


def derive_module_asset_specs(modules: Sequence[Any], *, existing_asset_ids: Sequence[str] = ()) -> tuple[dict[str, Any], ...]:
    existing = set(existing_asset_ids)
    rows: list[dict[str, Any]] = []
    for module in modules:
        render_kind = _MODULE_RENDER_KIND.get(str(module.kind))
        if not render_kind:
            continue
        asset_kind = "block" if render_kind.startswith("block.") else "item"
        asset_id = f"texture_{asset_kind}_{module.module_id}"
        if asset_id in existing:
            continue
        config = module.config if isinstance(module.config, Mapping) else {}
        display = str(config.get("visual_description") or config.get("display_name_en") or config.get("name") or module.module_id.replace("_", " ")).strip()
        parts = [display, str(module.kind)]
        if config.get("material"):
            parts.append(str(config["material"]))
        color = config.get("color") or config.get("main_color")
        if color:
            parts.append(f"primary color {color}")
        motifs = config.get("motifs")
        if isinstance(motifs, Sequence) and not isinstance(motifs, (str, bytes)):
            parts.extend(str(item) for item in motifs if str(item).strip())
        variant_count = 1
        if module.kind == "crop":
            try:
                variant_count = max(1, int(config.get("growth_stages", config.get("stage_count", 8))))
            except (TypeError, ValueError):
                variant_count = 8
        rows.append({
            "asset_id": asset_id, "kind": asset_kind,
            "visual_description": ", ".join(parts), "render_kind": render_kind,
            "subject_id": module.module_id, "owner_module_id": module.module_id,
            "container": "mod", "variant_count": variant_count,
        })
        existing.add(asset_id)
    return tuple(rows)


__all__ = ["ResolvedResourceAsset", "ResourceDocument", "SUPPORTED_RENDER_KINDS", "TextureSpec", "derive_module_asset_specs", "infer_render_kind", "resolve_asset"]
