from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .resource_catalog import host_binding, resource_geometry, texture_contract
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
    resource_contract: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceDocument:
    template_id: str
    target_path: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"template_id": self.template_id, "target_path": self.target_path, "payload": json.loads(json.dumps(self.payload, sort_keys=True))}


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
    raw = str(value or fallback)
    if not re.fullmatch(r"[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*", raw) or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise ValueError(f"Unsafe resource subject: {value!r}")
    return raw


def _texture_size(asset: Any, render_kind: str, binding: Mapping[str, Any]) -> tuple[int, int, str]:
    canonical = resource_geometry(binding, render_kind)
    width = getattr(asset, "requested_width", None)
    height = getattr(asset, "requested_height", None)
    if width is None and height is None:
        return *canonical, "host_catalog"
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Explicit resource texture dimensions require two positive integers.")
    if (width, height) != canonical:
        raise ValueError("Requested texture dimensions differ from HOST geometry.")
    return width, height, "host_catalog"


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


def _host_requires_resource_template(version_context: Any, *, leaf: str, template_id: str) -> bool:
    facts = getattr(version_context, "facts", None)
    bindings = facts.get("leaf_bindings") if isinstance(facts, Mapping) else None
    binding = bindings.get(leaf) if isinstance(bindings, Mapping) else None
    implementation = binding.get("implementation") if isinstance(binding, Mapping) else None
    if not isinstance(implementation, Mapping):
        raise ValueError(f"HOST resource leaf binding unavailable: {leaf}")
    extras = implementation.get("extra_templates", ())
    if extras is None:
        extras = ()
    if not isinstance(extras, Sequence) or isinstance(extras, (str, bytes)):
        raise ValueError(f"HOST resource leaf extra_templates invalid: {leaf}")
    return template_id in {str(item) for item in extras}


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


def resolve_asset(asset: Any, *, namespace: str, minecraft_version: str, version_context: Any | None = None,
                  owner_module: Any | None = None) -> ResolvedResourceAsset:
    namespace = str(namespace).strip()
    if not _RESOURCE_ID.fullmatch(namespace):
        raise ValueError(f"Invalid resource namespace: {namespace!r}")
    subject = _normalize_subject(getattr(asset, "subject_id", ""), getattr(asset, "asset_id", "asset"))
    render_kind = str(getattr(asset, "render_kind", "") or _DEFAULT_RENDER_KIND.get(str(asset.kind), "")).strip()
    if render_kind not in SUPPORTED_RENDER_KINDS:
        raise ValueError(f"Unsupported resource render kind: {render_kind!r}")
    container = str(getattr(asset, "container", "mod") or "mod")
    prefix = _prefix(container)
    binding = host_binding(version_context, subject)
    if owner_module is not None and subject != owner_module.module_id:
        raise ValueError("Asset subject differs from exact HOST owner module.")
    module_render_kind = _MODULE_RENDER_KIND.get(str(owner_module.kind)) if owner_module else None
    host_render_kind = binding.get("render_kind", module_render_kind or _DEFAULT_RENDER_KIND.get(str(asset.kind)))
    if host_render_kind != render_kind:
        raise ValueError("Asset render kind differs from HOST resource binding.")
    width, height, size_policy = _texture_size(asset, render_kind, binding)
    module_config = owner_module.config if owner_module else {}
    variant_count = binding.get("variant_count", module_config.get("growth_stages", module_config.get("stage_count"))) if render_kind == "block.crop" else 1
    if type(variant_count) is not int or variant_count < 1 or variant_count > 256:
        raise ValueError("HOST crop stage cardinality is unresolved or invalid.")
    if getattr(asset, "variant_count", 1) != variant_count:
        raise ValueError("Requested texture cardinality differs from HOST content binding.")

    textures: list[TextureSpec] = []
    if render_kind == "block.crop":
        for index in range(variant_count):
            textures.append(TextureSpec(role=f"stage{index}", topology="cutout_sprite", alpha_policy="cutout", size_policy=size_policy, uv_policy="cross", animation_policy="static", target_path=f"{prefix}assets/{namespace}/textures/block/{subject}_stage{index}.png", width=width, height=height))
    else:
        slots = _slots(render_kind)
        for role, topology, alpha, folder in slots:
            suffix = "" if len(slots) == 1 else f"_{role}"
            textures.append(TextureSpec(role=role, topology=topology, alpha_policy=alpha, size_policy=size_policy, uv_policy="fixed" if topology == "fixed_uv" else "model_slot", animation_policy="static", target_path=f"{prefix}assets/{namespace}/textures/{folder}/{subject}{suffix}.png", width=width, height=height))

    values = {"mod_id": namespace, "registry_path": subject, "subject": subject}
    documents: list[ResourceDocument] = []
    client_item = False
    client_block_item = False
    if version_context is not None:
        client_item = _host_requires_resource_template(version_context, leaf="minecraft/item/model", template_id=_RESOURCE_TEMPLATES["item.client_item"])
        client_block_item = _host_requires_resource_template(version_context, leaf="minecraft/block/model", template_id=_RESOURCE_TEMPLATES["item.client_block_item"])
    if render_kind.startswith("block."):
        if render_kind == "block.crop":
            for index in range(variant_count):
                documents.append(_render_document(_RESOURCE_TEMPLATES["block.crop_stage"], {**values, "stage": str(index)}, version_context))
            crop_blockstate_template = load_template(_RESOURCE_TEMPLATES["block.blockstate_crop"])
            if version_context is not None:
                version_context.admit_template(crop_blockstate_template)
            documents.append(ResourceDocument(_RESOURCE_TEMPLATES["block.blockstate_crop"], f"{prefix}assets/{namespace}/blockstates/{subject}.json", {"variants": {f"age={index}": {"model": f"{namespace}:block/{subject}_stage{index}"} for index in range(variant_count)}}))
        else:
            documents.append(_render_document(_RESOURCE_TEMPLATES[render_kind], values, version_context))
            documents.append(_render_document(_RESOURCE_TEMPLATES["block.blockstate_simple"], values, version_context))
        documents.append(_render_document(_RESOURCE_TEMPLATES["item.block_model_reuse"], values, version_context))
        if client_block_item:
            documents.append(_render_document(_RESOURCE_TEMPLATES["item.client_block_item"], values, version_context))
    elif render_kind in {"item.generated", "item.handheld"}:
        documents.append(_render_document(_RESOURCE_TEMPLATES[render_kind], values, version_context))
        if client_item:
            documents.append(_render_document(_RESOURCE_TEMPLATES["item.client_item"], values, version_context))
    if container == "resource_pack":
        rebased: list[ResourceDocument] = []
        source_prefix = "src/main/resources/"
        for document in documents:
            if not document.target_path.startswith(source_prefix):
                raise ValueError(f"Standalone resource document is not rooted under {source_prefix!r}: {document.target_path!r}")
            rebased.append(ResourceDocument(document.template_id, document.target_path[len(source_prefix):], document.payload))
        documents = rebased
    contracted = []
    for texture in textures:
        contract = texture_contract(binding=binding, render_kind=render_kind, namespace=namespace,
                                    minecraft_version=minecraft_version, target_path=texture.target_path,
                                    role=texture.role, width=width, height=height,
                                    topology=texture.topology, alpha=texture.alpha_policy)
        animation = contract["animation"]
        contracted.append(replace(texture, height=contract["geometry"]["canonical_height"],
                                  animation_policy="vertical_frames" if animation["animated"] else "static",
                                  resource_contract=contract))
        if animation["mcmeta_required"]:
            documents.append(ResourceDocument("asset/animation_plan", texture.target_path + ".mcmeta", {
                "animation": {"width": width, "height": height, "frametime": animation["frametime"],
                              "frames": list(range(animation["frame_count"]))}}))
    return ResolvedResourceAsset(str(asset.asset_id), subject, render_kind, container, tuple(contracted), tuple(documents))


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
            variant_count = config.get("growth_stages", config.get("stage_count"))
            if type(variant_count) is not int or not 1 <= variant_count <= 256:
                raise ValueError("HOST crop stage cardinality must be explicit.")
        from .resource_visual_spec import resolve_visual_spec
        visual = resolve_visual_spec(config.get("visual_spec"), ", ".join(parts))
        rows.append({"asset_id": asset_id, "kind": asset_kind, "visual_description": ", ".join(parts), "visual_spec": visual.to_dict(), "render_kind": render_kind, "subject_id": module.module_id, "owner_module_id": module.module_id, "container": "mod", "variant_count": variant_count})
        existing.add(asset_id)
    return tuple(rows)


__all__ = ["SUPPORTED_RENDER_KINDS", "ResolvedResourceAsset", "ResourceDocument", "TextureSpec", "_host_requires_resource_template", "derive_module_asset_specs", "infer_render_kind", "resolve_asset"]
