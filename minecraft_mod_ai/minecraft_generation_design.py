from __future__ import annotations

"""Complete generator-facing Minecraft fields without generator-side guessing.

This adapter runs after canonical content design. It deterministically preserves fields
that already have an authoritative upstream representation and asks the existing
one-property design template only for genuinely missing scalar generation decisions.
"""

import re
from dataclasses import replace
from typing import Any

from .complete_spec import ProductionModule
from .minecraft_generation_contract import (
    SUPPORTED_GENERATED_KINDS,
    required_generation_fields,
    validate_generation_config,
)
from .task_template_runner import run_record_template

_HEX_IN_VISUAL = re.compile(r"(?:^|,\s*)main_color:\s*(#[0-9A-Fa-f]{6})(?:,|$)")
_INT_FIELDS = frozenset(
    {"attack_damage", "hunger", "output_count", "processing_ticks", "max_level", "permission_level"}
)
_FLOAT_FIELDS = frozenset({"hardness", "attack_speed", "saturation"})
_SCALAR_GENERATION_FIELDS = frozenset(
    {
        "hardness",
        "attack_damage",
        "attack_speed",
        "hunger",
        "saturation",
        "seed_color",
        "input_item",
        "output_item",
        "output_count",
        "processing_ticks",
        "max_level",
        "literal",
        "message",
        "permission_level",
        "main_color",
    }
)


def _asset_main_color(graph: dict[str, Any], module_id: str) -> str | None:
    """Recover an already-authored visual color from the exact asset prompt projection."""
    suffix = "_" + module_id
    for asset in graph.get("assets", ()):
        if not str(getattr(asset, "asset_id", "")).endswith(suffix):
            continue
        prompt = str(getattr(asset, "prompt", ""))
        match = _HEX_IN_VISUAL.search(prompt)
        if match:
            return match.group(1)
    return None


def _coerce(field: str, raw: Any, module_id: str) -> Any:
    if field in _INT_FIELDS:
        text = str(raw).strip()
        if not re.fullmatch(r"-?[0-9]+", text):
            raise ValueError(f"MINECRAFT_DESIGN_FIELD_INVALID: {module_id}.{field}")
        return int(text)
    if field in _FLOAT_FIELDS:
        text = str(raw).strip()
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"MINECRAFT_DESIGN_FIELD_INVALID: {module_id}.{field}") from exc
    value = str(raw).strip()
    if not value:
        raise ValueError(f"MINECRAFT_DESIGN_FIELD_INVALID: {module_id}.{field}")
    return value


def _atomic_property(
    router: Any,
    *,
    prompt: str,
    module: ProductionModule,
    field: str,
    available_resource_ids: list[str],
    design_state: dict[str, Any],
    progress: dict[str, Any],
    checkpoint: Any,
) -> Any:
    if field not in _SCALAR_GENERATION_FIELDS:
        raise ValueError(f"MINECRAFT_DESIGN_FIELD_NOT_ATOMIC: {module.module_id}.{field}")

    def save(binding: str, accepted: Any) -> None:
        progress[binding] = accepted
        if checkpoint is not None:
            checkpoint(binding, accepted)

    result = run_record_template(
        router,
        "design/content_property",
        context={
            "requirement": prompt,
            "module_id": module.module_id,
            "module_kind": module.kind,
            "module_config": dict(module.config),
            "requested_property": field,
            "allowed_properties": [field],
            "required_properties": [field],
            "allowed_resource_ids": available_resource_ids,
            "design_state": design_state,
        },
        allowed_refs=(),
        progress=progress,
        checkpoint=save,
    )
    rows = result.get("records", ())
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("property") != field
    ):
        raise ValueError(f"MINECRAFT_DESIGN_FIELD_UNRESOLVED: {module.module_id}.{field}")
    return _coerce(field, rows[0].get("value"), module.module_id)


def complete_generation_fields(
    graph: dict[str, Any],
    router: Any,
    *,
    prompt: str,
    progress: dict[str, Any] | None = None,
    checkpoint: Any = None,
) -> dict[str, Any]:
    """Return graph with deterministic-generator modules fully specified.

    Data-only modules are intentionally not converted here: their canonical structured
    resource facts are owned by the artifact/resource pipeline, not by a model-authored
    serialized JSON blob.
    """
    from .atomic_design_pipeline import _sanitize_stem

    progress = progress if progress is not None else {}
    modules = tuple(graph.get("modules", ()))
    local_mod_id = _sanitize_stem(prompt) + "_mod"
    available_resource_ids = [
        f"{local_mod_id}:{module.module_id}"
        for module in modules
        if isinstance(module, ProductionModule) and module.kind in {"item", "block"}
    ]
    design_state = {
        "slots": graph.get("_design_slots", {}),
        "decisions": graph.get("_design_decisions", ()),
        "relations": graph.get("_content_relations", ()),
        "research_facts": graph.get("_research_facts", ()),
    }

    completed: list[ProductionModule] = []
    for module in modules:
        if not isinstance(module, ProductionModule) or module.kind not in SUPPORTED_GENERATED_KINDS:
            completed.append(module)
            continue
        if module.kind in {"recipe", "advancement", "loot"}:
            # These remain artifact/resource-owned until their canonical fact is lowered
            # by that subsystem. Never ask the model to manufacture serialized JSON here.
            completed.append(module)
            continue

        config = dict(module.config)
        if "display_name" in required_generation_fields(module.kind) and "display_name" not in config:
            authored = config.get("name")
            if not isinstance(authored, str) or not authored.strip():
                raise ValueError(f"MINECRAFT_DESIGN_FIELD_UNRESOLVED: {module.module_id}.display_name")
            config["display_name"] = authored.strip()
        if "main_color" in required_generation_fields(module.kind) and "main_color" not in config:
            recovered = _asset_main_color(graph, module.module_id)
            if recovered is not None:
                config["main_color"] = recovered

        for field in required_generation_fields(module.kind):
            if field in config:
                if field in _INT_FIELDS | _FLOAT_FIELDS:
                    config[field] = _coerce(field, config[field], module.module_id)
                continue
            config[field] = _atomic_property(
                router,
                prompt=prompt,
                module=replace(module, config=config),
                field=field,
                available_resource_ids=available_resource_ids,
                design_state=design_state,
                progress=progress,
                checkpoint=checkpoint,
            )
        validate_generation_config(module.kind, module.module_id, config)
        completed.append(replace(module, config=config))

    result = dict(graph)
    result["modules"] = completed
    return result


__all__ = ["complete_generation_fields"]
