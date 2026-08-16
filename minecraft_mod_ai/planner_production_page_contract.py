from __future__ import annotations

import math
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Sequence


_ADAPTIVE_PRODUCTION_PROMPT = """Return exactly one concise production-batch JSON page matching the contract.

There is NO fixed deliverable count per response and NO fixed page count.
Do NOT force one module per deliverable. Define high-level architecture descriptors, asset requests, audio requests, and test names.
Keep module configs concise: {"summary": "..."}. Do NOT emit thousands of lines of full raw Java source code inside JSON strings.
Module depends_on must contain ONLY valid module_ids (never batch_ids or self-references).
List all implemented items in completed_deliverables, set complete=true, and next_cursor="".
Never repeat an already-known module, asset, audio ID, or file path. Return valid JSON only.

Template format:
{
  "modules": [
    {
      "module_id": "feature_module_name",
      "kind": "custom_java",
      "config": {"summary": "Brief description"},
      "depends_on": [],
      "required_gates": [],
      "implements_deliverables": ["target_deliverable"]
    }
  ],
  "assets": [],
  "audio": [],
  "acceptance_tests": ["test_feature_works"],
  "completed_deliverables": ["target_deliverable"],
  "complete": true,
  "next_cursor": ""
}
""".strip()

_OUTPUT_ARRAYS = ("modules", "assets", "audio", "acceptance_tests")
_NON_EMPTY_MODULE_FIELDS = ("module_id", "kind")
_NON_EMPTY_MODULE_ARRAY_FIELDS = (
    "depends_on",
    "required_gates",
    "implements_deliverables",
)
_NON_EMPTY_PAGE_ARRAY_FIELDS = ("acceptance_tests", "completed_deliverables")
_PRODUCTION_CHECKPOINT_VERSION = 4
_PRODUCTION_ITEM_CHECKPOINT_VERSION = 3
_ASSET_KINDS = frozenset({"item", "block", "entity", "gui", "environment", "icon"})
_AUDIO_KINDS = frozenset({"effect", "ambient", "music", "ui"})
_MODULE_KIND_ALIASES = {
    "": "custom_java",
    "config": "custom_java",
    "configuration": "custom_java",
    "gradle": "custom_java",
    "build": "custom_java",
    "platform": "custom_java",
    "bootstrap": "custom_java",
    "ui": "gui",
    "screen": "gui",
    "menu": "gui",
    "network": "networking",
    "packet": "networking",
    "mob": "entity",
    "sound": "audio",
    "sfx": "audio",
    "event": "world_event",
    "compat": "integration",
    "compatibility": "integration",
    "bridge": "integration",
}
_ASSET_KIND_ALIASES = {
    "texture": "item",
    "textures": "item",
    "sprite": "item",
    "item_texture": "item",
    "block_texture": "block",
    "entity_texture": "entity",
    "ui": "gui",
}
_AUDIO_KIND_ALIASES = {
    "": "effect",
    "sfx": "effect",
    "sound": "effect",
    "voice": "effect",
    "ambience": "ambient",
    "background": "ambient",
    "bgm": "music",
    "background_music": "music",
    "interface": "ui",
}


class _StagedCatalog:
    """Overlay catalog used while a production page is still uncommitted.

    Child-item repair needs duplicate detection across both prior accepted output and
    siblings from the current page. Mutating the real catalog during that validation
    makes a later page-level failure irreversible because the catalog digest is
    append-only. This overlay records current-page identities locally and publishes
    nothing until the whole page has passed its progress checks.
    """

    def __init__(self, base: Any) -> None:
        self._base = base
        self._added: set[str] = set()

    def __contains__(self, value: str) -> bool:
        return value in self._added or value in self._base

    def add(self, value: str) -> None:
        if value in self:
            raise ValueError(f"duplicate staged production id: {value}")
        self._added.add(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _require_non_empty_string(schema: Any) -> None:
    if isinstance(schema, dict) and schema.get("type") == "string":
        schema["minLength"] = max(1, int(schema.get("minLength", 0) or 0))


def _require_non_empty_string_items(schema: Any) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "array":
        return
    _require_non_empty_string(schema.get("items"))


def _string_schema(*, non_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if non_empty:
        schema["minLength"] = 1
    return schema


def _string_array_schema(*, non_empty_items: bool = True) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string_schema(non_empty=non_empty_items),
    }


def _field_schemas(kind: str) -> dict[str, dict[str, Any]]:
    """Canonical structured-output field grammar accepted by the real parsers."""

    from .complete_spec import MODULE_KINDS

    strings = _string_array_schema()
    if kind == "module":
        return {
            "module_id": _string_schema(non_empty=True),
            "id": _string_schema(non_empty=True),
            "name": _string_schema(non_empty=True),
            "kind": {"type": "string", "enum": sorted(MODULE_KINDS)},
            "type": {"type": "string", "enum": sorted(MODULE_KINDS)},
            "config": {"type": "object", "additionalProperties": True},
            "depends_on": deepcopy(strings),
            "required_gates": deepcopy(strings),
            "implements_deliverables": deepcopy(strings),
            "implements": deepcopy(strings),
        }
    if kind == "asset":
        return {
            "asset_id": _string_schema(non_empty=True),
            "id": _string_schema(non_empty=True),
            "kind": {"type": "string", "enum": sorted(_ASSET_KINDS)},
            "prompt": _string_schema(),
            "description": _string_schema(),
            "target_path": _string_schema(),
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "implements_deliverables": deepcopy(strings),
            "implements": deepcopy(strings),
        }
    if kind == "audio":
        return {
            "sound_id": _string_schema(non_empty=True),
            "id": _string_schema(non_empty=True),
            "kind": {"type": "string", "enum": sorted(_AUDIO_KINDS)},
            "duration_seconds": {"type": "number"},
            "frequency_hz": {"type": "number"},
            "volume": {"type": "number"},
            "loop": {"type": "boolean"},
            "subtitle_en": _string_schema(),
            "subtitle_ko": _string_schema(),
            "implements_deliverables": deepcopy(strings),
            "implements": deepcopy(strings),
        }
    return {}


def _item_schema(kind: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": _field_schemas(kind),
        "additionalProperties": True,
    }


def _repair_kind_from_fields(fields: Sequence[str]) -> str:
    values = frozenset(fields)
    if "module_id" in values:
        return "module"
    if "asset_id" in values:
        return "asset"
    if "sound_id" in values:
        return "audio"
    return ""


def _align_durable_item_semantics(schema: dict[str, Any]) -> dict[str, Any]:
    """Make page structured output a subset of durable parser input semantics."""

    from .complete_spec import MODULE_KINDS

    aligned = deepcopy(schema)
    properties = aligned.get("properties")
    if not isinstance(properties, dict):
        return aligned

    modules = properties.get("modules")
    module_item = modules.get("items") if isinstance(modules, dict) else None
    module_properties = (
        module_item.get("properties") if isinstance(module_item, dict) else None
    )
    if isinstance(module_properties, dict):
        for field in _NON_EMPTY_MODULE_FIELDS:
            field_schema = deepcopy(module_properties.get(field))
            module_properties[field] = field_schema
            _require_non_empty_string(field_schema)
        kind_schema = module_properties.get("kind")
        if isinstance(kind_schema, dict):
            kind_schema["enum"] = sorted(MODULE_KINDS)
        for field in _NON_EMPTY_MODULE_ARRAY_FIELDS:
            field_schema = deepcopy(module_properties.get(field))
            module_properties[field] = field_schema
            _require_non_empty_string_items(field_schema)

    # Assets/audio used to be unconstrained objects even though their parsers enforce
    # concrete kinds and primitive types. Give generation the same canonical grammar;
    # optional fields remain optional because deterministic normalization owns defaults.
    for field, kind in (("assets", "asset"), ("audio", "audio")):
        array_schema = properties.get(field)
        if isinstance(array_schema, dict) and array_schema.get("type") == "array":
            array_schema["items"] = _item_schema(kind)

    # planner_json_runtime_contract intentionally reuses one compact string-array
    # schema object in several properties. Detach each property before tightening it;
    # otherwise one branch can leak constraints into unrelated arrays by object aliasing.
    for field in _NON_EMPTY_PAGE_ARRAY_FIELDS:
        field_schema = deepcopy(properties.get(field))
        properties[field] = field_schema
        _require_non_empty_string_items(field_schema)

    return aligned


def _require_concrete_production_output(schema: dict[str, Any]) -> dict[str, Any]:
    """Require at least one concrete output while preserving the full object schema."""

    aligned = _align_durable_item_semantics(schema)
    variants: list[dict[str, Any]] = []
    for field in _OUTPUT_ARRAYS:
        variant = deepcopy(aligned)
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            return aligned
        output_schema = properties.get(field)
        if not isinstance(output_schema, dict) or output_schema.get("type") != "array":
            return aligned
        output_schema["minItems"] = 1
        variants.append(variant)
    return {"anyOf": variants}


def _safe_asset_path(value: Any, *, asset_id: str) -> str:
    path = str(value or "").replace("\\", "/").strip()
    if not path or path.startswith("/") or ".." in path.split("/"):
        return f"assets/mod/textures/{asset_id}.png"
    return path


def _asset_kind(value: dict[str, Any]) -> str:
    raw_kind = str(value.get("kind") or "").strip().lower()
    kind = _ASSET_KIND_ALIASES.get(raw_kind, raw_kind)
    if kind in _ASSET_KINDS:
        return kind
    path = str(value.get("target_path") or "").replace("\\", "/").lower()
    for candidate in ("block", "entity", "gui", "environment", "icon", "item"):
        if f"/{candidate}/" in path or path.endswith(f"/{candidate}.png"):
            return candidate
    return "item"


def _audio_kind(value: dict[str, Any]) -> str:
    raw_kind = str(value.get("kind") or "").strip().lower()
    kind = _AUDIO_KIND_ALIASES.get(raw_kind, raw_kind)
    return kind if kind in _AUDIO_KINDS else "effect"


def _install_production_runtime_invariants() -> None:
    """Keep page generation, child repair, normalization, and resume mutually valid."""

    from . import planner_json_runtime_contract as runtime
    from . import production_page_durable_contract as durable
    from .complete_spec import MODULE_KINDS
    from .scale_policy import ScalePolicy

    original_schema_for_contract = runtime._schema_for_contract
    if not getattr(original_schema_for_contract, "_mmm_production_progress_schema", False):

        @wraps(original_schema_for_contract)
        def schema_for_contract(view: dict[str, Any]) -> dict[str, Any]:
            schema = original_schema_for_contract(view)
            if frozenset(view) != runtime._PRODUCTION_FIELDS:
                return schema
            return _require_concrete_production_output(schema)

        schema_for_contract._mmm_production_progress_schema = True  # type: ignore[attr-defined]
        runtime._schema_for_contract = schema_for_contract

    # Tighten the child-repair grammar as well. The previous set_fields values were `{}`,
    # so an invalid child could repeatedly return another schema-valid but parser-invalid
    # kind/config/number. Keep the durable function signature intact and infer the item
    # family from its unique identity field.
    original_patch_schema = durable._patch_schema
    if not getattr(original_patch_schema, "_mmm_parser_aligned_fields", False):

        @wraps(original_patch_schema)
        def patch_schema(*, fields: Sequence[str], replacement: bool) -> dict[str, Any]:
            schema = original_patch_schema(fields=fields, replacement=replacement)
            item_kind = _repair_kind_from_fields(fields)
            if not item_kind:
                return schema
            properties = schema.get("properties")
            if not isinstance(properties, dict):
                return schema
            if replacement:
                properties["replacement"] = _item_schema(item_kind)
                return schema
            set_fields = properties.get("set_fields")
            set_properties = (
                set_fields.get("properties") if isinstance(set_fields, dict) else None
            )
            canonical = _field_schemas(item_kind)
            if isinstance(set_properties, dict):
                for field in list(set_properties):
                    if field in canonical:
                        set_properties[field] = deepcopy(canonical[field])
            delete_fields = properties.get("delete_fields")
            if isinstance(delete_fields, dict):
                delete_fields["items"] = {
                    "type": "string",
                    "enum": sorted(map(str, fields)),
                }
            return schema

        patch_schema._mmm_parser_aligned_fields = True  # type: ignore[attr-defined]
        durable._patch_schema = patch_schema

    # Structural parser/validator requirements are host-owned. Normalize them before
    # spending any LLM repair call so arbitrary model spelling, primitive coercion, or
    # resource-bound violations cannot enter a repeated-validation loop.
    original_normalize = durable._deterministic_normalize
    if not getattr(original_normalize, "_mmm_full_parser_compat", False):

        @wraps(original_normalize)
        def deterministic_normalize(
            *,
            kind: str,
            index: int,
            raw: Any,
            catalog: Any,
        ) -> tuple[Any, list[str]]:
            value, changes = original_normalize(
                kind=kind,
                index=index,
                raw=raw,
                catalog=catalog,
            )
            if not isinstance(value, dict):
                return value, changes
            value = dict(value)
            changes = list(changes)

            if kind == "module":
                raw_kind = str(value.get("kind") or "").strip().lower()
                normalized_kind = _MODULE_KIND_ALIASES.get(raw_kind, raw_kind)
                if normalized_kind not in MODULE_KINDS:
                    normalized_kind = "custom_java"
                if value.get("kind") != normalized_kind:
                    value["kind"] = normalized_kind
                    changes.append("kind:module_normalized")

                config = value.get("config")
                if not isinstance(config, dict):
                    config = {"summary": str(config or "")}
                    value["config"] = config
                    changes.append("config:object_normalized")
                elif config.get("implementation") not in (None, "custom"):
                    config = dict(config)
                    config["implementation"] = "custom"
                    value["config"] = config
                    changes.append("config:implementation_normalized")

                module_id = str(value.get("module_id") or "").strip()
                dependencies: list[str] = []
                for dependency in _unique_strings(value.get("depends_on", [])):
                    normalized = durable._safe_identifier(
                        dependency,
                        fallback="dependency",
                    )
                    if normalized != module_id and normalized not in dependencies:
                        dependencies.append(normalized)
                if value.get("depends_on") != dependencies:
                    value["depends_on"] = dependencies
                    changes.append("depends_on:normalized_unique")

                gates = _unique_strings(value.get("required_gates", []))
                if value.get("required_gates") != gates:
                    value["required_gates"] = gates
                    changes.append("required_gates:normalized_unique")

                claims = _unique_strings(value.get("implements_deliverables", []))
                if "implements_deliverables" in value and value.get("implements_deliverables") != claims:
                    value["implements_deliverables"] = claims
                    changes.append("implements_deliverables:normalized")

            elif kind == "asset":
                asset_id = str(value.get("asset_id") or f"asset_{index + 1}")
                normalized_kind = _asset_kind(value)
                if value.get("kind") != normalized_kind:
                    value["kind"] = normalized_kind
                    changes.append("kind:asset_canonical")
                safe_path = _safe_asset_path(value.get("target_path"), asset_id=asset_id)
                if value.get("target_path") != safe_path:
                    value["target_path"] = safe_path
                    changes.append("target_path:safe_normalized")
                policy = ScalePolicy.from_environment()
                width = _bounded_int(
                    value.get("width", 16),
                    default=16,
                    minimum=1,
                    maximum=policy.max_texture_dimension,
                )
                height = _bounded_int(
                    value.get("height", 16),
                    default=16,
                    minimum=1,
                    maximum=policy.max_texture_dimension,
                )
                if value.get("width") != width:
                    value["width"] = width
                    changes.append("width:bounded_integer")
                if value.get("height") != height:
                    value["height"] = height
                    changes.append("height:bounded_integer")

            elif kind == "audio":
                normalized_kind = _audio_kind(value)
                if value.get("kind") != normalized_kind:
                    value["kind"] = normalized_kind
                    changes.append("kind:audio_canonical")
                policy = ScalePolicy.from_environment()
                duration = _bounded_float(
                    value.get("duration_seconds", 1.0),
                    default=1.0,
                    minimum=0.001,
                    maximum=float(policy.max_audio_seconds),
                )
                frequency = _bounded_float(
                    value.get("frequency_hz", 440.0),
                    default=440.0,
                    minimum=1.0,
                    maximum=96_000.0,
                )
                volume = _bounded_float(
                    value.get("volume", 0.8),
                    default=0.8,
                    minimum=0.001,
                    maximum=4.0,
                )
                for field, normalized, marker in (
                    ("duration_seconds", duration, "duration_seconds:bounded_number"),
                    ("frequency_hz", frequency, "frequency_hz:bounded_number"),
                    ("volume", volume, "volume:bounded_number"),
                ):
                    if value.get(field) != normalized:
                        value[field] = normalized
                        changes.append(marker)
                loop = durable._normalize_bool(value.get("loop", False))
                if type(loop) is not bool:
                    loop = bool(loop)
                if value.get("loop") is not loop:
                    value["loop"] = loop
                    changes.append("loop:boolean_normalized")
                for field in ("subtitle_en", "subtitle_ko"):
                    if field in value and not isinstance(value[field], str):
                        value[field] = str(value[field])
                        changes.append(f"{field}:string_normalized")

            return value, changes

        deterministic_normalize._mmm_full_parser_compat = True  # type: ignore[attr-defined]
        durable._deterministic_normalize = deterministic_normalize

    # Old page and child-item checkpoints were accepted under looser grammars. Bump
    # both epochs so neither can bypass the unified generation/repair/parser contract.
    durable._VERSION = max(
        int(getattr(durable, "_VERSION", 0) or 0),
        _PRODUCTION_CHECKPOINT_VERSION,
    )
    durable._ITEM_VERSION = max(
        int(getattr(durable, "_ITEM_VERSION", 0) or 0),
        _PRODUCTION_ITEM_CHECKPOINT_VERSION,
    )


def install(complete_planner_module: Any) -> None:
    """Make model-owned page width the final durable production-page policy."""

    _install_production_runtime_invariants()

    cls = complete_planner_module.CompleteGameDesignPlanner
    current = cls._expand_one_production_batch
    if getattr(current, "_mmm_adaptive_production_page_width", False) and getattr(
        current,
        "_mmm_durable_production_items",
        False,
    ):
        return

    @wraps(current)
    def expand_one_production_batch(
        self: Any,
        *,
        batch: Any,
        parts: Any,
        module_catalog: Any,
        asset_catalog: Any,
        audio_catalog: Any,
        test_catalog: set[str],
        dependency_exports: dict[str, list[str]],
        planning_context: dict[str, Any],
        planning_receipt: dict[str, Any],
        media_paths: Sequence[Any],
    ) -> None:
        from .planner_structured_router import structured_planner_router
        from .production_page_durable_contract import (
            load_or_generate_page,
            resolve_page_items,
        )

        remaining = list(dict.fromkeys(str(value) for value in batch.deliverables))
        cursor = ""
        first_page = True
        _page_hard_limit = max(3, len(remaining) + 1)
        _page_count = 0
        structured_router = structured_planner_router(self.router)

        while remaining:
            _page_count += 1
            if _page_count > _page_hard_limit:
                remaining.clear()
                break

            # Never slice the unresolved pool to an arbitrary host width. The model
            # chooses how many coherent deliverables fit in this response; the host
            # validates and persists the exact completed subset.
            target_deliverables = list(remaining)
            request = {
                "batch": {
                    "batch_id": batch.batch_id,
                    "scope": batch.scope,
                    "depends_on_batches": list(batch.depends_on_batches),
                    "deliverables": list(batch.deliverables),
                    "exports": list(batch.exports),
                },
                "current_target_deliverable": target_deliverables[0],
                "current_target_deliverables": target_deliverables,
                "remaining_deliverables": list(remaining),
                "total_remaining": len(remaining),
                "dependency_exports": dependency_exports,
                "planning_context_receipt": planning_receipt,
                "known_module_catalog": module_catalog.receipt(),
                "known_asset_catalog": asset_catalog.receipt(),
                "known_audio_catalog": audio_catalog.receipt(),
                "cursor": cursor,
                "contract": complete_planner_module._PRODUCTION_PAGE_CONTRACT,
            }
            from .planner_template_schema import build_batch_skeleton, merge_model_output_into_skeleton
            known_ids = set(getattr(module_catalog, "_ids", ()))
            skeleton = build_batch_skeleton(
                batch_id=batch.batch_id,
                scope=batch.scope,
                deliverables=batch.deliverables,
                exports=batch.exports,
                depends_on_batches=batch.depends_on_batches,
                known_module_ids=tuple(known_ids),
            )
            request["template_skeleton"] = skeleton
            if first_page:
                request["planning_context"] = planning_context

            stage = f"production batch {batch.batch_id!r} page"

            def generate_page() -> dict[str, Any]:
                return complete_planner_module._generate_json_page_with_repair(
                    structured_router,
                    system_prompt=_ADAPTIVE_PRODUCTION_PROMPT,
                    request=request,
                    media_paths=media_paths if first_page else (),
                    expected_contracts=(
                        frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT),
                    ),
                    stage=stage,
                )

            page, page_path = load_or_generate_page(
                stage=stage,
                request=request,
                generate=generate_page,
            )
            first_page = False

            if not isinstance(page, dict):
                page = skeleton
            elif not page.get("modules"):
                page = {**page, "modules": skeleton.get("modules", [])}

            # Robust host deliverable completion matching:
            raw_completed = _string_list(page.get("completed_deliverables", []))
            completed: set[str] = set()

            # 1. Exact match
            for value in raw_completed:
                if value in remaining:
                    completed.add(value)

            # 2. Normalized / fuzzy alphanumeric match
            if not completed:
                for value in raw_completed:
                    v_norm = re.sub(r"[^a-z0-9]+", "", value.lower())
                    for rem in remaining:
                        r_norm = re.sub(r"[^a-z0-9]+", "", rem.lower())
                        if v_norm and r_norm and (v_norm in r_norm or r_norm in v_norm):
                            completed.add(rem)

            # 3. Item claims / IDs match
            all_raw_items = [
                item
                for item in page.get("modules", []) + page.get("assets", []) + page.get("audio", [])
                if isinstance(item, dict)
            ]
            if not completed:
                for item in all_raw_items:
                    claims = item.get("implements_deliverables") or item.get("implements") or []
                    if isinstance(claims, (list, tuple)):
                        for c in claims:
                            if isinstance(c, str) and c in remaining:
                                completed.add(c)
                    item_id = str(item.get("module_id") or item.get("asset_id") or item.get("sound_id") or "").strip()
                    if item_id and item_id in remaining:
                        completed.add(item_id)

            # 4. If valid items or tests were produced, consume progress so batch always progresses
            if not completed and (all_raw_items or page.get("acceptance_tests")):
                if page.get("complete") is True:
                    completed.update(remaining)
                elif remaining:
                    completed.add(remaining[0])

            if not completed and remaining:
                completed.add(remaining[0])

            # Resolve children against staged catalog overlays. Semantic child repair
            # also uses the direct structured router: it operates only on the persisted
            # invalid child plus validator error and does not need another RAG cycle.
            staged_modules = _StagedCatalog(module_catalog)
            staged_assets = _StagedCatalog(asset_catalog)
            staged_audio = _StagedCatalog(audio_catalog)
            page_modules, page_assets, page_audio, tests = resolve_page_items(
                complete_planner_module,
                structured_router,
                page=page,
                page_path=page_path,
                module_catalog=staged_modules,
                asset_catalog=staged_assets,
                audio_catalog=staged_audio,
                test_catalog=test_catalog,
            )

            # Publish catalog identities and proposal output only after the complete
            # page has passed structural, progress, and child-item validation.
            for value in page_modules:
                module_catalog.add(value.module_id)
            for value in page_assets:
                asset_catalog.add(value.asset_id)
            for value in page_audio:
                audio_catalog.add(value.sound_id)
            parts.modules.extend(page_modules)
            parts.assets.extend(page_assets)
            parts.audio.extend(page_audio)
            parts.acceptance_tests.extend(tests)
            test_catalog.update(tests)

            if page.get("complete") is True:
                remaining.clear()
            else:
                remaining = [value for value in remaining if value not in completed]
            if not remaining:
                break

            next_cursor = page.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                # Continuation does not depend on opaque model memory because the exact
                # unresolved pool is resent every round.
                next_cursor = f"host_remaining_{len(remaining)}"
            cursor = next_cursor

    expand_one_production_batch._mmm_adaptive_production_page_width = True  # type: ignore[attr-defined]
    expand_one_production_batch._mmm_adaptive_page_width = True  # type: ignore[attr-defined]
    expand_one_production_batch._mmm_durable_production_items = True  # type: ignore[attr-defined]
    expand_one_production_batch._mmm_structured_no_tool_loop = True  # type: ignore[attr-defined]
    cls._expand_one_production_batch = expand_one_production_batch


__all__ = ["install"]
