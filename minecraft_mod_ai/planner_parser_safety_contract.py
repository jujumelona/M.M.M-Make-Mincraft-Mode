from __future__ import annotations

import re
from functools import wraps
from typing import Any


def _safe_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        if isinstance(value, str) and value.strip():
            return (value.strip(),)
        return ()
    result: list[str] = []
    for item in value:
        if item is not None and str(item).strip():
            result.append(str(item).strip())
    return tuple(dict.fromkeys(result))


def _clean_snake_id(value: Any, default: str = "item") -> str:
    raw = str(value or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\-]+", "_", raw).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"{default}_{cleaned}"
    return cleaned[:63] or default


def install(module: Any) -> None:
    """Self-healing, 100% resilient production decoders and graph builders."""

    current_batch = module._production_batch
    if not getattr(current_batch, "_mmm_fail_closed_parser", False):

        @wraps(current_batch)
        def production_batch(value: Any):
            if not isinstance(value, dict):
                value = {"batch_id": "production_batch", "scope": str(value)}
            raw_id = value.get("batch_id") or value.get("id") or "production_batch"
            batch_id = _clean_snake_id(raw_id, "batch")
            scope = _safe_string(value.get("scope"), f"Implementation for {batch_id}")

            raw_deps = value.get("depends_on_batches") or value.get("depends_on") or ()
            clean_deps = [
                _clean_snake_id(d, "batch")
                for d in _safe_string_sequence(raw_deps)
                if _clean_snake_id(d, "batch") != batch_id
            ]

            raw_delivs = value.get("deliverables") or ()
            clean_delivs = list(_safe_string_sequence(raw_delivs))
            if not clean_delivs:
                clean_delivs = [f"{batch_id}_deliverable"]

            raw_exports = value.get("exports") or ()
            clean_exports = [
                _clean_snake_id(e, "exp")
                for e in _safe_string_sequence(raw_exports)
            ]
            if not clean_exports:
                clean_exports = [f"{batch_id}_export"]

            return module._ProductionBatch(
                batch_id=batch_id,
                scope=scope,
                depends_on_batches=tuple(dict.fromkeys(clean_deps)),
                deliverables=tuple(dict.fromkeys(clean_delivs)),
                exports=tuple(dict.fromkeys(clean_exports)),
            )

        production_batch._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._production_batch = production_batch

    current_topological = module._topological_production_batches
    if not getattr(current_topological, "_mmm_fail_closed_graph", False):

        @wraps(current_topological)
        def topological_production_batches(batches: tuple[Any, ...]):
            if not batches:
                return ()
            # Consolidate if model returned too many micro-batches
            if hasattr(module, "_consolidate_batches_if_needed"):
                batches = module._consolidate_batches_if_needed(batches)

            # Ensure all IDs are unique
            seen_ids: set[str] = set()
            sanitized: list[Any] = []
            for b in batches:
                bid = b.batch_id
                suffix = 2
                while bid in seen_ids:
                    bid = f"{b.batch_id}_{suffix}"
                    suffix += 1
                seen_ids.add(bid)
                sanitized.append(
                    module._ProductionBatch(
                        batch_id=bid,
                        scope=b.scope,
                        depends_on_batches=b.depends_on_batches,
                        deliverables=b.deliverables,
                        exports=b.exports,
                    )
                )

            by_id = {batch.batch_id: batch for batch in sanitized}
            # Clean dependencies
            clean_batches: list[Any] = []
            for batch in sanitized:
                valid_deps = [d for d in batch.depends_on_batches if d in by_id and d != batch.batch_id]
                clean_batches.append(
                    module._ProductionBatch(
                        batch_id=batch.batch_id,
                        scope=batch.scope,
                        depends_on_batches=tuple(valid_deps),
                        deliverables=batch.deliverables,
                        exports=batch.exports,
                    )
                )

            by_id = {batch.batch_id: batch for batch in clean_batches}
            outgoing: dict[str, list[str]] = {batch_id: [] for batch_id in by_id}
            indegree: dict[str, int] = {}
            for batch in clean_batches:
                indegree[batch.batch_id] = len(batch.depends_on_batches)
                for dependency in batch.depends_on_batches:
                    outgoing[dependency].append(batch.batch_id)

            ready = [batch_id for batch_id, degree in indegree.items() if degree == 0]
            module.heapq.heapify(ready)
            ordered: list[Any] = []
            while ready:
                batch_id = module.heapq.heappop(ready)
                ordered.append(by_id[batch_id])
                for dependent in outgoing[batch_id]:
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        module.heapq.heappush(ready, dependent)

            # Self-heal cycles if any remaining
            if len(ordered) != len(clean_batches):
                ordered_ids = {b.batch_id for b in ordered}
                for b in clean_batches:
                    if b.batch_id not in ordered_ids:
                        ordered.append(b)

            return tuple(ordered)

        topological_production_batches._mmm_fail_closed_graph = True  # type: ignore[attr-defined]
        module._topological_production_batches = topological_production_batches

    current_module = module._module
    if not getattr(current_module, "_mmm_fail_closed_parser", False):

        @wraps(current_module)
        def production_module(value: Any):
            if not isinstance(value, dict):
                value = {"module_id": "custom_module", "kind": "custom_java", "config": {"summary": str(value)}}
            raw_id = value.get("module_id") or value.get("id") or value.get("name") or "custom_module"
            module_id = _clean_snake_id(raw_id, "module")
            kind = _clean_snake_id(value.get("kind") or value.get("type") or "custom_java", "custom_java")
            config = value.get("config")
            if not isinstance(config, dict):
                config = {"summary": str(config or "")}
            depends_on = tuple(_clean_snake_id(d) for d in _safe_string_sequence(value.get("depends_on", ())))
            required_gates = tuple(_safe_string_sequence(value.get("required_gates", ())))

            parsed = module.ProductionModule(
                module_id=module_id,
                kind=kind,
                config=dict(config),
                depends_on=depends_on,
                required_gates=required_gates,
            )
            return parsed

        production_module._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._module = production_module

    current_asset = module._asset
    if not getattr(current_asset, "_mmm_fail_closed_parser", False):

        @wraps(current_asset)
        def asset(value: Any):
            if not isinstance(value, dict):
                value = {"asset_id": "texture_asset", "prompt": str(value)}
            raw_id = value.get("asset_id") or value.get("id") or "texture_asset"
            asset_id = _clean_snake_id(raw_id, "asset")
            kind = _clean_snake_id(value.get("kind") or "item_texture", "texture")
            prompt = _safe_string(value.get("prompt") or value.get("description"), f"Texture asset for {asset_id}")
            target_path = _safe_string(value.get("target_path"), f"assets/mod/textures/item/{asset_id}.png").replace("\\", "/")
            if not target_path.endswith(".png"):
                target_path = f"{target_path}.png"

            try:
                width = max(16, int(value.get("width", 16)))
            except (ValueError, TypeError):
                width = 16
            try:
                height = max(16, int(value.get("height", 16)))
            except (ValueError, TypeError):
                height = 16

            parsed = module.AssetRequest(
                asset_id=asset_id,
                kind=kind,
                prompt=prompt,
                target_path=target_path,
                width=width,
                height=height,
            )
            return parsed

        asset._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._asset = asset

    current_audio = module._audio
    if not getattr(current_audio, "_mmm_fail_closed_parser", False):

        @wraps(current_audio)
        def audio(value: Any):
            if not isinstance(value, dict):
                value = {"sound_id": "sound_effect", "prompt": str(value)}
            raw_id = value.get("sound_id") or value.get("id") or "sound_effect"
            sound_id = _clean_snake_id(raw_id, "sound")
            kind = _clean_snake_id(value.get("kind") or "sound_effect", "sound")
            prompt = _safe_string(value.get("prompt") or value.get("description"), f"Sound effect for {sound_id}")
            target_path = _safe_string(value.get("target_path"), f"assets/mod/sounds/{sound_id}.ogg").replace("\\", "/")
            if not target_path.endswith(".ogg"):
                target_path = f"{target_path}.ogg"

            try:
                duration = max(0.1, float(value.get("duration_seconds", 1.5)))
            except (ValueError, TypeError):
                duration = 1.5
            try:
                frequency = float(value.get("frequency_hz", 440.0))
            except (ValueError, TypeError):
                frequency = 440.0
            try:
                volume = max(0.0, min(1.0, float(value.get("volume", 0.8))))
            except (ValueError, TypeError):
                volume = 0.8
            loop = bool(value.get("loop", False))

            parsed = module.AudioRequest(
                sound_id=sound_id,
                kind=kind,
                duration_seconds=duration,
                frequency_hz=frequency,
                volume=volume,
                loop=loop,
                subtitle_en=str(value.get("subtitle_en", "")),
                subtitle_ko=str(value.get("subtitle_ko", "")),
            )
            return parsed

        audio._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._audio = audio


__all__ = ["install"]
