from __future__ import annotations

from functools import wraps
from typing import Any


def _nonempty_string(value: Any, field: str, error_type: type[Exception]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{field} must be a non-empty string.")
    return value.strip()


def _string_sequence(value: Any, field: str, error_type: type[Exception]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise error_type(f"{field} must be a list of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise error_type(f"{field}[{index}] must be a non-empty string.")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise error_type(f"{field} must not contain duplicates.")
    return tuple(result)


def install(module: Any) -> None:
    """Replace permissive production decoders with typed, fail-closed parsers."""

    current_batch = module._production_batch
    if not getattr(current_batch, "_mmm_fail_closed_parser", False):

        @wraps(current_batch)
        def production_batch(value: Any):
            expected = {
                "batch_id",
                "scope",
                "depends_on_batches",
                "deliverables",
                "exports",
            }
            if not isinstance(value, dict) or set(value) != expected:
                raise module.SpecValidationError(
                    "Production batch descriptor fields are invalid."
                )
            batch_id = _nonempty_string(
                value["batch_id"],
                "production batch.batch_id",
                module.SpecValidationError,
            )
            if not module._BATCH_ID.fullmatch(batch_id):
                raise module.SpecValidationError(
                    f"Invalid production batch id: {batch_id!r}"
                )
            scope = _nonempty_string(
                value["scope"],
                f"production batch {batch_id}.scope",
                module.SpecValidationError,
            )
            dependencies = _string_sequence(
                value["depends_on_batches"],
                f"production batch {batch_id}.depends_on_batches",
                module.SpecValidationError,
            )
            deliverables = _string_sequence(
                value["deliverables"],
                f"production batch {batch_id}.deliverables",
                module.SpecValidationError,
            )
            exports = _string_sequence(
                value["exports"],
                f"production batch {batch_id}.exports",
                module.SpecValidationError,
            )
            if not deliverables:
                raise module.SpecValidationError(
                    f"Production batch {batch_id} must declare at least one deliverable."
                )
            if batch_id in dependencies:
                raise module.SpecValidationError(
                    f"Production batch {batch_id} may not depend on itself."
                )
            invalid_exports = [item for item in exports if not module._BATCH_ID.fullmatch(item)]
            if invalid_exports:
                raise module.SpecValidationError(
                    f"Production batch {batch_id} has invalid export ids: {invalid_exports[:4]}"
                )
            return module._ProductionBatch(
                batch_id=batch_id,
                scope=scope,
                depends_on_batches=dependencies,
                deliverables=deliverables,
                exports=exports,
            )

        production_batch._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._production_batch = production_batch

    current_topological = module._topological_production_batches
    if not getattr(current_topological, "_mmm_fail_closed_graph", False):

        @wraps(current_topological)
        def topological_production_batches(batches: tuple[Any, ...]):
            by_id: dict[str, Any] = {}
            unique_batches: list[Any] = []
            for b in batches:
                bid = b.batch_id
                suffix = 2
                while bid in by_id:
                    bid = f"{b.batch_id}_{suffix}"
                    suffix += 1
                new_b = module._ProductionBatch(
                    batch_id=bid,
                    scope=b.scope,
                    depends_on_batches=b.depends_on_batches,
                    deliverables=b.deliverables,
                    exports=b.exports,
                ) if bid != b.batch_id else b
                by_id[bid] = new_b
                unique_batches.append(new_b)

            outgoing: dict[str, list[str]] = {batch_id: [] for batch_id in by_id}
            indegree: dict[str, int] = {batch_id: 0 for batch_id in by_id}
            for batch in unique_batches:
                valid_deps = [d for d in batch.depends_on_batches if d in by_id and d != batch.batch_id]
                indegree[batch.batch_id] = len(valid_deps)
                for dependency in valid_deps:
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

            if len(ordered) != len(unique_batches):
                ordered_ids = {b.batch_id for b in ordered}
                for b in unique_batches:
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
                raise module.SpecValidationError(
                    "Every production module must be an object."
                )
            module_id = value.get("module_id") or value.get("id") or value.get("name")
            kind = value.get("kind") or value.get("type")
            module_id = _nonempty_string(
                module_id,
                "production module.module_id",
                module.SpecValidationError,
            )
            kind = _nonempty_string(
                kind,
                f"production module {module_id}.kind",
                module.SpecValidationError,
            )
            config = value.get("config")
            if not isinstance(config, dict):
                raise module.SpecValidationError(
                    f"Production module {module_id}.config must be an object."
                )
            depends_on = _string_sequence(
                value.get("depends_on", []),
                f"production module {module_id}.depends_on",
                module.SpecValidationError,
            )
            required_gates = _string_sequence(
                value.get("required_gates", []),
                f"production module {module_id}.required_gates",
                module.SpecValidationError,
            )
            parsed = module.ProductionModule(
                module_id=module_id,
                kind=kind,
                config=dict(config),
                depends_on=depends_on,
                required_gates=required_gates,
            )
            parsed.validate()
            return parsed

        production_module._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._module = production_module

    current_asset = module._asset
    if not getattr(current_asset, "_mmm_fail_closed_parser", False):

        @wraps(current_asset)
        def asset(value: Any):
            if not isinstance(value, dict):
                raise module.SpecValidationError("Every asset request must be an object.")
            asset_id = _nonempty_string(
                value.get("asset_id") or value.get("id"),
                "asset.asset_id",
                module.SpecValidationError,
            )
            kind = _nonempty_string(
                value.get("kind"),
                f"asset {asset_id}.kind",
                module.SpecValidationError,
            )
            prompt = _nonempty_string(
                value.get("prompt") or value.get("description"),
                f"asset {asset_id}.prompt",
                module.SpecValidationError,
            )
            target_path = _nonempty_string(
                value.get("target_path"),
                f"asset {asset_id}.target_path",
                module.SpecValidationError,
            )
            width = value.get("width", 16)
            height = value.get("height", 16)
            if type(width) is not int or type(height) is not int:
                raise module.SpecValidationError(
                    f"Asset {asset_id} width/height must be integers."
                )
            parsed = module.AssetRequest(
                asset_id=asset_id,
                kind=kind,
                prompt=prompt,
                target_path=target_path,
                width=width,
                height=height,
            )
            parsed.validate()
            return parsed

        asset._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._asset = asset

    current_audio = module._audio
    if not getattr(current_audio, "_mmm_fail_closed_parser", False):

        @wraps(current_audio)
        def audio(value: Any):
            if not isinstance(value, dict):
                raise module.SpecValidationError("Every audio request must be an object.")
            sound_id = _nonempty_string(
                value.get("sound_id") or value.get("id"),
                "audio.sound_id",
                module.SpecValidationError,
            )
            kind = _nonempty_string(
                value.get("kind"),
                f"audio {sound_id}.kind",
                module.SpecValidationError,
            )
            duration = value.get("duration_seconds")
            frequency = value.get("frequency_hz", 440.0)
            volume = value.get("volume", 0.8)
            loop = value.get("loop", False)
            for field, number in (
                ("duration_seconds", duration),
                ("frequency_hz", frequency),
                ("volume", volume),
            ):
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    raise module.SpecValidationError(
                        f"Audio {sound_id}.{field} must be numeric."
                    )
            if type(loop) is not bool:
                raise module.SpecValidationError(
                    f"Audio {sound_id}.loop must be boolean."
                )
            parsed = module.AudioRequest(
                sound_id=sound_id,
                kind=kind,
                duration_seconds=float(duration),
                frequency_hz=float(frequency),
                volume=float(volume),
                loop=loop,
                subtitle_en=str(value.get("subtitle_en", "")),
                subtitle_ko=str(value.get("subtitle_ko", "")),
            )
            parsed.validate()
            return parsed

        audio._mmm_fail_closed_parser = True  # type: ignore[attr-defined]
        module._audio = audio


__all__ = ["install"]
