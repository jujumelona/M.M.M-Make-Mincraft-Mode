from __future__ import annotations

from typing import Any, Mapping

from .platform_catalog import adapter_for_target


def install(module: Any) -> None:
    target_cls = module.TechnologyTarget

    def from_adapter(adapter: Any):
        return target_cls(
            edition=adapter.edition,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings=adapter.yarn_mappings,
            java_version=adapter.java_version,
            fabric_loader=adapter.fabric_loader,
            fabric_api=adapter.fabric_api,
        )

    def normalize(value: Any):
        if value is None:
            # Backward-compatible standalone technology calls retain the mature
            # 1.20.1 adapter. Plan-bound calls always pass their selected target.
            return from_adapter(adapter_for_target("1.20.1", "fabric"))
        if isinstance(value, target_cls):
            value.validate()
            return value
        if isinstance(value, module.PlatformLock):
            return from_adapter(
                adapter_for_target(value.minecraft_version, value.loader)
            )
        if isinstance(value, Mapping):
            version = str(value.get("minecraft_version", "")).strip()
            loader = str(value.get("loader", "fabric")).strip().lower()
            if not version:
                raise module.SpecValidationError(
                    "Technology target mapping requires minecraft_version."
                )
            try:
                adapter = adapter_for_target(version, loader)
            except ValueError as exc:
                raise module.SpecValidationError(str(exc)) from exc
            target = from_adapter(adapter)
            # If the caller supplied any target coordinate, it must match the adapter.
            aliases = {
                "edition": adapter.edition,
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
                "java_version": adapter.java_version,
                "fabric_loader": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
            }
            for field, expected in aliases.items():
                if field in value and str(value[field]) != expected:
                    raise module.SpecValidationError(
                        f"Technology target field {field} disagrees with reviewed "
                        f"adapter {adapter.adapter_id}: {value[field]!r} != {expected!r}."
                    )
            target.validate()
            return target
        raise module.SpecValidationError("Invalid technology target.")

    normalize._mmm_exact_adapter_normalization = True
    module.normalize_technology_target = normalize
