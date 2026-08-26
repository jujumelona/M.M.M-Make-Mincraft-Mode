from __future__ import annotations

"""Single owner for executable technology-target binding.

Technology classification may be request-derived, but exact integration evidence is
always bound to the host-selected executable platform provider. No historical target
is injected when the caller omitted platform selection.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

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

    def validate(self: Any) -> None:
        try:
            adapter = adapter_for_target(self.minecraft_version, self.loader)
        except ValueError as exc:
            raise module.SpecValidationError(str(exc)) from exc
        expected = {
            "edition": adapter.edition,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
            "java_version": adapter.java_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
        }
        for field, expected_value in expected.items():
            actual = getattr(self, field)
            if actual != expected_value:
                raise module.SpecValidationError(
                    f"Technology target is mixed at {field}: expected "
                    f"{expected_value!r}, got {actual!r}."
                )

    validate._mmm_provider_verified_target = True
    target_cls.validate = validate

    def normalize(value: Any):
        if value is None:
            raise module.SpecValidationError(
                "Technology analysis requires the host-selected platform target; "
                "targetless calls are not assigned a historical default."
            )
        if isinstance(value, target_cls):
            value.validate()
            return value
        if isinstance(value, module.PlatformLock):
            try:
                adapter = adapter_for_target(value.minecraft_version, value.loader)
            except ValueError as exc:
                raise module.SpecValidationError(str(exc)) from exc
            target = from_adapter(adapter)
            target.validate()
            return target
        if isinstance(value, Mapping):
            version = str(value.get("minecraft_version", "")).strip()
            loader = str(value.get("loader", "")).strip().casefold()
            if not version or not loader:
                raise module.SpecValidationError(
                    "Technology target mapping requires minecraft_version and loader."
                )
            try:
                adapter = adapter_for_target(version, loader)
            except ValueError as exc:
                raise module.SpecValidationError(str(exc)) from exc
            aliases = {
                "edition": adapter.edition,
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
                "java_version": adapter.java_version,
                "fabric_loader": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
            }
            for field, expected_value in aliases.items():
                if field in value and str(value[field]) != expected_value:
                    raise module.SpecValidationError(
                        f"Technology target field {field} disagrees with executable "
                        f"provider {adapter.adapter_id}: {value[field]!r} != "
                        f"{expected_value!r}."
                    )
            target = from_adapter(adapter)
            target.validate()
            return target
        raise module.SpecValidationError("Invalid technology target.")

    normalize._mmm_exact_adapter_normalization = True
    normalize._mmm_provider_verified_target = True
    module.normalize_technology_target = normalize

    original_make_requirement = module._make_requirement
    if not getattr(original_make_requirement, "_mmm_dynamic_platform_query", False):

        def make_requirement(domain: Any, capability: str, *, target: Any, flags: Any):
            requirement = original_make_requirement(
                domain,
                capability,
                target=target,
                flags=flags,
            )
            platform_query = (
                f"Minecraft {target.minecraft_version} {target.loader} mappings "
                f"{target.mappings} Java {target.java_version} "
                f"{capability.replace('_', ' ')} integration compatibility testing"
            )
            tests = tuple(
                "platform_integration" if name == "fabric_integration" else name
                for name in requirement.required_tests
            )
            return replace(
                requirement,
                research_queries=(requirement.research_queries[0], platform_query),
                required_tests=tests,
            )

        make_requirement._mmm_dynamic_platform_query = True
        module._make_requirement = make_requirement


__all__ = ["install"]
