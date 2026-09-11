from __future__ import annotations

"""Fail-closed lowering of atomic PromptFacts into concrete ArtifactJobs."""

import re
from collections.abc import Iterable
from typing import Any

from .artifact_job import ArtifactJob
from .artifact_ports import PortKind
from .implementation_fact import ImplementationFact
from .implementation_template_renderer import render_template
from .prompt_fact_types import FactType, PromptFact
from .task_template_catalog import load_template


class ArtifactExpansionError(ValueError):
    pass


_REGISTRY_PATH = re.compile(r"^[a-z0-9_.-]+$")

FACT_TO_CANONICAL_LEAVES: dict[FactType, tuple[str, ...]] = {
    FactType.ITEM_EXISTS: (
        "minecraft/item/registry",
        "minecraft/item/model",
        "minecraft/item/language",
        "minecraft/item/integration",
    ),
    FactType.ITEM_STACK_LIMIT: ("minecraft/item/properties",),
    FactType.BLOCK_EXISTS: (
        "minecraft/block/registry",
        "minecraft/block/state",
        "minecraft/block/model",
        "minecraft/language/key",
        "minecraft/block/integration",
    ),
    FactType.CRAFTING_RECIPE: ("minecraft/recipe/serializer",),
    FactType.SMELTING_RECIPE: ("minecraft/recipe/serializer",),
    FactType.REGISTRY_TAG: ("minecraft/tag/entries",),
    FactType.BLOCK_DROP: ("minecraft/block/drops",),
}

CANONICAL_LEAF_DEFAULT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "minecraft/item/registry": ("fabric/item/key", "fabric/item/register_basic"),
    "minecraft/item/properties": ("fabric/item/settings_max_stack",),
    "minecraft/item/model": ("fabric/item/client_item", "fabric/item/model_basic"),
    "minecraft/item/language": ("fabric/item/lang_en",),
    "minecraft/item/integration": ("fabric/item/initializer",),
    "minecraft/block/registry": ("fabric/block/key", "fabric/block/register_basic"),
    "minecraft/block/state": ("fabric/block/blockstate_basic",),
    "minecraft/block/model": ("fabric/block/model_cube_all",),
    "minecraft/language/key": ("fabric/block/lang_en",),
    "minecraft/block/integration": ("fabric/block/initializer",),
    "minecraft/block/drops": ("fabric/loot/block_drop",),
    "minecraft/recipe/serializer": ("fabric/recipe/shaped", "fabric/recipe/shapeless"),
    "minecraft/tag/entries": ("fabric/tag/registry",),
    "minecraft/loot/entry": ("fabric/loot/block_drop",),
}


def _templates_for_canonical_leaf(leaf_id: str, version_context=None) -> tuple[str, ...]:
    if version_context is not None:
        binding = version_context.require_leaf_binding(leaf_id)
        impl = binding.get("implementation", {})
        templates = []
        if "prerequisite_templates" in impl:
            templates.extend(impl["prerequisite_templates"])
        if "template" in impl and impl["template"]:
            templates.append(impl["template"])
        if "extra_templates" in impl:
            templates.extend(impl["extra_templates"])
        if templates:
            return tuple(dict.fromkeys(templates))
    return CANONICAL_LEAF_DEFAULT_TEMPLATES.get(leaf_id, ())


_SUPPORTED_EXPANSIONS: dict[FactType, tuple[str, ...]] = {
    fact_type: (
        ("fabric/recipe/smelting",)
        if fact_type == FactType.SMELTING_RECIPE
        else tuple(
            tid
            for leaf_id in leaf_ids
            for tid in CANONICAL_LEAF_DEFAULT_TEMPLATES.get(leaf_id, ())
        )
    )
    for fact_type, leaf_ids in FACT_TO_CANONICAL_LEAVES.items()
}

# Kept public for callers/tests, but every entry is verified before use.
FACT_EXPANSIONS = dict(_SUPPORTED_EXPANSIONS)

# These fact types are intentionally implemented by an existing module generator until
# their executable leaf catalog is complete.  Keeping this declaration explicit prevents
# a supported higher-level capability from being mistaken for an ArtifactJob leaf while
# still failing closed for every undeclared fact type.
DECLARED_GENERATOR_HANDOFFS = frozenset(
    {
        FactType.ENTITY_EXISTS,
        FactType.GUI_EXISTS,
        FactType.NETWORK_PACKET,
        FactType.BLOCK_ENTITY_EXISTS,
        FactType.DATA_COMPONENT,
        FactType.WORLDGEN_FEATURE,
        FactType.DIMENSION,
        FactType.BIOME,
        FactType.STATUS_EFFECT,
        FactType.SOUND_EVENT,
        FactType.PARTICLE_TYPE,
        FactType.ENTITY_LOOT,
        FactType.ADVANCEMENT,
        FactType.EQUIPMENT_ARMOR,
        FactType.CUSTOM_ITEM_BEHAVIOR,
        FactType.CUSTOM_BLOCK_BEHAVIOR,
        FactType.CONTENT_RELATION,
    }
)


def implementation_route(fact_type: FactType) -> str:
    if fact_type in FACT_EXPANSIONS:
        return "artifact"
    if fact_type in DECLARED_GENERATOR_HANDOFFS:
        return "generator"
    raise ArtifactExpansionError(
        f"ARTIFACT_FACT_UNSUPPORTED: no implementation route exists for {fact_type.value}"
    )


def _constant_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in value).strip("_")
    if not cleaned:
        raise ArtifactExpansionError(
            "ARTIFACT_SUBJECT_EMPTY: cannot derive a Java constant"
        )
    return cleaned.upper()


def _java_package_path(package_name: str) -> str:
    parts = [part for part in package_name.split(".") if part]
    if not parts or any(not part.replace("_", "a").isalnum() for part in parts):
        raise ArtifactExpansionError(
            f"ARTIFACT_PACKAGE_INVALID: invalid Java package {package_name!r}"
        )
    return "/".join(parts)


def _require_subject(fact: PromptFact) -> str:
    subject = str(fact.subject or "").strip()
    if not subject or not _REGISTRY_PATH.fullmatch(subject):
        raise ArtifactExpansionError(
            f"ARTIFACT_SUBJECT_INVALID: fact {fact.fact_id!r} has invalid registry path {subject!r}"
        )
    return subject


def _require_integer_value(fact: PromptFact, *, minimum: int, maximum: int) -> int:
    value = fact.value
    if type(value) is not int:
        raise ArtifactExpansionError(
            f"ARTIFACT_FACT_VALUE_REQUIRED: {fact.fact_type.value} requires an explicit integer value"
        )
    if not minimum <= value <= maximum:
        raise ArtifactExpansionError(
            f"ARTIFACT_FACT_VALUE_RANGE: {fact.fact_type.value} value {value} is outside "
            f"{minimum}..{maximum}"
        )
    return value


def validate_expansion_catalog() -> None:
    """Fail before generation if an expansion references a non-executable leaf."""
    for fact_type, identifiers in FACT_EXPANSIONS.items():
        if not identifiers:
            raise ArtifactExpansionError(
                f"ARTIFACT_EXPANSION_EMPTY: {fact_type.value} has no leaf templates"
            )
        for identifier in identifiers:
            try:
                template = load_template(identifier)
            except Exception as exc:
                raise ArtifactExpansionError(
                    f"ARTIFACT_TEMPLATE_MISSING: {fact_type.value} references {identifier!r}"
                ) from exc
            if not isinstance(template.get("render"), (str, dict)):
                raise ArtifactExpansionError(
                    f"ARTIFACT_TEMPLATE_NOT_EXECUTABLE: {identifier!r} has no render mold"
                )
            fixture = template.get("fixture")
            if not isinstance(fixture, dict) or not isinstance(
                fixture.get("input"), dict
            ):
                raise ArtifactExpansionError(
                    f"ARTIFACT_TEMPLATE_UNTESTED: {identifier!r} has no executable fixture"
                )
            target = template.get("target")
            if (
                not isinstance(target, dict)
                or not isinstance(target.get("file"), str)
                or not target["file"]
            ):
                raise ArtifactExpansionError(f"ARTIFACT_TARGET_REQUIRED: {identifier}")
            dependencies = template.get("dependencies")
            if not isinstance(dependencies, list) or any(
                not isinstance(d, str) or not d for d in dependencies
            ):
                raise ArtifactExpansionError(
                    f"ARTIFACT_DEPENDENCIES_REQUIRED: {identifier}"
                )
            ports = template.get("produces")
            if not isinstance(ports, list):
                raise ArtifactExpansionError(f"ARTIFACT_PORT_DECLARATION: {identifier}")
            for port in ports:
                if not isinstance(port, dict) or any(
                    not isinstance(port.get(key), str) or not port[key].strip()
                    for key in ("name", "binding", "kind", "target_type", "value")
                ):
                    raise ArtifactExpansionError(
                        f"ARTIFACT_PORT_DECLARATION: {identifier}"
                    )
                try:
                    PortKind(port["kind"])
                except ValueError as exc:
                    raise ArtifactExpansionError(
                        f"ARTIFACT_PORT_KIND: {identifier}"
                    ) from exc


def expand_facts_to_jobs(
    facts: Iterable[PromptFact | ImplementationFact],
    *,
    mod_id: str,
    package_name: str,
    main_class: str = "",
    minecraft_version: str = "",
    version_context=None,
) -> list[ArtifactJob]:
    """Lower only explicitly supported facts; never invent a fallback implementation."""
    validate_expansion_catalog()
    if version_context is not None:
        from .resolved_version_context import VersionContextError

        if minecraft_version and minecraft_version != version_context.minecraft:
            raise VersionContextError("VERSION_CONTEXT_MISMATCH", expected=version_context.minecraft, actual=minecraft_version)
        minecraft_version = version_context.minecraft
    mod_id = str(mod_id or "").strip()
    if not _REGISTRY_PATH.fullmatch(mod_id):
        raise ArtifactExpansionError(
            f"ARTIFACT_MOD_ID_INVALID: invalid mod id {mod_id!r}"
        )
    pkg_path = _java_package_path(package_name)

    jobs: list[ArtifactJob] = []
    seen_jobs: dict[str, ArtifactJob] = {}
    main_class_val = (
        main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"
    )

    for fact in facts:
        route = implementation_route(fact.fact_type)
        if route == "generator":
            # The ImplementationFact remains in the proposal and is consumed by the
            # module generator selected from its content capability.  It is deliberately
            # not represented as a fake/empty ArtifactJob.
            continue
        canonical_leaf_ids = FACT_TO_CANONICAL_LEAVES[fact.fact_type]
        if version_context is not None:
            for leaf_id in canonical_leaf_ids:
                version_context.require_leaf_binding(leaf_id)

        template_ids = []
        for leaf_id in canonical_leaf_ids:
            template_ids.extend(_templates_for_canonical_leaf(leaf_id, version_context))
        if not template_ids:
            template_ids = list(FACT_EXPANSIONS[fact.fact_type])

        resource_values = {}
        if fact.fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            from .resource_fact_inputs import resource_inputs

            identifier, resource_values = resource_inputs(fact, mod_id)
            template_ids = (identifier,)
        subject = _require_subject(fact)
        constant = _constant_name(subject)

        for template_id in template_ids:
            step_name = template_id.rsplit("/", 1)[-1]
            job_id = f"{subject}.{step_name}"

            deterministic_inputs: dict[str, Any] = {
                "mod_id": mod_id,
                "package_name": package_name,
                "package_path": pkg_path,
                "registry_path": subject,
                "java_constant": constant,
                "subject": subject,
                "main_class": main_class_val,
                "minecraft_version": minecraft_version,
                **resource_values,
            }
            # Fact values remain host-validated; leaf topology belongs to the catalog.
            if fact.fact_type == FactType.ITEM_STACK_LIMIT:
                deterministic_inputs["stack_limit"] = _require_integer_value(
                    fact, minimum=1, maximum=64
                )
            if template_id.endswith("/lang_en"):
                deterministic_inputs["display_name"] = getattr(
                    fact, "display_name", ""
                ) or " ".join(part.capitalize() for part in subject.split("_"))
            if fact.fact_type == FactType.BLOCK_DROP:
                if not fact.object:
                    raise ArtifactExpansionError(
                        "ARTIFACT_DROP_TARGET_REQUIRED: block drop needs an explicit item"
                    )
                deterministic_inputs["drop_item"] = fact.object

            template = load_template(template_id)
            if version_context is not None:
                version_context.admit_template(template)
            from .artifact_target_contract import validate_artifact_target

            validate_artifact_target(template, minecraft_version)

            def render_binding(pattern: str, values=deterministic_inputs) -> str:
                return render_template({"render": pattern}, values)

            target_path = render_binding(template["target"]["file"])
            anchor = render_binding(template["target"].get("anchor", ""))
            requires = [render_binding(value) for value in template["dependencies"]]
            registry_suffix = {
                "item": "registry_id",
                "block": "block_registry_id",
                "entity_type": "entity_registry_id",
            }.get(resource_values.get("registry_kind"), "registry_id")
            requires.extend(
                f"{ref.split(':', 1)[1]}.{registry_suffix}"
                for ref in resource_values.get("resource_references", [])
                if ref.split(":", 1)[0] == mod_id
            )
            required_types = {
                render_binding(name): types
                for name, types in template.get("dependency_types", {}).items()
            }
            for ref in resource_values.get("resource_references", []):
                if ref.split(":", 1)[0] == mod_id:
                    required_types[f"{ref.split(':', 1)[1]}.{registry_suffix}"] = {
                        "kind": "REGISTRY_ID",
                        "target_type": {
                            "registry_id": "Item",
                            "block_registry_id": "Block",
                            "entity_registry_id": "EntityType<?>",
                        }[registry_suffix],
                    }
            produces = [
                render_binding(port["binding"]) for port in template["produces"]
            ]

            candidate = ArtifactJob(
                job_id=job_id,
                template_id=template_id,
                owner_module=subject,
                target_path=target_path,
                anchor=anchor,
                operation=template["target"]["operation"],
                requires=tuple(requires),
                required_ports=tuple(
                    {"name": name, **types} for name, types in required_types.items()
                ),
                produces=tuple(produces),
                deterministic_inputs=deterministic_inputs,
                context_id=version_context.context_id if version_context is not None else "",
            )
            prior = seen_jobs.get(job_id)
            if prior is not None:
                if prior.to_dict() != candidate.to_dict():
                    raise ArtifactExpansionError(f"ARTIFACT_JOB_CONFLICT: {job_id}")
                continue
            seen_jobs[job_id] = candidate
            jobs.append(candidate)

    return jobs
