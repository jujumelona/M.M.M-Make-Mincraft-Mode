from __future__ import annotations

"""Fail-closed lowering of atomic PromptFacts into concrete ArtifactJobs."""

import re
from collections.abc import Iterable
from typing import Any

from .artifact_job import ArtifactJob
from .artifact_ports import PortKind
from .implementation_fact import ImplementationFact
from .implementation_identity import ExecutorType
from .implementation_template_renderer import render_template
from .prompt_fact_types import FactType, PromptFact
from .registered_leaf_binding import require_registered_leaf_binding
from .task_template_catalog import load_template


class ArtifactExpansionError(ValueError):
    pass


def _executor_type_from_string(exec_type_str: str) -> ExecutorType:
    """Convert a catalog executor type string to the runtime enum."""

    exec_type_map = {
        "deterministic_renderer": ExecutorType.DETERMINISTIC,
        "deterministic": ExecutorType.DETERMINISTIC,
        "python_generator": ExecutorType.PYTHON_GENERATOR,
        "template": ExecutorType.TEMPLATE,
        "model": ExecutorType.MODEL,
    }
    result = exec_type_map.get(exec_type_str)
    if result is None:
        raise ArtifactExpansionError(
            f"EXECUTOR_TYPE_UNKNOWN: {exec_type_str!r} not in {list(exec_type_map.keys())}"
        )
    return result


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
    FactType.ENTITY_EXISTS: ("minecraft/entity/registry",),
    FactType.GUI_EXISTS: ("minecraft/screen/registration",),
    FactType.NETWORK_PACKET: ("minecraft/network_payload/registration",),
    FactType.BLOCK_ENTITY_EXISTS: ("minecraft/block_entity/registry",),
    FactType.DATA_COMPONENT: ("minecraft/component/type",),
    FactType.WORLDGEN_FEATURE: ("minecraft/worldgen/configured_feature",),
    FactType.DIMENSION: ("minecraft/dimension/registry",),
    FactType.BIOME: ("minecraft/biome/registry",),
    FactType.STATUS_EFFECT: ("minecraft/effect/registry",),
    FactType.SOUND_EVENT: ("minecraft/sound/registration",),
    FactType.PARTICLE_TYPE: ("minecraft/particle/registry",),
    FactType.ENTITY_LOOT: ("minecraft/loot/entry",),
    FactType.ADVANCEMENT: ("minecraft/advancement/requirement",),
    FactType.EQUIPMENT_ARMOR: ("minecraft/item/properties",),
    FactType.CUSTOM_ITEM_BEHAVIOR: ("minecraft/item/interaction",),
    FactType.CUSTOM_BLOCK_BEHAVIOR: ("minecraft/block/interaction",),
    FactType.CONTENT_RELATION: ("minecraft/item/integration",),
}

CANONICAL_LEAF_DEFAULT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "minecraft/item/registry": ("fabric/item/key", "fabric/item/register_basic"),
    "minecraft/item/properties": ("fabric/item/settings_max_stack",),
    "minecraft/item/model": (
        "minecraft/resource/item/client_item",
        "minecraft/resource/item/model_generated",
    ),
    "minecraft/item/language": ("fabric/item/lang_en",),
    "minecraft/item/integration": ("fabric/item/initializer",),
    "minecraft/block/registry": ("fabric/block/key", "fabric/block/register_basic"),
    "minecraft/block/state": ("minecraft/resource/block/blockstate_simple",),
    "minecraft/block/model": ("minecraft/resource/block/model_cube_all",),
    "minecraft/language/key": ("fabric/block/lang_en",),
    "minecraft/block/integration": ("fabric/block/initializer",),
    "minecraft/block/drops": ("fabric/loot/block_drop",),
    "minecraft/recipe/serializer": ("fabric/recipe/shaped", "fabric/recipe/shapeless"),
    "minecraft/tag/entries": ("fabric/tag/registry",),
    "minecraft/loot/entry": ("fabric/loot/block_drop",),
}

_SUPPORTED_EXPANSIONS: dict[FactType, tuple[str, ...]] = {
    fact_type: (
        ("fabric/recipe/smelting",)
        if fact_type == FactType.SMELTING_RECIPE
        else tuple(
            template_id
            for leaf_id in leaf_ids
            for template_id in CANONICAL_LEAF_DEFAULT_TEMPLATES.get(leaf_id, ())
        )
    )
    for fact_type, leaf_ids in FACT_TO_CANONICAL_LEAVES.items()
}
FACT_EXPANSIONS = dict(_SUPPORTED_EXPANSIONS)


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


def _require_subject(fact: PromptFact | ImplementationFact) -> str:
    subject = str(fact.subject or "").strip()
    if not subject or not _REGISTRY_PATH.fullmatch(subject):
        raise ArtifactExpansionError(
            f"ARTIFACT_SUBJECT_INVALID: fact {fact.fact_id!r} has invalid registry path {subject!r}"
        )
    return subject


def _require_integer_value(
    fact: PromptFact | ImplementationFact,
    *,
    minimum: int,
    maximum: int,
) -> int:
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


def _leaf_binding(version_context: Any, leaf_id: str, *, source_candidate: bool):
    """Separate registration-level source candidates from production admission."""

    if source_candidate:
        return require_registered_leaf_binding(version_context, leaf_id)
    return version_context.require_leaf_binding(leaf_id)


def _templates_for_canonical_leaf(
    leaf_id: str,
    version_context: Any | None = None,
    *,
    source_candidate: bool = False,
) -> tuple[str, ...]:
    if version_context is None:
        return CANONICAL_LEAF_DEFAULT_TEMPLATES.get(leaf_id, ())
    binding = _leaf_binding(
        version_context,
        leaf_id,
        source_candidate=source_candidate,
    )
    impl = binding.get("implementation", {})
    templates: list[str] = []
    if "prerequisite_templates" in impl:
        templates.extend(impl["prerequisite_templates"])
    if "template" in impl and impl["template"]:
        templates.append(impl["template"])
    if "extra_templates" in impl:
        templates.extend(impl["extra_templates"])
    return tuple(dict.fromkeys(templates))


def _logical_step_name(template_id: str) -> str:
    """Keep graph identities stable while the host selects version-specific templates."""

    step_name = template_id.rsplit("/", 1)[-1]
    if template_id.startswith(("fabric/item/key", "fabric/block/key")):
        return "key"
    if template_id.startswith(("fabric/item/register", "fabric/block/register")):
        return "register_basic"
    return step_name


def validate_expansion_catalog() -> None:
    """Fail before generation if an expansion references a non-executable template."""

    for fact_type, identifiers in FACT_EXPANSIONS.items():
        if not identifiers:
            continue
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
                not isinstance(dependency, str) or not dependency
                for dependency in dependencies
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
    """Lower explicit facts without inventing a fallback implementation.

    Raw ``PromptFact`` lowering is a production boundary and therefore requires an
    admitted canonical leaf. ``ImplementationFact`` lowering is the planner's bounded
    source-candidate boundary: a structurally registered deterministic template may be
    materialized before execution evidence exists, but Python generators remain strict.
    """

    validate_expansion_catalog()
    if version_context is not None:
        from .resolved_version_context import VersionContextError

        if minecraft_version and minecraft_version != version_context.minecraft:
            raise VersionContextError(
                "VERSION_CONTEXT_MISMATCH",
                expected=version_context.minecraft,
                actual=minecraft_version,
            )
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
        source_candidate = isinstance(fact, ImplementationFact)
        canonical_leaf_ids = FACT_TO_CANONICAL_LEAVES.get(fact.fact_type)
        if not canonical_leaf_ids:
            raise ArtifactExpansionError(
                f"ARTIFACT_FACT_UNSUPPORTED: {fact.fact_type.value} not in FACT_TO_CANONICAL_LEAVES"
            )

        if version_context is not None:
            for leaf_id in canonical_leaf_ids:
                _leaf_binding(
                    version_context,
                    leaf_id,
                    source_candidate=source_candidate,
                )

        leaf_template_pairs: list[tuple[str, str]] = []
        for leaf_id in canonical_leaf_ids:
            templates = _templates_for_canonical_leaf(
                leaf_id,
                version_context,
                source_candidate=source_candidate,
            )
            if templates:
                leaf_template_pairs.extend((leaf_id, template_id) for template_id in templates)
                continue
            if version_context is None:
                raise ArtifactExpansionError("EXACT_HOST_IMPLEMENTATION_REQUIRED")
            binding = version_context.require_leaf_binding(leaf_id)
            if binding["implementation"]["executor_type"] != "python_generator":
                raise ArtifactExpansionError(
                    f"ARTIFACT_NO_TEMPLATE_NO_GENERATOR: {leaf_id}"
                )
            leaf_template_pairs.append((leaf_id, ""))

        resource_values: dict[str, Any] = {}
        if fact.fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            from .resource_fact_inputs import resource_inputs

            identifier, resource_values = resource_inputs(fact, mod_id)
            if version_context is not None and identifier not in {
                template_id for _, template_id in leaf_template_pairs
            }:
                raise ArtifactExpansionError("HOST_RESOURCE_TEMPLATE_NOT_BOUND")
            leaf_template_pairs = [(canonical_leaf_ids[0], identifier)]

        subject = _require_subject(fact)
        constant = _constant_name(subject)

        for canonical_leaf, template_id in leaf_template_pairs:
            if not template_id:
                if version_context is None:
                    raise ArtifactExpansionError(
                        f"ARTIFACT_PYTHON_GENERATOR_REQUIRES_CONTEXT: {canonical_leaf} needs version_context"
                    )
                binding = version_context.require_leaf_binding(canonical_leaf)
                impl_dict = binding.get("implementation", {})
                impl_id = impl_dict.get("implementation_id", "")
                exec_type = impl_dict.get("executor_type", "")
                if exec_type != "python_generator":
                    raise ArtifactExpansionError(
                        f"ARTIFACT_NO_TEMPLATE_NO_GENERATOR: {canonical_leaf} has no template and executor is {exec_type}"
                    )

                job_id = f"{subject}.{canonical_leaf.replace('/', '_')}"
                deterministic_inputs = {
                    "mod_id": mod_id,
                    "package_name": package_name,
                    "package_path": pkg_path,
                    "registry_path": subject,
                    "java_constant": constant,
                    "subject": subject,
                    "main_class": main_class_val,
                    "minecraft_version": minecraft_version,
                }
                candidate = ArtifactJob(
                    job_id=job_id,
                    template_id="",
                    owner_module=subject,
                    target_path="",
                    anchor="",
                    operation="generate",
                    requires=(),
                    required_ports=(),
                    produces=(),
                    deterministic_inputs=deterministic_inputs,
                    context_id=version_context.context_id,
                    canonical_leaf=canonical_leaf,
                    implementation_id=impl_id,
                    executor_type=_executor_type_from_string(exec_type),
                )
                prior = seen_jobs.get(job_id)
                if prior is not None:
                    if prior.to_dict() != candidate.to_dict():
                        raise ArtifactExpansionError(f"ARTIFACT_JOB_CONFLICT: {job_id}")
                    continue
                seen_jobs[job_id] = candidate
                jobs.append(candidate)
                continue

            step_name = _logical_step_name(template_id)
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
            if fact.fact_type == FactType.ITEM_STACK_LIMIT:
                deterministic_inputs["stack_limit"] = _require_integer_value(
                    fact,
                    minimum=1,
                    maximum=64,
                )
            if template_id.endswith("/lang_en"):
                deterministic_inputs["display_name"] = getattr(
                    fact,
                    "display_name",
                    "",
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
                render_binding(port["binding"])
                for port in template["produces"]
            ]

            impl_id = f"template:{template_id}"
            exec_type = "deterministic_renderer"
            if version_context is not None:
                binding = _leaf_binding(
                    version_context,
                    canonical_leaf,
                    source_candidate=source_candidate,
                )
                impl_dict = binding.get("implementation", {})
                impl_id = impl_dict.get("implementation_id", impl_id)
                exec_type = impl_dict.get("executor_type", exec_type)

            candidate = ArtifactJob(
                job_id=job_id,
                template_id=template_id,
                owner_module=subject,
                target_path=target_path,
                anchor=anchor,
                operation=template["target"]["operation"],
                requires=tuple(requires),
                required_ports=tuple(
                    {"name": name, **types}
                    for name, types in required_types.items()
                ),
                produces=tuple(produces),
                deterministic_inputs=deterministic_inputs,
                context_id=version_context.context_id if version_context is not None else "",
                canonical_leaf=canonical_leaf,
                implementation_id=impl_id,
                executor_type=_executor_type_from_string(exec_type),
            )
            prior = seen_jobs.get(job_id)
            if prior is not None:
                if prior.to_dict() != candidate.to_dict():
                    raise ArtifactExpansionError(f"ARTIFACT_JOB_CONFLICT: {job_id}")
                continue
            seen_jobs[job_id] = candidate
            jobs.append(candidate)

    return jobs
