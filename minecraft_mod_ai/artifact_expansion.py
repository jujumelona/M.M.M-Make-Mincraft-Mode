from __future__ import annotations

"""Fail-closed lowering of atomic PromptFacts into concrete ArtifactJobs."""

import re
from collections.abc import Iterable, Mapping
from typing import Any
from packaging.version import Version

from .artifact_job import ArtifactJob
from .artifact_ports import PortKind
from .implementation_fact import ImplementationFact
from .execution_contract_policy import (
    SCHEMA_CONTRACT_PROFILE_KEY,
    SCHEMA_STRING_CLASS_KEY,
    SOURCE_REPAIR_MAX_SOURCE_CHARS,
    SOURCE_REPAIR_SCHEMA_PROFILE,
    STRING_CLASS_SOURCE,
)
from .implementation_identity import ExecutorType
from .implementation_template_renderer import render_template
from .prompt_fact_types import FactType, PromptFact
from .registered_leaf_binding import require_registered_leaf_binding
from .task_template_catalog import load_template


class ArtifactExpansionError(ValueError):
    pass


def _runtime_executor_type(
    *,
    implementation_id: str,
    template_id: str = "",
    declared_executor_type: str = "",
) -> ExecutorType:
    """Resolve runtime dispatch from the concrete registered implementation identity."""

    if template_id:
        return ExecutorType.TEMPLATE
    if implementation_id.startswith("python_generator:"):
        return ExecutorType.PYTHON_GENERATOR
    direct = {
        "deterministic": ExecutorType.DETERMINISTIC,
        "python_generator": ExecutorType.PYTHON_GENERATOR,
        "template": ExecutorType.TEMPLATE,
        "model": ExecutorType.MODEL,
    }.get(str(declared_executor_type or "").strip())
    if direct is not None:
        return direct
    raise ArtifactExpansionError(
        "EXECUTOR_TYPE_UNKNOWN: "
        f"implementation={implementation_id!r}, declared={declared_executor_type!r}"
    )

_REGISTRY_PATH = re.compile(r"^[a-z0-9_.-]+$")

# P0-2: All facts now have ArtifactJob representations - no separate generator handoff
# Entity, GUI, networking등은 canonical leaf를 통해 처리됨

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
    # P0-2: Former generator handoffs now use canonical leaves
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
}

CANONICAL_LEAF_DEFAULT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "minecraft/item/registry": ("fabric/item/key", "fabric/item/register_basic"),
    "minecraft/item/properties": ("fabric/item/settings_max_stack",),
    "minecraft/item/model": ("minecraft/resource/item/client_item", "minecraft/resource/item/model_generated"),
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
    "minecraft/loot/entry": ("fabric/loot/entity_drop",),
}


def _templates_for_canonical_leaf(leaf_id: str, version_context=None) -> tuple[str, ...]:
    if version_context is not None:
        binding = require_registered_leaf_binding(version_context, leaf_id)
        impl = binding.get("implementation", {})
        templates = []
        if "prerequisite_templates" in impl:
            templates.extend(impl["prerequisite_templates"])
        if "template" in impl and impl["template"]:
            templates.append(impl["template"])
        if "extra_templates" in impl:
            templates.extend(impl["extra_templates"])
        return tuple(dict.fromkeys(templates))
    return CANONICAL_LEAF_DEFAULT_TEMPLATES.get(leaf_id, ())


# P0-2: REMOVED - No more separate generator handoff
# All facts expand to canonical leaves which then have implementations

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


# P0-2: REMOVED generator_implementation_profile() - no separate generator path
# All facts expand through canonical leaves with unified implementation registry


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


def _java_type_name(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", str(value))
    name = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not name:
        raise ArtifactExpansionError("CANONICAL_GENERATOR_CLASS_NAME_REQUIRED")
    if name[0].isdigit():
        name = "Generated" + name
    return name


def _canonical_candidate_inputs(
    fact: PromptFact | ImplementationFact,
    *,
    canonical_leaf: str,
    mod_id: str,
    package_name: str,
    package_path: str,
    subject: str,
    minecraft_version: str,
    context_id: str,
) -> dict[str, Any]:
    """Build the persisted host-owned input contract for a generator candidate."""

    manifest = load_template(canonical_leaf)
    side_values = tuple(str(value).upper() for value in manifest.get("side", ()))
    if len(side_values) != 1:
        raise ArtifactExpansionError(
            f"CANONICAL_GENERATOR_SIDE_AMBIGUOUS: {canonical_leaf}: {side_values}"
        )
    side = side_values[0]
    responsibility = canonical_leaf.rsplit("/", 1)[-1]
    class_name = _java_type_name(subject) + _java_type_name(responsibility)
    if side == "CLIENT":
        generated_package = package_name + ".client.generated"
        generated_package_path = package_path + "/client/generated"
    else:
        generated_package = package_name + ".generated"
        generated_package_path = package_path + "/generated"
    # Keep modern client-only classes out of the common compileJava source set.
    # Fabric 26.1+ is unobfuscated and the official scaffold has a client set.
    source_root = "src/client/java" if side == "CLIENT" and minecraft_version.startswith("26.") else "src/main/java"
    target_path = f"{source_root}/{generated_package_path}/{class_name}.java"
    requirement = str(getattr(fact, "source_clause", "") or "").strip()
    if not requirement:
        requirement = (
            str(getattr(fact, "display_name", "") or "").strip()
            or f"Implement {subject} through {canonical_leaf}."
        )
    display_name = (
        str(getattr(fact, "display_name", "") or "").strip()
        or " ".join(part.capitalize() for part in subject.split("_"))
    )
    bindings = {
        "mod_id": mod_id,
        "package_name": generated_package,
        "class_name": class_name,
        "registry_path": subject,
        "minecraft_version": minecraft_version,
        "display_name": display_name,
        "fact_type": fact.fact_type.value,
        "source_requirement": requirement[:12000],
    }
    parent_requirement = str(getattr(fact, "parent_requirement", "") or "").strip()
    if parent_requirement:
        bindings["parent_requirement"] = parent_requirement

    source_pattern = (
        rf"(?s)\bpackage\s+{re.escape(generated_package)}\s*;.*"
        rf"\bpublic\s+(?:final\s+)?class\s+{re.escape(class_name)}"
        rf"\s+implements\s+(?:net\.fabricmc\.api\.)?ClientModInitializer\b"
        rf".*\bonInitializeClient\s*\("
        if side == "CLIENT"
        else (
            rf"(?s)\bpackage\s+{re.escape(generated_package)}\s*;.*"
            rf"(?:class|record|interface|enum)\s+{re.escape(class_name)}\b"
        )
    )
    # The HOST owns the immutable mapping/API epoch. The small model supplies
    # only the leaf implementation; do not ask it to select mappings by inference.
    epoch_source_contract = ""
    if minecraft_version.startswith("26."):
        epoch_source_contract = (
            "Minecraft 26.1+ uses unobfuscated Mojang API names. "
            "Never use Yarn classes net.minecraft.client.gui.screen.*, "
            "net.minecraft.client.gui.widget.*, net.minecraft.text.*, "
            "net.minecraft.util.Identifier or net.minecraft.entity.effect.*. "
            "Client Screen is net.minecraft.client.gui.screens.Screen; "
            "text is net.minecraft.network.chat.Component. "
            "The ClientModInitializer registration class must NOT extend Screen: "
            "implement UI screens as separately named classes with valid constructors, "
            "and bind registration only to verified APIs. "
            "If an API or required screen/menu registration is not known, "
            "do not invent symbols or a no-op success implementation. "
        )
    output_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": SOURCE_REPAIR_MAX_SOURCE_CHARS,
        "pattern": source_pattern,
        SCHEMA_CONTRACT_PROFILE_KEY: SOURCE_REPAIR_SCHEMA_PROFILE,
        SCHEMA_STRING_CLASS_KEY: STRING_CLASS_SOURCE,
    }
    # 26.1 screen registrations are host-rendered, not AI-authored Java.
    # The semantic requirement stays in the model prompt, but only two
    # non-executable strings are requested from the small model.
    screen_contract = None
    if canonical_leaf == "minecraft/screen/registration" and minecraft_version.startswith("26."):
        from .host_screen_shell import basic_screen_candidate_contract

        screen_contract = basic_screen_candidate_contract(
            package_name=generated_package,
            class_name=class_name,
            mod_id=mod_id,
            subject=subject,
            default_title=display_name,
            requirement=requirement,
            minecraft_version=minecraft_version,
        )
    block_entity_contract = None
    if (
        canonical_leaf == "minecraft/block_entity/registry"
        and minecraft_version.startswith("26.")
    ):
        from .host_block_entity_shell import basic_block_entity_candidate_contract

        block_entity_contract = basic_block_entity_candidate_contract(
            package_name=generated_package,
            class_name=class_name,
            mod_id=mod_id,
            subject=subject,
            minecraft_version=minecraft_version,
        )
    if screen_contract:
        bindings["host_screen_capability"] = screen_contract["capability"]
        bindings["host_screen_command"] = screen_contract["command"]
    if block_entity_contract:
        bindings["host_block_entity_capability"] = block_entity_contract["capability"]
        bindings["host_block_entity_entrypoint"] = block_entity_contract["entrypoint"]
    host_contract = screen_contract or block_entity_contract
    spec = {
        "leaf_id": canonical_leaf,
        "context_id": context_id,
        "requirement": requirement[:12000],
        "target_path": target_path,
        "language": "java",
        "operation": "CREATE_FILE",
        "side": side,
        "bindings": bindings,
        "output_schema": output_schema,
        "render_mold": host_contract["render_mold"] if host_contract else "{{artifact_source}}",
        "slots": host_contract["slots"] if host_contract else [
            {
                "name": "artifact_source",
                "description": (
                    "Return complete compilable Java source only, with no Markdown. "
                    f"Use package {generated_package} and declare public final class {class_name}. "
                    + (
                        "The class must implement net.fabricmc.api.ClientModInitializer "
                        "and perform its registration from public void onInitializeClient(). "
                        if side == "CLIENT"
                        else ""
                    )
                    + epoch_source_contract
                    + f"Implement only canonical responsibility {canonical_leaf} for "
                    f"Minecraft {minecraft_version}. Preserve the supplied gameplay "
                    "requirement and verified bindings; do not invent unrelated systems."
                ),
                "schema": output_schema,
            }
        ],
        "java_filename": class_name + ".java",
    }

    inputs: dict[str, Any] = {}
    for port in manifest.get("inputs", ()):
        name = str(port.get("name") or "")
        port_type = str(port.get("type") or "")
        if port_type == "identifier":
            inputs[name] = f"{mod_id}:{subject}"
        elif port_type == "specification":
            inputs[name] = spec
        elif port.get("required", True):
            raise ArtifactExpansionError(
                f"CANONICAL_GENERATOR_INPUT_UNSUPPORTED: {canonical_leaf}: {port_type}"
            )
    if not inputs:
        raise ArtifactExpansionError(f"CANONICAL_GENERATOR_INPUTS_EMPTY: {canonical_leaf}")
    return inputs


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
    """Fail before generation if an expansion references a non-executable leaf.
    
    P0-2: Empty expansions are now allowed - they will use PYTHON_GENERATOR
    implementations from ImplementationRegistry instead of templates.
    """
    for fact_type, identifiers in FACT_EXPANSIONS.items():
        # P0-2: Empty is now valid - will use PYTHON_GENERATOR executor
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


def _require_declared_loot_binding(
    fact: PromptFact | ImplementationFact,
    registry_port_by_subject: Mapping[str, str],
) -> str:
    """Validate source and target of a drop fact against declared local registries."""
    drop_item = str(fact.object or "").strip()
    if not _REGISTRY_PATH.fullmatch(drop_item):
        raise ArtifactExpansionError(
            "ARTIFACT_DROP_TARGET_REQUIRED: "
            f"{fact.fact_type.value} needs a declared local item registry "
            "path (no invented default or unregistered external item): "
            f"{drop_item!r}"
        )
    if registry_port_by_subject.get(drop_item) != "registry_id":
        raise ArtifactExpansionError(
            "ARTIFACT_DROP_TARGET_UNREGISTERED: "
            f"{fact.fact_type.value} {fact.subject} -> {drop_item} "
            "requires an ITEM_EXISTS fact for the target"
        )
    expected_owner = (
        "entity_registry_id"
        if fact.fact_type == FactType.ENTITY_LOOT
        else "block_registry_id"
    )
    if registry_port_by_subject.get(fact.subject) != expected_owner:
        raise ArtifactExpansionError(
            "ARTIFACT_DROP_OWNER_UNREGISTERED: "
            f"{fact.fact_type.value} {fact.subject} requires its "
            f"registered {'entity' if fact.fact_type == FactType.ENTITY_LOOT else 'block'} owner"
        )
    return drop_item


def expand_facts_to_jobs(
    facts: Iterable[PromptFact | ImplementationFact],
    *,
    mod_id: str,
    package_name: str,
    main_class: str = "",
    minecraft_version: str = "",
    version_context=None,
) -> list[ArtifactJob]:
    """Lower facts only through a resolved HOST version authority."""
    validate_expansion_catalog()
    if version_context is None:
        raise ArtifactExpansionError("RESOLVED_VERSION_CONTEXT_REQUIRED")

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

    # Material facts, not the recipe template, own each local registry identity.
    # A recipe may consume items yet produce a block; using a single global
    # registry suffix creates a phantom `block.registry_id` dependency.
    facts = tuple(facts)
    registry_port_by_subject: dict[str, str] = {}
    registry_fact_ports = {
        FactType.ITEM_EXISTS: "registry_id",
        FactType.BLOCK_EXISTS: "block_registry_id",
        FactType.ENTITY_EXISTS: "entity_registry_id",
    }
    for registry_fact in facts:
        registry_port = registry_fact_ports.get(registry_fact.fact_type)
        if registry_port is None:
            continue
        prior_port = registry_port_by_subject.get(registry_fact.subject)
        if prior_port is not None and prior_port != registry_port:
            raise ArtifactExpansionError(
                "ARTIFACT_RESOURCE_REGISTRY_KIND_CONFLICT: "
                + str(registry_fact.subject)
            )
        registry_port_by_subject[registry_fact.subject] = registry_port

    jobs: list[ArtifactJob] = []
    seen_jobs: dict[str, ArtifactJob] = {}
    main_class_val = (
        main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"
    )

    for fact in facts:
        # Relations describe module topology, not a material artifact. They are
        # already lowered into ProductionModule.depends_on/executable_relations
        # before this stage and must never be reinterpreted as item integration.
        if fact.fact_type == FactType.CONTENT_RELATION:
            continue

        # P0-2: No more generator handoff - all material facts expand through
        # canonical leaves.
        canonical_leaf_ids = FACT_TO_CANONICAL_LEAVES.get(fact.fact_type)
        if not canonical_leaf_ids:
            raise ArtifactExpansionError(
                f"ARTIFACT_FACT_UNSUPPORTED: {fact.fact_type.value} not in FACT_TO_CANONICAL_LEAVES"
            )
        
        for leaf_id in canonical_leaf_ids:
            require_registered_leaf_binding(version_context, leaf_id)

        leaf_template_pairs: list[tuple[str, str]] = []
        for leaf_id in canonical_leaf_ids:
            templates = _templates_for_canonical_leaf(leaf_id, version_context)
            if templates:
                leaf_template_pairs.extend((leaf_id, tid) for tid in templates)
                continue
            binding = require_registered_leaf_binding(version_context, leaf_id)
            implementation = binding["implementation"]
            implementation_id = str(implementation.get("implementation_id") or "")
            if not implementation_id.startswith("python_generator:"):
                raise ArtifactExpansionError(
                    f"ARTIFACT_NO_TEMPLATE_NO_GENERATOR: {leaf_id}"
                )
            leaf_template_pairs.append((leaf_id, ""))

        resource_values = {}
        if fact.fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            from .resource_fact_inputs import resource_inputs

            identifier, resource_values = resource_inputs(fact, mod_id)
            if identifier not in {tid for _, tid in leaf_template_pairs}:
                raise ArtifactExpansionError("HOST_RESOURCE_TEMPLATE_NOT_BOUND")
            leaf_template_pairs = [(canonical_leaf_ids[0], identifier)]
        subject = _require_subject(fact)
        constant = _constant_name(subject)

        for canonical_leaf, template_id in leaf_template_pairs:
            # P0-2: Handle PYTHON_GENERATOR (empty template_id)
            if not template_id:
                # PYTHON_GENERATOR: Get implementation from registry
                binding = require_registered_leaf_binding(version_context, canonical_leaf)
                impl_dict = binding.get("implementation", {})
                impl_id = impl_dict.get("implementation_id", "")
                exec_type = impl_dict.get("executor_type", "")
                
                if not str(impl_id).startswith("python_generator:"):
                    raise ArtifactExpansionError(
                        f"ARTIFACT_NO_TEMPLATE_NO_GENERATOR: {canonical_leaf} has no registered Python generator implementation"
                    )
                
                # Create minimal ArtifactJob for PYTHON_GENERATOR
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
                    "_canonical_inputs": _canonical_candidate_inputs(
                        fact,
                        canonical_leaf=canonical_leaf,
                        mod_id=mod_id,
                        package_name=package_name,
                        package_path=pkg_path,
                        subject=subject,
                        minecraft_version=minecraft_version,
                        context_id=version_context.context_id,
                    ),
                }
                
                # Convert executor_type string to enum
                exec_type_enum = _runtime_executor_type(
                    implementation_id=impl_id,
                    declared_executor_type=exec_type,
                )
                
                candidate = ArtifactJob(
                    job_id=job_id,
                    template_id="",  # No template for PYTHON_GENERATOR
                    owner_module=subject,
                    target_path="",  # Will be determined by generator
                    anchor="",
                    operation="generate",
                    requires=(),
                    required_ports=(),
                    # The canonical entity registry generator produces one
                    # verified receipt and one source artifact. Publish their
                    # real typed outputs so entity loot can depend on actual
                    # registration generation, not an invented entity ID port.
                    produces=(
                        (f"{subject}.entity_registry_receipt",
                         f"{subject}.entity_registry_artifact")
                        if canonical_leaf == "minecraft/entity/registry"
                        else ()
                    ),
                    deterministic_inputs=deterministic_inputs,
                    context_id=version_context.context_id,
                    canonical_leaf=canonical_leaf,
                    implementation_id=impl_id,
                    executor_type=exec_type_enum,
                )
                
                prior = seen_jobs.get(job_id)
                if prior is not None:
                    if prior.to_dict() != candidate.to_dict():
                        raise ArtifactExpansionError(f"ARTIFACT_JOB_CONFLICT: {job_id}")
                    continue
                seen_jobs[job_id] = candidate
                jobs.append(candidate)
                continue
            
            # Normal template-based job creation
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
                "loot_directory": "loot_table" if Version(minecraft_version) >= Version("1.21") else "loot_tables",
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
            if fact.fact_type in {FactType.BLOCK_DROP, FactType.ENTITY_LOOT}:
                deterministic_inputs["drop_item"] = _require_declared_loot_binding(
                    fact, registry_port_by_subject
                )

            template = load_template(template_id)
            version_context.admit_template(template)
            from .artifact_target_contract import validate_artifact_target

            validate_artifact_target(template, minecraft_version)

            def render_binding(pattern: str, values=deterministic_inputs) -> str:
                return render_template({"render": pattern}, values)

            target_path = render_binding(template["target"]["file"])
            anchor = render_binding(template["target"].get("anchor", ""))
            # The reviewed client BlockItem model has no implicit dependency
            # array in its catalog mold. Bind it explicitly to the BlockItem
            # registry port emitted by fabric/block/register_basic.
            if template_id == "minecraft/resource/item/client_block_item":
                requires = [f"{subject}.registry_id"]
            else:
                requires = [render_binding(value) for value in template["dependencies"]]
            expected_tag_port = {
                "item": "registry_id",
                "block": "block_registry_id",
                "entity_type": "entity_registry_id",
            }.get(resource_values.get("registry_kind"))
            required_types = {
                render_binding(name): types
                for name, types in template.get("dependency_types", {}).items()
            }
            if template_id == "minecraft/resource/item/client_block_item":
                required_types[f"{subject}.registry_id"] = {
                    "kind": "REGISTRY_ID",
                    "target_type": "Item",
                }
            for ref in resource_values.get("resource_references", []):
                namespace, local_subject = ref.split(":", 1)
                if namespace != mod_id:
                    continue
                registry_suffix = registry_port_by_subject.get(
                    local_subject, expected_tag_port or "registry_id"
                )
                if fact.fact_type in {FactType.CRAFTING_RECIPE, FactType.SMELTING_RECIPE}:
                    # Recipe inputs and outputs are Item identities, even when
                    # their source fact declares a placeable Block. The Block
                    # producer publishes a distinct BlockItem REGISTRY_ID port;
                    # consuming its Block REGISTRY_ID must not type-check.
                    if registry_suffix not in {"registry_id", "block_registry_id"}:
                        raise ArtifactExpansionError(
                            "ARTIFACT_RECIPE_NON_ITEM_REFERENCE: " + ref
                        )
                    registry_suffix = "registry_id"
                if expected_tag_port is not None and registry_suffix != expected_tag_port:
                    raise ArtifactExpansionError(
                        "ARTIFACT_RESOURCE_TAG_REGISTRY_MISMATCH: "
                        + f"{ref}: expected {expected_tag_port}, found {registry_suffix}"
                    )
                port_name = f"{local_subject}.{registry_suffix}"
                requires.append(port_name)
                required_types[port_name] = {
                    "kind": "REGISTRY_ID",
                    "target_type": {
                        "registry_id": "Item",
                        "block_registry_id": "Block",
                        "entity_registry_id": "EntityType<?>",
                    }[registry_suffix],
                }
            produces = [
                render_binding(port["binding"])
                for port in (
                    [] if template_id == "minecraft/resource/item/client_block_item"
                    else template["produces"]
                )
            ]

            binding = require_registered_leaf_binding(
                version_context,
                canonical_leaf,
            )
            impl_dict = binding.get("implementation", {})
            impl_id = str(impl_dict.get("implementation_id") or "")
            exec_type = str(impl_dict.get("executor_type") or "")
            if not impl_id or not exec_type:
                raise ArtifactExpansionError(
                    f"ARTIFACT_IMPLEMENTATION_BINDING_INCOMPLETE: {canonical_leaf}"
                )
            
            # Convert executor_type string to enum
            exec_type_enum = _runtime_executor_type(
                implementation_id=impl_id,
                template_id=template_id,
                declared_executor_type=exec_type,
            )

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
                context_id=version_context.context_id,
                canonical_leaf=canonical_leaf,
                implementation_id=impl_id,
                executor_type=exec_type_enum,
            )
            prior = seen_jobs.get(job_id)
            if prior is not None:
                if prior.to_dict() != candidate.to_dict():
                    raise ArtifactExpansionError(f"ARTIFACT_JOB_CONFLICT: {job_id}")
                continue
            seen_jobs[job_id] = candidate
            jobs.append(candidate)

    return jobs
