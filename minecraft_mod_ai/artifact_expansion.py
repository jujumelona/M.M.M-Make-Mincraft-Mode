from __future__ import annotations

"""Fail-closed lowering of atomic PromptFacts into concrete ArtifactJobs."""

import re
from collections.abc import Iterable
from typing import Any

from .artifact_job import ArtifactJob
from .prompt_fact_types import FactType, PromptFact
from .task_template_catalog import load_template


class ArtifactExpansionError(ValueError):
    pass


_REGISTRY_PATH = re.compile(r"^[a-z0-9_.-]+$")
_SUPPORTED_EXPANSIONS: dict[FactType, tuple[str, ...]] = {
    FactType.ITEM_EXISTS: (
        "fabric/item/key",
        "fabric/item/register_basic",
        "fabric/item/client_item",
        "fabric/item/model_basic",
        "fabric/item/lang_en",
        "fabric/item/initializer",
    ),
    FactType.ITEM_STACK_LIMIT: (
        "fabric/item/settings_max_stack",
    ),
    FactType.BLOCK_EXISTS: (
        "fabric/block/key",
        "fabric/block/register_basic",
        "fabric/block/blockstate_basic",
        "fabric/block/model_cube_all",
        "fabric/block/lang_en",
        "fabric/block/initializer",
    ),
    FactType.BLOCK_DROP: (
        "fabric/loot/block_drop",
    ),
}

# Kept public for callers/tests, but every entry is verified before use.
FACT_EXPANSIONS = dict(_SUPPORTED_EXPANSIONS)


def _constant_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in value).strip("_")
    if not cleaned:
        raise ArtifactExpansionError("ARTIFACT_SUBJECT_EMPTY: cannot derive a Java constant")
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
            if not isinstance(fixture, dict) or not isinstance(fixture.get("input"), dict):
                raise ArtifactExpansionError(
                    f"ARTIFACT_TEMPLATE_UNTESTED: {identifier!r} has no executable fixture"
                )


def expand_facts_to_jobs(
    facts: Iterable[PromptFact],
    *,
    mod_id: str,
    package_name: str,
    main_class: str = "",
) -> list[ArtifactJob]:
    """Lower only explicitly supported facts; never invent a fallback implementation."""
    validate_expansion_catalog()
    mod_id = str(mod_id or "").strip()
    if not _REGISTRY_PATH.fullmatch(mod_id):
        raise ArtifactExpansionError(f"ARTIFACT_MOD_ID_INVALID: invalid mod id {mod_id!r}")
    pkg_path = _java_package_path(package_name)

    jobs: list[ArtifactJob] = []
    seen_job_ids: set[str] = set()
    main_class_val = main_class or "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"

    for fact in facts:
        template_ids = FACT_EXPANSIONS.get(fact.fact_type)
        if template_ids is None:
            raise ArtifactExpansionError(
                f"ARTIFACT_FACT_UNSUPPORTED: no validated leaf expansion exists for "
                f"{fact.fact_type.value}; do not silently skip or improvise it"
            )

        subject = _require_subject(fact)
        constant = _constant_name(subject)

        for template_id in template_ids:
            step_name = template_id.rsplit("/", 1)[-1]
            job_id = f"{subject}.{step_name}"
            if job_id in seen_job_ids:
                continue
            seen_job_ids.add(job_id)

            deterministic_inputs: dict[str, Any] = {
                "mod_id": mod_id,
                "package_name": package_name,
                "package_path": pkg_path,
                "registry_path": subject,
                "java_constant": constant,
                "subject": subject,
                "main_class": main_class_val,
            }
            target_path = ""
            anchor = ""
            requires: list[str] = []
            produces: list[str] = []

            if template_id == "fabric/item/key":
                target_path = f"src/main/java/{pkg_path}/registry/ModItemIds.java"
                anchor = "/* MMM:item_keys */"
                produces = [f"{subject}.key_symbol"]
            elif template_id == "fabric/item/register_basic":
                target_path = f"src/main/java/{pkg_path}/registry/ModItems.java"
                anchor = "/* MMM:item_registry */"
                requires = [f"{subject}.key_symbol"]
                produces = [f"{subject}.registry_id", f"{subject}.java_symbol"]
            elif template_id == "fabric/item/settings_max_stack":
                target_path = f"src/main/java/{pkg_path}/registry/ModItems.java"
                anchor = f"/* MMM:properties:{subject} */"
                deterministic_inputs["stack_limit"] = _require_integer_value(
                    fact, minimum=1, maximum=64
                )
                requires = [f"{subject}.java_symbol"]
            elif template_id == "fabric/item/client_item":
                target_path = f"src/main/resources/assets/{mod_id}/items/{subject}.json"
                requires = [f"{subject}.registry_id"]
                produces = [f"{subject}.client_item_ref"]
            elif template_id == "fabric/item/model_basic":
                target_path = (
                    f"src/main/resources/assets/{mod_id}/models/item/{subject}.json"
                )
                requires = [f"{subject}.registry_id"]
                produces = [f"{subject}.model_ref"]
            elif template_id == "fabric/item/lang_en":
                target_path = f"src/main/resources/assets/{mod_id}/lang/en_us.json"
                deterministic_inputs["display_name"] = " ".join(
                    part.capitalize() for part in subject.split("_")
                )
                requires = [f"{subject}.registry_id"]
                produces = [f"{subject}.translation_key"]
            elif template_id == "fabric/item/initializer":
                target_path = f"src/main/java/{pkg_path}/{main_class_val}.java"
                anchor = "/* MMM:init */"
                requires = [f"{subject}.java_symbol"]
            elif template_id == "fabric/block/key":
                target_path = f"src/main/java/{pkg_path}/registry/ModBlockIds.java"
                anchor = "/* MMM:block_keys */"
                produces = [f"{subject}.block_key_symbol"]
            elif template_id == "fabric/block/register_basic":
                target_path = f"src/main/java/{pkg_path}/registry/ModBlocks.java"
                anchor = "/* MMM:block_registry */"
                requires = [f"{subject}.block_key_symbol"]
                produces = [f"{subject}.block_registry_id", f"{subject}.block_symbol"]
            elif template_id == "fabric/block/blockstate_basic":
                target_path = f"src/main/resources/assets/{mod_id}/blockstates/{subject}.json"
                requires = [f"{subject}.block_registry_id"]
            elif template_id == "fabric/block/model_cube_all":
                target_path = f"src/main/resources/assets/{mod_id}/models/block/{subject}.json"
                requires = [f"{subject}.block_registry_id"]
            elif template_id == "fabric/block/lang_en":
                target_path = f"src/main/resources/assets/{mod_id}/lang/en_us.json"
                deterministic_inputs["display_name"] = " ".join(part.capitalize() for part in subject.split("_"))
                requires = [f"{subject}.block_registry_id"]
            elif template_id == "fabric/block/initializer":
                target_path = f"src/main/java/{pkg_path}/{main_class_val}.java"
                anchor = "/* MMM:init */"
                requires = [f"{subject}.block_symbol"]
            elif template_id == "fabric/loot/block_drop":
                target_path = f"src/main/resources/data/{mod_id}/loot_tables/blocks/{subject}.json"
                drop_item = fact.object or subject
                deterministic_inputs["drop_item"] = drop_item
                requires = [f"{subject}.block_registry_id", f"{drop_item}.registry_id"]
            else:
                raise ArtifactExpansionError(
                    f"ARTIFACT_EXPANSION_INTERNAL: unhandled validated leaf {template_id!r}"
                )

            jobs.append(
                ArtifactJob(
                    job_id=job_id,
                    template_id=template_id,
                    owner_module=subject,
                    target_path=target_path,
                    anchor=anchor,
                    requires=tuple(requires),
                    produces=tuple(produces),
                    deterministic_inputs=deterministic_inputs,
                )
            )

    return jobs
