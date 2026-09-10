from __future__ import annotations

"""Deterministic expansion of atomic PromptFacts into ArtifactJobs without AI selection."""

from collections.abc import Iterable
from typing import Any

from .artifact_job import ArtifactJob
from .prompt_fact_types import FactType, PromptFact


FACT_EXPANSIONS: dict[FactType, tuple[str, ...]] = {
    FactType.ITEM_EXISTS: (
        "fabric/item/register_basic",
        "fabric/item/model_basic",
        "fabric/item/lang_en",
    ),
    FactType.ITEM_STACK_LIMIT: (
        "fabric/item/settings_max_stack",
    ),
    FactType.ITEM_DURABILITY: (
        "fabric/item/settings_durability",
    ),
    FactType.ITEM_FOOD_NUTRITION: (
        "fabric/item/settings_food",
    ),
    FactType.ITEM_FIREPROOF: (
        "fabric/item/settings_fireproof",
    ),
    FactType.BLOCK_EXISTS: (
        "fabric/block/register_basic",
        "fabric/block/blockstate_basic",
        "fabric/block/model_cube_all",
        "fabric/block/lang_en",
    ),
    FactType.BLOCK_DROP: (
        "fabric/loot/block_drop",
    ),
}


def _constant_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in value).strip("_")
    return cleaned.upper() or "DEFAULT_CONSTANT"


def _java_package_path(package_name: str) -> str:
    return package_name.replace(".", "/")


def expand_facts_to_jobs(
    facts: Iterable[PromptFact],
    *,
    mod_id: str,
    package_name: str,
) -> list[ArtifactJob]:
    """Deterministically lower PromptFacts to ArtifactJobs based on static expansion rules."""
    jobs: list[ArtifactJob] = []
    seen_job_ids: set[str] = set()

    for fact in facts:
        template_ids = FACT_EXPANSIONS.get(fact.fact_type, ())
        subject = fact.subject
        constant = _constant_name(subject)
        pkg_path = _java_package_path(package_name)

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
            }

            target_path = ""
            anchor = ""
            requires: list[str] = []
            produces: list[str] = []

            if template_id == "fabric/item/register_basic":
                target_path = f"src/main/java/{pkg_path}/registry/ModItems.java"
                anchor = "mod_items_registry"
                produces = [f"{subject}.registry_id", f"{subject}.java_symbol"]
            elif template_id == "fabric/item/settings_max_stack":
                target_path = f"src/main/java/{pkg_path}/registry/ModItems.java"
                anchor = f"item_settings_{subject}"
                deterministic_inputs["stack_limit"] = fact.value or 16
                requires = [f"{subject}.java_symbol"]
            elif template_id == "fabric/item/settings_durability":
                target_path = f"src/main/java/{pkg_path}/registry/ModItems.java"
                anchor = f"item_settings_{subject}"
                deterministic_inputs["durability"] = fact.value or 100
                requires = [f"{subject}.java_symbol"]
            elif template_id == "fabric/item/model_basic":
                target_path = f"src/main/resources/assets/{mod_id}/models/item/{subject}.json"
                requires = [f"{subject}.registry_id"]
                produces = [f"{subject}.model_ref"]
            elif template_id == "fabric/item/lang_en":
                target_path = f"src/main/resources/assets/{mod_id}/lang/en_us.json"
                deterministic_inputs["display_name"] = " ".join(part.capitalize() for part in subject.split("_"))
                requires = [f"{subject}.registry_id"]
                produces = [f"{subject}.translation_key"]
            elif template_id == "fabric/block/register_basic":
                target_path = f"src/main/java/{pkg_path}/registry/ModBlocks.java"
                anchor = "mod_blocks_registry"
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
            elif template_id == "fabric/loot/block_drop":
                target_path = f"src/main/resources/data/{mod_id}/loot_tables/blocks/{subject}.json"
                drop_item = fact.object or subject
                deterministic_inputs["drop_item"] = drop_item
                requires = [f"{subject}.block_registry_id", f"{drop_item}.registry_id"]

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
