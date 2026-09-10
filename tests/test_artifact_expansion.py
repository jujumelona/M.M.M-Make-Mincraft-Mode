from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_expansion import (
    ArtifactExpansionError,
    expand_facts_to_jobs,
    validate_expansion_catalog,
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def test_expansion_catalog_contains_only_executable_tested_leaves():
    validate_expansion_catalog()


def test_expand_item_and_explicit_stack_limit_without_defaults():
    facts = [
        PromptFact(
            fact_id="fact_001",
            fact_type=FactType.ITEM_EXISTS,
            subject="raw_lunite",
            source_clause="raw lunite item exists",
        ),
        PromptFact(
            fact_id="fact_002",
            fact_type=FactType.ITEM_STACK_LIMIT,
            subject="raw_lunite",
            value=16,
            source_clause="stack limit is 16",
        ),
    ]
    jobs = expand_facts_to_jobs(
        facts, mod_id="space", package_name="com.foo.space"
    )
    assert [job.template_id for job in jobs] == [
        "fabric/item/register_basic",
        "fabric/item/model_basic",
        "fabric/item/lang_en",
        "fabric/item/settings_max_stack",
    ]
    register = jobs[0]
    assert register.produces == (
        "raw_lunite.registry_id",
        "raw_lunite.java_symbol",
    )
    model = jobs[1]
    assert model.requires == ("raw_lunite.registry_id",)
    stack = jobs[-1]
    assert stack.requires == ("raw_lunite.java_symbol",)
    assert stack.deterministic_inputs["stack_limit"] == 16


def test_missing_numeric_fact_is_not_replaced_by_magic_default():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_FACT_VALUE_REQUIRED"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.ITEM_STACK_LIMIT,
                    subject="raw_lunite",
                    value=None,
                    source_clause="make a special stack size",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )


def test_unimplemented_fact_fails_closed_instead_of_referencing_missing_template():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_FACT_UNSUPPORTED"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.BLOCK_EXISTS,
                    subject="lunite_ore",
                    source_clause="lunite ore block exists",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )


def test_invalid_registry_subject_is_rejected_before_render():
    with pytest.raises(ArtifactExpansionError, match="ARTIFACT_SUBJECT_INVALID"):
        expand_facts_to_jobs(
            [
                PromptFact(
                    fact_id="fact_001",
                    fact_type=FactType.ITEM_EXISTS,
                    subject="Raw Lunite",
                )
            ],
            mod_id="space",
            package_name="com.foo.space",
        )
