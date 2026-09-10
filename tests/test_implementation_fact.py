from __future__ import annotations

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.implementation_fact import (
    FactProvenance,
    ImplementationFact,
    prompt_fact_to_implementation_fact,
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def test_implementation_fact_creation_and_serialization():
    fact = ImplementationFact(
        fact_id="fact_lunarite_01",
        fact_type=FactType.ITEM_EXISTS,
        subject="lunarite_ingot",
        display_name="Lunarite Ingot",
        provenance=FactProvenance.DESIGN,
        parent_requirement="req_space_metals",
        evidence_refs=("research:space_ores",),
        source_clause="Add space metals",
    )
    assert fact.fact_id == "fact_lunarite_01"
    assert fact.provenance == FactProvenance.DESIGN
    assert fact.display_name == "Lunarite Ingot"

    data = fact.to_dict()
    assert data["fact_id"] == "fact_lunarite_01"
    assert data["fact_type"] == "ITEM_EXISTS"
    assert data["provenance"] == "design"
    assert data["display_name"] == "Lunarite Ingot"

    restored = ImplementationFact.from_dict(data)
    assert restored.fact_id == fact.fact_id
    assert restored.fact_type == fact.fact_type
    assert restored.provenance == FactProvenance.DESIGN
    assert restored.display_name == fact.display_name
    assert restored.evidence_refs == ("research:space_ores",)


def test_prompt_fact_lifting():
    p_fact = PromptFact(
        fact_id="p_fact_1",
        fact_type=FactType.ITEM_STACK_LIMIT,
        subject="starlight_bottle",
        value=16,
        source_clause="Starlight bottles stack up to 16",
    )
    impl_fact = prompt_fact_to_implementation_fact(p_fact)
    assert impl_fact.fact_id == "p_fact_1"
    assert impl_fact.fact_type == FactType.ITEM_STACK_LIMIT
    assert impl_fact.subject == "starlight_bottle"
    assert impl_fact.value == 16
    assert impl_fact.provenance == FactProvenance.PROMPT


def test_expand_facts_to_jobs_with_implementation_facts():
    facts = [
        ImplementationFact(
            fact_id="fact_item_1",
            fact_type=FactType.ITEM_EXISTS,
            subject="lunarite_crystal",
            display_name="루나이트 결정",
            provenance=FactProvenance.RESEARCH,
            parent_requirement="req_lunar_crystal",
        ),
        ImplementationFact(
            fact_id="fact_item_2",
            fact_type=FactType.ITEM_STACK_LIMIT,
            subject="lunarite_crystal",
            value=32,
            provenance=FactProvenance.DESIGN,
        ),
    ]

    jobs = expand_facts_to_jobs(
        facts,
        mod_id="cosmic",
        package_name="com.example.cosmic",
        main_class="CosmicMod",
    )

    assert len(jobs) == 7
    # Verify display name is retained in lang_en job inputs
    lang_job = next(j for j in jobs if j.template_id == "fabric/item/lang_en")
    assert lang_job.deterministic_inputs["display_name"] == "루나이트 결정"
