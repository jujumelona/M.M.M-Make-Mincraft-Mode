from __future__ import annotations

import inspect

from minecraft_mod_ai import central_research, parallel_runtime_contract, retrieval
from minecraft_mod_ai.platform_live_rag_contract import install


install(retrieval_module=retrieval)


def test_targetless_official_rag_returns_real_evidence() -> None:
    receipt = retrieval.retrieve_official_evidence(
        "Fabric data generation recipes loot tags models",
        limit=4,
    )

    assert receipt.hits
    assert receipt.minecraft_version == ""
    assert receipt.loader == ""
    assert receipt.mappings == ""


def test_partial_platform_target_does_not_abort_live_rag() -> None:
    receipt = retrieval.retrieve_official_evidence(
        "Fabric automated testing GameTest runtime validation",
        minecraft_version="1.21.1",
        loader="fabric",
        mappings=None,
        limit=4,
    )

    assert receipt.hits
    assert receipt.minecraft_version == ""
    assert receipt.loader == ""
    assert receipt.mappings == ""


def test_targetless_central_graph_retrieves_instead_of_deferring() -> None:
    brief = central_research.normalize_research_brief(
        "Create a Minecraft mod with resource gathering, trading and progression.",
        {},
    )
    assert "_mmm_platform_target" not in brief

    graph = central_research.retrieve_domain_evidence(brief)

    assert graph["target"] is None
    assert graph["deferred_official_domains"] == []
    official_domains = [
        domain
        for domain in graph["domains"]
        if domain.get("strategy") != "routed_to_other_providers"
    ]
    assert official_domains
    assert any(domain.get("queries") for domain in official_domains)
    assert all(
        domain.get("strategy") == "adaptive_generic_per_query"
        for domain in official_domains
    )


def test_parallel_targetless_wrapper_preserves_public_call_domain() -> None:
    signature = inspect.signature(
        parallel_runtime_contract.retrieve_domain_evidence,
        follow_wrapped=False,
    )
    parameters = list(signature.parameters.values())

    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ]


def test_parallel_targetless_wrapper_accepts_keyword_brief() -> None:
    brief = central_research.normalize_research_brief(
        "Create a Minecraft mod with resource gathering, trading and progression.",
        {},
    )

    graph = parallel_runtime_contract.retrieve_domain_evidence(research_brief=brief)

    assert graph["target"] is None
    assert graph["deferred_official_domains"] == []
