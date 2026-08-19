from __future__ import annotations

from minecraft_mod_ai.hybrid_route_hardening import (
    classify_code_evidence_need,
    portfolio_policy,
    route_query,
)


def test_fine_grained_code_route_classification() -> None:
    assert classify_code_evidence_need("JDT cannot find symbol PacketCodec") == "trace"
    assert classify_code_evidence_need("rename Foo and find all callers") == "ripple"
    assert classify_code_evidence_need("which registry callback API should be used") == "api"
    assert classify_code_evidence_need("implement save sync validate flow") == "procedural"
    assert classify_code_evidence_need("Minecraft 1.21.1 Fabric version mapping") == "exact_version"
    assert classify_code_evidence_need("what depends on RegistryBootstrap") == "dependency"


def test_trace_and_ripple_are_steered_to_relation_portfolio() -> None:
    trace = route_query("cannot resolve PacketCodec", "trace")
    ripple = route_query("rename oldMethod", "ripple")
    assert "dependency call chain" in trace
    assert "imports callers callees" in trace
    assert "all references affected usages" in ripple
    assert portfolio_policy("trace")["family_order"][0] == "lexical+relations"
    assert portfolio_policy("ripple")["family_order"][0] == "lexical+relations"


def test_api_and_procedural_routes_keep_retrieval_non_authoritative() -> None:
    api = portfolio_policy("api")
    procedural = portfolio_policy("procedural")
    assert api["family_order"][0] == "exact-symbol-lexical"
    assert "semantic+rerank" in procedural["family_order"]
    assert api["generic_similar_code_authoritative"] is False
    assert procedural["generic_similar_code_authoritative"] is False
    assert api["current_exact_source_authoritative"] is True


def test_runtime_installs_fine_grained_search_after_hybrid_engine() -> None:
    from minecraft_mod_ai.production_tools import ProductionToolService

    assert getattr(
        ProductionToolService.search_code_rag,
        "__mmm_research_fine_grained_code_route_v1__",
        False,
    )
