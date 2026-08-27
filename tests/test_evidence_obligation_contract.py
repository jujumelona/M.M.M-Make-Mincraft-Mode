from __future__ import annotations

from dataclasses import replace

from minecraft_mod_ai import evidence_obligation_contract as obligation
from minecraft_mod_ai.retrieval import RetrievalHit, RetrievalReceipt


def _catalog():
    return {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "catalog_sha256": "sha256:" + "1" * 64,
        "requirements": [
            {
                "requirement_id": "req_trade_123",
                "capability": "economy.trade",
                "semantic_statement": "Players trade gathered resources.",
            }
        ],
    }


def _receipt(query: str, *, document_id: str, title: str, excerpt: str, quality="strong"):
    hit = RetrievalHit(
        evidence_id="sha256:" + "2" * 64,
        document_id=document_id,
        title=title,
        url="https://docs.fabricmc.net/example",
        excerpt=excerpt,
        content_sha256="sha256:" + "3" * 64,
        revision="test",
        minecraft_versions=("*",),
        score=1.0,
        channels=("lexical",),
    )
    return RetrievalReceipt(
        schema_version="minecraft-mod-ai/retrieval-receipt-v1",
        query=query,
        canonical_query=query,
        query_family="project",
        minecraft_version="26.2",
        loader="fabric",
        mappings="mojang",
        query_hash="sha256:" + "4" * 64,
        corpus_snapshot_hash="sha256:" + "5" * 64,
        quality=quality,
        coverage=1.0,
        correction_required=False,
        correction_queries=(),
        hits=(hit,),
    )


def test_requirement_expands_to_independent_evidence_obligations():
    brief = obligation.build_evidence_obligation_brief("trade request", _catalog())
    dag = brief["evidence_obligation_dag"]
    nodes = dag["nodes"]

    assert len(nodes) == 7
    assert {item["kind"] for item in nodes} == {
        "reusable_implementation",
        "target_compatibility",
        "implementation_api",
        "dependency_closure",
        "license_provenance",
        "validation_mechanism",
        "asset_requirement",
    }
    asset = next(item for item in nodes if item["kind"] == "asset_requirement")
    assert asset["status"] == "PENDING_DESIGN_RESOLUTION"
    assert asset["retrieval_required"] is False


def test_atomic_domains_use_exactly_one_query_and_never_legacy_mixed_suffix():
    brief = obligation.build_evidence_obligation_brief("trade request", _catalog())

    assert len(brief["domains"]) == 6
    for domain in brief["domains"]:
        assert len(domain["queries"]) == 1
        assert (
            "minecraft java mod implementation dependencies assets license tests"
            not in domain["queries"][0].casefold()
        )
    assert brief["origin"] == "approved_requirement_graph"


def test_obligation_dependencies_form_external_dag():
    brief = obligation.build_evidence_obligation_brief("trade request", _catalog())
    nodes = {item["kind"]: item for item in brief["evidence_obligation_dag"]["nodes"]}

    assert nodes["implementation_api"]["depends_on"] == [
        nodes["target_compatibility"]["obligation_id"]
    ]
    assert nodes["dependency_closure"]["depends_on"] == [
        nodes["target_compatibility"]["obligation_id"]
    ]
    assert nodes["validation_mechanism"]["depends_on"] == [
        nodes["implementation_api"]["obligation_id"]
    ]


def test_generic_unrelated_hit_cannot_satisfy_implementation_api_obligation():
    meta = {
        "kind": "implementation_api",
        "capability": "economy.trade",
        "anchors": ["economy", "trade"],
    }
    unrelated = _receipt(
        "q",
        document_id="mcp-specification",
        title="Model Context Protocol Specification",
        excerpt="Tools resources prompts transports and security.",
    )

    assert obligation._obligation_satisfied(meta, unrelated) is False


def test_target_compatibility_requires_target_relevant_primary_source():
    meta = {"kind": "target_compatibility", "capability": "economy.trade", "anchors": []}
    wrong = _receipt(
        "q",
        document_id="fabric-automatic-testing",
        title="Fabric Automated Testing",
        excerpt="GameTest runtime validation.",
    )
    right = replace(
        wrong,
        hits=(
            replace(
                wrong.hits[0],
                document_id="fabric-project-creation",
                title="Fabric Project Creation",
                excerpt="Loader mappings and project structure.",
            ),
        ),
    )

    assert obligation._obligation_satisfied(meta, wrong) is False
    assert obligation._obligation_satisfied(meta, right) is True


def test_corrective_query_inherits_same_obligation_evaluator_metadata():
    query = "economy.trade Minecraft Fabric API Players trade gathered resources."
    meta = {
        "kind": "implementation_api",
        "capability": "economy.trade",
        "anchors": ["economy", "trade"],
    }
    obligation._QUERY_META[query] = meta
    unrelated = _receipt(
        query,
        document_id="mcp-specification",
        title="MCP",
        excerpt="tools resources prompts",
    )

    result = obligation._strict_retrieve(lambda _self, _query, **_kwargs: unrelated, object(), query)

    assert result.quality == "weak"
    assert result.coverage == 0.0
    assert result.correction_required is True
    assert len(result.correction_queries) == 1
    assert obligation._QUERY_META[result.correction_queries[0]]["kind"] == "implementation_api"


def test_coverage_is_fulfilled_obligations_over_required_obligations():
    q1, q2 = "q1", "q2"
    criteria = {q1: ("obligation:o1",), q2: ("obligation:o2",)}
    domains = {"d1": ["obligation:o1"], "d2": ["obligation:o2"]}
    graph = {
        "domains": [
            {
                "domain_id": "d1",
                "queries": [
                    {
                        "primary": {
                            "query": q1,
                            "quality": "strong",
                            "coverage": 1.0,
                            "hits": [{"id": "hit"}],
                        },
                        "corrections": [],
                    }
                ],
            },
            {
                "domain_id": "d2",
                "queries": [
                    {
                        "primary": {
                            "query": q2,
                            "quality": "weak",
                            "coverage": 0.0,
                            "hits": [{"id": "hit"}],
                        },
                        "corrections": [],
                    }
                ],
            },
        ]
    }

    result = obligation._attach_coverage(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback used")),
        graph,
        query_criteria=criteria,
        domain_criteria=domains,
    )

    assert result["coverage"]["fulfilled_obligations"] == 1
    assert result["coverage"]["required_obligations"] == 2
    assert result["coverage"]["ratio"] == 0.5
    assert result["coverage"]["complete"] is False
