from __future__ import annotations


def test_unauthenticated_external_query_budget_is_not_eight(monkeypatch):
    from minecraft_mod_ai import pre_design_external_source_contract as external

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("MMM_PREDESIGN_EXTERNAL_SOURCE_QUERIES", raising=False)
    assert external._max_queries() == 20


def test_all_approved_requirement_queries_are_attempted(monkeypatch):
    from minecraft_mod_ai import pre_design_external_source_contract as external

    calls = []
    monkeypatch.setattr(
        external,
        "_retrieve_github_source_body",
        lambda query: calls.append(query)
        or {
            "records": [],
            "search_requests": 1,
            "source_requests": 0,
            "provider_status": "available",
            "saturation_reason": "test",
            "errors": [],
        },
    )
    queries = [f"minecraft mod requirement {i} source" for i in range(10)]
    payload = {
        "schema_version": "mmm/corrective-retrieval-request-v1",
        "domains": [{"domain_id": "request", "queries": queries}],
    }
    bundle = {
        "domains": [
            {
                "domain_id": "request",
                "queries": [
                    {"query": q, "external_rag": {"records": []}} for q in queries
                ],
            }
        ]
    }
    external._augment_bundle(payload, bundle)
    assert calls == queries


def test_generic_github_repository_is_rejected_before_body():
    from minecraft_mod_ai import pre_design_external_source_contract as external

    assert not external._repository_candidate_relevant(
        "minecraft mod build space station modules",
        {
            "full_name": "microsoft/StudentsAtBuild",
            "description": "Minecraft student learning path",
            "topics": [],
        },
    )
    assert external._repository_candidate_relevant(
        "minecraft mod space rocket vehicle",
        {
            "full_name": "Advanced-Rocketry/AdvancedRocketry",
            "description": "Advanced Rocketry Minecraft space mod",
            "topics": ["minecraft", "mod"],
        },
    )
    assert not external._body_relevant(
        "minecraft mod build space station modules",
        "Student Zone learning path for Minecraft mod build tutorials.",
    )
    assert external._body_relevant(
        "minecraft mod space rocket vehicle",
        "Advanced Rocketry is a Minecraft mod for building rocket vehicles and travelling through space.",
    )


def test_corrective_queries_are_host_owned_and_do_not_call_model():
    from minecraft_mod_ai import pre_design_rag_corrective as corrective

    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model/project helper called: {name}")

    seen = {"minecraft alien mob entity mod"}
    result = corrective._generate_gap_queries(
        NeverModel(),
        NeverModel(),
        NeverModel(),
        domain={
            "domain_id": "request",
            "queries": [
                "minecraft alien mob entity mod",
                "minecraft colony settlement building mod",
            ],
        },
        gaps=["colonization evidence missing"],
        prior_queries=["minecraft alien mob entity mod"],
        seen=seen,
        raw_prompt="우주 식민지화 모드",
        progress_label="test",
    )
    assert result == ["minecraft colony settlement building mod"]
