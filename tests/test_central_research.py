from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.central_research import (
    external_discovery_routes,
    normalize_research_brief,
    retrieve_domain_evidence,
)
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.retrieval import RetrievalHit, RetrievalReceipt
from minecraft_mod_ai.spec import SpecValidationError


def _domain(
    domain_id: str,
    *,
    query: str | None = None,
    providers: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "objective": f"Research {domain_id} without assuming a genre template.",
        "requirements": [f"Preserve the requested {domain_id} capability."],
        "evidence_kinds": ["dependency", "compatibility", "license"],
        "queries": [query or f"{domain_id} implementation evidence"],
        "providers": providers or ["official_docs", "github"],
        "depends_on": depends_on or [],
    }


def _candidate(domains: list[dict[str, object]]) -> dict[str, object]:
    return {
        "summary": "A generic request-derived research DAG.",
        "domains": domains,
        "unresolved_questions": [],
    }


def _selected_design(version: str = "1.20.1", loader: str = "fabric") -> dict[str, object]:
    return {
        "_platform_selection": {
            "target": {
                "minecraft_version": version,
                "loader": loader,
            }
        }
    }


def test_normalize_research_brief_accepts_a_generic_planner_dag() -> None:
    candidate = _candidate(
        [
            _domain("request"),
            _domain(
                "simulation",
                providers=["official_docs", "project_rag", "modrinth"],
                depends_on=["request"],
            ),
            _domain(
                "presentation",
                providers=["github", "openverse_images", "openverse_audio"],
                depends_on=["simulation"],
            ),
        ]
    )

    normalized = normalize_research_brief(
        "Build the requested simulation.",
        {"title": "Generic Simulation"},
        candidate,
    )

    assert normalized["schema_version"] == "mmm/central-research-brief-v1"
    assert normalized["origin"] == "planner_classification"
    assert normalized["brief_sha256"].startswith("sha256:")
    assert [item["domain_id"] for item in normalized["domains"]] == [
        "request",
        "simulation",
        "presentation",
    ]
    assert normalized["domains"][2]["depends_on"] == ["simulation"]
    assert "_mmm_platform_target" not in normalized


def test_normalize_research_brief_preserves_host_selected_target() -> None:
    normalized = normalize_research_brief(
        "Build the requested simulation.",
        _selected_design("1.21.1", "fabric"),
        _candidate([_domain("request")]),
    )
    assert normalized["_mmm_platform_target"] == {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
    }


def test_normalize_research_brief_rejects_cycles_and_unknown_providers() -> None:
    cyclic = _candidate(
        [
            _domain("first", depends_on=["second"]),
            _domain("second", depends_on=["first"]),
        ]
    )
    with pytest.raises(SpecValidationError, match="cycle"):
        normalize_research_brief("generic request", {}, cyclic)

    unknown_provider = deepcopy(cyclic)
    unknown_provider["domains"][0]["depends_on"] = []
    unknown_provider["domains"][1]["depends_on"] = ["first"]
    unknown_provider["domains"][1]["providers"] = ["unreviewed_catalog"]
    with pytest.raises(SpecValidationError, match="unknown providers"):
        normalize_research_brief("generic request", {}, unknown_provider)


@pytest.mark.parametrize(
    ("prompt", "systems"),
    [
        (
            "Build a competitive handball league with passing, scoring, and seasons.",
            ("passing", "scoring", "seasons"),
        ),
        (
            "Build social deduction horror with trust voting, radio whispers, and hiding.",
            ("trust voting", "radio whispers", "hiding"),
        ),
        (
            "Build a cozy farming game with planting, watering, harvesting, and a market.",
            ("planting", "watering", "harvesting", "market"),
        ),
    ],
)
def test_fallback_is_request_derived_without_genre_content_injection(
    prompt: str,
    systems: tuple[str, ...],
) -> None:
    design = {
        "core_loop": list(systems[:2]),
        "progression": [systems[2]],
        "combat": {},
        "world": {},
        "modules": [
            {
                "plugin_id": "custom",
                "reason": f"Implement {systems[-1]} exactly as requested.",
            }
        ],
        "assets": [],
        "acceptance_tests": [f"Players can complete {systems[0]}."],
    }

    normalized = normalize_research_brief(prompt, design)
    serialized = " ".join(
        requirement
        for domain in normalized["domains"]
        for requirement in domain["requirements"]
    ).casefold()

    assert normalized["origin"] == "deterministic_fallback"
    assert prompt.casefold() in serialized
    assert all(system.casefold() in serialized for system in systems)
    assert not {"boss", "arena", "village", "dungeon"} & set(
        serialized.replace(".", "").replace(",", "").split()
    )


def test_simple_decorative_item_does_not_invent_audio_research_or_quality() -> None:
    prompt = "Add one decorative copper lantern item."
    design = {
        "title": "Copper lantern",
        "pitch": "A decorative inventory item.",
        "core_loop": ["Craft and display the lantern."],
        "progression": [],
        "combat": {},
        "mod_context": {},
        "modules": [],
        "assets": [
            {
                "id": "copper_lantern",
                "kind": "item",
                "brief": "Original copper lantern item texture.",
            }
        ],
        "acceptance_tests": ["The lantern item renders in the inventory."],
    }

    brief = normalize_research_brief(prompt, design)
    providers = {
        provider
        for domain in brief["domains"]
        for provider in domain["providers"]
    }
    evidence_kinds = {
        evidence_kind
        for domain in brief["domains"]
        for evidence_kind in domain["evidence_kinds"]
    }

    assert "requested_audio" not in {
        domain["domain_id"] for domain in brief["domains"]
    }
    assert "openverse_audio" not in providers
    assert "audio" not in evidence_kinds
    assert "openverse_images" in providers

    compiled = compile_production_contract(
        requested_prompt=prompt,
        game_design=design,
        research_brief=brief,
        modules=[
            {
                "module_id": "copper_lantern",
                "kind": "item",
                "config": {},
                "depends_on": [],
                "required_gates": [],
            }
        ],
        acceptance_tests=[],
    )
    dimensions = {
        item["dimension_id"]
        for item in compiled.contract["quality_dimension_catalog"]
    }
    assert "audio" not in dimensions


@pytest.mark.parametrize(
    "prompt",
    [
        "Add an original bell sound effect when the item is used.",
        "Add original background music for the requested feature.",
        "Add an original voice for the companion.",
    ],
)
def test_explicit_sound_music_and_voice_keep_audio_research(prompt: str) -> None:
    brief = normalize_research_brief(prompt, {})
    audio_domain = next(
        domain
        for domain in brief["domains"]
        if domain["domain_id"] == "requested_audio"
    )

    assert "audio" in audio_domain["evidence_kinds"]
    assert "openverse_audio" in audio_domain["providers"]


def test_generic_request_does_not_invent_visual_media_provider() -> None:
    brief = normalize_research_brief(
        "Add a server command that reports the current tick count.",
        {"assets": []},
    )

    assert all(
        "openverse_images" not in domain["providers"]
        and "visual_reference" not in domain["evidence_kinds"]
        for domain in brief["domains"]
    )


def test_fallback_decomposes_explicit_ai_voice_research_only_when_requested() -> None:
    voice = normalize_research_brief(
        "NPC가 마이크 음성을 알아듣고 동의한 내 목소리를 LoRA로 적응해 말하게 해줘.",
        {},
    )
    technology = next(
        domain
        for domain in voice["domains"]
        if domain["domain_id"] == "requested_technology"
    )

    assert {
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
        "latency_budget",
    } <= set(technology["evidence_kinds"])
    assert "huggingface_models" in technology["providers"]
    assert "openalex_works" in technology["providers"]
    assert "crossref_works" in technology["providers"]

    ordinary = normalize_research_brief(
        "계절 농사와 요리를 추가해줘.",
        {},
    )
    assert all(
        domain["domain_id"] != "requested_technology"
        for domain in ordinary["domains"]
    )


def test_korean_voice_adaptation_wording_gets_model_and_consent_routes() -> None:
    brief = normalize_research_brief(
        "사용자가 소유하며 명시적으로 동의한 목소리는 선택적으로 적응한다.",
        {},
    )
    technology = next(
        domain
        for domain in brief["domains"]
        if domain["domain_id"] == "requested_technology"
    )

    assert {
        "voice_adaptation",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
    } <= set(technology["evidence_kinds"])
    assert "huggingface_models" in technology["providers"]


def test_external_discovery_routes_emits_every_distinct_route_without_top_n() -> None:
    providers = [
        "modrinth",
        "github",
        "openverse_images",
        "openverse_audio",
    ]
    domains = [
        _domain(
            f"domain_{index:03d}",
            query=f"distinct external query {index:03d}",
            providers=providers,
            depends_on=[] if index == 0 else ["domain_000"],
        )
        for index in range(137)
    ]
    brief = normalize_research_brief("Research every requested domain.", {}, _candidate(domains))

    routes = external_discovery_routes(brief)

    expected = {
        (f"domain_{index:03d}", provider, f"distinct external query {index:03d}")
        for index in range(137)
        for provider in providers
    }
    actual = {
        (route["domain_id"], route["provider"], route["query"])
        for route in routes
    }
    assert len(routes) == 137 * len(providers)
    assert actual == expected


def _receipt(
    query: str,
    *,
    correction_queries: tuple[str, ...] = (),
) -> RetrievalReceipt:
    hit = RetrievalHit(
        evidence_id="sha256:" + "1" * 64,
        document_id="fabric-api-1201",
        title="Fabric API 1.20.1",
        url="https://maven.fabricmc.net/",
        excerpt=f"Evidence for {query}",
        content_sha256="sha256:" + "2" * 64,
        revision="fabric-api-0.92.11+1.20.1",
        minecraft_versions=("1.20.1",),
        score=1.0,
        channels=("test",),
    )
    return RetrievalReceipt(
        schema_version="minecraft-mod-ai/retrieval-receipt-v1",
        query=query,
        canonical_query=query,
        query_family="project",
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="yarn-1.20.1+build.1",
        query_hash="sha256:" + "3" * 64,
        corpus_snapshot_hash="sha256:" + "4" * 64,
        quality="strong",
        coverage=1.0,
        correction_required=bool(correction_queries),
        correction_queries=correction_queries,
        hits=(hit,),
    )


def test_targetless_official_research_is_deferred_without_retrieval() -> None:
    brief = normalize_research_brief(
        "Research all routed facts.",
        {},
        _candidate([_domain("official_one", providers=["official_docs"])]),
    )
    calls: list[str] = []

    def fake_retrieve(query: str, **_kwargs: object) -> RetrievalReceipt:
        calls.append(query)
        return _receipt(query)

    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)

    assert calls == []
    assert evidence["target"] is None
    assert evidence["deferred_official_domains"] == ["official_one"]
    assert evidence["unresolved_official_domains"] == []
    assert evidence["domains"][0]["strategy"] == "deferred_until_platform_selected"


def test_retrieve_domain_evidence_calls_every_query_and_every_correction() -> None:
    official_queries = (
        "official query alpha",
        "official query beta",
        "official query gamma",
    )
    brief = normalize_research_brief(
        "Research all routed facts.",
        _selected_design(),
        _candidate(
            [
                {
                    **_domain("official_one", providers=["official_docs"]),
                    "queries": list(official_queries[:2]),
                },
                {
                    **_domain("official_two", providers=["official_docs", "github"]),
                    "queries": [official_queries[2]],
                    "depends_on": ["official_one"],
                },
                {
                    **_domain(
                        "external_only",
                        query="must not use official retrieval",
                        providers=["github"],
                        depends_on=["official_one"],
                    ),
                    "evidence_kinds": ["source_code"],
                },
            ]
        ),
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_retrieve(query: str, **kwargs: object) -> RetrievalReceipt:
        calls.append((query, kwargs))
        if query in official_queries:
            return _receipt(
                query,
                correction_queries=(
                    f"{query} correction one",
                    f"{query} correction two",
                ),
            )
        return _receipt(query)

    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)

    expected_queries: list[str] = []
    for query in official_queries:
        expected_queries.extend(
            [
                query,
                f"{query} correction one",
                f"{query} correction two",
            ]
        )
    assert [query for query, _ in calls] == expected_queries
    assert all(
        kwargs["minecraft_version"] == "1.20.1"
        and kwargs["loader"] == "fabric"
        and kwargs["mappings"] == "yarn-1.20.1+build.1"
        for _, kwargs in calls
    )
    assert [kwargs["limit"] for _, kwargs in calls] == [8, 4, 4] * 3
    assert evidence["target"] == {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mappings": "yarn-1.20.1+build.1",
    }
    assert evidence["deferred_official_domains"] == []
    assert evidence["unresolved_official_domains"] == []
    assert evidence["retrieval_is_authority"] is False


def test_selected_target_drives_rag_coordinates_without_model_choice() -> None:
    brief = normalize_research_brief(
        "Research exact target evidence.",
        _selected_design("1.21.1", "fabric"),
        _candidate([_domain("official_one", providers=["official_docs"])]),
    )
    calls: list[dict[str, object]] = []

    def fake_retrieve(query: str, **kwargs: object) -> RetrievalReceipt:
        calls.append(kwargs)
        return _receipt(query)

    evidence = retrieve_domain_evidence(brief, retrieve=fake_retrieve)

    assert calls == [
        {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "1.21.1+build.3",
            "limit": 8,
        }
    ]
    assert evidence["target"] == {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
