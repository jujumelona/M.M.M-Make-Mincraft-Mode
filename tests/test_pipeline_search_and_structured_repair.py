from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import platform_optimizer
from minecraft_mod_ai.pipeline_hardening_v2 import _search_variants
from minecraft_mod_ai.pipeline_hardening_v4 import (
    _strict_provenance_repair,
    bounded_seed_query,
)


def test_generated_task_name_gets_bounded_semantic_search_variant() -> None:
    variants = _search_variants("TaskAlienPlanetInteractionSemanticImplementation")
    assert variants[0] == "TaskAlienPlanetInteractionSemanticImplementation"
    assert len(variants) == 2
    assert "Task" not in variants[1]
    assert "Semantic" not in variants[1]
    assert "Implementation" not in variants[1]
    assert "Alien" in variants[1]
    assert "Planet" in variants[1]


def test_mod_search_is_broad_then_exact_target_ranked() -> None:
    class Client:
        def __init__(self) -> None:
            self.search_calls = []
            self.inspect_calls = []

        def search(self, source, query, **kwargs):
            self.search_calls.append((source, query, dict(kwargs)))
            version = kwargs.get("minecraft_version")
            if version == "1.20.1":
                candidates = []
            elif version == "1.21.1":
                candidates = [{"candidate_id": "modrinth:compatible"}]
            else:
                candidates = [
                    {"candidate_id": "modrinth:compatible"},
                    {"candidate_id": "modrinth:other"},
                ]
            return {"candidates": candidates}

        def inspect_modrinth_project(self, project_id, *, minecraft_version, loader):
            self.inspect_calls.append((project_id, minecraft_version, loader))
            eligible = project_id == "compatible" and minecraft_version == "1.21.1"
            return {
                "license_policy": "permissive_candidate:mit",
                "versions": [
                    {
                        "eligible_for_selection": eligible,
                        "files": [{"sha512": "abc"}],
                        "dependencies": [],
                    }
                ],
            }

    client = Client()
    queries = ("alien planet",)
    found, errors = platform_optimizer._parallel_neutral_shallow(queries, client)
    assert not errors
    assert found["alien planet"]

    broad_calls = list(client.search_calls)
    assert broad_calls
    for _source, _query, kwargs in broad_calls:
        assert "minecraft_version" not in kwargs
        assert "loader" not in kwargs
        assert kwargs["target_profile"] == "minecraft_mod"

    probes = (
        SimpleNamespace(
            adapter_id="probe:fabric:1.21.1",
            minecraft_version="1.21.1",
            loader="fabric",
        ),
        SimpleNamespace(
            adapter_id="probe:fabric:1.20.1",
            minecraft_version="1.20.1",
            loader="fabric",
        ),
    )
    matrix, matrix_errors = platform_optimizer._parallel_support_matrix(
        probes,
        queries,
        client,
    )
    assert not matrix_errors
    assert matrix["probe:fabric:1.21.1"]["alien planet"] == (
        "modrinth:compatible",
    )
    assert matrix["probe:fabric:1.20.1"]["alien planet"] == ()

    exact_calls = client.search_calls[len(broad_calls):]
    assert exact_calls
    assert {call[2].get("minecraft_version") for call in exact_calls} == {
        "1.21.1",
        "1.20.1",
    }
    assert all(call[2].get("loader") == "fabric" for call in exact_calls)

    # Deep project/version inspection is deliberately deferred until after target
    # selection, avoiding target x project Cartesian API explosions.
    assert client.inspect_calls == []


def test_mod_search_transport_failure_is_not_treated_as_no_mod_support() -> None:
    class BrokenClient:
        def search(self, source, query, **kwargs):
            raise TimeoutError("modrinth unavailable")

    client = BrokenClient()
    queries = ("worldgen",)
    found, errors = platform_optimizer._parallel_neutral_shallow(queries, client)
    assert found["worldgen"] == ()
    assert errors

    probe = SimpleNamespace(
        adapter_id="probe:fabric:1.21.1",
        minecraft_version="1.21.1",
        loader="fabric",
    )
    with pytest.raises(ValueError, match="source unavailable"):
        platform_optimizer._parallel_support_matrix((probe,), queries, client)


def test_seed_query_stays_bounded_and_keeps_domain_terms() -> None:
    prompt = (
        "Please create Minecraft Fabric generated implementation planning semantic task "
        "for an AlienPlanetInteraction system with oxygen gravity radiation survival "
        + ("generic planning implementation module " * 200)
    )
    design = {
        "title": "Alien Planet Survival",
        "pitch": "Explore toxic planets with oxygen and gravity hazards.",
        "modules": [
            {
                "name": "OxygenGravityController",
                "kind": "survival_system",
                "reason": "oxygen gravity radiation atmosphere",
            }
        ],
    }
    query = bounded_seed_query(prompt, design)

    assert 0 < len(query) <= 320
    lowered = query.casefold()
    assert "alien" in lowered
    assert "planet" in lowered
    assert "oxygen" in lowered
    assert "gravity" in lowered
    assert "implementation" not in lowered
    assert "generated" not in lowered


def test_provenance_filter_never_invents_missing_evidence_refs() -> None:
    note = {
        "claims": [
            {"claim": "grounded", "evidence_refs": ["page:1"]},
            {"claim": "uncited", "evidence_refs": []},
            {"claim": "foreign", "evidence_refs": ["page:999"]},
        ],
        "gaps": [],
        "procedures": [],
        "sufficient": True,
    }
    repaired = _strict_provenance_repair(
        note,
        allowed_refs=("page:1", "page:2"),
    )

    assert repaired["claims"] == [
        {"claim": "grounded", "evidence_refs": ["page:1"]}
    ]
    assert all(
        claim["evidence_refs"]
        for claim in repaired["claims"]
    )
    assert any("omitted" in gap for gap in repaired["gaps"])


def test_machine_pack_metadata_preserves_three_value_contract(monkeypatch) -> None:
    from minecraft_mod_ai import platform_live_discovery as live

    monkeypatch.setattr(
        live,
        "_mojang_pack_versions",
        lambda version: ("61", "46"),
    )
    monkeypatch.setattr(
        live,
        "_mojang_target_url",
        lambda version: f"https://piston-meta.mojang.com/{version}.json",
    )

    data_pack, resource_pack, source_url = live._official_pack_versions("1.21.1")
    assert data_pack == "61"
    assert resource_pack == "46"
    assert source_url.endswith("/1.21.1.json")


def test_lossless_page_research_has_single_canonical_owner() -> None:
    from minecraft_mod_ai import pre_design_domain_research as domain_research

    assert callable(domain_research.research_document_domain)
    assert domain_research.research_document_domain.__module__ == (
        "minecraft_mod_ai.pre_design_domain_research"
    )
