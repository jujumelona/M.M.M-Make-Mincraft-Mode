from __future__ import annotations

import threading

from minecraft_mod_ai import central_research, ecosystem_discovery
from minecraft_mod_ai import parallel_runtime_contract as parallel


def test_discovery_prefetch_preserves_native_v2_target_semantics(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    calls: list[tuple[str, str | None, str | None]] = []
    lock = threading.Lock()

    class FakeDiscoveryClient:
        github_token = "test-token"
        openverse_token = ""
        transport = None
        timeout_seconds = 1.0

        def search(
            self,
            provider: str,
            query: str,
            *,
            cursor: str = "",
            limit: int = 20,
            minecraft_version: str | None = None,
            loader: str | None = None,
            target_profile: str = "minecraft_mod",
        ) -> dict[str, object]:
            del cursor, limit, target_profile
            with lock:
                calls.append((query, minecraft_version, loader))
            barrier.wait(timeout=2)
            return {
                "provider": provider,
                "query": query,
                "returned": 1,
                "minecraft_version": minecraft_version,
                "loader": loader,
            }

    brief = {
        "brief_sha256": "sha256:test",
        "_mmm_platform_target": {
            "minecraft_version": "1.20.1",
            "loader": "fabric",
        },
        "domains": [
            {
                "domain_id": "request",
                "objective": "research request",
                "requirements": ["preserve request"],
                "evidence_kinds": ["source_code"],
                "queries": ["route alpha", "route beta"],
                "providers": ["github"],
                "depends_on": [],
            }
        ],
    }
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "auto")
    monkeypatch.setenv("MMM_DISCOVERY_WORKERS", "2")
    monkeypatch.setattr(
        ecosystem_discovery,
        "EcosystemDiscoveryClient",
        FakeDiscoveryClient,
    )

    result = ecosystem_discovery.discover_seed_bundle(
        "build the requested mod",
        {},
        research_brief=brief,
    )

    assert result["schema_version"] == "mmm/ecosystem-seed-bundle-v2"
    assert [page["query"] for page in result["pages"]] == ["route alpha", "route beta"]
    assert sorted(calls) == sorted(
        [
            ("route alpha", "1.20.1", "fabric"),
            ("route beta", "1.20.1", "fabric"),
        ]
    )


def test_official_rag_prefetch_delegates_graph_payload_to_native_code(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    started: list[str] = []
    lock = threading.Lock()

    class Receipt:
        correction_queries: tuple[str, ...] = ()
        correction_required = False
        hits = (object(),)

        def __init__(self, query: str) -> None:
            self.query = query

        def to_dict(self) -> dict[str, object]:
            return {
                "query": self.query,
                "hits": [{"evidence_id": self.query}],
            }

    def retrieve(query: str, **_kwargs: object) -> Receipt:
        with lock:
            started.append(query)
        barrier.wait(timeout=2)
        return Receipt(query)

    native = central_research.retrieve_domain_evidence
    while hasattr(native, "__wrapped__"):
        native = native.__wrapped__
    monkeypatch.setattr(central_research, "retrieve_official_evidence", retrieve)
    wrapped = parallel._parallel_retrieve_domain_evidence_factory(
        central_research,
        native,
    )
    monkeypatch.setenv("MMM_RESEARCH_WORKERS", "2")
    brief = {
        "brief_sha256": "sha256:test",
        "_mmm_platform_target": {
            "minecraft_version": "1.20.1",
            "loader": "fabric",
        },
        "domains": [
            {
                "domain_id": "official",
                "objective": "research official API",
                "requirements": ["preserve request"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["official alpha", "official beta"],
                "providers": ["official_docs"],
                "depends_on": [],
            }
        ],
    }

    evidence = wrapped(brief)

    assert sorted(started) == ["official alpha", "official beta"]
    assert evidence["target"]["minecraft_version"] == "1.20.1"
    assert evidence["target"]["loader"] == "fabric"
    assert isinstance(evidence["target"].get("mappings"), str)
    assert evidence["target"]["mappings"]
    assert evidence["deferred_official_domains"] == []
    assert evidence["unresolved_official_domains"] == []
    assert evidence["domains"][0]["strategy"] == "adaptive_per_query"
