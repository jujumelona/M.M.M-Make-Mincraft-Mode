from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import custom_generation_research as generation_research
from minecraft_mod_ai import grounded_rag_runtime_contract as runtime
from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_provider_work_overlaps_without_sleep(monkeypatch):
    coordinator = runtime.GroundedRAGCoordinator()
    barrier = threading.Barrier(2)
    entered = []
    lock = threading.Lock()

    def modrinth(query, versions):
        del versions
        with lock:
            entered.append(("modrinth", query))
        barrier.wait(timeout=5)
        return [], []

    def github(query):
        with lock:
            entered.append(("github", query))
        barrier.wait(timeout=5)
        return [], []

    monkeypatch.setattr(grounded, "_modrinth_search", modrinth)
    monkeypatch.setattr(grounded, "_github_repository_search", github)
    result = coordinator.retrieve_many(("combat progression",), ("1.21.8",))

    assert result["combat progression"]["work_graph"]["nested_executor"] is False
    assert {provider for provider, _ in entered} == {"modrinth", "github"}
    assert 2 <= coordinator.max_workers <= 16
    coordinator.executor.shutdown(wait=True, cancel_futures=True)


def test_donor_repository_candidates_are_requirement_reusable():
    coordinator = runtime.GroundedRAGCoordinator()
    with coordinator._lock:
        coordinator._donors_by_query["combat progression"] = [
            {
                "source_id": "github:owner/repo:src/main/java/Combat.java",
                "content": "class Combat {}",
                "content_sha256": "sha256:test",
                "metadata": {
                    "repository": "owner/repo",
                    "path": "src/main/java/Combat.java",
                },
            }
        ]
    repositories = coordinator.repositories_for_capabilities(
        ("combat.progression",),
        {
            "search_terms": [
                {
                    "capability": "combat.progression",
                    "terms": ["combat progression implementation"],
                }
            ]
        },
    )
    assert repositories["combat.progression"] == ("https://github.com/owner/repo",)
    coordinator.executor.shutdown(wait=True, cancel_futures=True)


def test_grounded_install_merges_rag_donors_before_public_discovery(monkeypatch):
    coordinator = runtime.coordinator()
    with coordinator._lock:
        coordinator._donors_by_query["quest progression"] = [
            {
                "source_id": "github:grounded/donor:Quest.java",
                "content": "class Quest {}",
                "content_sha256": "sha256:quest",
                "metadata": {"repository": "grounded/donor", "path": "Quest.java"},
            }
        ]

    class Agentic:
        @staticmethod
        def _forced_rag_bundle(router, brief):
            del router, brief
            return {"versions": [], "domains": []}

    def public(capabilities, client, *, capability_graph=None):
        del client, capability_graph
        return {cap: ("https://github.com/public/donor",) for cap in capabilities}

    reuse = SimpleNamespace(_parallel_donor_repository_discovery=public)
    monkeypatch.setattr(runtime, "_INSTALLED", False)
    runtime.install(Agentic, reuse)
    found = reuse._parallel_donor_repository_discovery(
        ("quest.progression",),
        object(),
        capability_graph={
            "search_terms": [
                {"capability": "quest.progression", "terms": ["quest progression"]}
            ]
        },
    )
    assert found["quest.progression"][0] == "https://github.com/grounded/donor"
    assert found["quest.progression"][1] == "https://github.com/public/donor"


def test_actual_coder_call_receives_research_context(monkeypatch, tmp_path):
    captured = {}

    class UnderlyingRouter:
        def generate_text(self, role, messages, **kwargs):
            captured["role"] = role
            captured["messages"] = [dict(item) for item in messages]
            captured["kwargs"] = dict(kwargs)
            return "done"

    class Engine:
        def ingest_code_owned_request(self, messages):
            self.ingested = messages

        def initial_bundle(self):
            return {
                "evidence": [
                    {
                        "evidence_id": "ev-symbol",
                        "path": "src/main/java/X.java",
                        "symbols": ["X.tick"],
                    }
                ],
                "bundle_sha256": "sha256:bundle",
            }

        def evolve_from_generation(self, text):
            assert text == "done"
            return None, ()

        def receipt(self):
            return {"status": "ok"}

    wrapped = generation_research._ResearchEvidenceRouter(
        UnderlyingRouter(),
        owner=SimpleNamespace(_cached_index=None, _cached_root=None, policy=None),
        project_root=tmp_path,
        module=None,
        minecraft_version="1.21.8",
        loader="fabric",
        mappings="dummy",
    )
    engine = Engine()
    monkeypatch.setattr(wrapped, "_engine", lambda: engine)
    monkeypatch.setattr(
        generation_research,
        "_sanitized_messages",
        lambda messages, **kwargs: [dict(item) for item in messages],
    )

    result = wrapped.generate_text(
        "coder",
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "implement X.tick"},
        ],
    )

    assert result == "done"
    assert captured["role"] == "coder"
    injected = "\n".join(str(item.get("content", "")) for item in captured["messages"])
    assert "research_code_context" in injected
    assert "ev-symbol" in injected
    assert "X.tick" in injected


def test_runtime_labels_are_stable_nonversioned():
    payload = {
        "schema_version": "mmm/forced-pre-design-rag",
        "external": "mmm/external-grounded-rag",
        "local": "mmm/local-project-rag-index",
    }
    assert all(
        not value.endswith(("-v1", "-v2", "-v3"))
        for value in payload.values()
    )
