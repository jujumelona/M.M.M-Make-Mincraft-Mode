from __future__ import annotations

import threading

from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai.pre_design_research_pipeline import collect_design_research


def test_pre_design_research_parallelizes_evidence_and_defers_donor_search(
    monkeypatch,
) -> None:
    brief = {
        "domains": [
            {
                "domain_id": "fabric_api",
                "queries": ["Fabric API target behavior"],
            }
        ]
    }
    monkeypatch.setattr(agentic, "normalize_research_brief", lambda *_args: brief)

    barrier = threading.Barrier(2)

    def official(_brief):
        barrier.wait(timeout=2)
        return {"status": "available", "kind": "official"}

    def radar(_prompt, _brief, **_kwargs):
        barrier.wait(timeout=2)
        return {"status": "available", "kind": "radar"}

    monkeypatch.setattr(agentic, "retrieve_domain_evidence", official)
    monkeypatch.setattr(agentic, "collect_technology_radar", radar)
    monkeypatch.setattr(
        agentic,
        "collect_ecosystem_seed_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("donor discovery must not run before design freeze")
        ),
    )

    seen_deterministic = []

    def domain_agent(_router, **kwargs):
        deterministic = kwargs["deterministic"]
        seen_deterministic.append(set(deterministic))
        return {
            "domain_id": kwargs["domain"]["domain_id"],
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "sufficient": True,
        }

    monkeypatch.setattr(agentic, "_research_domain_with_agent", domain_agent)

    payload = collect_design_research(object(), "build a Fabric mechanic")

    assert set(payload["deterministic"]) == {"official_rag", "technology_radar"}
    assert "ecosystem_discovery" not in payload["deterministic"]
    assert seen_deterministic == [{"official_rag", "technology_radar"}]
    assert payload["domain_notes"][0]["domain_id"] == "fabric_api"
    assert "deferred" in payload["method"]["planning_search"]
