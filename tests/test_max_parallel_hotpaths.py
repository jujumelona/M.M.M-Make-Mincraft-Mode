from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import central_intelligence_amplifier as central


def _summary_note(domain_id: str = "mk_platform") -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "sufficient": True,
    }


def test_central_council_overlaps_independent_research(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    seen: list[str] = []
    lock = threading.Lock()

    def council(_router, _prompt):
        with lock:
            seen.append("council")
        barrier.wait(timeout=2)
        return {
            "authority": "advisory_only_user_request_is_authoritative",
            "chair_synthesis": {},
        }

    def research(_router, _prompt, *, trace_metadata=None):
        del trace_metadata
        with lock:
            seen.append("research")
        barrier.wait(timeout=2)
        return {"domain_notes": [], "method": {}}

    module = SimpleNamespace(
        collect_pre_design_research=research,
        _compact_research_for_design=lambda value: dict(value),
        generate_sectioned_game_design=lambda *_args, **_kwargs: {"title": "x"},
        supports_agentic_research_router=lambda _router: True,
        _research_domain_with_agent=lambda *_args, **_kwargs: _summary_note(),
    )
    monkeypatch.setattr(central, "build_central_committee", council)
    monkeypatch.setattr(central, "review_research_bundle", lambda *args, **kwargs: [])

    central.install(module)
    result = module.collect_pre_design_research(object(), "build only the requested mod")

    assert sorted(seen) == ["council", "research"]
    assert result["method"]["council_research_overlap"]
    assert result["_central_intelligence"]["committee"]["chair_synthesis"] == {}
