from __future__ import annotations

"""Fast design-evidence collection without premature donor discovery.

Pre-design research answers Minecraft/Fabric feasibility and compatibility questions.
Third-party donor discovery belongs to the frozen-design reuse phase, where the query is
specific enough to be useful. Keeping those phases separate avoids searching the same
Modrinth/GitHub space twice and prevents implementation candidates from biasing design.
"""

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


_DETERMINISTIC_STAGES = ("official_rag", "technology_radar")


def collect_design_research(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect design evidence, then run domain agents serially over one local model slot.

    The two deterministic evidence sources are independent and therefore run concurrently.
    Domain-agent turns remain serial: local llama deployments commonly expose one inference
    slot, so concurrent domain turns only create queueing/VRAM pressure instead of reducing
    wall time.
    """

    from . import agentic_research_game_design as agentic

    research_brief = agentic.normalize_research_brief(
        prompt,
        {"title": "pre-design research"},
    )
    deterministic: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    futures: dict[str, Future[Any]] = {}
    with ThreadPoolExecutor(
        max_workers=len(_DETERMINISTIC_STAGES),
        thread_name_prefix="mmm-design-evidence",
    ) as executor:
        futures["official_rag"] = executor.submit(
            agentic.retrieve_domain_evidence,
            research_brief,
        )
        futures["technology_radar"] = executor.submit(
            agentic.collect_technology_radar,
            prompt,
            research_brief,
            page_size=50,
            page_builder=agentic.build_technology_radar,
        )

        # Read in a stable order so receipts/error ordering stays deterministic even though
        # the independent work itself runs concurrently.
        for stage in _DETERMINISTIC_STAGES:
            try:
                deterministic[stage] = futures[stage].result()
            except Exception as exc:
                errors.append(agentic._error(stage, exc))
                deterministic[stage] = {"status": "unavailable"}

    domain_notes: list[dict[str, Any]] = []
    for domain in research_brief.get("domains", []):
        if not isinstance(domain, dict):
            continue
        domain_notes.append(
            agentic._research_domain_with_agent(
                router,
                prompt=prompt,
                domain=domain,
                deterministic=deterministic,
                trace_metadata=trace_metadata,
            )
        )

    payload = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": research_brief,
        "deterministic": deterministic,
        "domain_notes": domain_notes,
        "errors": errors,
        "method": {
            "reason_act": "ReAct-style stage-scoped research tool loop",
            "adaptive_retrieval": "Self-RAG/FLARE-style retrieve when evidence is missing",
            "corrective_retrieval": "CRAG-style official/project evidence correction",
            "reflection": "Reflexion-style gap feedback across research passes",
            "planning_search": "third-party donor search is deferred to frozen-design reuse planning",
        },
    }
    payload["research_sha256"] = agentic._json_sha256(payload)
    return payload


__all__ = ["collect_design_research"]
