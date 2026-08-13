from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import wraps
from typing import Any, Mapping, Sequence


_MARKER = "_mmm_central_intelligence_amplifier_v1"
_PARALLEL_CORE_MARKER = "_mmm_parallel_research_design_core_v1"
_WORD = re.compile(r"[A-Za-z0-9_./:+-]{2,}|[가-힣]{2,}")
_SEVERITY = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

_COUNCIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "must_preserve": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "must_not_invent": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "subproblems": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "must_preserve",
                "must_not_invent",
                "subproblems",
                "risks",
                "research_questions",
                "confidence",
            ],
            "additionalProperties": False,
        }
    },
    "required": ["analysis"],
    "additionalProperties": False,
}

_CHAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "synthesis": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "negative_constraints": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "subproblem_order": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "acceptance_observables": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": [
                "requirements",
                "negative_constraints",
                "subproblem_order",
                "acceptance_observables",
                "unresolved_questions",
            ],
            "additionalProperties": False,
        }
    },
    "required": ["synthesis"],
    "additionalProperties": False,
}

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "review": {
            "type": "object",
            "properties": {
                "missing_requirements": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "unsupported_additions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "contradictions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "research_gaps": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "affected_sections": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "severity": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "critical"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "missing_requirements",
                "unsupported_additions",
                "contradictions",
                "research_gaps",
                "affected_sections",
                "severity",
                "confidence",
            ],
            "additionalProperties": False,
        }
    },
    "required": ["review"],
    "additionalProperties": False,
}

_LENSES: tuple[tuple[str, str], ...] = (
    (
        "requirement_decomposer",
        "Use least-to-most decomposition. Extract atomic user requirements, exclusions, "
        "observable success conditions, and dependency-ordered subproblems. Do not solve "
        "the design and do not add wishes the user did not state.",
    ),
    (
        "systems_architect",
        "Act as a Minecraft systems architect. Identify cross-system dependencies, state "
        "ownership, persistence/networking/runtime risks, implementation research questions, "
        "and hidden integration constraints while preserving the exact request.",
    ),
    (
        "adversarial_skeptic",
        "Try to falsify an initial interpretation of the request. Look for ambiguous wording, "
        "easy-to-miss requirements, accidental scope expansion, contradictions, and concrete "
        "acceptance observations needed to catch a superficially plausible but wrong design.",
    ),
)


def install_parallel_core(agentic_module: Any) -> None:
    """Parallelize independent research providers, domain agents, and design sections."""

    current_collect = agentic_module.collect_pre_design_research
    if not getattr(current_collect, _PARALLEL_CORE_MARKER, False):

        @wraps(current_collect)
        def collect_parallel(
            router: Any,
            prompt: str,
            *,
            trace_metadata: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            research_brief = agentic_module.normalize_research_brief(
                prompt,
                {"title": "pre-design research"},
            )

            def official() -> Any:
                return agentic_module.retrieve_domain_evidence(research_brief)

            def technology() -> Any:
                return agentic_module.collect_technology_radar(
                    prompt,
                    research_brief,
                    page_size=50,
                    page_builder=agentic_module.build_technology_radar,
                )

            def ecosystem() -> Any:
                return agentic_module.collect_ecosystem_seed_bundle(
                    prompt,
                    {},
                    research_brief=research_brief,
                    route_limit=12,
                    page_builder=agentic_module.discover_seed_bundle,
                    allow_legacy_terminal=True,
                )

            provider_jobs = (
                ("official_rag", official),
                ("technology_radar", technology),
                ("ecosystem_discovery", ecosystem),
            )
            provider_results: dict[str, Any] = {}
            provider_errors: dict[str, dict[str, str]] = {}
            with ThreadPoolExecutor(
                max_workers=min(_worker_count(), len(provider_jobs)),
                thread_name_prefix="mmm_research_provider",
            ) as pool:
                futures = {pool.submit(fn): key for key, fn in provider_jobs}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        provider_results[key] = future.result()
                    except Exception as exc:
                        provider_results[key] = {"status": "unavailable"}
                        provider_errors[key] = agentic_module._error(key, exc)

            deterministic = {key: provider_results[key] for key, _fn in provider_jobs}
            errors = [
                provider_errors[key]
                for key, _fn in provider_jobs
                if key in provider_errors
            ]

            domains = [
                domain
                for domain in research_brief.get("domains", [])
                if isinstance(domain, dict)
            ]
            indexed_notes: dict[int, dict[str, Any]] = {}
            if domains:
                with ThreadPoolExecutor(
                    max_workers=min(_worker_count(), len(domains)),
                    thread_name_prefix="mmm_research_domain",
                ) as pool:
                    futures = {}
                    for index, domain in enumerate(domains):
                        # ContextVar state is per request. Copy it into each worker so the
                        # forced-RAG receipt remains isolated even when domain specialists
                        # execute concurrently on the same shared production router.
                        context = copy_context()
                        future = pool.submit(
                            context.run,
                            agentic_module._research_domain_with_agent,
                            router,
                            prompt=prompt,
                            domain=domain,
                            deterministic=deterministic,
                            trace_metadata=trace_metadata,
                        )
                        futures[future] = index
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            indexed_notes[index] = future.result()
                        except Exception as exc:
                            domain_id = str(
                                domains[index].get("domain_id", "unknown")
                            ).strip() or "unknown"
                            indexed_notes[index] = {
                                "domain_id": domain_id,
                                "claims": [],
                                "gaps": [f"{type(exc).__name__}: {exc}"],
                                "next_queries": list(domains[index].get("queries", [])),
                                "sufficient": False,
                                "worker_error": True,
                            }
            domain_notes = [indexed_notes[index] for index in range(len(domains))]

            payload = {
                "schema_version": "mmm/agentic-pre-design-research-v1",
                "research_brief": research_brief,
                "deterministic": deterministic,
                "domain_notes": domain_notes,
                "errors": errors,
                "method": {
                    "reason_act": "ReAct-style stage-scoped research tool loop",
                    "adaptive_retrieval": (
                        "Self-RAG/FLARE-style retrieve when evidence is missing"
                    ),
                    "corrective_retrieval": (
                        "CRAG-style official correction and ecosystem expansion"
                    ),
                    "reflection": "Reflexion-style gap feedback across research passes",
                    "parallel_provider_fanout": (
                        "official RAG, technology radar, and ecosystem discovery"
                    ),
                    "parallel_specialists": (
                        "independent research domains execute concurrently and merge "
                        "in deterministic domain order"
                    ),
                    "planning_search": (
                        "existing MMM verifier/candidate search remains downstream"
                    ),
                },
            }
            payload["research_sha256"] = agentic_module._json_sha256(payload)
            return payload

        setattr(collect_parallel, _PARALLEL_CORE_MARKER, True)
        collect_parallel.__wrapped__ = current_collect  # type: ignore[attr-defined]
        agentic_module.collect_pre_design_research = collect_parallel

    current_generate = agentic_module.generate_sectioned_game_design
    if getattr(current_generate, _PARALLEL_CORE_MARKER, False):
        return

    @wraps(current_generate)
    def generate_parallel(
        game_design_module: Any,
        router: Any,
        prompt: str,
        *,
        media_paths=(),
        research: Mapping[str, Any],
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        sections: dict[int, dict[str, Any]] = {}
        specs = tuple(agentic_module._SECTION_SPECS)
        with ThreadPoolExecutor(
            max_workers=min(_worker_count(), len(specs)),
            thread_name_prefix="mmm_design_section",
        ) as pool:
            futures = {
                pool.submit(
                    agentic_module._generate_section,
                    router,
                    prompt=prompt,
                    section_id=section_id,
                    fields=fields,
                    properties=properties,
                    research=research,
                    media_paths=media_paths if index == 0 else (),
                    trace_metadata=trace_metadata,
                ): index
                for index, (section_id, fields, properties) in enumerate(specs)
            }
            for future in as_completed(futures):
                sections[futures[future]] = future.result()

        merged: dict[str, Any] = {}
        for index in range(len(specs)):
            merged.update(sections[index])
        if merged.get("art_direction") == {}:
            merged.pop("art_direction", None)
        game_design_module._validate_design(merged)
        return merged

    setattr(generate_parallel, _PARALLEL_CORE_MARKER, True)
    generate_parallel.__wrapped__ = current_generate  # type: ignore[attr-defined]
    agentic_module.generate_sectioned_game_design = generate_parallel


def install(agentic_module: Any) -> None:
    """Amplify weak central planning with host-owned ensemble and verification.

    Model-generated committee text is never execution authority. The unchanged user request
    remains authoritative; council/debate/audit outputs are advisory evidence that exposes
    omissions before the validated planner commits to a design.
    """

    current_collect = agentic_module.collect_pre_design_research
    if not getattr(current_collect, _MARKER, False):

        @wraps(current_collect)
        def collect(router: Any, prompt: str, *, trace_metadata=None):
            if not _amplification_enabled(router):
                return current_collect(router, prompt, trace_metadata=trace_metadata)

            council = build_central_committee(router, prompt)
            result = current_collect(router, prompt, trace_metadata=trace_metadata)
            reviews = review_research_bundle(router, prompt, result, council=council)
            result = dict(result)
            result["_central_intelligence"] = {
                "schema_version": "mmm/central-intelligence-context-v1",
                "authority": "advisory_only_user_request_is_authoritative",
                "committee": council,
                "research_reviews": reviews,
            }

            gaps = _stable_unique(
                gap
                for item in reviews
                for gap in item.get("research_gaps", [])
                if isinstance(gap, str) and gap.strip()
            )[:8]
            worst = max(
                (_SEVERITY.get(str(item.get("severity", "none")), 0) for item in reviews),
                default=0,
            )
            if gaps and worst >= _SEVERITY["medium"]:
                correction_domain = {
                    "domain_id": "central_critical_gaps",
                    "objective": (
                        "Resolve only the adversarially identified evidence gaps before design."
                    ),
                    "providers": ["official_docs", "project_rag", "external_mcp"],
                    "queries": gaps,
                    "depends_on": [],
                }
                try:
                    correction = agentic_module._research_domain_with_agent(
                        router,
                        prompt=prompt,
                        domain=correction_domain,
                        deterministic=result.get("deterministic", {}),
                        trace_metadata={
                            **dict(trace_metadata or {}),
                            "adaptive_branch": "central_critical_gaps",
                        },
                    )
                except Exception as exc:
                    correction = {
                        "domain_id": "central_critical_gaps",
                        "claims": [],
                        "gaps": gaps,
                        "next_queries": [],
                        "sufficient": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                notes = list(result.get("domain_notes", []))
                notes.append(correction)
                result["domain_notes"] = notes
                result["_central_intelligence"]["adaptive_gap_branch"] = correction

            method = dict(result.get("method", {}))
            method.update(
                {
                    "least_to_most": "host-routed atomic requirement decomposition",
                    "self_consistency": "independent specialist council",
                    "mixture_of_agents": (
                        "specialist council followed by constrained chair synthesis"
                    ),
                    "multi_agent_debate": "independent coverage and skeptic reviews",
                    "adaptive_branching": (
                        "disagreement/gap-triggered specialist research"
                    ),
                }
            )
            result["method"] = method
            result["research_sha256"] = _sha(result)
            return result

        setattr(collect, _MARKER, True)
        collect.__wrapped__ = current_collect  # type: ignore[attr-defined]
        agentic_module.collect_pre_design_research = collect

    current_compact = agentic_module._compact_research_for_design
    if not getattr(current_compact, _MARKER, False):

        @wraps(current_compact)
        def compact(research: Mapping[str, Any]) -> dict[str, Any]:
            value = dict(current_compact(research))
            intelligence = research.get("_central_intelligence")
            if isinstance(intelligence, Mapping):
                value["central_intelligence"] = _compact_intelligence(intelligence)
            design_review = research.get("_central_design_review")
            if isinstance(design_review, Mapping):
                value["central_design_review"] = _compact_review(design_review)
            return value

        setattr(compact, _MARKER, True)
        compact.__wrapped__ = current_compact  # type: ignore[attr-defined]
        agentic_module._compact_research_for_design = compact

    current_generate = agentic_module.generate_sectioned_game_design
    if getattr(current_generate, _MARKER, False):
        return

    @wraps(current_generate)
    def generate(
        game_design_module: Any,
        router: Any,
        prompt: str,
        *,
        media_paths=(),
        research: Mapping[str, Any],
        trace_metadata=None,
    ) -> dict[str, Any]:
        if not _amplification_enabled(router):
            return current_generate(
                game_design_module,
                router,
                prompt,
                media_paths=media_paths,
                research=research,
                trace_metadata=trace_metadata,
            )

        first = current_generate(
            game_design_module,
            router,
            prompt,
            media_paths=media_paths,
            research=research,
            trace_metadata=trace_metadata,
        )
        first_reviews = review_design(router, prompt, first, research=research)
        first_score = _review_score(first_reviews)
        winner = first
        winner_reviews = first_reviews
        refined = False

        if _worst_severity(first_reviews) >= _SEVERITY["medium"]:
            refinement_context = {
                "schema_version": "mmm/central-design-review-v1",
                "instruction": (
                    "Refine the design only to fix reviewer-supported omissions, "
                    "contradictions, or unsupported additions. Preserve correct parts and "
                    "never broaden beyond the authoritative user request."
                ),
                "reviews": first_reviews,
            }
            second_research = {
                **dict(research),
                "_central_design_review": refinement_context,
            }
            second = current_generate(
                game_design_module,
                router,
                prompt,
                media_paths=media_paths,
                research=second_research,
                trace_metadata={
                    **dict(trace_metadata or {}),
                    "self_refine_pass": 1,
                },
            )
            second_reviews = review_design(
                router,
                prompt,
                second,
                research=second_research,
            )
            if _review_score(second_reviews) <= first_score:
                winner = second
                winner_reviews = second_reviews
                refined = True

        verification = {
            "schema_version": "mmm/central-design-verifier-v1",
            "authority": "advisory_only_user_request_is_authoritative",
            "initial_reviews": first_reviews,
            "winner_reviews": winner_reviews,
            "self_refine_attempted": (
                _worst_severity(first_reviews) >= _SEVERITY["medium"]
            ),
            "refined_candidate_selected": refined,
            "selection_rule": "lowest deterministic review severity/issue score",
        }
        if isinstance(research, dict):
            research["_central_design_verification"] = verification
            research["research_sha256"] = _sha(research)
        return winner

    setattr(generate, _MARKER, True)
    generate.__wrapped__ = current_generate  # type: ignore[attr-defined]
    agentic_module.generate_sectioned_game_design = generate


def build_central_committee(router: Any, prompt: str) -> dict[str, Any]:
    """Run independent bounded specialists in parallel, then synthesize consensus."""

    workers = min(_worker_count(), len(_LENSES))
    outputs: dict[str, dict[str, Any]] = {}

    def run(lens: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        lens_id, instruction = lens
        messages = [
            {
                "role": "system",
                "content": (
                    "You are one independent analysis specialist supporting a weak central "
                    "Minecraft-mod planner. Return only the compact JSON contract. Do not emit "
                    "chain-of-thought. The user request is the sole authority. Your analysis is "
                    "advisory: detect requirements and risks but never invent features. "
                    + instruction
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"authoritative_request": prompt, "lens": lens_id},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        raw = router.generate_text(
            "planner",
            messages,
            response_format="json",
            response_schema=_COUNCIL_SCHEMA,
            enable_tools=False,
        )
        return lens_id, _parse(raw, "analysis")

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="mmm_central_council",
    ) as pool:
        futures = {pool.submit(run, lens): lens[0] for lens in _LENSES}
        for future in as_completed(futures):
            lens_id = futures[future]
            try:
                key, value = future.result()
            except Exception as exc:
                outputs[lens_id] = {
                    "must_preserve": [],
                    "must_not_invent": [],
                    "subproblems": [],
                    "risks": [f"specialist_error:{type(exc).__name__}"],
                    "research_questions": [],
                    "confidence": 0.0,
                }
            else:
                outputs[key] = value

    ordered = [{"lens": lens_id, **outputs[lens_id]} for lens_id, _ in _LENSES]
    disagreement = _disagreement(ordered)
    chair = _chair_synthesis(router, prompt, ordered, disagreement)
    extra: dict[str, Any] | None = None
    if disagreement >= _disagreement_threshold():
        extra = _extra_disagreement_specialist(router, prompt, ordered)
    payload = {
        "schema_version": "mmm/central-specialist-council-v1",
        "authority": "advisory_only_user_request_is_authoritative",
        "parallel": workers > 1,
        "workers": workers,
        "specialists": ordered,
        "disagreement": round(disagreement, 6),
        "chair_synthesis": chair,
        "adaptive_disagreement_specialist": extra,
        "methods": [
            "least_to_most_decomposition",
            "self_consistency_by_independent_lenses",
            "mixture_of_agents_synthesis",
            "uncertainty_gated_branching",
        ],
    }
    payload["committee_sha256"] = _sha(payload)
    return payload


def review_research_bundle(
    router: Any,
    prompt: str,
    research: Mapping[str, Any],
    *,
    council: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = {
        "authoritative_request": prompt,
        "committee": _compact_intelligence({"committee": council}).get(
            "committee", {}
        ),
        "research_brief": research.get("research_brief"),
        "domain_notes": research.get("domain_notes", []),
        "errors": research.get("errors", []),
    }
    return _parallel_reviews(
        router,
        payload,
        target="pre-design research",
        reviewers=(
            (
                "coverage_reviewer",
                "Find explicit or implied user requirements that the research still does not cover.",
            ),
            (
                "skeptic_reviewer",
                "Try to falsify research claims: flag unsupported conclusions, contradictions, "
                "and questions whose answer could materially change the design.",
            ),
        ),
    )


def review_design(
    router: Any,
    prompt: str,
    design: Mapping[str, Any],
    *,
    research: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = {
        "authoritative_request": prompt,
        "candidate_design": {
            key: value for key, value in design.items() if not str(key).startswith("_")
        },
        "research_receipt": {
            "research_sha256": research.get("research_sha256"),
            "central_intelligence": _compact_intelligence(
                research.get("_central_intelligence", {})
                if isinstance(research.get("_central_intelligence"), Mapping)
                else {}
            ),
        },
    }
    return _parallel_reviews(
        router,
        payload,
        target="candidate game design",
        reviewers=(
            (
                "requirement_judge",
                "Check one-to-one coverage of the authoritative request. Missing requirements "
                "are worse than lack of embellishment. Do not demand features the user did not ask for.",
            ),
            (
                "hallucination_judge",
                "Attack scope creep, unsupported additions, contradictions, and plausible-looking "
                "details that are not justified by the request or research evidence.",
            ),
        ),
    )


def _parallel_reviews(
    router: Any,
    payload: Mapping[str, Any],
    *,
    target: str,
    reviewers: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    workers = min(_worker_count(), len(reviewers))
    results: dict[str, dict[str, Any]] = {}

    def run(spec: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        reviewer_id, instruction = spec
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an independent adversarial reviewer of {target}. "
                    "Return only the JSON contract and no chain-of-thought. The original user "
                    "request is authoritative; generated research/design is not. " + instruction
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            },
        ]
        raw = router.generate_text(
            "coder_safe",
            messages,
            response_format="json",
            response_schema=_REVIEW_SCHEMA,
            enable_tools=False,
        )
        return reviewer_id, _parse(raw, "review")

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="mmm_central_review",
    ) as pool:
        futures = {pool.submit(run, spec): spec[0] for spec in reviewers}
        for future in as_completed(futures):
            reviewer_id = futures[future]
            try:
                key, value = future.result()
            except Exception as exc:
                results[reviewer_id] = {
                    "missing_requirements": [],
                    "unsupported_additions": [],
                    "contradictions": [],
                    "research_gaps": [],
                    "affected_sections": [],
                    "severity": "low",
                    "confidence": 0.0,
                    "reviewer_error": f"{type(exc).__name__}: {exc}",
                }
            else:
                results[key] = value
    return [{"reviewer": key, **results[key]} for key, _ in reviewers]


def _chair_synthesis(
    router: Any,
    prompt: str,
    specialists: Sequence[Mapping[str, Any]],
    disagreement: float,
) -> dict[str, Any]:
    raw = router.generate_text(
        "planner",
        [
            {
                "role": "system",
                "content": (
                    "Act as a constrained Mixture-of-Agents chair. Synthesize only statements "
                    "supported by the authoritative request or at least one specialist as a "
                    "question/risk. Never turn a speculative specialist idea into a requirement. "
                    "Return only compact JSON and no chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "authoritative_request": prompt,
                        "specialists": specialists,
                        "host_disagreement": disagreement,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        response_format="json",
        response_schema=_CHAIR_SCHEMA,
        enable_tools=False,
    )
    try:
        return _parse(raw, "synthesis")
    except Exception as exc:
        return {
            "requirements": _stable_unique(
                value
                for item in specialists
                for value in item.get("must_preserve", [])
                if isinstance(value, str)
            )[:16],
            "negative_constraints": _stable_unique(
                value
                for item in specialists
                for value in item.get("must_not_invent", [])
                if isinstance(value, str)
            )[:12],
            "subproblem_order": _stable_unique(
                value
                for item in specialists
                for value in item.get("subproblems", [])
                if isinstance(value, str)
            )[:16],
            "acceptance_observables": [],
            "unresolved_questions": _stable_unique(
                value
                for item in specialists
                for value in item.get("research_questions", [])
                if isinstance(value, str)
            )[:12],
            "chair_error": f"{type(exc).__name__}: {exc}",
        }


def _extra_disagreement_specialist(
    router: Any,
    prompt: str,
    specialists: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = router.generate_text(
        "planner",
        [
            {
                "role": "system",
                "content": (
                    "The specialist council disagreed materially. Resolve only the disagreement "
                    "by re-reading the authoritative request. Prefer uncertainty over invention. "
                    "Return the analysis JSON contract, no chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "authoritative_request": prompt,
                        "disagreeing_specialists": specialists,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        response_format="json",
        response_schema=_COUNCIL_SCHEMA,
        enable_tools=False,
    )
    try:
        return _parse(raw, "analysis")
    except Exception as exc:
        return {
            "must_preserve": [],
            "must_not_invent": [],
            "subproblems": [],
            "risks": [f"disagreement_specialist_error:{type(exc).__name__}"],
            "research_questions": [],
            "confidence": 0.0,
        }


def _parse(raw: str, field: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get(field), dict):
            return dict(value[field])
    raise ValueError(f"model response did not contain {field!r} object")


def _tokens(values: Sequence[str]) -> set[str]:
    return {
        token.casefold()
        for value in values
        for token in _WORD.findall(value)
    }


def _disagreement(specialists: Sequence[Mapping[str, Any]]) -> float:
    sets = [
        _tokens(
            [
                *[str(x) for x in item.get("must_preserve", [])],
                *[str(x) for x in item.get("subproblems", [])],
            ]
        )
        for item in specialists
    ]
    pairs: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left, right = sets[left_index], sets[right_index]
            if not left and not right:
                pairs.append(0.0)
                continue
            pairs.append(1.0 - len(left & right) / max(1, len(left | right)))
    return sum(pairs) / len(pairs) if pairs else 0.0


def _worst_severity(reviews: Sequence[Mapping[str, Any]]) -> int:
    return max(
        (_SEVERITY.get(str(item.get("severity", "none")), 0) for item in reviews),
        default=0,
    )


def _review_score(reviews: Sequence[Mapping[str, Any]]) -> int:
    worst = _worst_severity(reviews)
    issues = sum(
        len(item.get(field, []))
        for item in reviews
        for field in (
            "missing_requirements",
            "unsupported_additions",
            "contradictions",
            "research_gaps",
        )
        if isinstance(item.get(field), list)
    )
    return worst * 100 + min(99, issues)


def _compact_intelligence(value: Mapping[str, Any]) -> dict[str, Any]:
    committee = value.get("committee")
    reviews = value.get("research_reviews")
    result: dict[str, Any] = {}
    if isinstance(committee, Mapping):
        result["committee"] = {
            "authority": committee.get("authority"),
            "disagreement": committee.get("disagreement"),
            "chair_synthesis": committee.get("chair_synthesis"),
            "adaptive_disagreement_specialist": committee.get(
                "adaptive_disagreement_specialist"
            ),
        }
    if isinstance(reviews, list):
        result["research_reviews"] = [_compact_review(item) for item in reviews]
    adaptive = value.get("adaptive_gap_branch")
    if isinstance(adaptive, Mapping):
        result["adaptive_gap_branch"] = {
            "domain_id": adaptive.get("domain_id"),
            "claims": adaptive.get("claims", []),
            "gaps": adaptive.get("gaps", []),
            "sufficient": adaptive.get("sufficient"),
        }
    return result


def _compact_review(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "reviewer",
            "missing_requirements",
            "unsupported_additions",
            "contradictions",
            "research_gaps",
            "affected_sections",
            "severity",
            "confidence",
        )
        if key in value
    }


def _amplification_enabled(router: Any) -> bool:
    if bool(getattr(router, "_mmm_enable_central_intelligence", False)):
        return True
    try:
        from .model_router import ModelRouter
    except Exception:
        return False
    return isinstance(router, ModelRouter)


def _stable_unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _worker_count() -> int:
    raw = os.environ.get("MMM_CENTRAL_AI_WORKERS", "").strip()
    try:
        requested = int(raw) if raw else 4
    except ValueError:
        requested = 4
    return max(1, min(8, requested))


def _disagreement_threshold() -> float:
    raw = os.environ.get("MMM_CENTRAL_AI_DISAGREEMENT", "").strip()
    try:
        value = float(raw) if raw else 0.62
    except ValueError:
        value = 0.62
    return max(0.0, min(1.0, value))


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "build_central_committee",
    "install",
    "install_parallel_core",
    "review_design",
    "review_research_bundle",
]
