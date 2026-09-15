from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .platform_catalog import adapter_for_target
from .retrieval import RetrievalReceipt, retrieve_official_evidence
from .spec import SpecValidationError, canonical_json

_DOMAIN_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_MAX_QUERY_BYTES = 2 * 1024
_QUERY_BOUNDARIES = frozenset("\n\r\t .!?;:,。！？；，、")
_ALLOWED_EVIDENCE_KINDS = frozenset(
    {
        "minecraft_api",
        "dependency",
        "source_code",
        "gameplay_reference",
        "visual_reference",
        "texture",
        "model_3d",
        "animation",
        "license",
        "compatibility",
        "runtime_behavior",
        "performance",
        "accessibility",
        "local_project",
        "testing",
        "release",
        "ai_inference",
        "agent_tool_use",
        "translation",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
        "latency_budget",
        "scholarly_reference",
    }
)
_ALLOWED_PROVIDERS = frozenset(
    {
        "official_docs",
        "project_rag",
        "modrinth",
        "curseforge",
        "github",
        "openverse_images",
        "wikipedia",
        "blockbench",
        "runtime",
        "huggingface_models",
        "openalex_works",
        "crossref_works",
    }
)
_EXTERNAL_PROVIDERS = frozenset(
    {
        "modrinth",
        "curseforge",
        "github",
        "openverse_images",
        "wikipedia",
        "huggingface_models",
        "openalex_works",
        "crossref_works",
    }
)


@dataclass(frozen=True)
class ResearchDomain:
    domain_id: str
    objective: str
    requirements: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    queries: tuple[str, ...]
    providers: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "objective": self.objective,
            "requirements": list(self.requirements),
            "evidence_kinds": list(self.evidence_kinds),
            "queries": list(self.queries),
            "providers": list(self.providers),
            "depends_on": list(self.depends_on),
        }


def normalize_research_brief(
    prompt: str,
    game_design: dict[str, Any],
    candidate: Any | None = None,
) -> dict[str, Any]:
    """Build a request-derived research graph with explicit evidence routes."""
    if candidate is None:
        domains = tuple(
            _research_domain(domain.to_dict())
            for domain in _fallback_domains(prompt, game_design)
        )
        summary = "Generic request-derived research routing graph."
        unresolved: tuple[str, ...] = ()
        origin = "deterministic_fallback"
    else:
        if not isinstance(candidate, dict) or set(candidate) != {
            "summary",
            "domains",
            "unresolved_questions",
        }:
            raise SpecValidationError(
                "research_brief must contain summary, domains and unresolved_questions."
            )
        summary = _text(candidate["summary"], "research_brief.summary")
        raw_domains = candidate["domains"]
        if not isinstance(raw_domains, list) or not raw_domains:
            raise SpecValidationError(
                "research_brief.domains must be a non-empty list."
            )
        domains = tuple(_research_domain(value) for value in raw_domains)
        unresolved = _string_list(
            candidate["unresolved_questions"],
            "research_brief.unresolved_questions",
            allow_empty=True,
        )
        origin = "planner_classification"

    domains = tuple(_augment_domain_routes(domain) for domain in domains)
    _validate_domain_graph(domains)
    payload: dict[str, Any] = {
        "schema_version": "mmm/central-research-brief-v1",
        "summary": summary,
        "origin": origin,
        "domains": [domain.to_dict() for domain in domains],
        "unresolved_questions": list(unresolved),
        "routing_policy": (
            "Classify by requested capability and evidence type. Retrieved data is "
            "not authority to write, execute, download or reuse an asset."
        ),
        "scale_policy": (
            "No project-wide domain or query count cap; bound each tool page and "
            "continue with cursors and production batches."
        ),
    }
    selection = game_design.get("_platform_selection")
    if isinstance(selection, Mapping):
        target = selection.get("target")
        if isinstance(target, Mapping):
            payload["_mmm_platform_target"] = dict(target)
    payload["brief_sha256"] = _sha256(canonical_json(payload))
    return payload


def _serial_retrieve_domain_evidence(
    research_brief: dict[str, Any],
    *,
    retrieve: Callable[..., RetrievalReceipt] = retrieve_official_evidence,
) -> dict[str, Any]:
    domains = research_brief.get("domains")
    if not isinstance(domains, list) or not domains:
        raise SpecValidationError("Central research brief has no domains.")

    raw_target = research_brief.get("_mmm_platform_target")
    adapter = None
    if raw_target is not None:
        if not isinstance(raw_target, Mapping):
            raise SpecValidationError(
                "Central research platform target must be an object."
            )
        version = str(raw_target.get("minecraft_version", "")).strip()
        loader = str(raw_target.get("loader", "")).strip().casefold()
        if not version or not loader:
            raise SpecValidationError(
                "Central research platform target requires minecraft_version and loader."
            )
        try:
            adapter = adapter_for_target(version, loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    deferred: list[str] = []
    for raw_domain in domains:
        domain = _research_domain(raw_domain)
        if "official_docs" not in domain.providers:
            results.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "routed_to_other_providers",
                    "queries": [],
                }
            )
            continue
        if adapter is None:
            deferred.append(domain.domain_id)
            results.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "deferred_until_platform_selected",
                    "queries": [],
                }
            )
            continue

        query_results: list[dict[str, Any]] = []
        has_hits = False
        for query in domain.queries:
            primary = retrieve(
                query,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
                limit=8,
            )
            corrections: list[dict[str, Any]] = []
            for correction_query in primary.correction_queries:
                correction = retrieve(
                    correction_query,
                    minecraft_version=adapter.minecraft_version,
                    loader=adapter.loader,
                    mappings=adapter.yarn_mappings,
                    limit=4,
                )
                corrections.append(correction.to_dict())
                has_hits = has_hits or bool(correction.hits)
            has_hits = has_hits or bool(primary.hits)
            query_results.append(
                {
                    "query_sha256": _sha256(query),
                    "strategy": (
                        "single"
                        if not primary.correction_required
                        else "corrective_multi_hop"
                    ),
                    "primary": primary.to_dict(),
                    "corrections": corrections,
                }
            )
        if not has_hits:
            unresolved.append(domain.domain_id)
        results.append(
            {
                "domain_id": domain.domain_id,
                "strategy": "adaptive_per_query",
                "queries": query_results,
            }
        )

    target_payload = (
        {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        }
        if adapter is not None
        else None
    )
    payload = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "target": target_payload,
        "domains": results,
        "deferred_official_domains": deferred,
        "unresolved_official_domains": unresolved,
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = _sha256(canonical_json(payload))
    return payload


def retrieve_domain_evidence(
    research_brief: dict[str, Any],
    *,
    retrieve: Callable[..., RetrievalReceipt] = retrieve_official_evidence,
) -> dict[str, Any]:
    from .parallel_runtime_contract import retrieve_domain_evidence as parallel_retrieve

    return parallel_retrieve(research_brief, retrieve=retrieve)


retrieve_domain_evidence._mmm_parallel_rag = True  # type: ignore[attr-defined]


def external_discovery_routes(
    research_brief: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    routes: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_domain in research_brief.get("domains", []):
        domain = _research_domain(raw_domain)
        for provider in domain.providers:
            if provider not in _EXTERNAL_PROVIDERS:
                continue
            target_profile = _external_target_profile(domain, provider)
            for query in domain.queries:
                key = (provider, query, target_profile)
                if key in seen:
                    continue
                seen.add(key)
                routes.append(
                    {
                        "domain_id": domain.domain_id,
                        "provider": provider,
                        "query": query,
                        "target_profile": target_profile,
                    }
                )
    return tuple(routes)


def _external_target_profile(domain: ResearchDomain, provider: str) -> str:
    kinds = set(domain.evidence_kinds)
    ai = {
        "ai_inference",
        "agent_tool_use",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "latency_budget",
    }
    if provider in {"wikipedia", "openalex_works", "crossref_works"}:
        return "general_reference"
    if kinds & ai:
        return "ai_runtime"
    return "minecraft_mod"


def _research_domain(value: Any) -> ResearchDomain:
    if not isinstance(value, dict):
        value = {"domain_id": "mk_custom", "purpose": "Custom mod features"}
    raw_domain_id = str(value.get("domain_id") or "mk_custom").strip()
    domain_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_domain_id).strip("_") or "mk_custom"

    raw_evidence = value.get("evidence_kinds")
    if isinstance(raw_evidence, list):
        unknown_kinds = sorted(set(raw_evidence) - _ALLOWED_EVIDENCE_KINDS)
        if unknown_kinds:
            raise SpecValidationError(
                f"Research domain {domain_id} has unknown evidence kinds: {unknown_kinds}"
            )
        evidence_kinds = [str(kind) for kind in raw_evidence]
    else:
        evidence_kinds = ["official_docs"]
    if not evidence_kinds:
        evidence_kinds = ["official_docs"]

    raw_providers = value.get("providers")
    if isinstance(raw_providers, list):
        unknown_providers = sorted(set(raw_providers) - _ALLOWED_PROVIDERS)
        if unknown_providers:
            raise SpecValidationError(
                f"Research domain {domain_id} has unknown providers: {unknown_providers}"
            )
        providers = [str(provider) for provider in raw_providers]
    else:
        providers = ["official_docs"]
    if not providers:
        providers = ["official_docs"]

    raw_queries = value.get("queries")
    if isinstance(raw_queries, list):
        source_queries = [str(query).strip() for query in raw_queries if str(query).strip()]
    elif isinstance(raw_queries, str) and raw_queries.strip():
        source_queries = [raw_queries.strip()]
    else:
        source_queries = [domain_id]
    queries = tuple(
        page
        for query in source_queries
        for page in _lossless_query_pages(query, _MAX_QUERY_BYTES)
        if page.strip()
    )

    raw_reqs = value.get("requirements", [])
    if isinstance(raw_reqs, list):
        requirements = tuple(str(item).strip() for item in raw_reqs if str(item).strip())
    elif isinstance(raw_reqs, str) and raw_reqs.strip():
        requirements = (raw_reqs.strip(),)
    else:
        requirements = ("Ensure correct domain APIs",)

    raw_deps = value.get("depends_on", [])
    dependencies = (
        tuple(str(item).strip() for item in raw_deps if str(item).strip())
        if isinstance(raw_deps, list)
        else ()
    )
    return ResearchDomain(
        domain_id=domain_id,
        objective=str(
            value.get("objective")
            or value.get("purpose")
            or f"Research {domain_id}"
        ).strip(),
        requirements=requirements,
        evidence_kinds=tuple(evidence_kinds),
        queries=queries,
        providers=tuple(providers),
        depends_on=dependencies,
    )


def _fallback_domains(
    prompt: str,
    game_design: dict[str, Any],
) -> tuple[ResearchDomain, ...]:
    entries: list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = [
        (
            "request",
            "Understand the requested game without assuming a genre template.",
            prompt,
            (
                "gameplay_reference",
                "minecraft_api",
                "dependency",
                "compatibility",
                "license",
            ),
            ("official_docs", "project_rag", "modrinth", "github", "wikipedia"),
        )
    ]

    media = _requested_media_routes(prompt, game_design)
    for index, statement in enumerate(media["visual"]):
        entries.append(
            (
                "requested_visual" if index == 0 else f"requested_visual_{index + 1}",
                "Resolve only the visual media explicitly requested by the design.",
                "Requested visual direction: " + statement,
                _visual_evidence_kinds(statement),
                ("project_rag", "github", "openverse_images"),
            )
        )

    for index, value in enumerate(game_design.get("core_loop", [])):
        entries.append(
            (
                f"loop_{index + 1}",
                "Resolve one requested player-loop capability.",
                str(value),
                ("minecraft_api", "runtime_behavior", "testing"),
                ("official_docs", "project_rag", "modrinth", "github"),
            )
        )
    for index, value in enumerate(game_design.get("progression", [])):
        entries.append(
            (
                f"progression_{index + 1}",
                "Resolve one requested progression capability.",
                str(value),
                ("minecraft_api", "runtime_behavior", "testing"),
                ("official_docs", "project_rag", "modrinth", "github"),
            )
        )

    design_index = 0
    for section in ("combat", "mod_context"):
        for path, value in _leaf_strings(game_design.get(section), section):
            design_index += 1
            entries.append(
                (
                    f"design_{design_index}",
                    "Resolve one request-derived mod behavior or integration requirement.",
                    f"{path}: {value}",
                    ("minecraft_api", "runtime_behavior", "testing"),
                    ("official_docs", "project_rag", "modrinth", "github"),
                )
            )

    for index, value in enumerate(game_design.get("modules", [])):
        if not isinstance(value, dict):
            continue
        statement = str(value.get("reason") or value.get("plugin_id") or "")
        entries.append(
            (
                f"system_{index + 1}",
                "Resolve one requested software or gameplay system.",
                statement,
                (
                    "minecraft_api",
                    "dependency",
                    "source_code",
                    "compatibility",
                    "license",
                    "testing",
                ),
                ("official_docs", "project_rag", "modrinth", "github"),
            )
        )

    for index, value in enumerate(game_design.get("assets", [])):
        if not isinstance(value, dict):
            continue
        statement = str(value.get("brief") or value.get("id") or "")
        asset_kind = str(value.get("kind") or "")
        entries.append(
            (
                f"visual_{index + 1}",
                "Resolve one requested visual, model or interface family.",
                statement,
                _visual_evidence_kinds(statement, asset_kind=asset_kind),
                ("project_rag", "github", "openverse_images"),
            )
        )

    for index, value in enumerate(game_design.get("acceptance_tests", [])):
        entries.append(
            (
                f"quality_{index + 1}",
                "Find evidence and a reproducible test for an observable requirement.",
                str(value),
                ("testing", "runtime_behavior", "performance"),
                ("official_docs", "project_rag", "runtime"),
            )
        )

    domains: list[ResearchDomain] = []
    seen_statements: set[str] = set()
    for domain_id, objective, statement, kinds, providers in entries:
        statement = statement.strip()
        if not statement or statement in seen_statements:
            continue
        seen_statements.add(statement)
        domains.append(
            ResearchDomain(
                domain_id=domain_id,
                objective=objective,
                requirements=(statement,),
                evidence_kinds=kinds,
                queries=_query_chunks(statement),
                providers=providers,
                depends_on=() if domain_id == "request" else ("request",),
            )
        )
    return tuple(domains)


def _requested_media_routes(
    prompt: str,
    game_design: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    sources = [prompt.strip()]
    sources.extend(
        value
        for path, value in _leaf_strings(game_design, "game_design")
        if not path.startswith("game_design.assets[")
    )
    visual = tuple(
        dict.fromkeys(
            source
            for source in sources
            if source and _contains_requested_media(source, family="visual")
        )
    )
    return {"visual": visual}


def _contains_requested_media(text: str, *, family: str) -> bool:
    folded = text.casefold()
    terms = {
        "visual": (
            "visual",
            "texture",
            "sprite",
            "image",
            "icon",
            "pixel art",
            "art style",
            "3d model",
            "3d-model",
            "animation",
            "shader",
            "particle",
            "render",
            "gui",
            "ui",
            "시각",
            "텍스처",
            "이미지",
            "아이콘",
            "픽셀 아트",
            "아트 스타일",
            "3d 모델",
            "3d모델",
            "애니메이션",
            "셰이더",
            "파티클",
        )
    }[family]
    return any(_contains_term(folded, term) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    if term.isascii() and re.fullmatch(r"[a-z0-9_]+", term):
        return bool(
            re.search(
                rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])",
                text,
            )
        )
    return term in text


def _visual_evidence_kinds(
    statement: str,
    *,
    asset_kind: str = "",
) -> tuple[str, ...]:
    folded = statement.casefold()
    kind = asset_kind.casefold().strip()
    evidence: list[str] = ["visual_reference"]
    if kind in {"item", "block", "gui", "environment"} or any(
        _contains_term(folded, term)
        for term in (
            "texture",
            "sprite",
            "image",
            "icon",
            "pixel art",
            "텍스처",
            "이미지",
            "아이콘",
            "픽셀 아트",
        )
    ):
        evidence.append("texture")
    if kind == "entity" or any(
        _contains_term(folded, term)
        for term in ("model", "3d", "mesh", "모델", "메시", "3차원")
    ):
        evidence.append("model_3d")
    if any(
        _contains_term(folded, term)
        for term in ("animation", "animated", "animate", "애니메이션", "움직임")
    ):
        evidence.append("animation")
    evidence.append("license")
    return tuple(dict.fromkeys(evidence))


def _validate_domain_graph(domains: tuple[ResearchDomain, ...]) -> None:
    if not domains:
        raise SpecValidationError("Central research brief must contain a domain.")
    ids = [domain.domain_id for domain in domains]
    if len(ids) != len(set(ids)):
        raise SpecValidationError("Central research domain IDs must be unique.")
    known = set(ids)
    for domain in domains:
        unknown = set(domain.depends_on) - known
        if unknown:
            raise SpecValidationError(
                f"Research domain {domain.domain_id} has unknown dependencies: {sorted(unknown)}"
            )
        if domain.domain_id in domain.depends_on:
            raise SpecValidationError(
                f"Research domain {domain.domain_id} depends on itself."
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {domain.domain_id: domain for domain in domains}

    def visit(domain_id: str) -> None:
        if domain_id in visited:
            return
        if domain_id in visiting:
            raise SpecValidationError("Central research domain graph has a cycle.")
        visiting.add(domain_id)
        for dependency in by_id[domain_id].depends_on:
            visit(dependency)
        visiting.remove(domain_id)
        visited.add(domain_id)

    for domain_id in ids:
        visit(domain_id)


def _augment_domain_routes(domain: ResearchDomain) -> ResearchDomain:
    required: list[str] = []
    kinds = set(domain.evidence_kinds)
    if kinds & {
        "minecraft_api",
        "dependency",
        "compatibility",
        "testing",
        "release",
        "runtime_behavior",
        "performance",
    }:
        required.append("official_docs")
    if "local_project" in kinds or "source_code" in kinds:
        required.append("project_rag")
    if "dependency" in kinds:
        required.extend(("modrinth", "curseforge", "github"))
    if "source_code" in kinds:
        required.append("github")
    if "gameplay_reference" in kinds:
        required.append("wikipedia")
    if kinds & {"visual_reference", "texture", "model_3d", "animation"}:
        required.append("openverse_images")
    if kinds & {"model_3d", "animation"}:
        required.extend(("github", "blockbench"))
    if kinds & {"runtime_behavior", "performance"}:
        required.append("runtime")
    providers = tuple(dict.fromkeys((*domain.providers, *required)))
    return ResearchDomain(
        domain_id=domain.domain_id,
        objective=domain.objective,
        requirements=domain.requirements,
        evidence_kinds=domain.evidence_kinds,
        queries=domain.queries,
        providers=providers,
        depends_on=domain.depends_on,
    )


def _leaf_strings(value: Any, root: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    stack: list[tuple[str, Any]] = [(root, value)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, str) and current.strip():
            result.append((path, current.strip()))
        elif isinstance(current, dict):
            for key in reversed(list(current)):
                stack.append((f"{path}.{key}", current[key]))
        elif isinstance(current, list):
            for index in range(len(current) - 1, -1, -1):
                stack.append((f"{path}[{index}]", current[index]))
    return tuple(result)


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "possibly empty " if allow_empty else ""
        raise SpecValidationError(f"{field} must be a {qualifier}list.")
    result: list[str] = []
    for item in value:
        text = _text(item, field)
        if text not in result:
            result.append(text)
    return tuple(result)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _lossless_query_pages(statement: str, max_bytes: int) -> tuple[str, ...]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if len(statement.encode("utf-8")) <= max_bytes:
        return (statement,)
    pages: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in statement:
        size = len(character.encode("utf-8"))
        if current and current_bytes + size > max_bytes:
            minimum = max_bytes // 2
            consumed = 0
            boundary = 0
            for index, value in enumerate(current, start=1):
                consumed += len(value.encode("utf-8"))
                if consumed >= minimum and value in _QUERY_BOUNDARIES:
                    boundary = index
            cut = boundary or len(current)
            pages.append("".join(current[:cut]))
            current = current[cut:]
            current_bytes = len("".join(current).encode("utf-8"))
            if current and current_bytes + size > max_bytes:
                pages.append("".join(current))
                current = []
                current_bytes = 0
        current.append(character)
        current_bytes += size
    if current:
        pages.append("".join(current))
    if "".join(pages) != statement:
        raise RuntimeError("Research query paging changed source text.")
    if any(len(page.encode("utf-8")) > max_bytes for page in pages):
        raise RuntimeError("Research query page exceeded its byte budget.")
    return tuple(pages)


def _query_chunks(statement: str) -> tuple[str, ...]:
    suffix = " Minecraft Java mod implementation dependencies assets license tests"
    budget = _MAX_QUERY_BYTES - len(suffix.encode("utf-8"))
    if budget <= 0:
        raise RuntimeError("Central research query suffix exceeds its byte policy.")
    return tuple(
        page.strip() + suffix
        for page in _lossless_query_pages(statement, budget)
        if page.strip()
    )


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ResearchDomain",
    "external_discovery_routes",
    "normalize_research_brief",
    "retrieve_domain_evidence",
]
