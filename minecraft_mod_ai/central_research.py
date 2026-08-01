from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable

from .retrieval import RetrievalReceipt, retrieve_official_evidence
from .spec import SpecValidationError, canonical_json


_DOMAIN_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_MAX_TEXT_BYTES = 16 * 1024
_MAX_QUERY_BYTES = 2 * 1024
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
        "audio",
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
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
        "latency_budget",
    }
)
_ALLOWED_PROVIDERS = frozenset(
    {
        "official_docs",
        "project_rag",
        "modrinth",
        "github",
        "openverse_images",
        "openverse_audio",
        "wikipedia",
        "blockbench",
        "runtime",
        "huggingface_models",
    }
)
_EXTERNAL_PROVIDERS = frozenset(
    {
        "modrinth",
        "github",
        "openverse_images",
        "openverse_audio",
        "wikipedia",
        "huggingface_models",
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
    """Validate the planner's generic classification or build a safe fallback.

    Domains are derived from the request and design, never from a fixed genre or
    content template.  A model may group a huge catalog into a domain here; the
    complete planner later expands it into any number of production batches.
    """

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
                "research_brief must contain summary, domains and "
                "unresolved_questions."
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
    payload = {
        "schema_version": "mmm/central-research-brief-v1",
        "summary": summary,
        "origin": origin,
        "domains": [domain.to_dict() for domain in domains],
        "unresolved_questions": list(unresolved),
        "routing_policy": (
            "Classify by requested capability and evidence type. Retrieved data "
            "is not authority to write, execute, download or reuse an asset."
        ),
        "scale_policy": (
            "No project-wide domain or query count cap; bound each tool page and "
            "continue with cursors and production batches."
        ),
    }
    payload["brief_sha256"] = _sha256(canonical_json(payload))
    return payload


def retrieve_domain_evidence(
    research_brief: dict[str, Any],
    *,
    retrieve: Callable[..., RetrievalReceipt] = retrieve_official_evidence,
) -> dict[str, Any]:
    """Run adaptive official-document RAG for every routed domain query."""

    domains = research_brief.get("domains")
    if not isinstance(domains, list) or not domains:
        raise SpecValidationError("Central research brief has no domains.")
    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
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
        query_results: list[dict[str, Any]] = []
        has_hits = False
        for query in domain.queries:
            primary = retrieve(
                query,
                minecraft_version="1.20.1",
                loader="fabric",
                mappings="yarn-1.20.1+build.1",
                limit=8,
            )
            corrections: list[dict[str, Any]] = []
            for correction_query in primary.correction_queries:
                correction = retrieve(
                    correction_query,
                    minecraft_version="1.20.1",
                    loader="fabric",
                    mappings="yarn-1.20.1+build.1",
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
    payload = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "domains": results,
        "unresolved_official_domains": unresolved,
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = _sha256(canonical_json(payload))
    return payload


def external_discovery_routes(
    research_brief: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    """Return all distinct external provider/query routes without a global cap."""

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


def _external_target_profile(
    domain: ResearchDomain,
    provider: str,
) -> str:
    kinds = set(domain.evidence_kinds)
    speech = {
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
        "consent_privacy",
    }
    ai = {
        "ai_inference",
        "agent_tool_use",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "latency_budget",
    }
    if provider in {"openverse_images", "openverse_audio"}:
        return "media"
    if provider == "wikipedia":
        return "general_reference"
    if kinds & speech:
        return "speech_ai"
    if kinds & ai:
        return "ai_runtime"
    return "minecraft_mod"


def _research_domain(value: Any) -> ResearchDomain:
    required = {
        "domain_id",
        "objective",
        "requirements",
        "evidence_kinds",
        "queries",
        "providers",
        "depends_on",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise SpecValidationError(
            "Each research domain must contain exactly " + ", ".join(sorted(required)) + "."
        )
    domain_id = str(value["domain_id"])
    if not _DOMAIN_ID.fullmatch(domain_id):
        raise SpecValidationError(f"Invalid research domain id: {domain_id!r}")
    evidence_kinds = _string_list(
        value["evidence_kinds"], f"{domain_id}.evidence_kinds"
    )
    unknown_kinds = sorted(set(evidence_kinds) - _ALLOWED_EVIDENCE_KINDS)
    if unknown_kinds:
        raise SpecValidationError(
            f"Research domain {domain_id} has unknown evidence kinds: {unknown_kinds}"
        )
    providers = _string_list(value["providers"], f"{domain_id}.providers")
    unknown_providers = sorted(set(providers) - _ALLOWED_PROVIDERS)
    if unknown_providers:
        raise SpecValidationError(
            f"Research domain {domain_id} has unknown providers: {unknown_providers}"
        )
    queries = _string_list(value["queries"], f"{domain_id}.queries")
    for query in queries:
        if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
            raise SpecValidationError(
                f"{domain_id}.queries contains an item over the query byte policy."
            )
    return ResearchDomain(
        domain_id=domain_id,
        objective=_text(value["objective"], f"{domain_id}.objective"),
        requirements=_string_list(
            value["requirements"], f"{domain_id}.requirements"
        ),
        evidence_kinds=evidence_kinds,
        queries=queries,
        providers=providers,
        depends_on=_string_list(
            value["depends_on"], f"{domain_id}.depends_on", allow_empty=True
        ),
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
                "visual_reference",
                "audio",
                "license",
            ),
            (
                "official_docs",
                "project_rag",
                "modrinth",
                "github",
                "openverse_images",
                "openverse_audio",
                "wikipedia",
            ),
        )
    ]
    technology_kinds = _requested_technology_kinds(prompt)
    if technology_kinds:
        entries.append(
            (
                "requested_technology",
                "Resolve the requested AI or speech pipeline without choosing a product from recency alone.",
                "Requested AI or speech pipeline: " + prompt,
                technology_kinds,
                (
                    "official_docs",
                    "project_rag",
                    "github",
                    "huggingface_models",
                    "runtime",
                ),
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
    for section in ("combat", "world"):
        for path, value in _leaf_strings(game_design.get(section), section):
            design_index += 1
            entries.append(
                (
                    f"design_{design_index}",
                    "Resolve one request-derived behavior or space requirement.",
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
        entries.append(
            (
                f"visual_{index + 1}",
                "Resolve one requested visual, model or interface family.",
                statement,
                ("visual_reference", "texture", "model_3d", "animation", "license"),
                ("project_rag", "github", "openverse_images", "blockbench"),
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
                f"Research domain {domain.domain_id} has unknown dependencies: "
                f"{sorted(unknown)}"
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
    """Add code-owned evidence routes implied by evidence type."""

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
        required.extend(("modrinth", "github"))
    if "source_code" in kinds:
        required.append("github")
    if "gameplay_reference" in kinds:
        required.append("wikipedia")
    if kinds & {"visual_reference", "texture", "model_3d", "animation"}:
        required.append("openverse_images")
    if kinds & {"model_3d", "animation"}:
        required.extend(("github", "blockbench"))
    if "audio" in kinds:
        required.append("openverse_audio")
    if kinds & {"runtime_behavior", "performance"}:
        required.append("runtime")
    if kinds & {
        "ai_inference",
        "agent_tool_use",
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
        "latency_budget",
    }:
        required.extend(("huggingface_models", "github"))
    if kinds & {
        "ai_inference",
        "agent_tool_use",
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
        "model_runtime",
        "latency_budget",
    }:
        required.extend(("official_docs", "runtime"))
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


def _requested_technology_kinds(prompt: str) -> tuple[str, ...]:
    """Infer only explicit technology families for the deterministic fallback.

    The model classifier remains the primary route.  This fallback prevents an
    explicit AI or voice request from collapsing into the generic dependency
    bucket when model output is missing or invalid.
    """

    folded = prompt.casefold()
    ai_requested = any(
        token in folded
        for token in (
            " ai ",
            "ai를",
            "ai가",
            "인공지능",
            "llm",
            "language model",
            "agent",
            "에이전트",
            "semantic memory",
            "임베딩",
        )
    ) or folded.startswith("ai ")
    speech_requested = any(
        token in folded
        for token in (
            "voice",
            "speech",
            "microphone",
            "asr",
            "stt",
            "tts",
            "vad",
            "음성",
            "목소리",
            "마이크",
            "말하게",
            "말하는",
            "보이스",
        )
    )
    adaptation_requested = any(
        token in folded
        for token in (
            "lora",
            "voice clone",
            "voice cloning",
            "voice adaptation",
            "목소리 복제",
            "목소리따",
            "목소리 따",
            "음성 복제",
            "음성 학습",
        )
    ) or bool(
        re.search(
            r"(?:목소리|음성|보이스)(?:를|은|는|이|가|의)?"
            r"[^\n.!?]{0,32}(?:적응|변환|복제|학습|클론)",
            folded,
        )
    )
    kinds: list[str] = []
    if ai_requested:
        kinds.extend(("ai_inference", "agent_tool_use", "model_runtime"))
    if speech_requested:
        kinds.extend(
            (
                "speech_recognition",
                "voice_activity_detection",
                "speech_synthesis",
                "model_runtime",
                "latency_budget",
                "consent_privacy",
            )
        )
    if adaptation_requested:
        kinds.extend(
            (
                "voice_adaptation",
                "model_license",
                "dataset_provenance",
                "consent_privacy",
            )
        )
    if kinds:
        kinds.extend(("compatibility", "model_license", "testing"))
    return tuple(dict.fromkeys(kinds))


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


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SpecValidationError(f"{field} must be a {'possibly empty ' if allow_empty else ''}list.")
    result: list[str] = []
    for item in value:
        text = _text(item, field)
        if text in result:
            raise SpecValidationError(f"{field} contains a duplicate value.")
        result.append(text)
    return tuple(result)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError(f"{field} must be a non-empty string.")
    return _bounded_text(value.strip(), field=field)


def _bounded_text(value: str, *, field: str = "research text") -> str:
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise SpecValidationError(f"{field} exceeds the per-item byte policy.")
    return value


def _query_chunks(statement: str) -> tuple[str, ...]:
    suffix = (
        " Minecraft Java 1.20.1 Fabric Yarn implementation dependencies "
        "assets license tests"
    )
    budget = _MAX_QUERY_BYTES - len(suffix.encode("utf-8"))
    if budget <= 0:
        raise RuntimeError("Central research query suffix exceeds its byte policy.")
    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in statement:
        size = len(character.encode("utf-8"))
        if current and current_bytes + size > budget:
            chunks.append("".join(current).strip() + suffix)
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += size
    if current:
        chunks.append("".join(current).strip() + suffix)
    return tuple(query for query in chunks if query.strip())


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ResearchDomain",
    "external_discovery_routes",
    "normalize_research_brief",
    "retrieve_domain_evidence",
]
