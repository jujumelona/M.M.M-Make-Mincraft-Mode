from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from .spec import PlatformLock, SpecValidationError, canonical_json


_MAX_TEXT_BYTES = 16 * 1024
_MAX_PAGE_SIZE = 100
_CURSOR = re.compile(r"^technology:(\d+):([0-9a-f]{16})$")
_REVISION = re.compile(r"^(?:sha256:)?[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RECEIPT_MAC = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_REQUIREMENT_ID = re.compile(r"^[a-z][a-z0-9_]{1,127}$")

TOPOLOGIES = (
    "in_process_java",
    "local_sidecar",
    "remote_api",
    "offline_build_tool",
)

VOICE_CAPABILITIES = (
    "speech_recognition",
    "voice_activity_detection",
    "speech_synthesis",
    "voice_transport",
    "language_intersection",
)

_CAPABILITY_KINDS = frozenset(
    {
        "ai_inference",
        "agent_tool_use",
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "voice_transport",
        "language_intersection",
        "translation",
    }
)
_MODEL_CAPABILITIES = frozenset(
    {
        "ai_inference",
        "agent_tool_use",
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
    }
)
_VOICE_MODEL_CAPABILITIES = frozenset(
    {
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
    }
)
_SAFE_MODEL_FORMATS = frozenset(
    {
        "safetensors",
        "gguf",
        "onnx",
        "tflite",
        "openvino_ir",
    }
)
_UNSAFE_MODEL_FORMATS = frozenset(
    {
        "pickle",
        "pkl",
        "pt",
        "pth",
        "ckpt",
        "bin_pickle",
    }
)
_UNKNOWN_LICENSES = frozenset(
    {"", "unknown", "none", "noassertion", "unlicensed", "other"}
)
_OFFICIAL_TARGET_HOSTS = frozenset(
    {
        "fabricmc.net",
        "docs.fabricmc.net",
        "maven.fabricmc.net",
        "meta.fabricmc.net",
        "minecraft.net",
        "www.minecraft.net",
        "help.minecraft.net",
        "openjdk.org",
        "www.oracle.com",
        "docs.oracle.com",
    }
)

_EVIDENCE_TO_CAPABILITY = {
    "ai_inference": "ai_inference",
    "agent_tool_use": "agent_tool_use",
    "speech_recognition": "speech_recognition",
    "voice_activity_detection": "voice_activity_detection",
    "speech_synthesis": "speech_synthesis",
    "voice_adaptation": "voice_adaptation",
    "voice_conversion": "voice_conversion",
    "translation": "translation",
}

_AI_CUES = (
    r"(?<![a-z0-9])ai(?![a-z0-9])",
    r"(?<![a-z0-9])llm(?![a-z0-9])",
    r"artificial intelligence",
    r"language model",
    r"machine learning",
    r"generative",
    r"inference",
    r"인공지능",
    r"에이아이",
    r"언어\s*모델",
    r"생성형",
    r"추론\s*모델",
)
_AGENT_CUES = (
    r"\bagent(?:ic)?\b",
    r"tool[ -]?use",
    r"function[ -]?call",
    r"(?<![a-z0-9])mcp(?![a-z0-9])",
    r"에이전트",
    r"도구\s*사용",
    r"함수\s*호출",
)
_VOICE_CUES = (
    r"\bvoice\b",
    r"\bspeech\b",
    r"\bmicrophone\b",
    r"(?<![a-z0-9])asr(?![a-z0-9])",
    r"(?<![a-z0-9])stt(?![a-z0-9])",
    r"(?<![a-z0-9])tts(?![a-z0-9])",
    r"(?<![a-z0-9])vad(?![a-z0-9])",
    r"speech[ -]?to[ -]?text",
    r"text[ -]?to[ -]?speech",
    r"voice chat",
    r"음성",
    r"목소리",
    r"마이크",
    r"발화",
    r"말로\s*(?:조작|대화|명령)",
)
_ADAPTATION_CUES = (
    r"voice[ -]?(?:clone|cloning|adaptation|conversion)",
    r"speaker[ -]?adaptation",
    r"(?<![a-z0-9])lora(?![a-z0-9])",
    r"fine[ -]?tun(?:e|ing)",
    r"목소리(?:를|은|는|이|가|의)?[^\n.!?]{0,32}(?:복제|클론|따|학습|적응|변환)",
    r"화자(?:를|은|는|이|가|의)?\s*(?:적응|학습|변환)",
    r"보이스(?:를|는|의)?\s*(?:복제|클론|적응|변환)",
    r"로라",
)
_TRANSLATION_CUES = (
    r"translat(?:e|ion)",
    r"interpret(?:er|ation)",
    r"번역",
    r"통역",
)
_OFFLINE_CUES = (
    r"\boffline\b",
    r"local[ -]?only",
    r"air[ -]?gapped",
    r"오프라인",
    r"로컬만",
    r"인터넷\s*(?:없이|차단)",
)
_CPU_ONLY_CUES = (r"cpu[ -]?only", r"cpu만", r"gpu\s*(?:없이|없음)")
_REALTIME_CUES = (
    r"real[ -]?time",
    r"streaming",
    r"실시간",
    r"스트리밍",
)


@dataclass(frozen=True)
class TechnologyTarget:
    edition: str = ""
    minecraft_version: str = ""
    loader: str = ""
    mappings: str = ""
    java_version: str = ""
    fabric_loader: str = ""
    fabric_api: str = ""

    def validate(self) -> None:
        required = (
            self.edition, self.minecraft_version, self.loader, self.mappings,
            self.java_version, self.fabric_loader, self.fabric_api,
        )
        if not all(str(value).strip() for value in required):
            raise SpecValidationError(
                "Technology assessment requires an explicit executable platform target."
            )

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TechniqueRequirement:
    requirement_id: str
    domain_id: str
    capability_kind: str
    objective: str
    target: TechnologyTarget
    allowed_topologies: tuple[str, ...]
    authority: Mapping[str, str]
    hardware: Mapping[str, Any]
    latency: Mapping[str, Any]
    privacy: Mapping[str, Any]
    offline_required: bool
    required_gates: tuple[str, ...]
    required_tests: tuple[str, ...]
    deterministic_fallback: str
    research_queries: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/technique-requirement-v1",
            "requirement_id": self.requirement_id,
            "domain_id": self.domain_id,
            "capability_kind": self.capability_kind,
            "objective": self.objective,
            "target": self.target.to_dict(),
            "allowed_topologies": list(self.allowed_topologies),
            "authority": dict(self.authority),
            "hardware": dict(self.hardware),
            "latency": dict(self.latency),
            "privacy": dict(self.privacy),
            "offline_required": self.offline_required,
            "required_gates": list(self.required_gates),
            "required_tests": list(self.required_tests),
            "deterministic_fallback": self.deterministic_fallback,
            "research_queries": list(self.research_queries),
        }


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_technology_radar(
    prompt: str,
    research_brief: Mapping[str, Any] | None = None,
    *,
    page_size: int = 50,
    cursor: str = "",
    target: TechnologyTarget | Mapping[str, Any] | PlatformLock | None = None,
) -> dict[str, Any]:
    """Classify request-derived AI/voice needs into a paginated technology page.

    The radar contains capability and evidence gates, not an embedded list of
    products or models. Discovery is intentionally rerun for each planning or
    execution session so a newer technique can be evaluated without silently
    making "newest" equivalent to "compatible" or "best".
    """

    prompt = _text(prompt, "prompt")
    if type(page_size) is not int or not 1 <= page_size <= _MAX_PAGE_SIZE:
        raise SpecValidationError(
            f"page_size must be between 1 and {_MAX_PAGE_SIZE}."
        )
    normalized_target = normalize_technology_target(target)
    domains = _research_domains(prompt, research_brief)
    requirements = _derive_requirements(prompt, domains, normalized_target)
    source_payload = {
        "prompt_sha256": _sha256(prompt),
        "research_brief_sha256": _research_brief_sha(research_brief),
        "target": normalized_target.to_dict(),
        "target_evidence_policy": {
            "coordinates_are_declared_constraints": True,
            "official_exact_version_receipt_required": True,
            "current_documentation_requires_target_translation": True,
            "receipt_schema": "mmm/official-target-evidence-v1",
            "authenticated_code_owned_mac_required": True,
            "executed_tests_bind_candidate_snapshot": True,
        },
        "requirement_ids": [item.requirement_id for item in requirements],
    }
    source_sha256 = _sha256(canonical_json(source_payload))
    offset = _decode_cursor(
        cursor,
        source_sha256=source_sha256,
        page_size=page_size,
    )
    if offset > len(requirements):
        raise SpecValidationError("Technology cursor is beyond the result set.")
    page = requirements[offset : offset + page_size]
    next_offset = offset + len(page)
    next_cursor = (
        _encode_cursor(
            next_offset,
            source_sha256=source_sha256,
            page_size=page_size,
        )
        if next_offset < len(requirements)
        else ""
    )
    flags = _request_flags(prompt, domains)
    voice_components = [
        item.capability_kind
        for item in requirements
        if item.capability_kind in {*VOICE_CAPABILITIES, "voice_adaptation", "voice_conversion"}
    ]
    payload: dict[str, Any] = {
        "schema_version": "mmm/technology-radar-page-v1",
        "source_sha256": source_sha256,
        "target": normalized_target.to_dict(),
        "target_evidence_policy": {
            "coordinates_are_declared_constraints": True,
            "official_exact_version_receipt_required": True,
            "current_documentation_requires_target_translation": True,
            "receipt_schema": "mmm/official-target-evidence-v1",
            "authenticated_code_owned_mac_required": True,
            "executed_tests_bind_candidate_snapshot": True,
        },
        "classification": {
            "ai_requested": flags["ai"],
            "agent_tools_requested": flags["agent"],
            "voice_requested": flags["voice"],
            "voice_adaptation_requested": flags["adaptation"],
            "translation_requested": flags["translation"],
            "offline_required": flags["offline"],
            "real_time_requested": flags["real_time"],
        },
        "voice_contract": _voice_contract(
            activated=bool(voice_components),
            adaptation_requested=flags["adaptation"],
            components=voice_components,
        ),
        "requirements": [item.to_dict() for item in page],
        "pagination": {
            "offset": offset,
            "page_size": page_size,
            "returned": len(page),
            "total_requirements": len(requirements),
            "next_cursor": next_cursor,
        },
        "discovery_policy": {
            "catalog": "request-derived queries; no embedded product or model list",
            "refresh": "rerun discovery at each planning or execution session",
            "untrusted_results": True,
            "download_or_execution_authorized": False,
            "selection": (
                "Pass exact compatibility, authority, license, provenance, safe-"
                "format, privacy, benchmark, test and fallback gates first; use "
                "maintenance and recency only as later tie-breakers."
            ),
        },
        "scale_policy": (
            "Only this response page is bounded. Continue next_cursor until empty; "
            "there is no project-wide capability or domain count cap."
        ),
    }
    payload["radar_sha256"] = _sha256(canonical_json(payload))
    return payload


def technology_research_routes(
    radar_page: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return evidence routes for every requirement on one radar page."""

    if radar_page.get("schema_version") != "mmm/technology-radar-page-v1":
        raise SpecValidationError("Unsupported technology radar schema.")
    raw_requirements = radar_page.get("requirements")
    if not isinstance(raw_requirements, list):
        raise SpecValidationError("Technology radar requirements must be a list.")
    routes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_requirements:
        requirement = normalize_technology_requirement(raw)
        providers = ["official_docs", "github", "runtime"]
        if requirement.capability_kind in _MODEL_CAPABILITIES:
            providers.insert(1, "huggingface_models")
        profile = (
            "speech_ai"
            if requirement.capability_kind
            in {*_VOICE_MODEL_CAPABILITIES, "voice_transport", "language_intersection"}
            else "ai_runtime"
        )
        for provider in providers:
            for query in requirement.research_queries:
                key = (requirement.requirement_id, provider, query)
                if key in seen:
                    continue
                seen.add(key)
                routes.append(
                    {
                        "requirement_id": requirement.requirement_id,
                        "domain_id": requirement.domain_id,
                        "capability_kind": requirement.capability_kind,
                        "provider": provider,
                        "target_profile": (
                            "minecraft_mod" if provider == "official_docs" else profile
                        ),
                        "query": query,
                        "query_sha256": _sha256(query),
                        "authorization": "read_only_evidence",
                    }
                )
    return tuple(routes)


def normalize_technology_target(
    value: TechnologyTarget | Mapping[str, Any] | PlatformLock | None,
) -> TechnologyTarget:
    if value is None:
        raise SpecValidationError(
            "Technology analysis requires an explicit host-selected platform target."
        )
    elif isinstance(value, TechnologyTarget):
        target = value
    elif isinstance(value, PlatformLock):
        target = TechnologyTarget(
            edition=value.edition,
            minecraft_version=value.minecraft_version,
            loader=value.loader,
            mappings=value.yarn_mappings,
            java_version=value.java_version,
            fabric_loader=value.fabric_loader,
            fabric_api=value.fabric_api,
        )
    elif isinstance(value, Mapping):
        allowed = set(asdict(TechnologyTarget()))
        unknown = set(value) - allowed
        if unknown:
            raise SpecValidationError(
                f"Unknown technology target fields: {sorted(unknown)}"
            )
        target = TechnologyTarget(
            **{key: str(item) for key, item in value.items()}
        )
    else:
        raise SpecValidationError("Invalid technology target.")
    target.validate()
    return target


def normalize_technology_requirement(value: Any) -> TechniqueRequirement:
    if isinstance(value, TechniqueRequirement):
        value.target.validate()
        return value
    if not isinstance(value, Mapping):
        raise SpecValidationError("Technology requirement must be an object.")
    if value.get("schema_version") != "mmm/technique-requirement-v1":
        raise SpecValidationError("Unsupported technique requirement schema.")
    requirement_id = _text(value.get("requirement_id"), "requirement_id")
    if not _REQUIREMENT_ID.fullmatch(requirement_id):
        raise SpecValidationError("Invalid technology requirement ID.")
    capability_kind = _text(value.get("capability_kind"), "capability_kind")
    if capability_kind not in _CAPABILITY_KINDS:
        raise SpecValidationError(
            f"Unsupported technology capability: {capability_kind!r}"
        )
    topologies = _string_tuple(value.get("allowed_topologies"), "allowed_topologies")
    if set(topologies) - set(TOPOLOGIES):
        raise SpecValidationError("Technology requirement has an invalid topology.")
    return TechniqueRequirement(
        requirement_id=requirement_id,
        domain_id=_text(value.get("domain_id"), "domain_id"),
        capability_kind=capability_kind,
        objective=_text(value.get("objective"), "objective"),
        target=normalize_technology_target(value.get("target")),
        allowed_topologies=topologies,
        authority=_mapping(value.get("authority"), "authority"),
        hardware=_mapping(value.get("hardware"), "hardware"),
        latency=_mapping(value.get("latency"), "latency"),
        privacy=_mapping(value.get("privacy"), "privacy"),
        offline_required=_boolean(value.get("offline_required"), "offline_required"),
        required_gates=_string_tuple(value.get("required_gates"), "required_gates"),
        required_tests=_string_tuple(value.get("required_tests"), "required_tests"),
        deterministic_fallback=_text(
            value.get("deterministic_fallback"), "deterministic_fallback"
        ),
        research_queries=_string_tuple(value.get("research_queries"), "research_queries"),
    )


def normalize_technology_candidate(
    value: Any,
    *,
    expected_capability: str | None = None,
) -> dict[str, Any]:
    """Normalize reviewed evidence without interpreting model-card text as commands."""

    if not isinstance(value, Mapping):
        raise SpecValidationError("Technology candidate must be an object.")
    candidate_id = _text(value.get("candidate_id"), "candidate_id")
    capability = str(value.get("capability_kind") or expected_capability or "").strip()
    if capability not in _CAPABILITY_KINDS:
        raise SpecValidationError("Technology candidate has an invalid capability kind.")
    topology = str(value.get("topology") or "").strip()
    if topology and topology not in TOPOLOGIES:
        raise SpecValidationError("Technology candidate has an invalid topology.")
    licenses = _mapping(value.get("licenses", {}), "licenses")
    return {
        "schema_version": "mmm/technology-candidate-v1",
        "candidate_id": candidate_id,
        "capability_kind": capability,
        "topology": topology,
        "revision": str(value.get("revision") or "").strip().lower(),
        "artifact_sha256": str(value.get("artifact_sha256") or "").strip().lower(),
        "evidence_sha256": str(value.get("evidence_sha256") or "").strip().lower(),
        "formats": list(_optional_string_tuple(value.get("formats"), "formats")),
        "licenses": {
            "code": _normalize_license(licenses.get("code")),
            "model": _normalize_license(licenses.get("model")),
            "data": _normalize_license(licenses.get("data")),
        },
        "dataset_provenance": _mapping(
            value.get("dataset_provenance", {}), "dataset_provenance"
        ),
        "official_target_evidence": _mapping(
            value.get("official_target_evidence", {}),
            "official_target_evidence",
        ),
        "compatibility": _mapping(value.get("compatibility", {}), "compatibility"),
        "authority": _mapping(value.get("authority", {}), "authority"),
        "runtime": _mapping(value.get("runtime", {}), "runtime"),
        "benchmarks": _mapping(value.get("benchmarks", {}), "benchmarks"),
        "privacy": _mapping(value.get("privacy", {}), "privacy"),
        "voice_rights": _mapping(value.get("voice_rights", {}), "voice_rights"),
        "tests": _mapping(value.get("tests", {}), "tests"),
        "fallback": _mapping(value.get("fallback", {}), "fallback"),
        "maintenance": _mapping(value.get("maintenance", {}), "maintenance"),
        "external_text_is_instructions": False,
    }


def assess_technology_candidate(
    requirement: TechniqueRequirement | Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Public fail-closed assessment without caller-selected trust material."""

    return _assess_technology_candidate_with_receipt_key(
        requirement,
        candidate,
        receipt_key=None,
    )


def _assess_technology_candidate_with_receipt_key(
    requirement: TechniqueRequirement | Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    receipt_key: bytes | None = None,
) -> dict[str, Any]:
    """Fail closed on a candidate before it can be selected or installed."""

    requirement = normalize_technology_requirement(requirement)
    normalized = normalize_technology_candidate(
        candidate,
        expected_capability=requirement.capability_kind,
    )
    gates: list[GateResult] = []

    def add(gate_id: str, status: str, reason: str) -> None:
        gates.append(GateResult(gate_id, status, reason))

    if normalized["capability_kind"] == requirement.capability_kind:
        add("capability_match", "pass", "Candidate matches the routed capability.")
    else:
        add("capability_match", "fail", "Candidate is for a different capability.")

    topology = normalized["topology"]
    if not topology:
        add("topology", "unresolved", "Execution topology was not recorded.")
    elif topology in requirement.allowed_topologies:
        add("topology", "pass", "Topology is allowed by the request-derived policy.")
    else:
        add("topology", "fail", "Topology conflicts with the request-derived policy.")

    compatibility = normalized["compatibility"]
    expected = requirement.target.to_dict()
    _assess_official_target_evidence(
        requirement,
        normalized,
        add,
        receipt_key=receipt_key,
    )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if str(compatibility.get(key, "")) != expected_value
    ]
    if mismatches:
        add(
            "exact_minecraft_bridge",
            "fail" if compatibility else "unresolved",
            "Exact target evidence is missing or mismatched for: " + ", ".join(mismatches),
        )
    elif compatibility.get("bridge_verified") is not True:
        add(
            "exact_minecraft_bridge",
            "unresolved",
            "Target fields match, but the Fabric bridge has not been verified.",
        )
    else:
        add(
            "exact_minecraft_bridge",
            "pass",
            f"Bridge is verified for Minecraft {requirement.target.minecraft_version}, {requirement.target.loader}, mappings {requirement.target.mappings} and Java {requirement.target.java_version}.",
        )

    revision = normalized["revision"]
    if _REVISION.fullmatch(revision):
        add("immutable_revision", "pass", "An immutable revision is recorded.")
    else:
        add(
            "immutable_revision",
            "unresolved",
            "A mutable tag or missing revision cannot bind this assessment.",
        )

    evidence_sha = normalized["evidence_sha256"]
    if _SHA256.fullmatch(evidence_sha):
        add("evidence_receipt", "pass", "Evidence is bound to a SHA-256 receipt.")
    else:
        add("evidence_receipt", "unresolved", "Evidence SHA-256 is missing.")

    _assess_licenses(requirement, normalized, add)
    _assess_artifact_format(requirement, normalized, add)
    _assess_dataset_provenance(requirement, normalized, add)
    _assess_authority(requirement, normalized, add)
    _assess_runtime(requirement, normalized, add)
    _assess_privacy(requirement, normalized, add)
    _assess_voice_rights(requirement, normalized, add)
    _assess_tests_and_fallback(
        requirement,
        normalized,
        add,
        receipt_key=receipt_key,
    )

    failures = [gate.gate_id for gate in gates if gate.status == "fail"]
    unresolved = [gate.gate_id for gate in gates if gate.status == "unresolved"]
    status = "blocked" if failures else "needs_evidence" if unresolved else "eligible"
    payload: dict[str, Any] = {
        "schema_version": "mmm/technology-assessment-v1",
        "requirement_id": requirement.requirement_id,
        "candidate_id": normalized["candidate_id"],
        "capability_kind": requirement.capability_kind,
        "status": status,
        "eligible": status == "eligible",
        "blocking_gates": failures,
        "unresolved_gates": unresolved,
        "gates": [gate.to_dict() for gate in gates],
        "candidate": normalized,
        "selection_policy": {
            "latest_is_automatically_best": False,
            "recency_used_only_after_required_gates": True,
            "benchmark_on_declared_target_required": True,
            "auto_download_or_execution": False,
        },
    }
    payload["assessment_sha256"] = _sha256(canonical_json(payload))
    return payload


def assess_technology_compatibility(
    requirement: TechniqueRequirement | Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Public alias; authenticated verification is owned by the MCP service."""

    return assess_technology_candidate(
        requirement,
        candidate,
    )


def _build_signed_official_target_evidence(
    requirement: TechniqueRequirement | Mapping[str, Any],
    *,
    receipt_key: bytes,
) -> dict[str, Any]:
    """Issue an authenticated receipt from the reviewed code-owned target corpus.

    The MAC key remains owned by the calling service. Query text and candidate
    metadata never control these sources or the exact facts attached to them.
    """

    normalized = normalize_technology_requirement(requirement)
    expected = normalized.target.to_dict()

    # Exact coordinates are owned by the already-resolved platform receipt.  The
    # technology gate binds those coordinates to stable official authorities rather
    # than to historical document IDs for one Minecraft release.  This keeps the
    # receipt valid for future provider-supported targets without weakening the MAC
    # or the exact-fact verification performed below.
    source_facts = (
        (
            "fabric-meta-target",
            "https://meta.fabricmc.net/",
            "Fabric official metadata API",
            {
                "edition": expected["edition"],
                "minecraft_version": expected["minecraft_version"],
                "loader": expected["loader"],
                "mappings": expected["mappings"],
            },
        ),
        (
            "fabric-api-maven-target",
            "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/",
            "Fabric official Maven repository",
            {"fabric_api": expected["fabric_api"]},
        ),
        (
            "fabric-loader-maven-target",
            "https://maven.fabricmc.net/net/fabricmc/fabric-loader/",
            "Fabric official Maven repository",
            {"fabric_loader": expected["fabric_loader"]},
        ),
        (
            "java-runtime-target",
            "https://openjdk.org/",
            "OpenJDK official project",
            {"java_version": expected["java_version"]},
        ),
    )
    observed_at = datetime.now().astimezone().isoformat()
    sources: list[dict[str, Any]] = []
    for document_id, source_url, authority, facts in source_facts:
        evidence_record = {
            "source_url": source_url,
            "facts": facts,
            "retrieval_document_id": document_id,
            "authority": authority,
        }
        sources.append(
            {
                **evidence_record,
                "observed_at": observed_at,
                "content_sha256": _sha256(canonical_json(evidence_record)),
                "retrieval_revision": normalized.target.minecraft_version,
                "trust_tier": "official_primary",
            }
        )
    body: dict[str, Any] = {
        "schema_version": "mmm/official-target-evidence-v1",
        "retrieved_by": "mmm_authoritative_retriever",
        "authorization": "read_only_evidence",
        "target": expected,
        "sources": sources,
    }
    return _seal_technology_receipt(body, receipt_key)


def compute_voice_language_intersection(
    asr_languages: Sequence[str],
    tts_languages: Sequence[str],
    translation_pairs: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compute only full ASR-to-output paths; never advertise a component union."""

    asr = _languages(asr_languages, "asr_languages")
    tts = _languages(tts_languages, "tts_languages")
    if translation_pairs:
        paths: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, pair in enumerate(translation_pairs):
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise SpecValidationError(
                    f"translation_pairs[{index}] must contain source and target."
                )
            source = _language(pair[0], f"translation_pairs[{index}].source")
            target = _language(pair[1], f"translation_pairs[{index}].target")
            if source in asr and target in tts and (source, target) not in seen:
                seen.add((source, target))
                paths.append({"input": source, "output": target})
        direct: list[str] = []
    else:
        direct = sorted(set(asr) & set(tts))
        paths = [{"input": language, "output": language} for language in direct]
    payload: dict[str, Any] = {
        "schema_version": "mmm/voice-language-intersection-v1",
        "asr_languages": list(asr),
        "tts_languages": list(tts),
        "direct_languages": direct,
        "full_pipeline_paths": paths,
        "advertise_component_union": False,
    }
    payload["intersection_sha256"] = _sha256(canonical_json(payload))
    return payload


def _research_domains(
    prompt: str,
    research_brief: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    if research_brief is None:
        return (
            {
                "domain_id": "request",
                "objective": prompt,
                "requirements": [prompt],
                "evidence_kinds": [],
                "queries": [prompt],
            },
        )
    if not isinstance(research_brief, Mapping):
        raise SpecValidationError("research_brief must be an object.")
    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise SpecValidationError("research_brief.domains must be a non-empty list.")
    result: list[dict[str, Any]] = []
    seen_domain_ids: set[str] = set()
    for index, raw in enumerate(raw_domains):
        if not isinstance(raw, Mapping):
            raise SpecValidationError(f"research_brief.domains[{index}] must be an object.")
        domain_id = _slug(str(raw.get("domain_id") or f"domain_{index + 1}"), index)
        if domain_id in seen_domain_ids:
            raise SpecValidationError(
                f"Duplicate normalized technology domain ID: {domain_id}"
            )
        seen_domain_ids.add(domain_id)
        objective = str(raw.get("objective") or "").strip()
        requirements = _loose_strings(raw.get("requirements"))
        evidence_kinds = _loose_strings(raw.get("evidence_kinds"))
        queries = _loose_strings(raw.get("queries"))
        statement = objective or (requirements[0] if requirements else prompt)
        result.append(
            {
                "domain_id": domain_id,
                "objective": _bounded_text(statement),
                "requirements": requirements,
                "evidence_kinds": evidence_kinds,
                "queries": queries or [statement],
            }
        )
    return tuple(result)


def _derive_requirements(
    prompt: str,
    domains: Sequence[Mapping[str, Any]],
    target: TechnologyTarget,
) -> tuple[TechniqueRequirement, ...]:
    global_flags = _request_flags(prompt, domains)
    result: list[TechniqueRequirement] = []
    seen: set[tuple[str, str]] = set()
    voice_domain_found = False
    specific_voice_domain = any(
        bool(
            set(str(item) for item in domain.get("evidence_kinds", []))
            & {
                "speech_recognition",
                "voice_activity_detection",
                "speech_synthesis",
                "voice_adaptation",
                "voice_conversion",
            }
        )
        or _matches(_domain_text(domain), _VOICE_CUES)
        or _matches(_domain_text(domain), _ADAPTATION_CUES)
        for domain in domains
    )
    specific_ai_domain = any(
        "ai_inference" in domain.get("evidence_kinds", [])
        or _matches(_domain_text(domain), _AI_CUES)
        for domain in domains
    )
    specific_agent_domain = any(
        "agent_tool_use" in domain.get("evidence_kinds", [])
        or _matches(_domain_text(domain), _AGENT_CUES)
        for domain in domains
    )
    specific_translation_domain = any(
        "translation" in domain.get("evidence_kinds", [])
        or _matches(_domain_text(domain), _TRANSLATION_CUES)
        for domain in domains
    )
    for index, domain in enumerate(domains):
        domain_text = _domain_text(domain)
        kinds = set(str(item) for item in domain.get("evidence_kinds", []))
        requested = {
            mapped
            for kind, mapped in _EVIDENCE_TO_CAPABILITY.items()
            if kind in kinds
        }
        domain_voice = bool(
            kinds
            & {
                "speech_recognition",
                "voice_activity_detection",
                "speech_synthesis",
                "voice_adaptation",
                "voice_conversion",
            }
        ) or _matches(domain_text, _VOICE_CUES) or _matches(domain_text, _ADAPTATION_CUES)
        if (
            not domain_voice
            and global_flags["voice"]
            and not specific_voice_domain
            and not voice_domain_found
        ):
            domain_voice = True
        if domain_voice:
            voice_domain_found = True
            requested.update(VOICE_CAPABILITIES)
            if "voice_adaptation" in kinds or _matches(domain_text, _ADAPTATION_CUES):
                requested.add("voice_adaptation")
            if "voice_conversion" in kinds:
                requested.add("voice_conversion")
        if "translation" in kinds or _matches(domain_text, _TRANSLATION_CUES):
            requested.add("translation")
        if "agent_tool_use" in kinds or _matches(domain_text, _AGENT_CUES):
            requested.add("agent_tool_use")
        if "ai_inference" in kinds or _matches(domain_text, _AI_CUES):
            requested.add("ai_inference")
        if not requested and index == 0:
            if global_flags["agent"] and not specific_agent_domain:
                requested.add("agent_tool_use")
            if global_flags["ai"] and not specific_ai_domain:
                requested.add("ai_inference")
            if global_flags["translation"] and not specific_translation_domain:
                requested.add("translation")
        ordered = sorted(requested, key=_capability_order)
        for capability in ordered:
            key = (str(domain["domain_id"]), capability)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                _make_requirement(
                    domain,
                    capability,
                    target=target,
                    flags=global_flags,
                )
            )
    return tuple(result)


def _make_requirement(
    domain: Mapping[str, Any],
    capability: str,
    *,
    target: TechnologyTarget,
    flags: Mapping[str, bool],
) -> TechniqueRequirement:
    domain_id = str(domain["domain_id"])
    objective = _bounded_text(str(domain.get("objective") or capability))
    topologies = _topologies_for(capability, offline=flags["offline"])
    authority = {
        "capture": "client" if capability in _VOICE_MODEL_CAPABILITIES else "not_applicable",
        "presentation": "client",
        "game_state_mutation": "server_only",
        "client_messages": "schema_validated_and_rate_limited_by_server",
    }
    required_tests = ["unit", "fabric_integration", "performance"]
    if capability in _VOICE_MODEL_CAPABILITIES:
        required_tests.extend(("client_runtime", "audio_quality"))
    if capability in {"speech_recognition", "voice_activity_detection"}:
        required_tests.extend(
            (
                "microphone_permission_and_capture",
                "echo_noise_and_silence_conditions",
                "barge_in_and_cancellation",
            )
        )
    if capability == "speech_synthesis":
        required_tests.extend(
            ("streaming_playback_backpressure", "barge_in_and_cancellation")
        )
    if capability == "voice_transport":
        required_tests.extend(
            (
                "client_runtime",
                "gametest",
                "jitter_packet_loss_and_backpressure",
                "barge_in_and_cancellation",
            )
        )
    if capability == "language_intersection":
        required_tests.append("full_pipeline_language_matrix")
    if capability == "voice_adaptation":
        required_tests.append("consent_revocation_deletion")
    research_parts = list(
        dict.fromkeys(
            item
            for item in [
                objective,
                *[
                    str(value).strip()
                    for value in domain.get("queries", [])
                ],
            ]
            if item
        )
    )
    research_basis = " ".join(research_parts).strip()
    query_prefix = _bounded_text(research_basis, maximum=4096)
    queries = (
        _bounded_text(
            f"{query_prefix} {capability.replace('_', ' ')} model card license provenance benchmark",
            maximum=4096,
        ),
        (
            f"Minecraft {target.minecraft_version} {target.loader} mappings {target.mappings} "
            f"Java {target.java_version} {capability.replace('_', ' ')} integration compatibility testing"
        ),
    )
    return TechniqueRequirement(
        requirement_id=_requirement_id(domain_id, capability),
        domain_id=domain_id,
        capability_kind=capability,
        objective=objective,
        target=target,
        allowed_topologies=topologies,
        authority=authority,
        hardware={
            "benchmark_on_declared_target": True,
            "requested_devices": ["cpu"] if flags["cpu_only"] else ["cpu", "gpu"],
            "record_cpu_gpu_ram_and_startup": True,
            "no_unmeasured_hardware_claims": True,
        },
        latency={
            "real_time_required": flags["real_time"],
            "p95_budget_ms": _latency_budget(" ".join([objective, query_prefix])),
            "measure_p50_p95_and_concurrency": True,
            "measure_real_time_factor": capability in _VOICE_MODEL_CAPABILITIES,
        },
        privacy={
            "raw_input_sensitive": capability in _VOICE_MODEL_CAPABILITIES,
            "remote_transfer_requires_explicit_opt_in": True,
            "record_retention_and_deletion": True,
            "prefer_local_when_request_is_silent": True,
        },
        offline_required=flags["offline"],
        required_gates=(
            "official_target_evidence",
            "exact_minecraft_bridge",
            "immutable_revision",
            "evidence_receipt",
            "licenses",
            "dataset_provenance",
            "safe_artifact_format",
            "authority",
            "target_hardware_benchmark",
            "privacy",
            "tests",
            "deterministic_fallback",
        ),
        required_tests=tuple(dict.fromkeys(required_tests)),
        deterministic_fallback=_fallback_for(capability),
        research_queries=queries,
    )


def _assess_official_target_evidence(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
    *,
    receipt_key: bytes | None,
) -> None:
    evidence = candidate["official_target_evidence"]
    if not evidence:
        add(
            "official_target_evidence",
            "unresolved",
            "A code-retrieved exact-version official evidence receipt is required.",
        )
        return
    if evidence.get("schema_version") != "mmm/official-target-evidence-v1":
        add(
            "official_target_evidence",
            "fail",
            "The official target evidence receipt schema is invalid.",
        )
        return
    if evidence.get("retrieved_by") != "mmm_authoritative_retriever":
        add(
            "official_target_evidence",
            "fail",
            "Target facts must come from the code-owned authoritative retriever.",
        )
        return
    if evidence.get("authorization") != "read_only_evidence":
        add(
            "official_target_evidence",
            "fail",
            "The target receipt does not preserve the read-only evidence boundary.",
        )
        return
    sources = evidence.get("sources")
    if not isinstance(sources, list) or not sources:
        add(
            "official_target_evidence",
            "unresolved",
            "The official target receipt has no source records.",
        )
        return
    if receipt_key is None:
        add(
            "official_target_evidence",
            "unresolved",
            "Authenticated target evidence must be verified by the code-owned service.",
        )
        return
    if not _verify_technology_receipt(evidence, receipt_key):
        add(
            "official_target_evidence",
            "fail",
            "The official target receipt is missing a valid code-owned MAC or its body was changed.",
        )
        return

    expected = requirement.target.to_dict()
    covered: set[str] = set()
    failures: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            failures.append(f"sources[{index}] is not an object")
            continue
        source_url = str(source.get("source_url") or "").strip()
        parsed = urlparse(source_url)
        if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_TARGET_HOSTS:
            failures.append(f"sources[{index}] is not an allowlisted official HTTPS URL")
        if not _valid_observed_at(source.get("observed_at")):
            failures.append(f"sources[{index}] has no timezone-aware observed_at")
        content_sha256 = str(source.get("content_sha256") or "").lower()
        if not _SHA256.fullmatch(content_sha256):
            failures.append(f"sources[{index}] has no content SHA-256")
        facts = source.get("facts")
        if not isinstance(facts, Mapping) or not facts:
            failures.append(f"sources[{index}] has no exact target facts")
            continue
        for key, value in facts.items():
            if key not in expected:
                failures.append(f"sources[{index}] contains an unknown target fact {key}")
            elif str(value) != expected[key]:
                failures.append(f"sources[{index}] mismatches target fact {key}")
            else:
                covered.add(str(key))
    missing = sorted(set(expected) - covered)
    if missing:
        failures.append("missing target facts: " + ", ".join(missing))
    if failures:
        add(
            "official_target_evidence",
            "fail",
            "Official exact-version evidence failed: " + "; ".join(failures),
        )
    else:
        add(
            "official_target_evidence",
            "pass",
            "Official allowlisted sources cover every exact target coordinate and are receipt-bound.",
        )


def _valid_observed_at(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _assess_licenses(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    licenses = candidate["licenses"]
    required = ["code"]
    if requirement.capability_kind in _MODEL_CAPABILITIES:
        required.extend(("model", "data"))
    unresolved: list[str] = []
    denied: list[str] = []
    for kind in required:
        license_value = licenses[kind]
        identifier = str(license_value.get("id", "")).casefold()
        if identifier in _UNKNOWN_LICENSES or license_value.get("reviewed") is not True:
            unresolved.append(kind)
        elif license_value.get("use_allowed") is False:
            denied.append(kind)
        elif license_value.get("use_allowed") is not True:
            unresolved.append(kind)
    if denied:
        add(
            "licenses",
            "fail",
            "Reviewed license terms do not permit the intended use for: "
            + ", ".join(denied),
        )
    elif unresolved:
        add(
            "licenses",
            "unresolved",
            "Reviewed code/model/data license and intended-use evidence is missing for: "
            + ", ".join(unresolved),
        )
    else:
        add("licenses", "pass", "Required license identifiers and reviews are recorded.")


def _assess_artifact_format(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    if requirement.capability_kind not in _MODEL_CAPABILITIES:
        add("safe_artifact_format", "not_applicable", "No model artifact is used.")
        return
    if candidate["topology"] == "remote_api":
        add(
            "safe_artifact_format",
            "not_applicable",
            "Remote service exposes no local weight artifact; API evidence remains required.",
        )
        return
    formats = {str(item).casefold().lstrip(".") for item in candidate["formats"]}
    unsafe = formats & _UNSAFE_MODEL_FORMATS
    safe = formats & _SAFE_MODEL_FORMATS
    unreviewed = formats - _UNSAFE_MODEL_FORMATS - _SAFE_MODEL_FORMATS
    if unsafe:
        add(
            "safe_artifact_format",
            "fail",
            "Candidate includes executable or pickle-like weight formats: "
            + ", ".join(sorted(unsafe)),
        )
    elif not safe:
        add(
            "safe_artifact_format",
            "unresolved",
            "No reviewed safe model format is recorded.",
        )
    elif unreviewed:
        add(
            "safe_artifact_format",
            "unresolved",
            "Unreviewed model artifact formats remain: "
            + ", ".join(sorted(unreviewed)),
        )
    elif not _SHA256.fullmatch(candidate["artifact_sha256"]):
        add(
            "safe_artifact_format",
            "unresolved",
            "Safe format is present but the artifact SHA-256 is missing.",
        )
    else:
        add("safe_artifact_format", "pass", "Safe format and artifact hash are bound.")


def _assess_dataset_provenance(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    if requirement.capability_kind not in _MODEL_CAPABILITIES:
        add("dataset_provenance", "not_applicable", "No trained model is selected.")
        return
    provenance = candidate["dataset_provenance"]
    sources = provenance.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or provenance.get("verified") is not True
    ):
        add(
            "dataset_provenance",
            "unresolved",
            "Dataset sources and a verified provenance review are required.",
        )
    else:
        add("dataset_provenance", "pass", "Dataset provenance is explicitly verified.")


def _assess_authority(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    authority = candidate["authority"]
    if candidate["topology"] == "offline_build_tool":
        if authority.get("generated_output_reviewed") is True:
            add("authority", "pass", "Offline generated output is reviewed before packaging.")
        else:
            add("authority", "unresolved", "Offline generated output review is not recorded.")
        return
    if (
        authority.get("game_state_mutation") == "server_only"
        and authority.get("client_messages_validated") is True
    ):
        add("authority", "pass", "Server authority and client-message validation are enforced.")
    else:
        add(
            "authority",
            "fail" if authority else "unresolved",
            "AI output must not directly grant client authority over game state.",
        )


def _assess_runtime(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    if requirement.capability_kind not in _MODEL_CAPABILITIES:
        add("target_hardware_benchmark", "not_applicable", "No model runtime is selected.")
        return
    benchmark = candidate["benchmarks"]
    measured_on = str(benchmark.get("measured_on") or "").strip()
    p50 = _number_or_none(benchmark.get("p50_latency_ms"))
    p95 = _number_or_none(benchmark.get("p95_latency_ms"))
    memory = _number_or_none(benchmark.get("peak_memory_mb"))
    startup = _number_or_none(benchmark.get("startup_ms"))
    concurrency = _number_or_none(benchmark.get("concurrency"))
    if (
        not measured_on
        or p50 is None
        or p95 is None
        or memory is None
        or startup is None
        or concurrency is None
        or concurrency < 1
    ):
        add(
            "target_hardware_benchmark",
            "unresolved",
            "Target hardware, startup, p50/p95 latency, concurrency and peak "
            "memory measurements are required.",
        )
        return
    budget = requirement.latency.get("p95_budget_ms")
    if isinstance(budget, (int, float)) and p95 > float(budget):
        add("target_hardware_benchmark", "fail", "Measured p95 latency exceeds the request budget.")
        return
    if (
        requirement.latency.get("real_time_required")
        and requirement.capability_kind in _VOICE_MODEL_CAPABILITIES
    ):
        rtf = _number_or_none(benchmark.get("real_time_factor"))
        if rtf is None:
            add("target_hardware_benchmark", "unresolved", "Real-time factor was not measured.")
            return
        if rtf > 1.0:
            add("target_hardware_benchmark", "fail", "Real-time factor is above 1.0.")
            return
    if requirement.hardware.get("requested_devices") == ["cpu"]:
        device = str(candidate["runtime"].get("device") or "").casefold()
        if device != "cpu":
            add("target_hardware_benchmark", "fail", "CPU-only request lacks a CPU benchmark.")
            return
    add("target_hardware_benchmark", "pass", "Candidate was benchmarked on a declared target.")


def _assess_privacy(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    privacy = candidate["privacy"]
    runtime = candidate["runtime"]
    if requirement.offline_required and (
        candidate["topology"] == "remote_api" or runtime.get("supports_offline") is not True
    ):
        add("privacy", "fail", "Candidate cannot satisfy the requested offline boundary.")
        return
    leaves_device = privacy.get("raw_input_leaves_device")
    if leaves_device is True:
        if (
            privacy.get("explicit_transfer_consent") is not True
            or not str(privacy.get("retention_policy") or "").strip()
            or privacy.get("deletion_supported") is not True
        ):
            add(
                "privacy",
                "fail",
                "Remote raw-data transfer lacks consent, retention or deletion controls.",
            )
            return
    elif leaves_device is None and requirement.privacy.get("raw_input_sensitive"):
        add("privacy", "unresolved", "Raw voice-data flow has not been documented.")
        return
    add("privacy", "pass", "Offline and raw-data transfer policy is satisfied.")


def _assess_voice_rights(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
) -> None:
    if requirement.capability_kind not in {"voice_adaptation", "voice_conversion"}:
        add("voice_rights", "not_applicable", "No speaker identity adaptation is requested.")
        return
    rights = candidate["voice_rights"]
    missing = [
        field
        for field in (
            "explicit_consent",
            "authorized_speaker",
            "provenance_verified",
            "revocation_supported",
            "deletion_supported",
        )
        if rights.get(field) is not True
    ]
    if not str(rights.get("provenance") or "").strip():
        missing.append("provenance")
    if missing:
        add(
            "voice_rights",
            "fail",
            "Voice adaptation remains blocked without: " + ", ".join(missing),
        )
    else:
        add(
            "voice_rights",
            "pass",
            "Explicit speaker rights, provenance, revocation and deletion are recorded.",
        )


def _assess_tests_and_fallback(
    requirement: TechniqueRequirement,
    candidate: Mapping[str, Any],
    add: Any,
    *,
    receipt_key: bytes | None,
) -> None:
    tests = candidate["tests"]
    missing_tests = [
        name
        for name in requirement.required_tests
        if not _valid_technology_test_receipt(
            tests.get(name),
            test_id=name,
            requirement_id=requirement.requirement_id,
            candidate=candidate,
            receipt_key=receipt_key,
        )
    ]
    if missing_tests:
        add(
            "tests",
            "unresolved",
            "Required executed test receipts are missing: "
            + ", ".join(missing_tests),
        )
    else:
        add("tests", "pass", "All request-derived test receipts passed.")
    fallback = candidate["fallback"]
    if (
        fallback.get("deterministic") is True
        and str(fallback.get("description") or "").strip()
        and _valid_technology_test_receipt(
            fallback.get("test_receipt"),
            test_id="deterministic_fallback",
            requirement_id=requirement.requirement_id,
            candidate=candidate,
            receipt_key=receipt_key,
        )
    ):
        add(
            "deterministic_fallback",
            "pass",
            "A deterministic fallback has an executed test receipt.",
        )
    else:
        add(
            "deterministic_fallback",
            "unresolved",
            "Selection requires a tested deterministic fallback.",
        )


def _valid_technology_test_receipt(
    value: Any,
    *,
    test_id: str,
    requirement_id: str,
    candidate: Mapping[str, Any],
    receipt_key: bytes | None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema_version") != "mmm/technology-test-receipt-v1":
        return False
    if value.get("executed_by") != "mmm_quality_runner":
        return False
    if value.get("status") != "pass":
        return False
    if value.get("test_id") != test_id:
        return False
    if value.get("requirement_id") != requirement_id:
        return False
    if not _valid_observed_at(value.get("observed_at")):
        return False
    for field in ("environment_sha256", "result_sha256"):
        if not _SHA256.fullmatch(str(value.get(field) or "").lower()):
            return False
    for field in (
        "candidate_id",
        "revision",
        "artifact_sha256",
        "evidence_sha256",
    ):
        if field not in value or str(value.get(field)) != str(candidate.get(field)):
            return False
    if value.get("candidate_snapshot_sha256") != (
        _technology_candidate_snapshot_sha256(candidate)
    ):
        return False
    return _verify_technology_receipt(value, receipt_key)


def _technology_candidate_snapshot_sha256(
    candidate: Mapping[str, Any],
) -> str:
    """Bind test evidence to every assessed field without creating a receipt cycle."""

    normalized = normalize_technology_candidate(candidate)
    snapshot = {
        key: value
        for key, value in normalized.items()
        if key not in {"tests", "fallback"}
    }
    fallback_policy = dict(normalized["fallback"])
    fallback_policy.pop("test_receipt", None)
    snapshot["fallback"] = fallback_policy
    return _sha256(canonical_json(snapshot))


def _seal_technology_receipt(
    body: Mapping[str, Any],
    receipt_key: bytes,
) -> dict[str, Any]:
    """Hash and authenticate a receipt with a process-owned secret."""

    if not isinstance(receipt_key, bytes) or len(receipt_key) < 32:
        raise SpecValidationError("Technology receipt key must contain at least 32 bytes.")
    if "receipt_sha256" in body or "receipt_mac" in body:
        raise SpecValidationError("Technology receipt body contains a reserved field.")
    sealed = dict(body)
    sealed["receipt_sha256"] = _sha256(canonical_json(sealed))
    sealed["receipt_mac"] = "hmac-sha256:" + hmac.new(
        receipt_key,
        canonical_json(sealed).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return sealed


def _verify_technology_receipt(
    value: Mapping[str, Any],
    receipt_key: bytes | None,
) -> bool:
    if not isinstance(receipt_key, bytes) or len(receipt_key) < 32:
        return False
    receipt_mac = str(value.get("receipt_mac") or "").lower()
    if not _RECEIPT_MAC.fullmatch(receipt_mac):
        return False
    mac_body = dict(value)
    mac_body.pop("receipt_mac", None)
    expected_mac = "hmac-sha256:" + hmac.new(
        receipt_key,
        canonical_json(mac_body).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(receipt_mac, expected_mac):
        return False
    receipt_sha256 = str(mac_body.get("receipt_sha256") or "").lower()
    hash_body = dict(mac_body)
    hash_body.pop("receipt_sha256", None)
    return bool(
        _SHA256.fullmatch(receipt_sha256)
        and hmac.compare_digest(receipt_sha256, _sha256(canonical_json(hash_body)))
    )


def _voice_contract(
    *,
    activated: bool,
    adaptation_requested: bool,
    components: Sequence[str],
) -> dict[str, Any]:
    return {
        "activated": activated,
        "components": list(dict.fromkeys(components)) if activated else [],
        "speaker_identity": "speech_synthesis_or_voice_model",
        "expression": {
            "owner": "utterance_local_pattern_trace",
            "representation": "time_series",
            "fields": ["time", "energy", "entropy", "f0", "attack", "pause"],
            "prohibited": ["single_embedding", "conversation_average"],
        },
        "language_support": "full_asr_translation_tts_intersection_only",
        "adaptation": {
            "requested": adaptation_requested,
            "default": "disabled",
            "status": (
                "blocked_until_consent_provenance_revocation_and_deletion_pass"
                if adaptation_requested
                else "not_requested"
            ),
            "celebrity_or_unowned_voice_imitation": "blocked",
        },
    }


def _request_flags(prompt: str, domains: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    text = " ".join([prompt, *[_domain_text(domain) for domain in domains]])
    kinds = {
        str(kind)
        for domain in domains
        for kind in domain.get("evidence_kinds", [])
    }
    adaptation = bool(kinds & {"voice_adaptation", "voice_conversion"}) or _matches(
        text, _ADAPTATION_CUES
    )
    return {
        "ai": "ai_inference" in kinds or _matches(text, _AI_CUES),
        "agent": "agent_tool_use" in kinds or _matches(text, _AGENT_CUES),
        "voice": bool(
            kinds
            & {
                "speech_recognition",
                "voice_activity_detection",
                "speech_synthesis",
                "voice_adaptation",
                "voice_conversion",
            }
        )
        or _matches(text, _VOICE_CUES)
        or adaptation,
        "adaptation": adaptation,
        "translation": "translation" in kinds or _matches(text, _TRANSLATION_CUES),
        "offline": _matches(text, _OFFLINE_CUES),
        "cpu_only": _matches(text, _CPU_ONLY_CUES),
        "real_time": _matches(text, _REALTIME_CUES),
    }


def _topologies_for(capability: str, *, offline: bool) -> tuple[str, ...]:
    by_capability = {
        "voice_activity_detection": ("in_process_java", "local_sidecar"),
        "voice_transport": ("in_process_java",),
        "language_intersection": ("offline_build_tool",),
        "speech_recognition": ("in_process_java", "local_sidecar", "remote_api"),
        "speech_synthesis": TOPOLOGIES,
        "voice_adaptation": ("local_sidecar", "remote_api", "offline_build_tool"),
        "voice_conversion": ("local_sidecar", "remote_api", "offline_build_tool"),
        "translation": TOPOLOGIES,
        "ai_inference": TOPOLOGIES,
        "agent_tool_use": ("in_process_java", "local_sidecar", "remote_api"),
    }
    result = by_capability[capability]
    if offline:
        result = tuple(item for item in result if item != "remote_api")
    return result


def _fallback_for(capability: str) -> str:
    return {
        "speech_recognition": "Keep typed text input available.",
        "voice_activity_detection": "Use explicit push-to-talk input.",
        "speech_synthesis": "Show localized subtitles and text dialogue.",
        "voice_adaptation": "Use an authorized base voice without adaptation.",
        "voice_conversion": "Play the authorized source voice without conversion.",
        "voice_transport": "Keep typed, schema-validated commands available.",
        "language_intersection": "Advertise only the verified direct-language path.",
        "translation": "Keep original text and disable unsupported language paths.",
        "agent_tool_use": "Use the deterministic server-owned action state machine.",
        "ai_inference": "Use deterministic scripted gameplay behavior.",
    }[capability]


def _capability_order(value: str) -> tuple[int, str]:
    order = {
        "voice_activity_detection": 0,
        "speech_recognition": 1,
        "translation": 2,
        "speech_synthesis": 3,
        "voice_adaptation": 4,
        "voice_conversion": 5,
        "voice_transport": 6,
        "language_intersection": 7,
        "ai_inference": 8,
        "agent_tool_use": 9,
    }
    return order.get(value, 99), value


def _normalize_license(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "id": value.strip(),
            "url": "",
            "reviewed": False,
            "use_allowed": None,
        }
    if isinstance(value, Mapping):
        return {
            "id": str(value.get("id") or "").strip(),
            "url": str(value.get("url") or "").strip(),
            "reviewed": value.get("reviewed") is True,
            "use_allowed": (
                value.get("use_allowed")
                if type(value.get("use_allowed")) is bool
                else None
            ),
        }
    return {"id": "", "url": "", "reviewed": False, "use_allowed": None}


def _research_brief_sha(research_brief: Mapping[str, Any] | None) -> str:
    if research_brief is None:
        return _sha256("")
    try:
        return _sha256(canonical_json(research_brief))
    except (TypeError, ValueError) as exc:
        raise SpecValidationError("research_brief must be JSON serializable.") from exc


def _encode_cursor(offset: int, *, source_sha256: str, page_size: int) -> str:
    signature = hashlib.sha256(
        f"{source_sha256}\0{page_size}\0{offset}".encode("utf-8")
    ).hexdigest()[:16]
    return f"technology:{offset}:{signature}"


def _decode_cursor(cursor: str, *, source_sha256: str, page_size: int) -> int:
    if not cursor:
        return 0
    if not isinstance(cursor, str) or len(cursor) > 96:
        raise SpecValidationError("Technology cursor is invalid.")
    match = _CURSOR.fullmatch(cursor)
    if not match:
        raise SpecValidationError("Technology cursor is invalid.")
    offset = int(match.group(1))
    if _encode_cursor(offset, source_sha256=source_sha256, page_size=page_size) != cursor:
        raise SpecValidationError(
            "Technology cursor does not match this request, target and page size."
        )
    return offset


def _latency_budget(value: str) -> float | None:
    match = re.search(r"(?<!\d)(\d{1,7}(?:\.\d+)?)\s*(?:ms|milliseconds?|밀리초)", value, re.I)
    if not match:
        return None
    return float(match.group(1))


def _domain_text(domain: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(domain.get("objective") or ""),
            *[str(item) for item in domain.get("requirements", [])],
            *[str(item) for item in domain.get("queries", [])],
        ]
    )


def _matches(value: str, patterns: Iterable[str]) -> bool:
    folded = value.casefold()
    return any(re.search(pattern, folded, re.I) is not None for pattern in patterns)


def _requirement_id(domain_id: str, capability: str) -> str:
    value = _slug(f"{domain_id}_{capability}", 0)
    if len(value) <= 127:
        return value
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[:114].rstrip('_')}_{suffix}"


def _slug(value: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"domain_{index + 1}_{slug}".rstrip("_")
    if len(slug) > 63:
        suffix = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[:52].rstrip('_')}_{suffix}"
    return slug


def _language(value: Any, field: str) -> str:
    text = _text(value, field).replace("_", "-").casefold()
    if not re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{1,8})*", text):
        raise SpecValidationError(f"{field} is not a normalized language tag.")
    return text


def _languages(values: Sequence[str], field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SpecValidationError(f"{field} must be a list.")
    result: list[str] = []
    for index, value in enumerate(values):
        language = _language(value, f"{field}[{index}]")
        if language not in result:
            result.append(language)
    return tuple(result)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{field} must be an object.")
    return dict(value)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    result = _optional_string_tuple(value, field)
    if not result:
        raise SpecValidationError(f"{field} must be a non-empty list.")
    return result


def _optional_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SpecValidationError(f"{field} must be a list.")
    result: list[str] = []
    for item in value:
        text = _text(item, field)
        if text not in result:
            result.append(text)
    return tuple(result)


def _loose_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_bounded_text(str(item)) for item in value if str(item).strip()]


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError(f"{field} must be a JSON boolean.")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError(f"{field} must be a non-empty string.")
    text = value.strip()
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise SpecValidationError(f"{field} exceeds the text byte policy.")
    return text


def _bounded_text(value: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    value = value.strip()
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value
    clipped = encoded[:maximum]
    while True:
        try:
            return clipped.decode("utf-8").rstrip()
        except UnicodeDecodeError:
            clipped = clipped[:-1]


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 else None


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
