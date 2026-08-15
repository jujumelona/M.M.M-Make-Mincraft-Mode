"""Target-bound official retrieval without historical platform defaults.

The static corpus is deliberately target-neutral. Exact Minecraft version, loader,
mappings and toolchain coordinates are admitted only through the executable platform
provider selected by the host. Query text may rank trusted records but cannot invent
or select a platform target.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from .platform_catalog import adapter_for_target
from .spec import SpecValidationError, canonical_json

_ALLOWED_SOURCE_PREFIXES = (
    "https://docs.fabricmc.net/",
    "https://maven.fabricmc.net/",
    "https://meta.fabricmc.net/",
    "https://github.com/FabricMC/",
    "https://www.blockbench.net/",
    "https://blockbench.net/",
    "https://web.blockbench.net/",
    "https://github.com/JannisX11/",
    "https://wiki.geckolib.com/",
    "https://github.com/bernie-g/geckolib",
    "https://modelcontextprotocol.io/",
    "https://github.com/modelcontextprotocol/",
    "https://py.sdk.modelcontextprotocol.io/",
    "https://docs.oracle.com/",
    "https://docs.modrinth.com/",
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9_.:+-]+|[가-힣]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    title: str
    url: str
    authority: str
    trust_tier: str
    license_id: str
    revision: str
    verified_on: str
    minecraft_versions: tuple[str, ...]
    loader: str
    mappings: str
    families: tuple[str, ...]
    topics: tuple[str, ...]
    content: str
    related_ids: tuple[str, ...] = ()

    @property
    def content_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", self.document_id):
            raise SpecValidationError(f"Invalid corpus document id: {self.document_id!r}")
        if not any(self.url.startswith(prefix) for prefix in _ALLOWED_SOURCE_PREFIXES):
            raise SpecValidationError(
                f"Corpus URL is outside the primary-source allowlist: {self.url}"
            )
        parsed = urlparse(self.url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise SpecValidationError(f"Corpus URL must use HTTPS: {self.url}")
        if self.trust_tier != "official_primary":
            raise SpecValidationError("Built-in RAG documents must be official primary.")
        if not self.content.strip():
            raise SpecValidationError(
                f"Corpus document has no indexed content: {self.document_id}"
            )
        # Static source code must never freeze a Minecraft target. Exact version and
        # mapping authority belongs to the live executable-provider receipt.
        if self.minecraft_versions != ("*",):
            raise SpecValidationError(
                f"Built-in corpus document {self.document_id} is not target-neutral."
            )
        if self.mappings not in {"agnostic", "provider-selected"}:
            raise SpecValidationError(
                f"Built-in corpus document {self.document_id} freezes mappings."
            )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "url": self.url,
            "authority": self.authority,
            "trust_tier": self.trust_tier,
            "license_id": self.license_id,
            "revision": self.revision,
            "verified_on": self.verified_on,
            "minecraft_versions": list(self.minecraft_versions),
            "loader": self.loader,
            "mappings": self.mappings,
            "families": list(self.families),
            "topics": list(self.topics),
            "content_sha256": self.content_sha256,
            "retrieval_policy": "data_only",
        }


def _concept_document(
    document_id: str,
    title: str,
    url: str,
    *,
    authority: str,
    license_id: str,
    loader: str,
    families: tuple[str, ...],
    topics: tuple[str, ...],
    content: str,
    related_ids: tuple[str, ...] = (),
) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        title=title,
        url=url,
        authority=authority,
        trust_tier="official_primary",
        license_id=license_id,
        revision="target-neutral-concept-source",
        verified_on="2026-08-15",
        minecraft_versions=("*",),
        loader=loader,
        mappings="agnostic",
        families=families,
        topics=topics,
        content=content,
        related_ids=related_ids,
    )


BUILTIN_CORPUS: tuple[CorpusDocument, ...] = (
    _concept_document(
        "fabric-project-creation",
        "Fabric Documentation - Creating a Project",
        "https://docs.fabricmc.net/develop/getting-started/creating-a-project",
        authority="Fabric official documentation",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        loader="fabric",
        families=("profile", "build", "project"),
        topics=("project", "loader", "mappings", "loom", "gradle", "dependency"),
        content=(
            "Project structure and dependency concepts are retrieved here only as "
            "target-neutral guidance. Exact coordinates must come from the selected "
            "executable platform provider."
        ),
        related_ids=("fabric-building", "fabric-mod-json"),
    ),
    _concept_document(
        "fabric-building",
        "Fabric Documentation - Building a Mod",
        "https://docs.fabricmc.net/develop/getting-started/building-a-mod",
        authority="Fabric official documentation",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        loader="fabric",
        families=("build", "project"),
        topics=("gradle", "build", "jar", "artifact", "validation"),
        content=(
            "Build and JAR production guidance. The host-selected provider owns the "
            "actual Gradle, Java, loader and toolchain coordinates."
        ),
        related_ids=("fabric-project-creation",),
    ),
    _concept_document(
        "fabric-datagen",
        "Fabric Documentation - Data Generation",
        "https://docs.fabricmc.net/develop/data-generation/setup",
        authority="Fabric official documentation",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        loader="fabric",
        families=("datagen", "content"),
        topics=("datagen", "recipe", "loot", "tag", "language", "model"),
        content=(
            "Data-generation concepts for recipes, loot, tags, language and models. "
            "Version-specific symbols are intentionally not stored in this corpus."
        ),
    ),
    _concept_document(
        "fabric-automatic-testing",
        "Fabric Documentation - Automated Testing",
        "https://docs.fabricmc.net/develop/automatic-testing",
        authority="Fabric official documentation",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        loader="fabric",
        families=("test", "entity", "project"),
        topics=("test", "gametest", "runtime", "server", "validation"),
        content=(
            "Automated runtime validation concepts. Exact test APIs must be grounded "
            "against live target evidence before code generation."
        ),
    ),
    _concept_document(
        "fabric-mod-json",
        "Fabric Documentation - Mod Metadata",
        "https://docs.fabricmc.net/develop/loader/fabric-mod-json",
        authority="Fabric official documentation",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        loader="fabric",
        families=("profile", "build", "project"),
        topics=("metadata", "entrypoint", "dependency", "loader", "compatibility"),
        content=(
            "Loader metadata concepts. Dependency ranges and target coordinates are "
            "bound from the provider receipt, never from this static source."
        ),
        related_ids=("fabric-project-creation",),
    ),
    _concept_document(
        "mcp-specification",
        "Model Context Protocol Specification",
        "https://modelcontextprotocol.io/specification/",
        authority="Model Context Protocol official specification",
        license_id="project-license",
        loader="agnostic",
        families=("mcp", "project"),
        topics=("mcp", "tools", "resources", "prompts", "transport", "security"),
        content=(
            "MCP tools, resources, prompts and transport semantics are platform-neutral "
            "evidence and do not authorize writes or execution."
        ),
    ),
    _concept_document(
        "blockbench-documentation",
        "Blockbench Documentation",
        "https://www.blockbench.net/wiki/",
        authority="Blockbench official documentation",
        license_id="project-license",
        loader="agnostic",
        families=("asset", "project"),
        topics=("model", "texture", "animation", "geometry", "asset"),
        content=(
            "Model, texture and animation authoring concepts independent of a Minecraft "
            "target. Runtime integration still requires target-specific evidence."
        ),
    ),
)


@dataclass(frozen=True)
class RetrievalHit:
    evidence_id: str
    document_id: str
    title: str
    url: str
    excerpt: str
    content_sha256: str
    revision: str
    minecraft_versions: tuple[str, ...]
    score: float
    channels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "minecraft_versions": list(self.minecraft_versions),
            "channels": list(self.channels),
        }


@dataclass(frozen=True)
class RetrievalReceipt:
    schema_version: str
    query: str
    canonical_query: str
    query_family: str
    minecraft_version: str
    loader: str
    mappings: str
    query_hash: str
    corpus_snapshot_hash: str
    quality: str
    coverage: float
    correction_required: bool
    correction_queries: tuple[str, ...]
    hits: tuple[RetrievalHit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "correction_queries": list(self.correction_queries),
            "hits": [hit.to_dict() for hit in self.hits],
        }


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text))


def _trigrams(text: str) -> frozenset[str]:
    normalized = " ".join(_tokens(text))
    if len(normalized) < 3:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[index : index + 3] for index in range(len(normalized) - 2))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _classify_query(query: str) -> str:
    lowered = query.lower()
    families: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("mcp", ("mcp", "tool", "resource", "prompt", "stdio", "프로토콜")),
        ("networking", ("network", "packet", "payload", "codec", "멀티플레이", "패킷")),
        ("native_minecraft", ("structure", "biome", "dimension", "world_event", "구조물", "바이옴", "차원")),
        ("asset", ("3d", "model", "texture", "animation", "blockbench", "geckolib", "모델", "텍스처", "애니메이션")),
        ("entity", ("entity", "mob", "boss", "ai", "goal", "몹", "보스", "엔티티")),
        ("test", ("test", "gametest", "verify", "검증", "테스트")),
        ("datagen", ("datagen", "recipe", "loot", "tag", "lang", "레시피", "전리품", "태그")),
        ("build", ("build", "gradle", "jar", "loom", "빌드")),
        ("profile", ("version", "loader", "mapping", "java", "버전", "로더", "매핑")),
    )
    positions: list[tuple[int, str]] = []
    for family, keywords in families:
        position = max((lowered.rfind(keyword) for keyword in keywords), default=-1)
        if position >= 0:
            positions.append((position, family))
    return max(positions)[1] if positions else "project"


def _canonical_query(query: str, family: str) -> str:
    terms = list(dict.fromkeys(_tokens(query)))
    family_terms = {
        "mcp": ("mcp", "tools", "resources", "prompts", "security"),
        "networking": ("server", "validation", "packet", "codec"),
        "native_minecraft": ("registry", "world", "validation", "compatibility"),
        "asset": ("model", "texture", "animation", "runtime"),
        "entity": ("entity", "attributes", "goals", "runtime", "test"),
        "test": ("test", "report", "runtime", "evidence"),
        "datagen": ("datagen", "recipe", "loot", "tag", "model"),
        "build": ("gradle", "jar", "loader", "toolchain"),
        "profile": ("minecraft", "loader", "mapping", "provider"),
        "project": ("project", "requirements", "provider", "evidence"),
    }[family]
    for term in family_terms:
        if term not in terms:
            terms.append(term)
    return " ".join(terms[:32])


class OfficialCorpusIndex:
    """Deterministic multi-signal ranking over target-neutral primary sources."""

    def __init__(self, documents: Iterable[CorpusDocument] = BUILTIN_CORPUS) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise SpecValidationError("The official RAG corpus is empty.")
        by_id: dict[str, CorpusDocument] = {}
        for document in self.documents:
            document.validate()
            if document.document_id in by_id:
                raise SpecValidationError(f"Duplicate corpus document: {document.document_id}")
            by_id[document.document_id] = document
        for document in self.documents:
            unknown = set(document.related_ids) - set(by_id)
            if unknown:
                raise SpecValidationError(
                    f"Corpus graph has unknown relations for {document.document_id}: {sorted(unknown)}"
                )
        self._by_id = by_id

    def close(self) -> None:
        return None

    def __enter__(self) -> "OfficialCorpusIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def snapshot_hash(self) -> str:
        payload = [
            document.public_metadata()
            for document in sorted(self.documents, key=lambda item: item.document_id)
        ]
        return "sha256:" + hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            document.public_metadata()
            for document in sorted(self.documents, key=lambda item: item.document_id)
        )

    def retrieve(
        self,
        query: str,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
        limit: int = 6,
    ) -> RetrievalReceipt:
        query = query.strip()
        if not 2 <= len(query) <= 2_000:
            raise SpecValidationError("RAG query length must be between 2 and 2000.")
        minecraft_version = str(minecraft_version).strip()
        loader = str(loader).strip().casefold()
        mappings = str(mappings).strip()
        if not minecraft_version or not loader or not mappings:
            raise SpecValidationError(
                "Official retrieval requires an explicit Minecraft version, loader and mappings."
            )
        if type(limit) is not int or not 1 <= limit <= 12:
            raise SpecValidationError("RAG result limit must be between 1 and 12.")
        try:
            adapter = adapter_for_target(minecraft_version, loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
        if mappings != adapter.yarn_mappings:
            raise SpecValidationError(
                "Retrieval mappings do not match the executable provider receipt."
            )

        family = _classify_query(query)
        canonical = _canonical_query(query, family)
        eligible = {
            document.document_id: document
            for document in self.documents
            if document.loader in {adapter.loader, "agnostic"}
        }
        query_terms = frozenset(_tokens(canonical))
        query_grams = _trigrams(canonical)
        graph_boost: dict[str, float] = {document_id: 0.0 for document_id in eligible}
        lexical: dict[str, float] = {}
        semantic: dict[str, float] = {}
        family_score: dict[str, float] = {}
        for document_id, document in eligible.items():
            searchable = " ".join((document.title, document.content, *document.topics))
            document_terms = frozenset(_tokens(searchable))
            lexical[document_id] = (
                len(query_terms & document_terms) / max(1, len(query_terms))
            )
            semantic[document_id] = _jaccard(query_grams, _trigrams(searchable))
            family_score[document_id] = 1.0 if family in document.families else 0.0
        lexical_order = sorted(
            eligible,
            key=lambda document_id: (-lexical[document_id], document_id),
        )
        for rank, document_id in enumerate(lexical_order[:5], start=1):
            graph_boost[document_id] += 1.0 / rank
            for related_id in eligible[document_id].related_ids:
                if related_id in graph_boost:
                    graph_boost[related_id] += 0.45 / rank

        score: dict[str, float] = {}
        channels: dict[str, tuple[str, ...]] = {}
        for document_id in eligible:
            score[document_id] = (
                0.42 * lexical[document_id]
                + 0.28 * semantic[document_id]
                + 0.20 * family_score[document_id]
                + 0.10 * min(1.0, graph_boost[document_id])
            )
            active: list[str] = []
            if lexical[document_id] > 0:
                active.append("lexical")
            if semantic[document_id] > 0:
                active.append("semantic")
            if family_score[document_id] > 0:
                active.append("family")
            if graph_boost[document_id] > 0:
                active.append("graph")
            channels[document_id] = tuple(active)
        ordered = sorted(eligible, key=lambda document_id: (-score[document_id], document_id))[:limit]

        hits: list[RetrievalHit] = []
        for rank, document_id in enumerate(ordered, start=1):
            document = eligible[document_id]
            evidence_seed = canonical_json(
                {
                    "query": canonical,
                    "document_id": document_id,
                    "content_sha256": document.content_sha256,
                    "rank": rank,
                    "snapshot": self.snapshot_hash,
                    "target": {
                        "minecraft_version": adapter.minecraft_version,
                        "loader": adapter.loader,
                        "mappings": adapter.yarn_mappings,
                    },
                }
            ).encode("utf-8")
            hits.append(
                RetrievalHit(
                    evidence_id="sha256:" + hashlib.sha256(evidence_seed).hexdigest(),
                    document_id=document_id,
                    title=document.title,
                    url=document.url,
                    excerpt=document.content,
                    content_sha256=document.content_sha256,
                    revision=document.revision,
                    minecraft_versions=("*",),
                    score=round(score[document_id], 8),
                    channels=channels[document_id],
                )
            )

        family_hits = sum(family in eligible[hit.document_id].families for hit in hits)
        signal_hits = sum(bool(hit.channels) for hit in hits)
        coverage = min(
            1.0,
            0.6 * family_hits / max(1, min(2, len(hits)))
            + 0.4 * signal_hits / max(1, min(3, len(hits))),
        )
        quality = (
            "strong"
            if hits and signal_hits >= min(2, len(hits)) and (family_hits > 0 or family == "project")
            else "weak"
        )
        correction_required = quality != "strong"
        corrections = (
            (
                f"{family} official API for Minecraft {adapter.minecraft_version} {adapter.loader}",
                f"{family} mapping symbols for {adapter.yarn_mappings}",
                f"{family} deterministic runtime validation",
            )
            if correction_required
            else ()
        )
        query_hash = "sha256:" + hashlib.sha256(
            canonical_json(
                {
                    "query": query,
                    "canonical": canonical,
                    "family": family,
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "mappings": adapter.yarn_mappings,
                }
            ).encode("utf-8")
        ).hexdigest()
        return RetrievalReceipt(
            schema_version="minecraft-mod-ai/retrieval-receipt-v1",
            query=query,
            canonical_query=canonical,
            query_family=family,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings=adapter.yarn_mappings,
            query_hash=query_hash,
            corpus_snapshot_hash=self.snapshot_hash,
            quality=quality,
            coverage=round(coverage, 6),
            correction_required=correction_required,
            correction_queries=corrections,
            hits=tuple(hits),
        )


def retrieve_official_evidence(
    query: str,
    *,
    minecraft_version: str,
    loader: str,
    mappings: str,
    limit: int = 6,
) -> RetrievalReceipt:
    with OfficialCorpusIndex() as index:
        return index.retrieve(
            query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            limit=limit,
        )


def corpus_manifest() -> dict[str, Any]:
    with OfficialCorpusIndex() as index:
        return {
            "schema_version": "minecraft-mod-ai/rag-corpus-v1",
            "snapshot_hash": index.snapshot_hash,
            "retrieval_policy": "data_only",
            "target_policy": "explicit-provider-bound-only",
            "documents": list(index.catalog()),
        }


__all__ = [
    "BUILTIN_CORPUS",
    "CorpusDocument",
    "OfficialCorpusIndex",
    "RetrievalHit",
    "RetrievalReceipt",
    "corpus_manifest",
    "retrieve_official_evidence",
]
