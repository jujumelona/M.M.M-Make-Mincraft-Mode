"""Version-aware, provenance-bearing retrieval over a code-owned corpus.

This is the local RAG lane used by the planner and MCP adapter.  It is not a
web-search simulator: every built-in document is a reviewed primary source with
an explicit version scope and content hash.  Query text can rank records but
cannot add sources, instructions, capabilities, or permissions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from .spec import SpecValidationError, canonical_json


SUPPORTED_VERSIONS = frozenset({"1.20.1"})
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
            raise SpecValidationError(
                f"Invalid corpus document id: {self.document_id!r}"
            )
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
        if not self.minecraft_versions:
            raise SpecValidationError(
                f"Corpus document has no version scope: {self.document_id}"
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


def _fabric_document(
    document_id: str,
    title: str,
    url: str,
    *,
    families: tuple[str, ...],
    topics: tuple[str, ...],
    content: str,
    exact: bool = False,
    related_ids: tuple[str, ...] = (),
) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        title=title,
        url=url,
        authority="Fabric official documentation or artifact repository",
        trust_tier="official_primary",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        revision=(
            "minecraft-1.20.1/fabric-api-0.92.11+yarn-1.20.1+build.1"
            if exact
            else "live-concept-documentation-checked-2026-07-29"
        ),
        verified_on="2026-07-29",
        minecraft_versions=("1.20.1",) if exact else ("*",),
        loader="fabric",
        mappings="yarn-1.20.1+build.1" if exact else "concept-only",
        families=families,
        topics=topics,
        content=content,
        related_ids=related_ids,
    )


BUILTIN_CORPUS: tuple[CorpusDocument, ...] = (
    _fabric_document(
        "fabric-yarn-1201",
        "Yarn 1.20.1+build.1 Javadoc",
        "https://maven.fabricmc.net/docs/yarn-1.20.1%2Bbuild.1/",
        families=("source", "entity", "world", "networking"),
        topics=("class", "method", "symbol", "mapping", "entity", "world", "server"),
        content=(
            "Exact named Minecraft 1.20.1 API surface for Yarn mappings build 1. "
            "Use this lane for class and method signatures; do not substitute current-version examples."
        ),
        exact=True,
        related_ids=("fabric-api-1201",),
    ),
    _fabric_document(
        "fabric-api-1201",
        "Fabric API 0.92.11+1.20.1 artifacts",
        "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/0.92.11%2B1.20.1/",
        families=("profile", "source", "build"),
        topics=("fabric api", "dependency", "artifact", "version", "maven"),
        content=(
            "Pinned Fabric API artifact lane for Minecraft 1.20.1. Dependency resolution "
            "and compiled signatures are authoritative for the selected target profile."
        ),
        exact=True,
        related_ids=("fabric-yarn-1201", "fabric-datagen-1201", "fabric-gametest-1201"),
    ),
    _fabric_document(
        "fabric-datagen-1201",
        "Fabric API 1.20.1 datagen package",
        "https://maven.fabricmc.net/docs/fabric-api-0.92.11%2B1.20.1/net/fabricmc/fabric/api/datagen/v1/package-summary.html",
        families=("datagen", "content", "world"),
        topics=("recipe", "loot", "tag", "language", "model", "data generation"),
        content=(
            "Exact Fabric API datagen package for the pinned 1.20.1 profile. JSON resources "
            "must be generated or schema-validated and then checked for reference integrity."
        ),
        exact=True,
        related_ids=("fabric-api-1201",),
    ),
    _fabric_document(
        "fabric-gametest-1201",
        "Fabric API 1.20.1 GameTest package",
        "https://maven.fabricmc.net/docs/fabric-api-0.92.11%2B1.20.1/net/fabricmc/fabric/api/gametest/v1/package-summary.html",
        families=("test", "entity", "world", "content"),
        topics=("gametest", "server", "test", "seed", "runtime"),
        content=(
            "Exact Fabric GameTest API lane for Minecraft 1.20.1. A successful model response "
            "is not test evidence; record real test identifiers, exit status, and report hashes."
        ),
        exact=True,
        related_ids=("fabric-api-1201",),
    ),
    _fabric_document(
        "fabric-meta",
        "Fabric Meta API",
        "https://meta.fabricmc.net/",
        families=("profile", "build"),
        topics=("loader", "mapping", "minecraft version", "metadata", "dependency"),
        content=(
            "Official Fabric metadata service used to resolve loader and mapping versions. "
            "Resolution results must be frozen into a target profile before generation."
        ),
        exact=True,
        related_ids=("fabric-api-1201",),
    ),
    _fabric_document(
        "fabric-project-structure",
        "Fabric project structure",
        "https://docs.fabricmc.net/develop/getting-started/project-structure",
        families=("project", "build", "client-server"),
        topics=("source set", "client", "server", "resources", "entrypoint"),
        content=(
            "Concept documentation for separating common and client-only source and resources. "
            "The live examples may target a newer Minecraft version, so exact signatures require the pinned source lane."
        ),
        related_ids=("fabric-yarn-1201", "fabric-api-1201"),
    ),
    _fabric_document(
        "fabric-build",
        "Building a Fabric mod",
        "https://docs.fabricmc.net/develop/getting-started/building-a-mod",
        families=("build", "release"),
        topics=("gradle", "build", "jar", "artifact", "remap"),
        content=(
            "Concept documentation for Gradle build and JAR output. Release status still requires "
            "a real pinned build, GameTest report, and post-build JAR inspection."
        ),
        related_ids=("fabric-api-1201",),
    ),
    _fabric_document(
        "fabric-datagen-concepts",
        "Fabric data generation setup",
        "https://docs.fabricmc.net/develop/data-generation/setup",
        families=("datagen", "content"),
        topics=("advancement", "loot", "recipe", "tag", "translation", "model"),
        content=(
            "Datagen can produce recipes, advancements, tags, models, language files and loot tables. "
            "The current documentation is conceptual evidence only for a 1.20.1 build."
        ),
        related_ids=("fabric-datagen-1201",),
    ),
    _fabric_document(
        "fabric-worldgen-concepts",
        "Fabric feature generation",
        "https://docs.fabricmc.net/develop/data-generation/features",
        families=("world", "datagen"),
        topics=("configured feature", "placed feature", "biome modification", "ore", "worldgen"),
        content=(
            "Feature generation separates configured features, placed features and biome modifications. "
            "This source alone does not prove arbitrary villages, dungeons, structures or dimensions are implemented."
        ),
        related_ids=("fabric-datagen-1201",),
    ),
    _fabric_document(
        "fabric-networking-concepts",
        "Fabric networking",
        "https://docs.fabricmc.net/develop/networking",
        families=("networking", "security", "multiplayer"),
        topics=("packet", "payload", "codec", "client", "server", "validation"),
        content=(
            "Networking bridges logical client and server state. Serverbound data must be validated "
            "on the server, including target existence, distance, type, permission and replay-sensitive state."
        ),
        related_ids=("fabric-yarn-1201",),
    ),
    _fabric_document(
        "fabric-entity-concepts",
        "Creating a Fabric entity",
        "https://docs.fabricmc.net/develop/entities/first-entity",
        families=("entity", "asset", "client-server"),
        topics=("entity type", "attribute", "goal", "renderer", "model", "texture"),
        content=(
            "Entity logic and behavior are server concerns while rendering is client-side. "
            "Current examples are not 1.20.1 signatures and must be translated through the exact source lane."
        ),
        related_ids=("fabric-yarn-1201", "fabric-networking-concepts"),
    ),
    CorpusDocument(
        document_id="blockbench-formats",
        title="Blockbench formats",
        url="https://www.blockbench.net/wiki/blockbench/formats/",
        authority="Blockbench official documentation",
        trust_tier="official_primary",
        license_id="Blockbench-documentation-license",
        revision="live-docs-checked-2026-07-29",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="agnostic",
        mappings="agnostic",
        families=("asset", "model", "animation"),
        topics=("bbmodel", "geckolib", "model", "texture", "animation", "export"),
        content=(
            "Blockbench supports multiple project formats. Java mod animation workflows may use "
            "GeckoLib; project source and runtime export are separate artifacts."
        ),
        related_ids=("blockbench-bbmodel", "geckolib-official"),
    ),
    CorpusDocument(
        document_id="blockbench-bbmodel",
        title="Blockbench BBModel format notes",
        url="https://www.blockbench.net/wiki/docs/bbmodel/",
        authority="Blockbench official documentation",
        trust_tier="official_primary",
        license_id="Blockbench-documentation-license",
        revision="live-docs-checked-2026-07-29",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="agnostic",
        mappings="agnostic",
        families=("asset", "model"),
        topics=("bbmodel", "internal format", "backup", "plugin"),
        content=(
            "BBModel is Blockbench's internal project format and is not promised as a stable complete interchange specification. "
            "Preserve originals and validate any runtime export independently."
        ),
        related_ids=("blockbench-formats",),
    ),
    CorpusDocument(
        document_id="geckolib-official",
        title="GeckoLib official repository",
        url="https://github.com/bernie-g/geckolib",
        authority="GeckoLib official repository",
        trust_tier="official_primary",
        license_id="MIT",
        revision="version-matrix-required",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="fabric",
        mappings="version-dependent",
        families=("asset", "animation", "entity"),
        topics=("geckolib", "bone", "animation controller", "geo json", "texture"),
        content=(
            "GeckoLib versions and asset paths vary by Minecraft target. Select a compatible major "
            "in the target profile and validate bone, texture and animation references before runtime use."
        ),
        related_ids=("blockbench-formats", "fabric-entity-concepts"),
    ),
    CorpusDocument(
        document_id="mcp-server-primitives",
        title="MCP server primitives",
        url="https://modelcontextprotocol.io/specification/2025-11-25/server/index",
        authority="Model Context Protocol official specification",
        trust_tier="official_primary",
        license_id="MCP-documentation-license",
        revision="2025-11-25",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="agnostic",
        mappings="agnostic",
        families=("mcp", "security"),
        topics=("tools", "resources", "prompts", "server", "protocol"),
        content=(
            "MCP servers expose tools, resources and prompts. Tool descriptions and annotations do not grant authority; "
            "application policy must independently enforce scope and approval."
        ),
        related_ids=("mcp-transports", "mcp-python-sdk"),
    ),
    CorpusDocument(
        document_id="mcp-transports",
        title="MCP transports",
        url="https://modelcontextprotocol.io/specification/2025-11-25/basic/transports",
        authority="Model Context Protocol official specification",
        trust_tier="official_primary",
        license_id="MCP-documentation-license",
        revision="2025-11-25",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="agnostic",
        mappings="agnostic",
        families=("mcp", "security"),
        topics=("stdio", "streamable http", "json-rpc", "origin", "authentication"),
        content=(
            "Stdio reserves stdout for protocol messages. Streamable HTTP servers must validate Origin, "
            "bind local services to loopback where appropriate, and implement authentication."
        ),
        related_ids=("mcp-server-primitives",),
    ),
    CorpusDocument(
        document_id="mcp-python-sdk",
        title="MCP Python SDK 2.0.0",
        url="https://github.com/modelcontextprotocol/python-sdk",
        authority="Model Context Protocol official Python SDK",
        trust_tier="official_primary",
        license_id="MIT",
        revision="v2.0.0",
        verified_on="2026-07-29",
        minecraft_versions=("*",),
        loader="agnostic",
        mappings="agnostic",
        families=("mcp", "implementation"),
        topics=("python", "mcpserver", "client", "stdio", "structured output"),
        content=(
            "The official Python SDK 2.0.0 exposes typed tools, resources and prompts and supports "
            "stdio and Streamable HTTP. This project pins the SDK version so protocol behavior cannot drift silently."
        ),
        related_ids=("mcp-server-primitives", "mcp-transports"),
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
        ("world", ("world", "map", "village", "field", "dungeon", "arena", "월드", "맵", "마을", "필드", "던전", "아레나")),
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
        "world": ("worldgen", "structure", "path", "seed"),
        "asset": ("model", "texture", "animation", "runtime"),
        "entity": ("entity", "attributes", "goals", "gametest"),
        "test": ("gametest", "report", "runtime", "evidence"),
        "datagen": ("datagen", "recipe", "loot", "tag", "model"),
        "build": ("gradle", "loom", "jar", "fabric"),
        "profile": ("minecraft", "fabric", "loader", "mapping"),
        "project": ("fabric", "project", "requirements"),
    }[family]
    for term in family_terms:
        if term not in terms:
            terms.append(term)
    return " ".join(terms[:32])


class OfficialCorpusIndex:
    """SQLite FTS5/BM25 index plus deterministic semantic and graph reranking."""

    def __init__(self, documents: Iterable[CorpusDocument] = BUILTIN_CORPUS) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise SpecValidationError("The official RAG corpus is empty.")
        by_id: dict[str, CorpusDocument] = {}
        for document in self.documents:
            document.validate()
            if document.document_id in by_id:
                raise SpecValidationError(
                    f"Duplicate corpus document: {document.document_id}"
                )
            by_id[document.document_id] = document
        for document in self.documents:
            unknown = set(document.related_ids) - set(by_id)
            if unknown:
                raise SpecValidationError(
                    f"Corpus graph has unknown relations for {document.document_id}: "
                    f"{sorted(unknown)}"
                )
        self._by_id = by_id
        self._connection = sqlite3.connect(":memory:")
        self._build_index()

    def close(self) -> None:
        self._connection.close()

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

    def _build_index(self) -> None:
        connection = self._connection
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                topics TEXT NOT NULL,
                families TEXT NOT NULL
            )
            """
        )
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE documents_fts USING fts5(
                    document_id UNINDEXED,
                    title,
                    content,
                    topics,
                    families,
                    tokenize='unicode61'
                )
                """
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False
        rows = [
            (
                document.document_id,
                document.title,
                document.content,
                " ".join(document.topics),
                " ".join(document.families),
            )
            for document in self.documents
        ]
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        if self._fts_available:
            connection.executemany(
                "INSERT INTO documents_fts VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        connection.commit()

    def retrieve(
        self,
        query: str,
        *,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        mappings: str = "yarn-1.20.1+build.1",
        limit: int = 6,
    ) -> RetrievalReceipt:
        query = query.strip()
        if not 2 <= len(query) <= 2_000:
            raise SpecValidationError("RAG query length must be between 2 and 2000.")
        if minecraft_version not in SUPPORTED_VERSIONS:
            raise SpecValidationError(
                f"No reviewed RAG profile for Minecraft {minecraft_version}."
            )
        if loader != "fabric":
            raise SpecValidationError("The reviewed local RAG profile supports Fabric only.")
        if mappings != "yarn-1.20.1+build.1":
            raise SpecValidationError("The reviewed local RAG profile uses pinned Yarn mappings.")
        if type(limit) is not int or not 1 <= limit <= 12:
            raise SpecValidationError("RAG result limit must be between 1 and 12.")

        family = _classify_query(query)
        canonical = _canonical_query(query, family)
        eligible = {
            document.document_id: document
            for document in self.documents
            if (
                minecraft_version in document.minecraft_versions
                or "*" in document.minecraft_versions
            )
            and document.loader in {loader, "agnostic"}
        }
        lexical = self._lexical_ranking(canonical, eligible)
        semantic = self._semantic_ranking(canonical, eligible)
        graph = self._graph_ranking(lexical, eligible)
        family_rank = sorted(
            eligible,
            key=lambda document_id: (
                family not in eligible[document_id].families,
                document_id,
            ),
        )
        rankings = {
            "bm25": lexical,
            "semantic_fallback": semantic,
            "graph": graph,
            "family_filter": family_rank,
        }
        weights = {
            "bm25": 1.0,
            "semantic_fallback": 0.55,
            "graph": 0.35,
            "family_filter": 0.7,
        }
        scores: dict[str, float] = {document_id: 0.0 for document_id in eligible}
        channels: dict[str, list[str]] = {document_id: [] for document_id in eligible}
        for channel, ranking in rankings.items():
            for rank, document_id in enumerate(ranking, start=1):
                scores[document_id] += weights[channel] / (60 + rank)
                if rank <= max(limit * 2, 8):
                    channels[document_id].append(channel)
        ordered = sorted(
            eligible,
            key=lambda document_id: (
                -scores[document_id],
                eligible[document_id].document_id,
            ),
        )[:limit]

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
                    minecraft_versions=document.minecraft_versions,
                    score=round(scores[document_id], 8),
                    channels=tuple(channels[document_id]),
                )
            )

        exact_hits = sum(
            minecraft_version in hit.minecraft_versions for hit in hits
        )
        family_hits = sum(
            family in eligible[hit.document_id].families for hit in hits
        )
        coverage = min(
            1.0,
            (exact_hits / max(1, min(2, len(hits)))) * 0.6
            + (family_hits / max(1, min(3, len(hits)))) * 0.4,
        )
        quality = (
            "strong"
            if len(hits) >= min(3, limit) and exact_hits >= 1 and family_hits >= 1
            else "weak"
        )
        correction_required = quality != "strong"
        corrections = (
            (
                f"{family} exact API Minecraft {minecraft_version} {loader}",
                f"{family} Yarn {mappings} symbol signature",
                f"{family} deterministic validation GameTest",
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
                    "minecraft_version": minecraft_version,
                    "loader": loader,
                    "mappings": mappings,
                }
            ).encode("utf-8")
        ).hexdigest()
        return RetrievalReceipt(
            schema_version="minecraft-mod-ai/retrieval-receipt-v1",
            query=query,
            canonical_query=canonical,
            query_family=family,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            query_hash=query_hash,
            corpus_snapshot_hash=self.snapshot_hash,
            quality=quality,
            coverage=round(coverage, 6),
            correction_required=correction_required,
            correction_queries=corrections,
            hits=tuple(hits),
        )

    def _lexical_ranking(
        self,
        query: str,
        eligible: dict[str, CorpusDocument],
    ) -> list[str]:
        terms = list(dict.fromkeys(_tokens(query)))[:24]
        if self._fts_available and terms:
            match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
            try:
                rows = self._connection.execute(
                    """
                    SELECT document_id, bm25(documents_fts, 0.0, 2.0, 1.0, 1.5, 1.2)
                    FROM documents_fts
                    WHERE documents_fts MATCH ?
                    ORDER BY 2 ASC, document_id ASC
                    """,
                    (match_query,),
                ).fetchall()
                ranked = [
                    str(document_id)
                    for document_id, _score in rows
                    if document_id in eligible
                ]
            except sqlite3.OperationalError:
                ranked = []
        else:
            ranked = []
        missing = sorted(set(eligible) - set(ranked))
        return [*ranked, *missing]

    @staticmethod
    def _semantic_ranking(
        query: str,
        eligible: dict[str, CorpusDocument],
    ) -> list[str]:
        query_grams = _trigrams(query)
        return sorted(
            eligible,
            key=lambda document_id: (
                -_jaccard(
                    query_grams,
                    _trigrams(
                        " ".join(
                            (
                                eligible[document_id].title,
                                eligible[document_id].content,
                                *eligible[document_id].topics,
                            )
                        )
                    ),
                ),
                document_id,
            ),
        )

    @staticmethod
    def _graph_ranking(
        lexical: list[str],
        eligible: dict[str, CorpusDocument],
    ) -> list[str]:
        scores = {document_id: 0.0 for document_id in eligible}
        for rank, document_id in enumerate(lexical[:5], start=1):
            scores[document_id] += 1.0 / rank
            for related_id in eligible[document_id].related_ids:
                if related_id in scores:
                    scores[related_id] += 0.45 / rank
        return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))


def retrieve_official_evidence(
    query: str,
    *,
    minecraft_version: str = "1.20.1",
    loader: str = "fabric",
    mappings: str = "yarn-1.20.1+build.1",
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
