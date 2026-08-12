from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from .spec import EvidenceSource, SpecValidationError, canonical_json


# Hosts remain code-owned; version numbers do not. A model can select/rank only
# records on these official domains, while the selected Minecraft version is bound by
# the separate live platform receipt.
OFFICIAL_EVIDENCE_HOSTS = frozenset(
    {
        "docs.fabricmc.net",
        "fabricmc.net",
        "maven.fabricmc.net",
        "meta.fabricmc.net",
    }
)
# Backward-compatible public symbol only. It is not used as an allowlist anymore.
SUPPORTED_MINECRAFT_VERSIONS = frozenset({"1.20.1", "1.21.1"})


@dataclass(frozen=True)
class _CatalogRecord:
    source_id: str
    title: str
    url: str
    authority: str
    version_scope: str
    verified_on: str
    minecraft_versions: tuple[str, ...]
    topics: tuple[str, ...]


_ALL = ("*",)
_CATALOG_RECORDS: tuple[_CatalogRecord, ...] = (
    _CatalogRecord(
        source_id="fabric-project-creation",
        title="Fabric Documentation - Creating a Project",
        url="https://docs.fabricmc.net/develop/getting-started/creating-a-project",
        authority="Fabric official documentation",
        version_scope="Version-selected Fabric documentation; exact coordinates are frozen by the live platform receipt",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("project", "structure", "version", "fabric", "loom", "gradle"),
    ),
    _CatalogRecord(
        source_id="fabric-building",
        title="Fabric Documentation - Building a Mod",
        url="https://docs.fabricmc.net/develop/getting-started/building-a-mod",
        authority="Fabric official documentation",
        version_scope="Version-selected Gradle/JAR guidance",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("build", "gradle", "jar", "artifact", "fabric"),
    ),
    _CatalogRecord(
        source_id="fabric-data-generation",
        title="Fabric Documentation - Data Generation Setup",
        url="https://docs.fabricmc.net/develop/data-generation/setup",
        authority="Fabric official documentation",
        version_scope="Version-selected data-generation guidance",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("data", "generation", "recipe", "model", "loot", "tag", "resource"),
    ),
    _CatalogRecord(
        source_id="fabric-automatic-testing",
        title="Fabric Documentation - Automated Testing",
        url="https://docs.fabricmc.net/develop/automatic-testing",
        authority="Fabric official documentation",
        version_scope="Version-selected GameTest guidance",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("test", "gametest", "entity", "runtime", "server"),
    ),
    _CatalogRecord(
        source_id="fabric-mod-json",
        title="Fabric Documentation - fabric.mod.json",
        url="https://docs.fabricmc.net/develop/loader/fabric-mod-json",
        authority="Fabric official documentation",
        version_scope="Fabric Loader metadata contract for the selected documentation version",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("metadata", "fabric.mod.json", "loader", "entrypoint", "dependency"),
    ),
    _CatalogRecord(
        source_id="fabric-develop-live",
        title="Fabric Develop - Latest Versions",
        url="https://fabricmc.net/develop/",
        authority="Fabric official website",
        version_scope="Live recommended Loader, Loom and Fabric API coordinates",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("version", "loader", "loom", "fabric", "api", "latest"),
    ),
    _CatalogRecord(
        source_id="fabric-meta",
        title="Fabric Meta API",
        url="https://meta.fabricmc.net/",
        authority="Fabric official API",
        version_scope="Live game, loader and Yarn metadata",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("version", "loader", "mapping", "yarn", "metadata", "api"),
    ),
    _CatalogRecord(
        source_id="fabric-api-maven",
        title="Fabric API Maven repository",
        url="https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/",
        authority="Fabric official Maven repository",
        version_scope="Live Fabric API artifacts for the selected Minecraft target",
        verified_on="2026-08-12",
        minecraft_versions=_ALL,
        topics=("dependency", "maven", "fabric", "api", "artifact", "version"),
    ),
    # Exact old Javadocs are retained as optional evidence for legacy targets only.
    _CatalogRecord(
        source_id="yarn-1201-javadoc",
        title="Yarn 1.20.1+build.1 Javadoc",
        url="https://maven.fabricmc.net/docs/yarn-1.20.1%2Bbuild.1/",
        authority="Fabric official Maven Javadoc",
        version_scope="Exact named Minecraft API surface for Yarn 1.20.1+build.1",
        verified_on="2026-07-28",
        minecraft_versions=("1.20.1",),
        topics=("yarn", "mapping", "javadoc", "class", "method", "entity", "structure"),
    ),
    _CatalogRecord(
        source_id="yarn-1211-javadoc",
        title="Yarn 1.21.1+build.3 Javadoc",
        url="https://maven.fabricmc.net/docs/yarn-1.21.1%2Bbuild.3/",
        authority="Fabric official Maven Javadoc",
        version_scope="Exact named Minecraft API surface for Yarn 1.21.1+build.3",
        verified_on="2026-08-12",
        minecraft_versions=("1.21.1",),
        topics=("yarn", "mapping", "javadoc", "class", "method", "identifier", "item", "block"),
    ),
)


def _applies(record: _CatalogRecord, minecraft_version: str) -> bool:
    return "*" in record.minecraft_versions or minecraft_version in record.minecraft_versions


def _record_payload(record: _CatalogRecord) -> dict[str, object]:
    return {
        "source_id": record.source_id,
        "title": record.title,
        "url": record.url,
        "authority": record.authority,
        "version_scope": record.version_scope,
        "verified_on": record.verified_on,
        "minecraft_versions": list(record.minecraft_versions),
        "topics": list(record.topics),
        "trust_tier": "official_primary",
        "retrieval_policy": "data_only",
    }


def _record_sha256(record: _CatalogRecord) -> str:
    encoded = canonical_json(_record_payload(record)).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _as_evidence(record: _CatalogRecord) -> EvidenceSource:
    return EvidenceSource(
        source_id=record.source_id,
        title=record.title,
        url=record.url,
        authority=record.authority,
        version_scope=record.version_scope,
        verified_on=record.verified_on,
        trust_tier="official_primary",
        retrieval_policy="data_only",
        record_sha256=_record_sha256(record),
    )


def evidence_catalog_for_version(minecraft_version: str) -> tuple[EvidenceSource, ...]:
    version = str(minecraft_version).strip()
    if not version:
        raise SpecValidationError("Minecraft version is required for evidence selection.")
    result = tuple(
        _as_evidence(record)
        for record in _CATALOG_RECORDS
        if _applies(record, version)
    )
    if not result:
        raise SpecValidationError(f"No official evidence applies to Minecraft {version}.")
    return result


FABRIC_1201_EVIDENCE: tuple[EvidenceSource, ...] = evidence_catalog_for_version("1.20.1")
FABRIC_1211_EVIDENCE: tuple[EvidenceSource, ...] = evidence_catalog_for_version("1.21.1")
_TRUSTED_BY_ID = {record.source_id: record for record in _CATALOG_RECORDS}


class AuthoritativeEvidenceRetriever:
    """Rank only code-owned official sources; target versions come from live discovery."""

    def search(
        self,
        query: str,
        *,
        minecraft_version: str = "1.20.1",
        limit: int = 4,
    ) -> tuple[EvidenceSource, ...]:
        available = [
            record for record in _CATALOG_RECORDS
            if _applies(record, minecraft_version)
        ]
        if type(limit) is not int or not 1 <= limit <= len(available):
            raise SpecValidationError("Evidence search limit is outside the official-source range.")

        terms = frozenset(re.findall(r"[a-z0-9_.-]+", query.lower()))
        ranked: list[tuple[int, str, EvidenceSource]] = []
        for record in available:
            searchable = frozenset(
                (*record.topics, *re.findall(r"[a-z0-9_.-]+", record.title.lower()))
            )
            score = len(terms & searchable)
            ranked.append((-score, record.source_id, _as_evidence(record)))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit])


def validate_trusted_evidence(
    sources: tuple[EvidenceSource, ...],
    *,
    minecraft_version: str | None = None,
) -> None:
    if not sources:
        raise SpecValidationError("At least one authoritative evidence source is required.")
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise SpecValidationError(f"Duplicate evidence source: {source.source_id}")
        seen.add(source.source_id)
        parsed = urlparse(source.url)
        if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_EVIDENCE_HOSTS:
            raise SpecValidationError(
                f"Evidence source host is not on the official allowlist: {source.url}"
            )
        if source.retrieval_policy != "data_only":
            raise SpecValidationError("Retrieved evidence must use the data_only policy.")
        if source.trust_tier != "official_primary":
            raise SpecValidationError("Evidence is not in the official_primary trust tier.")
        record = _TRUSTED_BY_ID.get(source.source_id)
        if record is None or source != _as_evidence(record):
            raise SpecValidationError(
                f"Evidence source is not an exact code-owned official record: {source.source_id}"
            )
        if minecraft_version is not None and not _applies(record, minecraft_version):
            raise SpecValidationError(
                f"Evidence {source.source_id} does not apply to Minecraft {minecraft_version}."
            )


def evidence_snapshot_hash(sources: tuple[EvidenceSource, ...]) -> str:
    validate_trusted_evidence(sources)
    ordered = sorted((asdict(source) for source in sources), key=lambda item: item["source_id"])
    encoded = canonical_json(ordered).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def evidence_for_target(
    query: str | None = None,
    *,
    minecraft_version: str,
) -> tuple[EvidenceSource, ...]:
    catalog = evidence_catalog_for_version(minecraft_version)
    if query is None:
        return catalog

    mandatory_ids = {
        "fabric-project-creation",
        "fabric-building",
        "fabric-mod-json",
        "fabric-develop-live",
        "fabric-meta",
        "fabric-api-maven",
    }
    selected = {
        source.source_id: source
        for source in catalog
        if source.source_id in mandatory_ids
    }
    ranked = AuthoritativeEvidenceRetriever().search(
        query,
        minecraft_version=minecraft_version,
        limit=len(catalog),
    )
    for source in ranked:
        selected.setdefault(source.source_id, source)
        if len(selected) >= min(8, len(catalog)):
            break
    result = tuple(selected[source_id] for source_id in sorted(selected))
    validate_trusted_evidence(result, minecraft_version=minecraft_version)
    return result


def evidence_for_mvp(query: str | None = None) -> tuple[EvidenceSource, ...]:
    return evidence_for_target(query, minecraft_version="1.20.1")
