from __future__ import annotations

"""Exact-target external code exemplars for the coder research hot path.

The generated project remains the primary repository context. This module adds a
separate, fail-closed lane of curated public repositories whose branch/commit is
resolved against the host-selected Minecraft target before any source excerpt can
enter the model context.

External reference code is evidence only: it never expands dependency authority,
never writes into the generated project, and never bypasses the existing official
documentation or validation gates.
"""

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any
from urllib.parse import quote

from .platform_catalog import PlatformAdapter, adapter_for_target
from .research_code_context import _tokens
from .reuse_license import is_reusable_source_license
from .source_transplant import (
    SourceTransplantError,
    _build_metadata_text,
    _fetch_blob_bytes,
    _github_client,
    _github_json,
    _repository_tree_entries,
    _target_compatibility_evidence,
)

_SOURCE_SUFFIXES = (".java", ".kt")
_SOURCE_MARKERS = ("/src/main/", "/src/client/", "/src/common/", "/src/")
_PROPERTY_ASSIGN = re.compile(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*=\s*([^\s#]+)\s*$")
_TYPE = re.compile(r"\b(?:class|interface|record|enum|object)\s+([A-Za-z_$][A-Za-z0-9_$]*)")
_METHOD = re.compile(
    r"\b(?:public|protected|private|static|final|abstract|synchronized|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)

_REFERENCE_LOCK = RLock()
_RESOLUTION_CACHE: dict[tuple[str, str, str], "_ResolvedReference"] = {}
_BRANCH_CACHE: dict[str, tuple[str, ...]] = {}
_EXCERPT_CACHE: dict[tuple[str, ...], tuple["ReferenceExcerpt", ...]] = {}
_MAX_CACHE_ENTRIES = 64


@dataclass(frozen=True)
class ReferenceFamily:
    repository: str
    capabilities: tuple[str, ...]
    priority: int
    ref_strategy: str = "version_branch"

    def matches_capability(self, query: str, capability: str) -> float:
        requested = _tokens(" ".join((query, capability)))
        if not requested:
            return 0.0
        labels = _tokens(" ".join(self.capabilities))
        overlap = len(requested & labels) / max(1, len(requested))
        if capability:
            capability_tokens = _tokens(capability)
            if capability_tokens and capability_tokens & labels:
                overlap += 0.35
        return min(1.0, overlap)


@dataclass(frozen=True)
class ReferenceExcerpt:
    repository: str
    commit_sha: str
    license_id: str
    minecraft_version: str
    loader: str
    ref_name: str
    path: str
    start_line: int
    end_line: int
    text: str
    sha256: str
    symbols: tuple[str, ...]
    score: float
    compatibility: Mapping[str, Any]

    def evidence_id(self) -> str:
        return (
            f"external:{self.repository}@{self.commit_sha}:"
            f"{self.path}:{self.start_line}:{self.end_line}"
        )


@dataclass(frozen=True)
class _ResolvedReference:
    family: ReferenceFamily
    ref_name: str
    commit_sha: str
    license_id: str
    blobs: Mapping[str, str]
    compatibility: Mapping[str, Any]


# Families are deliberately curated rather than discovered from arbitrary search results.
# Each family is still admitted per target only after its exact ref metadata and reusable
# source license pass the fail-closed resolver below.
_BUILTIN_REFERENCE_FAMILIES: tuple[ReferenceFamily, ...] = (
    ReferenceFamily(
        repository="FabricMC/fabric-api",
        capabilities=(
            "fabric api events registry networking packets lifecycle worldgen world generation rendering "
            "resources datagen data commands blocks items entities server client "
            "이벤트 레지스트리 네트워킹 패킷 월드젠 월드 생성 렌더링 데이터 블록 아이템 엔티티 서버 클라이언트"
        ).split(),
        priority=100,
        ref_strategy="exact_version",
    ),
    ReferenceFamily(
        repository="TechReborn/TechReborn",
        capabilities=(
            "machine machines automation industrial processing energy inventory recipe storage fluids cables "
            "block entity persistence crafting power transfer 기계 자동화 산업 처리 에너지 인벤토리 레시피 저장 유체 케이블 전력"
        ).split(),
        priority=96,
    ),
    ReferenceFamily(
        repository="shedaniel/RoughlyEnoughItems",
        capabilities=(
            "gui screen inventory recipe rendering networking client search widgets configuration serialization "
            "화면 인벤토리 레시피 렌더링 검색 위젯 설정 직렬화"
        ).split(),
        priority=92,
    ),
    ReferenceFamily(
        repository="TerraformersMC/ModMenu",
        capabilities=(
            "gui screen configuration client rendering integration metadata widgets "
            "화면 설정 메뉴 클라이언트 렌더링 통합 메타데이터 위젯"
        ).split(),
        priority=84,
    ),
)


def reference_families() -> tuple[ReferenceFamily, ...]:
    """Return the immutable built-in curated reference family catalogue."""

    return _BUILTIN_REFERENCE_FAMILIES


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(high, value))


def _github_token() -> str:
    return str(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()


def _bounded_cache_put(cache: dict[Any, Any], key: Any, value: Any) -> None:
    cache[key] = value
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))


def _branches(client: Any, repository: str) -> tuple[str, ...]:
    with _REFERENCE_LOCK:
        cached = _BRANCH_CACHE.get(repository)
        if cached is not None:
            return cached

    pages = _env_int("MMM_REFERENCE_BRANCH_PAGES", 6, 1, 20)
    values: list[str] = []
    for page in range(1, pages + 1):
        payload = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/branches",
            params={"per_page": "100", "page": str(page)},
        )
        if not isinstance(payload, list):
            break
        page_values = [
            str(item.get("name") or "").strip()
            for item in payload
            if isinstance(item, Mapping) and str(item.get("name") or "").strip()
        ]
        values.extend(page_values)
        if len(payload) < 100:
            break
    result = tuple(dict.fromkeys(values))
    if result:
        with _REFERENCE_LOCK:
            _bounded_cache_put(_BRANCH_CACHE, repository, result)
    return result


def _version_boundary_match(name: str, version: str) -> bool:
    pattern = rf"(?<![0-9A-Za-z]){re.escape(version)}(?![0-9A-Za-z])"
    return bool(re.search(pattern, name))


def _candidate_refs(
    client: Any,
    family: ReferenceFamily,
    minecraft_version: str,
) -> tuple[str, ...]:
    if family.ref_strategy == "exact_version":
        return (minecraft_version,)

    branches = _branches(client, family.repository)
    candidates = [minecraft_version]
    candidates.extend(
        branch
        for branch in branches
        if _version_boundary_match(branch, minecraft_version)
    )

    def rank(value: str) -> tuple[int, int, int, str]:
        exact = 0 if value == minecraft_version else 1
        suffix = 0 if value.endswith(minecraft_version) else 1
        return (exact, suffix, len(value), value)

    return tuple(sorted(dict.fromkeys(candidates), key=rank))


def _resolve_commit(client: Any, repository: str, ref_name: str) -> str:
    try:
        payload = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/commits/{quote(ref_name, safe='')}",
        )
    except SourceTransplantError:
        return ""
    if not isinstance(payload, Mapping):
        return ""
    value = str(payload.get("sha") or "").strip().casefold()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else ""


def _license_at_commit(client: Any, repository: str, commit_sha: str) -> str:
    try:
        payload = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/license",
            params={"ref": commit_sha},
        )
    except SourceTransplantError:
        return ""
    if not isinstance(payload, Mapping):
        return ""
    license_value = payload.get("license")
    if not isinstance(license_value, Mapping):
        return ""
    return str(license_value.get("spdx_id") or "").strip()


def _metadata_compatibility(
    metadata_text: str,
    *,
    adapter: PlatformAdapter,
) -> dict[str, Any] | None:
    evidence = _target_compatibility_evidence(metadata_text, adapter=adapter)
    if evidence.status != "metadata_exact":
        return None

    properties = {
        key.casefold(): value
        for key, value in _PROPERTY_ASSIGN.findall(metadata_text)
    }
    java_value = (
        properties.get("java_version")
        or properties.get("java.version")
        or properties.get("java")
        or ""
    )
    if java_value and java_value.isdigit() and str(adapter.java_version).isdigit():
        # A donor compiled for an older Java language level is usable from the newer
        # target. The unsafe direction is donor source requiring a newer Java than the
        # host-selected target can compile.
        if int(java_value) > int(adapter.java_version):
            return None

    raw_yarn = (
        properties.get("yarn_mappings")
        or properties.get("yarn_version")
        or ""
    )
    yarn_value = (
        adapter.minecraft_version + raw_yarn
        if raw_yarn.startswith("+")
        else raw_yarn
    )
    if adapter.mappings_applicable and yarn_value:
        # Mapping build numbers may differ, but a donor mapping must be anchored to the
        # exact same Minecraft version. This rejects cross-version names while allowing
        # compatible Yarn build revisions.
        if not (
            yarn_value == adapter.yarn_mappings
            or yarn_value.startswith(adapter.minecraft_version + "+")
        ):
            return None

    fabric_api = (
        properties.get("fabric_version")
        or properties.get("fabric_api_version")
        or properties.get("fabric-api.version")
        or properties.get("fabric_api")
        or properties.get("fapi_version")
        or ""
    )
    loader_version = (
        properties.get("loader_version")
        or properties.get("fabric_loader_version")
        or properties.get("fabricloader_version")
        or ""
    )
    return {
        **evidence.to_dict(),
        "status": "metadata_exact",
        "target_mappings": adapter.yarn_mappings,
        "target_java_version": adapter.java_version,
        "target_fabric_loader": adapter.fabric_loader,
        "target_fabric_api": adapter.fabric_api,
        "donor_java_version": java_value,
        "donor_yarn_mappings": yarn_value,
        "donor_loader_version": loader_version,
        "donor_fabric_api": fabric_api,
        "java_compatible": bool(
            not java_value
            or not java_value.isdigit()
            or not str(adapter.java_version).isdigit()
            or int(java_value) <= int(adapter.java_version)
        ),
        "fabric_api_exact": bool(fabric_api and fabric_api == adapter.fabric_api),
        "loader_version_exact": bool(
            loader_version and loader_version == adapter.fabric_loader
        ),
        "mappings_exact": bool(
            (not adapter.mappings_applicable)
            or (yarn_value and yarn_value == adapter.yarn_mappings)
        ),
    }


def _resolve_family(
    family: ReferenceFamily,
    *,
    adapter: PlatformAdapter,
) -> _ResolvedReference | None:
    key = (family.repository, adapter.minecraft_version, adapter.loader)
    with _REFERENCE_LOCK:
        if key in _RESOLUTION_CACHE:
            return _RESOLUTION_CACHE[key]

    client = _github_client(_github_token())
    resolved: _ResolvedReference | None = None
    try:
        for ref_name in _candidate_refs(client, family, adapter.minecraft_version):
            commit_sha = _resolve_commit(client, family.repository, ref_name)
            if not commit_sha:
                continue
            license_id = _license_at_commit(client, family.repository, commit_sha)
            if not is_reusable_source_license(license_id):
                continue
            try:
                entries = _repository_tree_entries(
                    client, family.repository, commit_sha
                )
            except SourceTransplantError:
                continue
            blobs = {
                str(item.get("path")): str(item.get("sha"))
                for item in entries
                if isinstance(item, Mapping)
                and item.get("type") == "blob"
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha"), str)
            }
            if not blobs:
                continue
            metadata_text = _build_metadata_text(
                client,
                repository=family.repository,
                blobs=blobs,
            )
            compatibility = _metadata_compatibility(
                metadata_text,
                adapter=adapter,
            )
            if compatibility is None:
                continue
            resolved = _ResolvedReference(
                family=family,
                ref_name=ref_name,
                commit_sha=commit_sha,
                license_id=license_id,
                blobs=blobs,
                compatibility=compatibility,
            )
            break
    finally:
        client.close()

    if resolved is not None:
        with _REFERENCE_LOCK:
            _bounded_cache_put(_RESOLUTION_CACHE, key, resolved)
    return resolved


def _path_score(path: str, query_tokens: set[str], family_tokens: set[str]) -> float:
    path_tokens = _tokens(path.replace("/", " ").replace("_", " "))
    if not path_tokens:
        return 0.0
    direct = len(query_tokens & path_tokens) / max(1, len(query_tokens))
    family = len(family_tokens & path_tokens) / max(1, len(family_tokens))
    main_bonus = 0.12 if any(marker in f"/{path.casefold()}" for marker in _SOURCE_MARKERS) else 0.0
    test_penalty = 0.18 if "/test/" in f"/{path.casefold()}/" else 0.0
    return direct + 0.35 * family + main_bonus - test_penalty


def _source_paths(
    resolved: _ResolvedReference,
    *,
    query: str,
    capability: str,
) -> tuple[str, ...]:
    query_tokens = _tokens(" ".join((query, capability)))
    family_tokens = _tokens(" ".join(resolved.family.capabilities))
    candidates = [
        path
        for path in resolved.blobs
        if path.casefold().endswith(_SOURCE_SUFFIXES)
        and "/src/" in f"/{path.casefold()}"
    ]
    candidates.sort(
        key=lambda path: (
            -_path_score(path, query_tokens, family_tokens),
            len(path),
            path,
        )
    )
    return tuple(candidates[: _env_int("MMM_REFERENCE_PATH_CANDIDATES", 24, 4, 96)])


def _symbols(text: str) -> tuple[str, ...]:
    values = set(_TYPE.findall(text)) | set(_METHOD.findall(text))
    return tuple(sorted(values))[:24]


def _window_excerpt(
    text: str,
    *,
    query: str,
    max_bytes: int,
) -> tuple[int, int, str, float]:
    lines = text.splitlines()
    if not lines:
        return (1, 1, "", 0.0)
    query_tokens = _tokens(query)
    if not query_tokens:
        query_tokens = {"minecraft"}
    line_scores: list[float] = []
    for line in lines:
        tokens = _tokens(line)
        line_scores.append(len(tokens & query_tokens) / max(1, len(query_tokens)))
    peak = max(range(len(line_scores)), key=line_scores.__getitem__)
    radius = _env_int("MMM_REFERENCE_EXCERPT_LINES", 80, 20, 160) // 2
    start = max(0, peak - radius)
    end = min(len(lines), peak + radius + 1)
    excerpt_lines = lines[start:end]
    while excerpt_lines and len("\n".join(excerpt_lines).encode("utf-8")) > max_bytes:
        left_distance = peak - start
        right_distance = (end - 1) - peak
        if right_distance >= left_distance and end > peak + 1:
            end -= 1
        elif start < peak:
            start += 1
        else:
            break
        excerpt_lines = lines[start:end]
    excerpt = "\n".join(excerpt_lines)
    score = max(line_scores[start:end], default=0.0)
    return (start + 1, end, excerpt, score)


def _content_score(
    query: str,
    path: str,
    text: str,
    *,
    family_score: float,
    priority: int,
) -> float:
    requested = _tokens(query)
    content = _tokens(" ".join((path, text)))
    lexical = len(requested & content) / max(1, len(requested))
    priority_score = max(0.0, min(1.0, priority / 100.0))
    return max(
        0.0,
        min(
            1.0,
            0.58 * lexical + 0.22 * family_score + 0.20 * priority_score,
        ),
    )


class VersionedReferenceCatalog:
    """Resolve and retrieve exact-version curated external exemplars."""

    def __init__(
        self,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
    ) -> None:
        self.adapter = adapter_for_target(minecraft_version, loader)
        if self.adapter.mappings_applicable and mappings != self.adapter.yarn_mappings:
            raise ValueError(
                "Versioned reference mappings disagree with the executable platform target."
            )
        if not self.adapter.mappings_applicable and mappings:
            raise ValueError(
                "Native/unobfuscated targets must not carry a legacy mapping coordinate."
            )

    def pool(self, *, query: str, capability: str = "") -> tuple[_ResolvedReference, ...]:
        scored_families = [
            (family.matches_capability(query, capability), family)
            for family in reference_families()
        ]
        ranked_families = sorted(
            (
                (score, family)
                for score, family in scored_families
                if score > 0.0 or family.repository == "FabricMC/fabric-api"
            ),
            key=lambda item: (
                -item[0],
                -item[1].priority,
                item[1].repository,
            ),
        )
        limit = _env_int("MMM_REFERENCE_FAMILY_LIMIT", 3, 1, len(ranked_families))
        result: list[_ResolvedReference] = []
        for _score, family in ranked_families[:limit]:
            resolved = _resolve_family(family, adapter=self.adapter)
            if resolved is not None:
                result.append(resolved)
        return tuple(result)

    def retrieve(
        self,
        query: str,
        *,
        capability: str = "",
        max_examples: int = 3,
        byte_budget: int = 12 * 1024,
    ) -> tuple[ReferenceExcerpt, ...]:
        normalized = " ".join(str(query).split())
        if not normalized:
            return ()
        max_examples = max(1, min(6, int(max_examples)))
        byte_budget = max(2048, min(64 * 1024, int(byte_budget)))
        cache_key = (
            self.adapter.minecraft_version,
            self.adapter.loader,
            self.adapter.yarn_mappings,
            normalized.casefold(),
            str(capability).casefold(),
            str(max_examples),
            str(byte_budget),
        )
        with _REFERENCE_LOCK:
            cached = _EXCERPT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        family_pool = self.pool(query=normalized, capability=capability)
        if not family_pool:
            return ()

        client = _github_client(_github_token())
        candidates: list[ReferenceExcerpt] = []
        try:
            per_file_budget = max(
                1536,
                min(
                    6144,
                    byte_budget // max(1, max_examples),
                ),
            )
            for resolved in family_pool:
                family_score = resolved.family.matches_capability(normalized, capability)
                paths = _source_paths(
                    resolved,
                    query=normalized,
                    capability=capability,
                )
                fetch_limit = _env_int(
                    "MMM_REFERENCE_FETCH_FILES_PER_FAMILY",
                    6,
                    2,
                    24,
                )
                for path in paths[:fetch_limit]:
                    blob_sha = str(resolved.blobs.get(path) or "")
                    if not blob_sha:
                        continue
                    try:
                        raw = _fetch_blob_bytes(
                            client,
                            resolved.family.repository,
                            blob_sha,
                        )
                    except SourceTransplantError:
                        continue
                    text = raw.decode("utf-8", errors="replace")
                    start, end, excerpt, window_score = _window_excerpt(
                        text,
                        query=" ".join((normalized, capability)),
                        max_bytes=per_file_budget,
                    )
                    if not excerpt.strip():
                        continue
                    score = _content_score(
                        " ".join((normalized, capability)),
                        path,
                        excerpt,
                        family_score=family_score,
                        priority=resolved.family.priority,
                    )
                    score = max(
                        score,
                        min(1.0, 0.7 * score + 0.3 * min(1.0, window_score)),
                    )
                    minimum_score = (
                        _env_int("MMM_REFERENCE_MIN_SCORE_PERCENT", 32, 0, 100)
                        / 100.0
                    )
                    if score < minimum_score:
                        continue
                    digest = "sha256:" + hashlib.sha256(
                        excerpt.encode("utf-8")
                    ).hexdigest()
                    candidates.append(
                        ReferenceExcerpt(
                            repository=resolved.family.repository,
                            commit_sha=resolved.commit_sha,
                            license_id=resolved.license_id,
                            minecraft_version=self.adapter.minecraft_version,
                            loader=self.adapter.loader,
                            ref_name=resolved.ref_name,
                            path=path,
                            start_line=start,
                            end_line=end,
                            text=excerpt,
                            sha256=digest,
                            symbols=_symbols(excerpt),
                            score=score,
                            compatibility=resolved.compatibility,
                        )
                    )
        finally:
            client.close()

        candidates.sort(
            key=lambda item: (
                -item.score,
                item.repository,
                item.path,
                item.start_line,
            )
        )
        selected: list[ReferenceExcerpt] = []
        used = 0
        repositories: set[str] = set()
        for item in candidates:
            size = len(item.text.encode("utf-8"))
            if used + size > byte_budget:
                continue
            # Prefer repository diversity before taking a second excerpt from the same
            # donor. This reduces correlated example errors for a small coder model.
            if item.repository in repositories and len(repositories) < min(
                max_examples, len(family_pool)
            ):
                continue
            selected.append(item)
            repositories.add(item.repository)
            used += size
            if len(selected) >= max_examples:
                break
        if len(selected) < max_examples:
            selected_ids = {item.evidence_id() for item in selected}
            for item in candidates:
                if item.evidence_id() in selected_ids:
                    continue
                size = len(item.text.encode("utf-8"))
                if used + size > byte_budget:
                    continue
                selected.append(item)
                selected_ids.add(item.evidence_id())
                used += size
                if len(selected) >= max_examples:
                    break
        result = tuple(selected)
        if result:
            with _REFERENCE_LOCK:
                _bounded_cache_put(_EXCERPT_CACHE, cache_key, result)
        return result


__all__ = [
    "ReferenceExcerpt",
    "ReferenceFamily",
    "VersionedReferenceCatalog",
    "reference_families",
]
