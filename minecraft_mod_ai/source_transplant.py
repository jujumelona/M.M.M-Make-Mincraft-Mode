from __future__ import annotations

"""Pinned, capability-slice source transplantation for permissive OSS donors.

A repository hit is never a reusable implementation by itself. Reuse is admitted
only after an immutable commit, SPDX license, exact target metadata (or an explicit
adaptation classification), a resource-budgeted dependency-complete source slice,
and hashes for every source blob have been recorded. Execution refetches the pinned
blobs and verifies the same hashes before exposing them to the coder.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .artifact_dependency_graph import (
    ArtifactDependencyGraph,
    ArtifactEdge,
    ArtifactKind,
    ArtifactNode,
    UnresolvedArtifactEdge,
)
from .capability_implementation_locator import CapabilityImplementationLocator
from .deadline_executor import (
    ParallelExecutionTimeout,
    ParallelTaskError,
    iter_completed_with_deadlines,
)
from .ecosystem_discovery import (
    EcosystemDiscoveryUnavailable,
    ProviderRequestFailure,
    provider_request_failure,
)
from .github_public_source import (
    PublicGitSourceError,
    inspect_public_repository,
    public_git_available,
)
from .platform_catalog import PlatformAdapter
from .repository_artifact_index import RepositoryArtifactIndex
from .reuse_license import is_reusable_source_license

_TYPE_DECL = re.compile(
    r"\b(?:(?:data|sealed|value|annotation|enum)\s+class|class|interface|object|record|enum)"
    r"\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_METHOD_DECL = re.compile(
    r"\b(?:public|protected|private|static|final|synchronized|abstract|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_KOTLIN_FUN_DECL = re.compile(
    r"\bfun\s+(?:<[^>]+>\s*)?(?:[A-Za-z_][A-Za-z0-9_?.<>]*\.)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_MINECRAFT_PROPERTIES = frozenset({
    "minecraft_version", "minecraftVersion", "minecraft.version", "mc_version", "mcVersion",
})
_MINECRAFT_VERSION = re.compile(r"(?:\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?|\d{2}w\d{2}[a-z])")
_METADATA_NAMES = frozenset({
    "gradle.properties", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts", "libs.versions.toml", "fabric.mod.json", "mods.toml",
    "neoforge.mods.toml",
})
_GRADLE_DEP = re.compile(
    r"(?m)^\s*(?:modImplementation|implementation|api|compileOnly|runtimeOnly)"
    r"\s*[\( ]\s*['\"]([^'\"]+)"
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _slice_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_SLICE_BYTE_BUDGET",
        8 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _single_blob_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_SINGLE_BLOB_BYTE_BUDGET",
        16 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _response_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_RESPONSE_BYTE_BUDGET",
        16 * 1024 * 1024,
        minimum=256 * 1024,
        maximum=512 * 1024 * 1024,
    )


def _archive_byte_budget() -> int:
    return _env_int(
        "MMM_SOURCE_ARCHIVE_BYTE_BUDGET", 64 * 1024 * 1024,
        minimum=64 * 1024, maximum=512 * 1024 * 1024,
    )


def _tree_request_budget() -> int:
    # Work budget only. It never truncates a successfully enumerated tree.
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_TREE_REQUEST_BUDGET",
        2048,
        minimum=8,
        maximum=100_000,
    )


def _snapshot_cache_entries() -> int:
    return _env_int(
        "MMM_SOURCE_TRANSPLANT_SNAPSHOT_CACHE_ENTRIES",
        32,
        minimum=1,
        maximum=512,
    )


def _blob_cache_byte_budget() -> int:
    configured = _env_int(
        "MMM_SOURCE_TRANSPLANT_BLOB_CACHE_BYTE_BUDGET",
        256 * 1024 * 1024,
        minimum=64 * 1024,
        maximum=2 * 1024 * 1024 * 1024,
    )
    return max(configured, _single_blob_byte_budget())


def _materialization_download_workers() -> int:
    """Bound concurrent immutable donor downloads by an explicit I/O resource budget."""

    return _env_int(
        "MMM_SOURCE_TRANSPLANT_MATERIALIZE_DOWNLOAD_WORKERS",
        8,
        minimum=1,
        maximum=32,
    )


@dataclass
class _SourceRequestFlight:
    event: Event = field(default_factory=Event)
    value: Any = None
    error: Exception | None = None


_SNAPSHOT_LOCK = Lock()
_SNAPSHOT_CACHE: OrderedDict[tuple[str, str, str], Mapping[str, Any]] = OrderedDict()
_SNAPSHOT_INFLIGHT: dict[tuple[str, str, str], _SourceRequestFlight] = {}
_BLOB_LOCK = Lock()
_BLOB_CACHE: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_BLOB_CACHE_BYTES = 0
_BLOB_INFLIGHT: dict[tuple[str, str], _SourceRequestFlight] = {}


@dataclass(frozen=True)
class CompatibilityEvidence:
    minecraft_version: str
    loader: str
    loader_version: str = ""
    java_version: str = ""
    fabric_api_dependency: str = ""
    mixin_count: int = 0
    has_access_widener: bool = False
    status: str = "unverified"  # "metadata_exact" | "metadata_adapt" | "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "minecraft_version": self.minecraft_version,
            "loader": self.loader,
            "loader_version": self.loader_version,
            "java_version": self.java_version,
            "fabric_api_dependency": self.fabric_api_dependency,
            "mixin_count": self.mixin_count,
            "has_access_widener": self.has_access_widener,
            "status": self.status,
        }


@dataclass(frozen=True)
class DonorFile:
    path: str
    blob_sha: str
    sha256: str
    size_bytes: int
    symbols: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DonorFile:
        size = value.get("size_bytes")
        if size is None:
            size = value.get("size", 0)
        return cls(
            path=str(value["path"]),
            blob_sha=str(value["blob_sha"]),
            sha256=str(value["sha256"]),
            size_bytes=int(size),
            symbols=tuple(value.get("symbols", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_sha": self.blob_sha,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "symbols": list(self.symbols),
        }


def _artifact_kind_from_dict(value: Mapping[str, Any]) -> ArtifactKind:
    raw = str(value.get("kind") or value.get("artifact_kind") or "").strip()
    if raw:
        try:
            return ArtifactKind(raw.upper())
        except ValueError:
            pass
    path = str(value.get("id") or value.get("path") or "")
    return ArtifactDependencyGraph.kind_for_path(path)


def _artifact_node_from_dict(value: Mapping[str, Any]) -> ArtifactNode:
    path = str(value.get("id") or value.get("path") or "")
    return ArtifactNode(
        id=path,
        kind=_artifact_kind_from_dict(value),
        namespace=str(value.get("namespace") or "common"),
        logical_id=str(value.get("logical_id") or ""),
        environment=str(value.get("environment") or "common"),
        source_set=str(value.get("source_set") or "main"),
        rel_path=str(value.get("rel_path") or path),
        symbols_defined=tuple(
            value.get("symbols_defined") or value.get("declared_symbols") or ()
        ),
        symbols_referenced=tuple(
            value.get("symbols_referenced") or value.get("referenced_symbols") or ()
        ),
    )


def _artifact_edge_from_dict(value: Mapping[str, Any]) -> ArtifactEdge:
    return ArtifactEdge(
        source_id=str(value.get("source_id") or value.get("source_path") or ""),
        target_id=str(value.get("target_id") or value.get("target_path") or ""),
        dependency_type=str(
            value.get("dependency_type") or value.get("relation") or "reference"
        ),
        is_unresolved=bool(value.get("is_unresolved", False)),
        is_mandatory=bool(value.get("is_mandatory", True)),
    )


def _unresolved_edge_from_dict(value: Mapping[str, Any]) -> UnresolvedArtifactEdge:
    return UnresolvedArtifactEdge(
        source_id=str(value.get("source_id") or value.get("source_path") or ""),
        requested_target=str(
            value.get("requested_target")
            or value.get("target_id")
            or value.get("target_path")
            or ""
        ),
        relation=str(value.get("relation") or value.get("dependency_type") or "reference"),
        reason=str(value.get("reason") or "REPOSITORY_GRAPH_UNRESOLVED"),
    )


@dataclass(frozen=True)
class DonorSlice:
    capability: str
    repository: str
    commit_sha: str
    license_id: str
    source_url: str
    target_compatibility: str
    files: tuple[DonorFile, ...]
    seed_files: tuple[str, ...]
    source_symbols: tuple[str, ...]
    required_dependencies: tuple[str, ...]
    donor_tests: tuple[str, ...]
    confidence: float
    adaptation_cost: float = 0.0
    closure_complete: bool = True
    truncation_reason: str = ""
    artifact_nodes: tuple[ArtifactNode, ...] = ()
    artifact_edges: tuple[ArtifactEdge, ...] = ()
    unresolved_edges: tuple[UnresolvedArtifactEdge, ...] = ()
    compatibility_evidence: CompatibilityEvidence | None = None

    @property
    def metadata_match(self) -> bool:
        return self.target_compatibility in {"exact", "metadata_exact"} and self.closure_complete

    @property
    def exact_target(self) -> bool:
        return self.metadata_match

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DonorSlice:
        files = tuple(DonorFile.from_dict(f) for f in value.get("files", ()))
        artifact_nodes = tuple(
            _artifact_node_from_dict(n)
            for n in value.get("artifact_nodes", ())
            if isinstance(n, Mapping)
        )
        artifact_edges = tuple(
            _artifact_edge_from_dict(e)
            for e in value.get("artifact_edges", ())
            if isinstance(e, Mapping)
        )
        unresolved_edges = tuple(
            _unresolved_edge_from_dict(e)
            for e in value.get("unresolved_edges", ())
            if isinstance(e, Mapping)
        )
        compat_raw = value.get("compatibility_evidence")
        compat = None
        if compat_raw:
            compat = CompatibilityEvidence(
                minecraft_version=str(compat_raw.get("minecraft_version", "")),
                loader=str(compat_raw.get("loader", "")),
                loader_version=str(compat_raw.get("loader_version", "")),
                java_version=str(compat_raw.get("java_version", "")),
                fabric_api_dependency=str(compat_raw.get("fabric_api_dependency", "")),
                mixin_count=int(compat_raw.get("mixin_count", 0)),
                has_access_widener=bool(compat_raw.get("has_access_widener", False)),
                status=str(compat_raw.get("status", "unverified")),
            )
        return cls(
            capability=str(value.get("capability", "")),
            repository=str(value.get("repository", "")),
            commit_sha=str(value.get("commit_sha", "")),
            license_id=str(value.get("license_id", "")),
            source_url=str(value.get("source_url", "")),
            target_compatibility=str(value.get("target_compatibility", "exact")),
            files=files,
            seed_files=tuple(value.get("seed_files", ())),
            source_symbols=tuple(value.get("source_symbols", ())),
            required_dependencies=tuple(value.get("required_dependencies", ())),
            donor_tests=tuple(value.get("donor_tests", ())),
            confidence=float(value.get("confidence", 0.0)),
            adaptation_cost=float(value.get("adaptation_cost", 0.0)),
            closure_complete=bool(value.get("closure_complete", True)),
            truncation_reason=str(value.get("truncation_reason", "")),
            artifact_nodes=artifact_nodes,
            artifact_edges=artifact_edges,
            unresolved_edges=unresolved_edges,
            compatibility_evidence=compat,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/source-transplant-slice-v1",
            "capability": self.capability,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "license_id": self.license_id,
            "source_url": self.source_url,
            "target_compatibility": self.target_compatibility,
            "files": [item.to_dict() for item in self.files],
            "seed_files": list(self.seed_files),
            "source_symbols": list(self.source_symbols),
            "required_dependencies": list(self.required_dependencies),
            "donor_tests": list(self.donor_tests),
            "confidence": self.confidence,
            "adaptation_cost": self.adaptation_cost,
            "closure_complete": self.closure_complete,
            "truncation_reason": self.truncation_reason,
            "artifact_nodes": [n.to_dict() for n in self.artifact_nodes],
            "artifact_edges": [e.to_dict() for e in self.artifact_edges],
            "unresolved_edges": [e.to_dict() for e in self.unresolved_edges],
            "compatibility_evidence": self.compatibility_evidence.to_dict() if self.compatibility_evidence else None,
        }


class SourceTransplantError(RuntimeError):
    def __init__(self, message: str = "", *, request_failure: ProviderRequestFailure | None = None) -> None:
        super().__init__(message)
        self.request_failure = request_failure


@dataclass
class _RepositoryInspectionSession:
    """Plan-local observations; immutable successes remain in the shared caches."""

    snapshots: dict[tuple[str, str, str], Mapping[str, Any] | None] = field(default_factory=dict)
    snapshot_errors: dict[tuple[str, str, str], Exception] = field(default_factory=dict)
    rejected_targets: set[tuple[str, str, str]] = field(default_factory=set)
    blob_errors: dict[tuple[str, str], Exception] = field(default_factory=dict)
    provider_failures: dict[str, ProviderRequestFailure] = field(default_factory=dict)
    active: bool = True

    def remember_failure(self, exc: BaseException, *, hostname: str = "") -> ProviderRequestFailure | None:
        failure = provider_request_failure(exc, provider="github", hostname=hostname)
        if self.active and failure is not None and failure.backend_unavailable and failure.hostname:
            self.provider_failures.setdefault(failure.hostname, failure)
        return failure

    def require_network(self, hostname: str) -> None:
        failure = self.provider_failures.get(hostname) if self.active else None
        if failure is not None:
            raise SourceTransplantError(
                f"{hostname} source inspection is unavailable for this reuse plan.",
                request_failure=failure,
            )


_INSPECTION_SESSION: ContextVar[_RepositoryInspectionSession | None] = ContextVar(
    "source_repository_inspection_session", default=None,
)


@contextmanager
def repository_inspection_session():
    """Keep negative network observations only until one reuse plan finishes."""
    session = _RepositoryInspectionSession()
    token = _INSPECTION_SESSION.set(session)
    try:
        yield session
    finally:
        _INSPECTION_SESSION.reset(token)
        session.active = False
        session.snapshots.clear()
        session.snapshot_errors.clear()
        session.rejected_targets.clear()
        session.blob_errors.clear()
        session.provider_failures.clear()


def repository_inspection_failures() -> tuple[ProviderRequestFailure, ...]:
    session = _INSPECTION_SESSION.get()
    return tuple(session.provider_failures.values()) if session is not None else ()


def repository_inspection_requires_unavailable_network(
    repository: str, *, minecraft_version: str, loader: str,
) -> bool:
    """Defer only if neither cached evidence nor public Git can supply a snapshot."""
    session = _INSPECTION_SESSION.get()
    if session is None or "api.github.com" not in session.provider_failures:
        return False
    keys = ((repository, minecraft_version, loader), (repository, "", ""))
    if any(session.snapshots.get(key) is not None for key in keys):
        return False
    with _SNAPSHOT_LOCK:
        if any(key in _SNAPSHOT_CACHE for key in keys):
            return False
    return not public_git_available() or "github.com" in session.provider_failures


def _client_inspection_session(client: httpx.Client) -> _RepositoryInspectionSession | None:
    # HTTP clients cross download-worker boundaries, where ContextVars are not
    # inherited. Bind the plan state to the client as well as the calling context.
    session = _INSPECTION_SESSION.get() or getattr(client, "_mmm_inspection_session", None)
    return session if session is not None and session.active else None


def _raise_unavailable_blob_failure(
    client: httpx.Client, repository: str, blobs: Mapping[str, str], paths: Sequence[str],
) -> None:
    """Propagate the transport that actually failed, preserving semantic no-match."""
    session = _client_inspection_session(client)
    if session is None:
        return
    for path in paths:
        error = session.blob_errors.get((repository, blobs.get(path, "")))
        if error is not None:
            failure = provider_request_failure(error)
            if failure is not None and failure.backend_unavailable:
                raise error


def _donor_test_paths(blobs: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for path in blobs
            if path.casefold().endswith((".java", ".kt"))
            and any(
                marker in f"/{path.casefold()}/"
                for marker in ("/test/", "/gametest/")
            )
        )
    )


def _closure_paths(
    graph: ArtifactDependencyGraph,
    seed_paths: Sequence[str],
) -> tuple[str, ...]:
    """Return the complete graph closure without an ordinal file-count cutoff."""

    selected: list[str] = []
    seen: set[str] = set()
    for path in seed_paths:
        if path not in seen:
            seen.add(path)
            selected.append(path)
    for closure in graph.compute_directional_closures(seed_paths):
        for path in closure:
            if path not in seen:
                seen.add(path)
                selected.append(path)
    return tuple(selected)


def _repository_tree_entries(
    client: httpx.Client,
    repository: str,
    commit_sha: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve the complete immutable Git tree, recovering GitHub truncation.

    The recursive Git Trees endpoint is used as the fast path. If GitHub marks it
    truncated (or the response exceeds the configured transport byte budget), the
    repository is walked subtree-by-subtree. A repository is never discarded just
    because it crosses a local file-count threshold.
    """

    commit = _github_json(
        client,
        f"https://api.github.com/repos/{repository}/git/commits/{quote(commit_sha, safe='')}",
    )
    tree_meta = commit.get("tree") if isinstance(commit, Mapping) else None
    root_sha = str(tree_meta.get("sha") or "") if isinstance(tree_meta, Mapping) else ""
    if not re.fullmatch(r"[0-9a-f]{40,64}", root_sha):
        raise SourceTransplantError("Pinned donor commit did not expose an immutable tree SHA.")

    tree_url = f"https://api.github.com/repos/{repository}/git/trees/{quote(root_sha, safe='')}"
    try:
        recursive = _github_json(client, tree_url, params={"recursive": "1"})
    except SourceTransplantError as exc:
        failure = provider_request_failure(exc)
        if failure is not None and failure.backend_unavailable:
            raise
        recursive = None
    if isinstance(recursive, Mapping) and recursive.get("truncated") is not True:
        entries = recursive.get("tree")
        if isinstance(entries, list):
            return tuple(item for item in entries if isinstance(item, Mapping))

    budget = _tree_request_budget()
    requests = 0
    queue: deque[tuple[str, str]] = deque([(root_sha, "")])
    seen_trees: set[tuple[str, str]] = set()
    resolved: list[Mapping[str, Any]] = []
    while queue:
        if requests >= budget:
            raise SourceTransplantError(
                "Complete donor tree traversal exhausted the configured request budget."
            )
        tree_sha, prefix = queue.popleft()
        tree_identity = (tree_sha, prefix)
        if tree_identity in seen_trees:
            continue
        seen_trees.add(tree_identity)
        requests += 1
        payload = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/git/trees/{quote(tree_sha, safe='')}",
        )
        if not isinstance(payload, Mapping) or payload.get("truncated") is True:
            raise SourceTransplantError("Non-recursive donor subtree response was incomplete.")
        entries = payload.get("tree")
        if not isinstance(entries, list):
            raise SourceTransplantError("GitHub donor subtree response had no tree entries.")
        for raw in entries:
            if not isinstance(raw, Mapping):
                continue
            leaf = str(raw.get("path") or "").strip("/")
            if not leaf:
                continue
            full_path = f"{prefix}/{leaf}".strip("/")
            kind = str(raw.get("type") or "")
            sha = str(raw.get("sha") or "")
            if kind == "tree" and sha:
                queue.append((sha, full_path))
                continue
            item = dict(raw)
            item["path"] = full_path
            resolved.append(item)
    return tuple(resolved)


def repository_from_candidate(candidate: Mapping[str, Any]) -> str:
    """Return owner/repo from a GitHub discovery candidate without guessing."""

    candidate_id = str(candidate.get("candidate_id") or "")
    if candidate_id.startswith("github:"):
        value = candidate_id.split(":", 1)[1].strip()
        if value.count("/") == 1:
            return value
    source = str(candidate.get("source_url") or "")
    parsed = urlparse(source)
    if parsed.hostname in {"github.com", "www.github.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return ""


def inspect_repository_slice(
    *,
    repository: str,
    capability: str,
    adapter: PlatformAdapter,
    discovery_client: Any,
) -> DonorSlice | None:
    """External GitHub donor inspection is disabled."""
    raise SourceTransplantError("GitHub donor inspection disabled: CurseForge only.")

    session = _INSPECTION_SESSION.get()
    target_key = (repository, adapter.minecraft_version, adapter.loader)
    if session is not None and target_key in session.rejected_targets:
        return None
    snapshot = _repository_snapshot(repository, discovery_client,
        minecraft_version=adapter.minecraft_version, loader=adapter.loader)
    if not isinstance(snapshot, Mapping):
        return None
    license_id = str(snapshot.get("license_id") or "")
    commit_sha = str(snapshot.get("commit_sha") or "")
    blobs = snapshot.get("blobs")
    if (
        not is_reusable_source_license(license_id)
        or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha)
        or not isinstance(blobs, Mapping)
    ):
        return None

    token = str(getattr(discovery_client, "github_token", "") or "").strip()
    client = _github_client(token)
    try:
        metadata = _build_metadata_sources(
            client, repository=repository, blobs=blobs, commit_sha=commit_sha)

        def source_contexts():
            contexts = {}
            for path in blobs:
                if path.casefold().endswith((".java", ".kt")):
                    module = _source_module(path, metadata)
                    if module not in contexts:
                        contexts[module] = _module_metadata_context(metadata, module)
            return contexts

        contexts = source_contexts()
        if (not any(version == adapter.minecraft_version for version, _text in contexts.values())
                and snapshot.get('selected_branch') != snapshot.get('default_branch')):
            fallback = _repository_snapshot(repository, discovery_client)
            if fallback:
                snapshot = fallback
                commit_sha, license_id, blobs = snapshot['commit_sha'], snapshot['license_id'], snapshot['blobs']
                metadata = _build_metadata_sources(
                    client, repository=repository, blobs=blobs, commit_sha=commit_sha)
                contexts = source_contexts()
        target_version_present = any(
            version == adapter.minecraft_version for version, _text in contexts.values()
        )
        if not target_version_present:
            _raise_unavailable_blob_failure(
                client, repository, blobs,
                tuple(path for path, text in metadata.items() if text is None),
            )
            print('SOURCE_REUSE_VERSION: ' + json.dumps({
                'repository': repository, 'requested': adapter.minecraft_version,
                'declared': sorted({version for version, _text in contexts.values() if version}),
                'status': 'reference_only_source_version_mismatch',
            }), flush=True)
            # Do not discard a declared older-version repository: its pinned,
            # licensed source can still be a non-executable authoring reference.
            # Unresolved metadata is not evidence and cannot select source code.
            if not any(version for version, _text in contexts.values()):
                if session is not None:
                    session.rejected_targets.add(target_key)
                return None

        def version_for(path):
            module = _source_module(path, metadata)
            if module not in contexts:
                contexts[module] = _module_metadata_context(metadata, module)
            return contexts[module][0]

        tree_items = tuple(
            {"path": path, "sha": blob_sha, "type": "blob"}
            for path, blob_sha in blobs.items()
        )
        blob_paths = {blob_sha: path for path, blob_sha in blobs.items()}
        index = RepositoryArtifactIndex.build_from_tree(
            repository,
            commit_sha,
            tree_items,
            blob_fetcher=lambda repo, blob_sha: _fetch_blob_bytes(
                client, repo, blob_sha, commit_sha=commit_sha, path=blob_paths.get(blob_sha, "")
            ),
            resolver_scope=session,
        )
        graph = index.build_dependency_graph()
        unreadable = tuple(index.metadata.get("unreadable_artifacts") or ())

        seed_evidence = CapabilityImplementationLocator.locate_seeds(
            capability,
            index,
        )
        # Locate a real implementation under one declared module version.
        # Prefer the exact target; otherwise preserve only one older-version
        # source closure as an adaptation reference, never executable reuse.
        seeded_versions = tuple(
            version_for(evidence.node_id)
            for evidence in seed_evidence
            if evidence.node_id in graph.nodes and version_for(evidence.node_id)
        )
        source_version = (
            adapter.minecraft_version if adapter.minecraft_version in seeded_versions
            else next(iter(seeded_versions), "")
        )
        seed_paths = tuple(
            evidence.node_id
            for evidence in seed_evidence
            if evidence.node_id in graph.nodes
            and source_version and version_for(evidence.node_id) == source_version
        )
        if not seed_paths:
            _raise_unavailable_blob_failure(client, repository, blobs, unreadable)
            return None

        selected = list(_closure_paths(graph, seed_paths))
        selected_set = set(selected)
        truncation_reason = ""

        contents: dict[str, bytes] = {}
        total_bytes = 0
        slice_byte_budget = _slice_byte_budget()
        unresolved_edges: list[UnresolvedArtifactEdge] = [
            edge
            for edge in (*graph.unresolved_edges, *graph.ambiguous_edges)
            if edge.source_id in selected_set
            and (
                edge in graph.ambiguous_edges
                or graph.is_mandatory_unresolved(edge)
            )
        ]
        for path in unreadable:
            if path in selected_set:
                unresolved_edges.append(
                    UnresolvedArtifactEdge(
                        source_id="closure_root",
                        requested_target=path,
                        relation="materialization",
                        reason="SELECTED_ARTIFACT_UNREADABLE",
                    )
                )
        for path in selected:
            # Missing or foreign-version dependencies make a partial reference;
            # a sibling module must never lend its version to these bytes.
            if (_metadata_scope(path) is None and not _is_license_path(path)
                    and version_for(path) != source_version):
                unresolved_edges.append(UnresolvedArtifactEdge(
                    source_id="closure_root", requested_target=path,
                    relation="source_version", reason="SOURCE_TRANSPLANT_MINECRAFT_VERSION_MISMATCH",
                ))
                continue
            raw = index.artifact_bytes(path)
            if raw is None:
                unresolved_edges.append(
                    UnresolvedArtifactEdge(
                        source_id="closure_root",
                        requested_target=path,
                        relation="materialization",
                        reason="INDEXED_ARTIFACT_BYTES_MISSING",
                    )
                )
                continue
            if total_bytes + len(raw) > slice_byte_budget:
                if not truncation_reason:
                    truncation_reason = f"Exceeded configured slice byte budget ({slice_byte_budget})"
                unresolved_edges.append(
                    UnresolvedArtifactEdge(
                        source_id="closure_root",
                        requested_target=path,
                        relation="size_overflow",
                        reason="SOURCE_TRANSPLANT_SLICE_BYTE_BUDGET_EXCEEDED",
                    )
                )
                continue
            contents[path] = raw
            total_bytes += len(raw)

        selected = [path for path in selected if path in contents]
        selected_set = set(selected)
        artifact_nodes = tuple(
            graph.nodes[path]
            for path in selected
            if path in graph.nodes
        )
        artifact_edges = tuple(
            ArtifactEdge(
                source_id=source_id,
                target_id=target_id,
                dependency_type="reference",
            )
            for source_id in selected
            for target_id in sorted(graph.adjacency.get(source_id, ()))
            if target_id in selected_set
        )
        closure_complete = (
            not truncation_reason
            and not unresolved_edges
            and graph.is_closure_complete(selected)
        )

        if not selected:
            _raise_unavailable_blob_failure(client, repository, blobs, unreadable)
            return None

        selected_modules = {_source_module(path, metadata) for path in selected
                            if _metadata_scope(path) is None and not _is_license_path(path)}
        metadata_text = "\n".join(contexts[module][1] for module in sorted(selected_modules))
        ev = _compatibility_evidence(metadata_text, source_version, adapter=adapter)
        if not source_version or not ev.loader:
            return None
        compatibility = (
            "reference_only" if source_version != adapter.minecraft_version
            else ("exact" if ev.status == "metadata_exact" else "adapt")
        )
        required_dependencies = _declared_dependencies(metadata_text)
        donor_tests = _donor_test_paths({path: sha for path, sha in blobs.items()
                                       if version_for(path) == source_version})

        files: list[DonorFile] = []
        symbols: set[str] = set()
        for path in selected:
            raw = contents[path]
            text = raw.decode("utf-8", errors="replace")
            local_symbols = tuple(
                sorted(
                    set(_TYPE_DECL.findall(text))
                    | set(_METHOD_DECL.findall(text))
                    | (set(_KOTLIN_FUN_DECL.findall(text)) if path.casefold().endswith(".kt") else set())
                )
            )
            symbols.update(local_symbols)
            files.append(
                DonorFile(
                    path=path,
                    blob_sha=blobs[path],
                    sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw),
                    symbols=local_symbols,
                )
            )

        structural_seed_count = sum(
            any(kind != "path_token_match" for kind in item.evidence_types)
            for item in seed_evidence
        )
        confidence = min(
            0.99,
            0.55
            + (0.25 if compatibility in {"exact", "metadata_exact"} else 0.05)
            + min(0.15, 0.03 * structural_seed_count)
            + min(0.04, 0.01 * len(files))
            - (0.20 if not closure_complete else 0.0),
        )
        adaptation_cost = round(
            10.0 * len(required_dependencies)
            + (0.0 if compatibility in {"exact", "metadata_exact"} else 25.0)
            + (5.0 if "mixin" in metadata_text.casefold() else 0.0)
            + 0.002 * (total_bytes / 1024.0)
            + (15.0 if not closure_complete else 0.0),
            2,
        )
        return DonorSlice(
            capability=capability,
            repository=repository,
            commit_sha=commit_sha,
            license_id=license_id,
            source_url=str(snapshot.get("source_url") or f"https://github.com/{repository}"),
            target_compatibility=compatibility,
            files=tuple(files),
            seed_files=seed_paths,
            source_symbols=tuple(sorted(symbols)),
            required_dependencies=required_dependencies,
            donor_tests=donor_tests,
            confidence=round(max(0.1, confidence), 4),
            adaptation_cost=adaptation_cost,
            closure_complete=closure_complete,
            truncation_reason=truncation_reason,
            artifact_nodes=artifact_nodes,
            artifact_edges=artifact_edges,
            unresolved_edges=tuple(unresolved_edges),
            compatibility_evidence=ev,
        )
    except SourceTransplantError as exc:
        failure = provider_request_failure(exc)
        if failure is not None and failure.backend_unavailable:
            raise
        return None
    finally:
        client.close()


def materialize_source_slices(
    project_root: str | Path,
    reuse_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Refetch pinned donor blobs, verify hashes, and expose them as immutable evidence."""

    root = Path(project_root).expanduser().resolve()
    decisions = reuse_plan.get("capabilities") if isinstance(reuse_plan, Mapping) else None
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        return {"schema_version": "mmm/reuse-materialization-v1", "donors": [], "count": 0}

    validated: list[tuple[Mapping[str, Any], DonorSlice]] = []
    bound_target = reuse_plan.get("bound_target")
    target_version = str(bound_target.get("minecraft_version") or "") if isinstance(bound_target, Mapping) else ""
    for decision in decisions:
        if not isinstance(decision, Mapping) or decision.get("mode") not in {"source_transplant", "adapt"}:
            continue
        donor = validated_reuse_donor(decision)
        if (not target_version or donor.compatibility_evidence is None
                or donor.compatibility_evidence.minecraft_version != target_version):
            raise SourceTransplantError("Pinned donor Minecraft version does not match the bound target.")
        validated.append((decision, donor))
    if not validated:
        return {"schema_version": "mmm/reuse-materialization-v1", "donors": [], "count": 0}

    target_root = root / ".minecraft_ai" / "reuse" / "donors"
    target_root.mkdir(parents=True, exist_ok=True)

    token = str(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    client = _github_client(token)
    download_workers = _materialization_download_workers()
    receipts: list[dict[str, Any]] = []
    try:
        for decision, parsed_donor in validated:
            donor = parsed_donor.to_dict()
            repository = parsed_donor.repository
            commit_sha = parsed_donor.commit_sha
            files = donor["files"]
            donor_key = _donor_materialization_key(decision, donor, files)
            donor_root = target_root / donor_key
            donor_root.mkdir(parents=True, exist_ok=True)

            prepared: list[tuple[Mapping[str, Any], str, str, str, int, str, str]] = []
            for item in files:
                if not isinstance(item, Mapping):
                    raise SourceTransplantError("Malformed donor file manifest.")
                path = str(item.get("path") or "").replace("\\", "/")
                blob_sha = str(item.get("blob_sha") or "")
                expected = str(item.get("sha256") or "")
                expected_size = item.get("size_bytes")
                if not path or path.startswith("/") or ".." in path.split("/"):
                    raise SourceTransplantError("Unsafe donor source path.")
                if type(expected_size) is not int:
                    raise SourceTransplantError(
                        f"Pinned donor size is invalid for {repository}@{commit_sha}:{path}."
                    )
                prepared.append((item, path, blob_sha, expected, expected_size, repository, commit_sha))

            def fetch_and_verify(
                spec: tuple[Mapping[str, Any], str, str, str, int, str, str],
            ) -> tuple[Mapping[str, Any], str, str, str, bytes]:
                (
                    item,
                    path,
                    blob_sha,
                    expected,
                    expected_size,
                    repository_name,
                    pinned_commit,
                ) = spec
                raw = _fetch_blob_bytes(
                    client, repository_name, blob_sha, commit_sha=pinned_commit, path=path)
                if not raw:
                    raise SourceTransplantError(
                        f"Pinned donor blob is empty for {repository_name}@{pinned_commit}:{path}."
                    )
                actual = "sha256:" + hashlib.sha256(raw).hexdigest()
                if actual != expected:
                    raise SourceTransplantError(
                        f"Pinned donor hash mismatch for {repository_name}@{pinned_commit}:{path}."
                    )
                if len(raw) != expected_size:
                    raise SourceTransplantError(
                        f"Pinned donor size mismatch for {repository_name}@{pinned_commit}:{path}."
                    )
                return item, path, blob_sha, actual, raw

            # Fetch and verify the whole immutable slice before the first write. Completion
            # order is allowed to vary, but the materialized manifest keeps donor order.
            if download_workers <= 1 or len(prepared) <= 1:
                fetched = [fetch_and_verify(spec) for spec in prepared]
            else:
                indexed_specs = tuple(enumerate(prepared))

                def fetch_indexed(
                    indexed: tuple[int, tuple[Mapping[str, Any], str, str, str, int, str, str]],
                ) -> tuple[int, tuple[Mapping[str, Any], str, str, str, bytes]]:
                    index, spec = indexed
                    return index, fetch_and_verify(spec)

                fetched_by_index: dict[
                    int, tuple[Mapping[str, Any], str, str, str, bytes]
                ] = {}
                try:
                    for _indexed, result in iter_completed_with_deadlines(
                        indexed_specs,
                        fetch_indexed,
                        max_workers=min(download_workers, len(indexed_specs)),
                        stage="source-transplant-download",
                        sort_key=lambda item: item[0],
                    ):
                        index, payload = result
                        fetched_by_index[index] = payload
                except ParallelTaskError as exc:
                    cause = exc.cause
                    if isinstance(cause, SourceTransplantError):
                        raise cause
                    raise SourceTransplantError(
                        f"Pinned donor download worker failed: {cause}"
                    ) from cause
                except ParallelExecutionTimeout as exc:
                    raise SourceTransplantError(
                        f"Pinned donor download exceeded its execution deadline: {exc}"
                    ) from exc
                if len(fetched_by_index) != len(prepared):
                    raise SourceTransplantError(
                        "Pinned donor download completed without every prepared artifact."
                    )
                fetched = [fetched_by_index[index] for index in range(len(prepared))]

            written: list[dict[str, Any]] = []
            for item, path, blob_sha, actual, raw in fetched:
                destination = (donor_root / path).resolve()
                destination.relative_to(donor_root.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                written.append(
                    {
                        "path": str(destination),
                        "source_path": path,
                        "blob_sha": blob_sha,
                        "sha256": actual,
                        "size_bytes": len(raw),
                        "symbols": list(item.get("symbols") or ()),
                    }
                )
            manifest = {
                "repository": repository,
                "commit_sha": commit_sha,
                "license_id": donor.get("license_id"),
                "capability": decision.get("capability"),
                "required_dependencies": list(donor.get("required_dependencies") or ()),
                "donor_tests": list(donor.get("donor_tests") or ()),
                "files": written,
            }
            (donor_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            receipts.append(manifest)
    finally:
        client.close()
    return {
        "schema_version": "mmm/reuse-materialization-v1",
        "donors": receipts,
        "count": len(receipts),
    }


def _donor_materialization_key(
    decision: Mapping[str, Any],
    donor: Mapping[str, Any],
    files: Sequence[Any],
) -> str:
    identity = {
        "repository": str(donor.get("repository") or ""),
        "commit_sha": str(donor.get("commit_sha") or ""),
        "capability": str(decision.get("capability") or ""),
        "files": [
            [
                str(item.get("path") or ""),
                str(item.get("blob_sha") or ""),
                str(item.get("sha256") or ""),
                int(item.get("size_bytes") or 0),
            ]
            for item in files
            if isinstance(item, Mapping)
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _repository_snapshot(repository: str, discovery_client: Any, *, minecraft_version="", loader="") -> Mapping[str, Any] | None:
    """GitHub snapshots are disabled."""
    raise SourceTransplantError("GitHub snapshot disabled: CurseForge only.")

    cache_key = (repository, minecraft_version, loader)
    session = _INSPECTION_SESSION.get()
    if session is not None and cache_key in session.snapshots:
        return session.snapshots[cache_key]
    owner = False
    with _SNAPSHOT_LOCK:
        if cache_key in _SNAPSHOT_CACHE:
            cached = _SNAPSHOT_CACHE[cache_key]
            _SNAPSHOT_CACHE.move_to_end(cache_key)
            if session is not None:
                session.snapshots[cache_key] = cached
            return cached
        if session is not None and minecraft_version and "api.github.com" in session.provider_failures:
            # A previously pinned default branch remains usable as reference
            # evidence; module metadata below must still prove the exact version.
            default_key = (repository, "", "")
            cached = session.snapshots.get(default_key) or _SNAPSHOT_CACHE.get(default_key)
            if cached is not None:
                session.snapshots[cache_key] = cached
                return cached
        if session is not None and cache_key in session.snapshot_errors:
            raise session.snapshot_errors[cache_key]
        flight = _SNAPSHOT_INFLIGHT.get(cache_key)
        if flight is None:
            flight = _SourceRequestFlight()
            _SNAPSHOT_INFLIGHT[cache_key] = flight
            owner = True
    if not owner:
        if not flight.event.wait(timeout=30.0):
            raise SourceTransplantError("Timed out waiting for a concurrent repository inspection.")
        if flight.error is not None:
            if session is not None:
                failure = session.remember_failure(flight.error, hostname="api.github.com")
                if minecraft_version and failure is not None and failure.backend_unavailable:
                    default_key = (repository, "", "")
                    with _SNAPSHOT_LOCK:
                        cached = session.snapshots.get(default_key) or _SNAPSHOT_CACHE.get(default_key)
                    if cached is not None:
                        session.snapshots[cache_key] = cached
                        return cached
                session.snapshot_errors[cache_key] = flight.error
            raise flight.error
        if session is not None:
            session.snapshots[cache_key] = flight.value
        return flight.value

    snapshot: Mapping[str, Any] | None = None
    failure_error: Exception | None = None
    cache_snapshot_globally = True
    try:
        if session is not None:
            session.require_network("api.github.com")
        target_inspector = getattr(discovery_client, 'inspect_github_repository_for_target', None)
        evidence = (target_inspector(repository, minecraft_version=minecraft_version, loader=loader)
                    if minecraft_version and callable(target_inspector) else
                    discovery_client.inspect_github_repository(repository))
        if not isinstance(evidence, Mapping):
            return None
        commit_sha = str(evidence.get("commit_sha") or "")
        license_id = str(evidence.get("license_id") or "")
        if not is_reusable_source_license(license_id) or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha):
            return None
        token = str(getattr(discovery_client, "github_token", "") or "").strip()
        client = _github_client(token)
        try:
            entries = _repository_tree_entries(client, repository, commit_sha)
            blobs = {str(item["path"]): str(item["sha"]) for item in entries
                     if item.get("type") == "blob" and isinstance(item.get("path"), str)
                     and isinstance(item.get("sha"), str)}
            try:
                _prefetch_pinned_archive(client, repository, commit_sha, blobs)
            except (httpx.HTTPError, SourceTransplantError, OSError, zipfile.BadZipFile) as exc:
                # The immutable blob API remains a hash-checked fallback for
                # unavailable/oversized archives and shallow test clients.
                if session is not None:
                    session.remember_failure(exc, hostname="codeload.github.com")
        finally:
            client.close()
        blobs = {
            str(item.get("path")): str(item.get("sha"))
            for item in entries
            if isinstance(item, Mapping)
            and item.get("type") == "blob"
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha"), str)
        }
        snapshot = {
            "commit_sha": commit_sha,
            "license_id": license_id,
            "source_url": str(evidence.get("source_url") or f"https://github.com/{repository}"),
            "blobs": blobs,
            "default_branch": evidence.get("default_branch", ""),
            "selected_branch": evidence.get("selected_branch", ""),
        }
        return snapshot
    except (SourceTransplantError, EcosystemDiscoveryUnavailable, httpx.HTTPError) as exc:
        failure = (session.remember_failure(exc, hostname="api.github.com") if session is not None
                   else provider_request_failure(exc))
        unavailable = failure is not None and failure.backend_unavailable
        if unavailable and session is not None and minecraft_version:
            default_key = (repository, "", "")
            with _SNAPSHOT_LOCK:
                snapshot = session.snapshots.get(default_key) or _SNAPSHOT_CACHE.get(default_key)
            if snapshot is not None:
                # Do not persist this fallback as the target branch snapshot:
                # a new plan must be able to resolve its intended branch again.
                cache_snapshot_globally = False
                return snapshot
        if unavailable and public_git_available():
            try:
                snapshot = _public_repository_snapshot(
                    repository, discovery_client, minecraft_version=minecraft_version, loader=loader)
                return snapshot
            except (PublicGitSourceError, SourceTransplantError, httpx.HTTPError, OSError) as git_exc:
                git_failure = (session.remember_failure(git_exc, hostname="github.com")
                               if session is not None else provider_request_failure(git_exc))
                failure_error = SourceTransplantError(
                    "GitHub API is unavailable and public Git source inspection failed.",
                    request_failure=git_failure or failure,
                )
                raise failure_error from git_exc
        if isinstance(exc, SourceTransplantError) and not unavailable:
            return None
        failure_error = exc
        raise
    except Exception as exc:
        failure_error = exc
        raise
    finally:
        if session is not None:
            if failure_error is not None:
                session.snapshot_errors[cache_key] = failure_error
            else:
                session.snapshots[cache_key] = snapshot
        with _SNAPSHOT_LOCK:
            if snapshot is not None and cache_snapshot_globally:
                _SNAPSHOT_CACHE[cache_key] = snapshot
                _SNAPSHOT_CACHE.move_to_end(cache_key)
                while len(_SNAPSHOT_CACHE) > _snapshot_cache_entries():
                    _SNAPSHOT_CACHE.popitem(last=False)
            elif cache_snapshot_globally:
                _SNAPSHOT_CACHE.pop(cache_key, None)
            pending = _SNAPSHOT_INFLIGHT.pop(cache_key, None)
            if pending is not None:
                pending.value = snapshot
                pending.error = failure_error
                pending.event.set()


def _public_repository_snapshot(
    repository: str, discovery_client: Any, *, minecraft_version: str, loader: str,
) -> Mapping[str, Any] | None:
    """Get immutable tree/license evidence without spending the REST API quota."""
    session = _INSPECTION_SESSION.get()
    if session is not None:
        session.require_network("github.com")
    evidence = inspect_public_repository(
        repository, minecraft_version=minecraft_version, loader=loader,
        response_byte_budget=_response_byte_budget(), archive_byte_budget=_archive_byte_budget(),
        single_blob_byte_budget=_single_blob_byte_budget(),
    )
    if not isinstance(evidence, Mapping):
        return None
    snapshot = {key: value for key, value in evidence.items()
                if key not in {"verified_blobs", "verified_license_bytes"}}
    commit_sha = str(snapshot.get("commit_sha") or "")
    blobs = snapshot.get("blobs")
    if (not is_reusable_source_license(snapshot.get("license_id"))
            or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha) or not isinstance(blobs, Mapping)):
        return None
    for blob_sha, raw in evidence.get("verified_blobs", {}).items():
        _verify_git_blob(blob_sha, raw)
        _remember_verified_blob(repository, blob_sha, raw)
    client = _github_client(str(getattr(discovery_client, "github_token", "") or "").strip())
    try:
        try:
            _prefetch_pinned_archive(client, repository, commit_sha, blobs)
        except (httpx.HTTPError, SourceTransplantError, OSError, zipfile.BadZipFile) as exc:
            # Pinned raw reads below retain the independent Git tree hash gate.
            if session is not None:
                session.remember_failure(exc, hostname="codeload.github.com")
    finally:
        client.close()
    snapshot["source_transport"] = "public_git"
    print("SOURCE_REUSE_SOURCE_ROUTE: " + json.dumps({
        "repository": repository, "transport": "public_git", "commit_sha": commit_sha,
        "license_id": snapshot["license_id"], "api_quota_required": False,
    }), flush=True)
    return snapshot


def _github_client(token: str) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mmm-source-transplant"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client = httpx.Client(timeout=15.0, headers=headers, follow_redirects=False)
    client._mmm_inspection_session = _INSPECTION_SESSION.get()
    return client


def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:
    session = _client_inspection_session(client)
    hostname = urlparse(url).hostname or ""
    client_failure = getattr(client, "_mmm_api_request_failure", None)
    if hostname == "api.github.com" and isinstance(client_failure, ProviderRequestFailure):
        raise SourceTransplantError(
            "GitHub API is unavailable for this source download client.", request_failure=client_failure)
    if session is not None:
        session.require_network(hostname)
    try:
        response = client.get(url, params=params)
        if response.status_code in {301, 302, 307, 308}:
            from urllib.parse import urljoin

            redirect = urljoin(str(response.url), response.headers.get("location", ""))
            origin = urlparse(url)
            target = urlparse(redirect)
            if (
                origin.scheme != "https"
                or origin.hostname != "api.github.com"
                or target.scheme != "https"
                or target.hostname != "api.github.com"
                or target.username is not None
                or target.password is not None
                or not target.path.startswith(("/repos/", "/repositories/"))
            ):
                raise SourceTransplantError("GitHub donor redirect left the trusted API origin.")
            response = client.get(redirect)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        hostname = urlparse(url).hostname or ""
        failure = (session.remember_failure(exc, hostname=hostname) if session is not None
                   else provider_request_failure(exc, provider="github", hostname=hostname))
        if hostname == "api.github.com" and failure is not None and failure.backend_unavailable:
            client._mmm_api_request_failure = failure
        raise SourceTransplantError(
            f"GitHub donor request failed: {url}", request_failure=failure,
        ) from exc
    limit = _response_byte_budget()
    if len(response.content) > limit:
        raise SourceTransplantError(
            f"GitHub response exceeded configured source-transplant response budget ({limit} bytes)."
        )
    try:
        return response.json()
    except ValueError as exc:
        raise SourceTransplantError(
            f"GitHub donor response was not valid JSON: {url}"
        ) from exc


def _remember_verified_blob(repository: str, blob_sha: str, raw: bytes) -> None:
    global _BLOB_CACHE_BYTES
    key = (repository, blob_sha)
    with _BLOB_LOCK:
        existing = _BLOB_CACHE.pop(key, None)
        if existing is not None:
            _BLOB_CACHE_BYTES -= len(existing)
        byte_budget = _blob_cache_byte_budget()
        while _BLOB_CACHE and _BLOB_CACHE_BYTES + len(raw) > byte_budget:
            _old_key, old_value = _BLOB_CACHE.popitem(last=False)
            _BLOB_CACHE_BYTES -= len(old_value)
        _BLOB_CACHE[key] = raw
        _BLOB_CACHE_BYTES += len(raw)


def _verify_git_blob(blob_sha: str, raw: bytes) -> None:
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", blob_sha):
        raise SourceTransplantError("Donor blob has no valid immutable Git identity.")
    identity = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    actual = (hashlib.sha1(identity).hexdigest() if len(blob_sha) == 40
              else hashlib.sha256(identity).hexdigest())
    if actual != blob_sha:
        raise SourceTransplantError("Pinned source Git blob hash mismatch.")


def _prefetch_pinned_archive(client: httpx.Client, repository: str, commit_sha: str,
                             blobs: Mapping[str, str]) -> int:
    """One commit archive replaces API-per-file downloads, retaining Git hashes."""
    from .repository_artifact_index import _is_repository_artifact

    if not callable(getattr(client, "stream", None)):
        return 0
    session = _client_inspection_session(client)
    if session is not None:
        session.require_network("codeload.github.com")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha):
        raise SourceTransplantError("Archive source is not an immutable repository commit.")
    budget = _archive_byte_budget()
    verified: list[tuple[str, bytes]] = []
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as downloaded:
        with client.stream("GET", f"https://codeload.github.com/{repository}/zip/{commit_sha}",
                           headers={"Authorization": ""}, timeout=30.0) as response:
            response.raise_for_status()
            total = 0
            for chunk in response.iter_bytes(64 * 1024):
                total += len(chunk)
                if total > budget:
                    raise SourceTransplantError("Pinned source archive exceeded download byte budget.")
                downloaded.write(chunk)
        downloaded.seek(0)
        with zipfile.ZipFile(downloaded) as archive:
            decoded = 0
            for member in archive.infolist():
                _root, separator, path = member.filename.partition("/")
                if (not separator or path not in blobs
                        or not (_is_repository_artifact(path) or _is_license_path(path))):
                    continue
                if member.file_size > _single_blob_byte_budget():
                    raise SourceTransplantError("Pinned source archive file exceeded blob byte budget.")
                decoded += member.file_size
                if decoded > _blob_cache_byte_budget():
                    raise SourceTransplantError("Pinned source archive exceeded decoded byte budget.")
                raw = archive.read(member)
                _verify_git_blob(blobs[path], raw)
                verified.append((blobs[path], raw))
    for blob_sha, raw in verified:
        _remember_verified_blob(repository, blob_sha, raw)
    return len(verified)


def _fetch_pinned_raw_blob(
    client: httpx.Client, repository: str, blob_sha: str, *, commit_sha: str, path: str,
) -> bytes:
    """Fetch a pinned public file without sending the REST token to another host."""
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or any(part in {".", ".."} for part in repository.split("/"))
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_sha)
            or not path or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise SourceTransplantError("Pinned raw donor location is invalid.")
    session = _client_inspection_session(client)
    if session is not None:
        session.require_network("raw.githubusercontent.com")
    url = f"https://raw.githubusercontent.com/{repository}/{commit_sha}/{quote(path, safe='/')}"
    data = bytearray()
    try:
        with client.stream("GET", url, headers={"Authorization": ""}, timeout=30.0) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(64 * 1024):
                if len(data) + len(chunk) > _single_blob_byte_budget():
                    raise SourceTransplantError("Pinned raw donor exceeded the single blob byte budget.")
                data.extend(chunk)
    except httpx.HTTPError as exc:
        failure = (session.remember_failure(exc, hostname="raw.githubusercontent.com")
                   if session is not None else provider_request_failure(exc, provider="github"))
        raise SourceTransplantError("Pinned raw source request failed.", request_failure=failure) from exc
    raw = bytes(data)
    _verify_git_blob(blob_sha, raw)
    return raw


def _fetch_blob_bytes(
    client: httpx.Client, repository: str, blob_sha: str, *, commit_sha: str = "", path: str = "",
) -> bytes:

    if not re.fullmatch(r"[0-9a-f]{40,64}", blob_sha):
        raise SourceTransplantError("Donor blob is not immutable.")
    key = (repository, blob_sha)
    session = _client_inspection_session(client)
    owner = False
    with _BLOB_LOCK:
        cached = _BLOB_CACHE.get(key)
        if cached is not None:
            _BLOB_CACHE.move_to_end(key)
            return cached
        if session is not None and key in session.blob_errors:
            raise session.blob_errors[key]
        flight = _BLOB_INFLIGHT.get(key)
        if flight is None:
            flight = _SourceRequestFlight()
            _BLOB_INFLIGHT[key] = flight
            owner = True
    if not owner:
        if not flight.event.wait(timeout=30.0):
            raise SourceTransplantError("Timed out waiting for a concurrent donor blob download.")
        with _BLOB_LOCK:
            cached = _BLOB_CACHE.get(key)
        if cached is not None:
            return cached
        if flight.error is not None:
            if session is not None:
                session.blob_errors[key] = flight.error
                session.remember_failure(flight.error, hostname="api.github.com")
            raise flight.error
        if flight.value is not None:
            return flight.value
        raise SourceTransplantError("Concurrent donor blob download failed.")

    fetched: bytes | None = None
    failure_error: Exception | None = None
    try:
        try:
            value = _github_json(
                client,
                f"https://api.github.com/repos/{repository}/git/blobs/{quote(blob_sha, safe='')}",
            )
            if not isinstance(value, Mapping) or value.get("encoding") != "base64":
                raise SourceTransplantError("GitHub donor blob is not base64 encoded.")
            try:
                raw = base64.b64decode(
                    str(value.get("content") or "").replace("\n", ""), validate=True,
                )
            except (binascii.Error, ValueError) as exc:
                raise SourceTransplantError("GitHub donor blob contained invalid base64.") from exc
        except (SourceTransplantError, httpx.HTTPError) as api_exc:
            failure = provider_request_failure(api_exc)
            if not commit_sha or not path or failure is None or not failure.backend_unavailable:
                raise
            raw = _fetch_pinned_raw_blob(
                client, repository, blob_sha, commit_sha=commit_sha, path=path)
        single_blob_budget = _single_blob_byte_budget()
        if len(raw) > single_blob_budget:
            raise SourceTransplantError(
                f"Single donor blob exceeded configured byte budget ({single_blob_budget} bytes)."
            )
        _verify_git_blob(blob_sha, raw)
        _remember_verified_blob(repository, blob_sha, raw)
        fetched = raw
        return raw
    except (SourceTransplantError, httpx.HTTPError, OSError) as exc:
        failure_error = exc
        if session is not None:
            session.blob_errors[key] = exc
            session.remember_failure(exc)
        raise
    finally:
        with _BLOB_LOCK:
            pending = _BLOB_INFLIGHT.pop(key, None)
            if pending is not None:
                pending.value = fetched
                pending.error = failure_error
                pending.event.set()


def validate_donor_slice_manifest(donor_slice: DonorSlice) -> None:
    """Validate immutable donor identity and every manifest path/hash before I/O."""

    repository = str(donor_slice.repository or "").strip()
    if repository.count("/") != 1 or any(
        not part or part in {".", ".."} for part in repository.split("/")
    ):
        raise SourceTransplantError("Donor repository identity is invalid.")
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_slice.commit_sha or "")):
        raise SourceTransplantError("Donor commit is not an immutable full SHA.")
    if not is_reusable_source_license(donor_slice.license_id):
        raise SourceTransplantError("Donor source license is not admitted for reuse.")
    if not donor_slice.files:
        raise SourceTransplantError("Donor source-slice manifest is empty.")

    seen_paths: set[str] = set()
    for donor_file in donor_slice.files:
        path = str(donor_file.path or "")
        normalized = path.replace("\\", "/").strip()
        parts = normalized.split("/")
        if (
            not normalized
            or normalized != path
            or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or any(part in {"", ".", ".."} for part in parts)
            or normalized in seen_paths
        ):
            raise SourceTransplantError("Donor manifest contains an unsafe or duplicate path.")
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_file.blob_sha or "")):
            raise SourceTransplantError("Donor manifest contains a non-immutable blob SHA.")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(donor_file.sha256 or "").casefold()):
            raise SourceTransplantError("Donor manifest contains an invalid SHA-256 binding.")
        if donor_file.size_bytes < 0:
            raise SourceTransplantError("Donor manifest contains a negative file size.")
        seen_paths.add(normalized)


def donor_closure_sha256(donor_slice: DonorSlice) -> str:
    """Bind proof identity to every immutable donor file attribute."""

    payload = [
        [item.path, item.blob_sha, item.sha256, item.size_bytes]
        for item in sorted(donor_slice.files, key=lambda entry: entry.path)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validated_reuse_donor(decision: Mapping[str, Any]) -> DonorSlice:
    """Validate one source-reuse decision and its executable proof as one unit."""

    if str(decision.get("mode") or "").strip().casefold() not in {
        "source_transplant",
        "adapt",
    }:
        raise SourceTransplantError("Reuse decision is not a source donor action.")
    raw_donor = decision.get("donor")
    if not isinstance(raw_donor, Mapping):
        raise SourceTransplantError("Reuse decision has no donor manifest.")
    if raw_donor.get("schema_version") != "mmm/source-transplant-slice-v1":
        raise SourceTransplantError("Reuse donor schema is invalid.")
    try:
        donor = DonorSlice.from_dict(raw_donor)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceTransplantError("Reuse donor manifest is malformed.") from exc
    validate_donor_slice_manifest(donor)
    if not donor.closure_complete or donor.target_compatibility not in {"exact", "adapt"}:
        raise SourceTransplantError("Reuse donor closure or target compatibility is unverified.")

    capability = str(decision.get("capability") or "").strip().casefold()
    donor_capability = str(donor.capability or "").strip().casefold()
    if capability.removeprefix("capability:") != donor_capability.removeprefix("capability:"):
        raise SourceTransplantError("Reuse decision capability does not match its donor.")

    proof = decision.get("proof_receipt")
    if not isinstance(proof, Mapping):
        raise SourceTransplantError("Reuse decision has no executable proof receipt.")
    if proof.get("schema_version") != "mmm/reuse-proof-receipt-v1":
        raise SourceTransplantError("Reuse proof schema is invalid.")
    from .proof_level import ProofLevel

    if not ProofLevel.from_value(proof.get("proof_level")).is_verified():
        raise SourceTransplantError("Reuse proof level is not verified.")
    if proof.get("authoritative_compile") is not True or proof.get("compile_passed") is not True:
        raise SourceTransplantError("Reuse proof has no authoritative compile pass.")
    candidate_id = f"{donor.repository}@{donor.commit_sha}"
    if str(proof.get("candidate_id") or "") != candidate_id:
        raise SourceTransplantError("Reuse proof candidate identity does not match its donor.")
    if str(proof.get("commit_sha") or "") != donor.commit_sha:
        raise SourceTransplantError("Reuse proof commit does not match its donor.")
    proof_capability = str(proof.get("capability") or "").strip().casefold()
    if proof_capability.removeprefix("capability:") != donor_capability.removeprefix("capability:"):
        raise SourceTransplantError("Reuse proof capability does not match its donor.")
    if str(proof.get("closure_hash") or "") != donor_closure_sha256(donor):
        raise SourceTransplantError("Reuse proof closure hash does not match its donor manifest.")
    verified_capabilities = {
        str(item).strip().casefold().removeprefix("capability:")
        for item in proof.get("verified_capabilities", ())
        if str(item).strip()
    }
    if donor_capability.removeprefix("capability:") not in verified_capabilities:
        raise SourceTransplantError("Reuse proof did not verify the selected capability.")
    verified_artifacts = {
        str(item).replace("\\", "/")
        for item in proof.get("verified_artifacts", ())
        if str(item).strip()
    }
    donor_paths = {item.path for item in donor.files}
    if not donor_paths or not donor_paths <= verified_artifacts:
        raise SourceTransplantError("Reuse proof did not verify the complete donor artifact set.")
    return donor


def materialize_pinned_donor(
    donor_slice: DonorSlice,
    discovery_client: Any = None,
) -> dict[str, bytes]:
    """GitHub pinned donor downloads are disabled."""
    raise SourceTransplantError("GitHub donor download disabled: CurseForge only.")
    validate_donor_slice_manifest(donor_slice)
    token = str(getattr(discovery_client, "github_token", "") or os.environ.get("GITHUB_TOKEN")
                or os.environ.get("GH_TOKEN") or "").strip()
    client = getattr(discovery_client, "_client", None)
    own_client = False
    if client is None:
        client = _github_client(token)
        own_client = True

    try:
        materialized: dict[str, bytes] = {}
        for donor_file in donor_slice.files:
            raw = _fetch_blob_bytes(
                client, donor_slice.repository, donor_file.blob_sha,
                commit_sha=donor_slice.commit_sha, path=donor_file.path)
            if not raw:
                raise SourceTransplantError(f"Failed to fetch blob for {donor_file.path}")
            actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
            if actual_sha.casefold() != donor_file.sha256.casefold():
                raise SourceTransplantError(
                    f"SHA-256 hash mismatch for {donor_file.path}: expected {donor_file.sha256}, got {actual_sha}"
                )
            if len(raw) != donor_file.size_bytes:
                raise SourceTransplantError(
                    f"Pinned donor size mismatch for {donor_file.path}: expected {donor_file.size_bytes}, got {len(raw)}"
                )
            materialized[donor_file.path] = raw
        return materialized
    finally:
        if own_client:
            client.close()


def _metadata_scope(path: str) -> str | None:
    """Return the Gradle/source module owning a conventional metadata path."""
    parts = PurePosixPath(path).parts
    if not parts or parts[-1] not in _METADATA_NAMES:
        return None
    if "src" in parts and parts[-1] in {"fabric.mod.json", "mods.toml", "neoforge.mods.toml"}:
        return "/".join(parts[:parts.index("src")])
    if len(parts) >= 2 and parts[-2:] == ("gradle", "libs.versions.toml"):
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


def _is_license_path(path: str) -> bool:
    return PurePosixPath(path).name.casefold() in {
        "license", "license.txt", "license.md", "copying", "copying.txt", "copying.md",
    }


def _source_module(path: str, metadata: Mapping[str, str | None]) -> str:
    parts = PurePosixPath(path).parts
    if "src" in parts:
        return "/".join(parts[:parts.index("src")])
    scopes = {scope for item in metadata if (scope := _metadata_scope(item)) is not None}
    return max((scope for scope in scopes if not scope or path.startswith(scope + "/")),
               key=len, default="")


def _build_metadata_sources(client: httpx.Client, *, repository: str,
                            blobs: Mapping[str, str], commit_sha: str = "") -> dict[str, str | None]:
    sources = {}
    for path, blob in sorted(blobs.items()):
        if _metadata_scope(path) is None:
            continue
        try:
            sources[path] = _fetch_blob_bytes(
                client, repository, blob, commit_sha=commit_sha, path=path,
            ).decode("utf-8", errors="replace")
        except SourceTransplantError:
            # An unreadable overriding module file cannot inherit a sibling's
            # or ancestor's version as though the missing evidence were empty.
            sources[path] = None
    return sources


def _build_metadata_text(client: httpx.Client, *, repository: str,
                         blobs: Mapping[str, str]) -> str:
    return "\n".join(f"\n# MMM_METADATA_PATH {path}\n{text}"
                     for path, text in _build_metadata_sources(
                         client, repository=repository, blobs=blobs).items() if text is not None)


def _without_metadata_comments(text: str) -> str:
    # Preserve quoted strings (including URLs and Gradle interpolation) while
    # removing comments before extracting any version authority.
    return re.sub(r'''(?:"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(/\*[\s\S]*?\*/|//[^\n]*|\#[^\n]*)''',
                  lambda match: match.group(1) or "\n" * match.group(0).count("\n"), text)


def _metadata_assignments(path: str, text: str) -> dict[str, str]:
    if path.endswith("libs.versions.toml"):
        section = ""
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("["):
                section = line.strip()
            elif section == "[versions]":
                lines.append(line)
        text = "\n".join(lines)
    separator = r"[=:]" if path.endswith("gradle.properties") else "="
    pattern = (r"(?m)^\s*(?:(?:val|var|def)\s+)?(?:(?:ext|project|rootProject)\.)?"
               r"([A-Za-z_][\w.-]*)(?:[ \t]*:[ \t]*[\w?.<>]+)?[ \t]*" + separator + r"[ \t]*([^\n]+)")
    values = {match[1]: match[2].strip().rstrip(";") for match in re.finditer(pattern, text)}
    for match in re.finditer(r'''\b(?:extra|ext)\[(["'])([A-Za-z_][\w.-]*)\1\]\s*=\s*([^\n]+)''', text):
        values[match[2]] = match[3].strip().rstrip(";")
    if path.endswith("libs.versions.toml"):
        values = {"libs.versions." + key: value for key, value in values.items()}
    return values


def _resolve_metadata_value(value: str, properties: Mapping[str, str], seen=frozenset()) -> str:
    value = value.strip().rstrip(";").strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1]
    key = re.sub(r"^(?:project\.|ext\.)+", "", value)
    key = key.replace("rootProject.ext.", "rootProject.", 1)
    accessor = re.fullmatch(
        r'''(?:(rootProject|project)\.)?(?:findProperty|property)\(["']([\w.-]+)["']\)(?:\s+as\s+String)?''', value)
    provider = re.fullmatch(r'''providers\.gradleProperty\(["']([\w.-]+)["']\)\.get\(\)''', value)
    if accessor:
        key = ("rootProject." if accessor[1] == "rootProject" else "") + accessor[2]
    elif provider:
        key = provider[1]
    if key.startswith("libs.versions.") and key.endswith(".get()"):
        key = key[:-6]
    if key in properties and key not in seen:
        return _resolve_metadata_value(properties[key], properties, seen | {key})

    def interpolate(match):
        reference = match[1] or match[2]
        resolved = _resolve_metadata_value(reference, properties, seen)
        # Missing variables must stay unresolved; replacing them with empty
        # text could turn an unknown version expression into an exact release.
        return resolved if resolved != reference else match[0]

    return re.sub(r"\$\{([^{}]+)\}|\$([A-Za-z_][\w.]*)", interpolate, value)


def _single_minecraft_version(values: Sequence[str]) -> str:
    versions = set(values)
    if len(versions) != 1:
        return ""
    value = next(iter(versions))
    return value if _MINECRAFT_VERSION.fullmatch(value) else ""


def _gradle_owned_metadata(text: str, *, inherited: bool) -> tuple[str, str] | None:
    """Separate local dependencies from explicitly shared project configuration.

    Project selectors and conditional Minecraft mutations need Gradle evaluation;
    they cannot supply global version authority to an unrelated source module.
    """
    quoted = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' '''.rstrip())
    masked = quoted.sub(lambda match: ' ' * len(match[0]), text)
    properties, coordinates = [], []
    stack = [(True, not inherited, False, 0)]
    previous = header_start = 0
    authority = re.compile(r'(?:com\.mojang:minecraft:|net\.minecraftforge:forge:)')
    for token in re.finditer(r'[{}]', masked):
        start = token.start()
        include_properties, include_coordinates, unknown, opened = stack[-1]
        segment = text[previous:start]
        blank = ''.join('\n' if char == '\n' else ' ' for char in segment)
        properties.append(segment if include_properties else blank)
        coordinates.append(segment if include_coordinates else blank)
        if token[0] == '{':
            header = masked[header_start:start]
            shared = bool(re.search(r'\b(?:allprojects|subprojects)\s*$', header))
            subprojects = bool(re.search(r'\bsubprojects\s*$', header))
            # Only these declarative project contexts grant authority. A task,
            # function, conditional or project selector is not a global scope.
            scoped = not bool(re.search(r'(?<![\w.])(?:ext|dependencies|allprojects|subprojects)\s*$', header))
            stack.append((include_properties and not scoped and not (subprojects and not inherited),
                          (include_coordinates or (shared and inherited)) and not scoped
                          and not (subprojects and not inherited), scoped, token.end()))
        elif len(stack) > 1:
            if unknown:
                body = text[opened:start]
                if authority.search(body):
                    return None
                mutations = set(_metadata_assignments('build.gradle', body))
                mutations.update(match[1] for match in re.finditer(
                    r'''(?:setProperty|extraProperties\.set|extra\.set)\(\s*["']([\w.-]+)["']''', body))
                # Keep unknown writes in the property graph. Silently dropping
                # an alias write would revive its earlier literal version.
                properties.append('\n' + '\n'.join(
                    key + ' = <unresolved-scoped-mutation>' for key in sorted(mutations)) + '\n')
            stack.pop()
        else:
            return None
        # Braces are not version literals. Keeping line boundaries is enough
        # for the assignment/dependency extractor, including inline blocks.
        properties.append('\n')
        coordinates.append('\n')
        previous = token.end()
        header_start = previous
    if len(stack) != 1:
        return None
    tail = text[previous:]
    properties.append(tail)
    coordinates.append(tail if not inherited else ''.join('\n' if char == '\n' else ' ' for char in tail))
    return ''.join(properties), ''.join(coordinates)


def _module_metadata_context(metadata: Mapping[str, str | None], module: str) -> tuple[str, str]:
    """Resolve declarative module/ancestor properties, never sibling modules.

    This handles conventional properties, simple Gradle interpolation and version
    catalogs; executable Gradle expressions remain unknown rather than guessed.
    """
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for path, text in metadata.items():
        scope = _metadata_scope(path)
        if scope is None or (scope and module != scope and not module.startswith(scope + "/")):
            continue
        if PurePosixPath(path).name in {"fabric.mod.json", "mods.toml", "neoforge.mods.toml"} and scope != module:
            continue  # Resource manifests are not inherited by child projects.
        groups.setdefault(scope, []).append((path, text))
    properties: dict[str, str] = {}
    version = ""
    has_build_version = False
    shared_coordinates: list[str] = []
    chunks = []
    for scope in sorted(groups, key=lambda value: (len(value), value)):
        files = groups[scope]
        if any(text is None for _path, text in files):
            return "", ""  # Missing nearer metadata cannot authorize inherited source.
        assigned = set()
        cleaned = []
        root_properties = {}
        for path, text in sorted(files, key=lambda item: (not item[0].endswith("gradle.properties"), item[0])):
            chunks.append(f"\n# MMM_METADATA_PATH {path}\n{text}")
            clean = _without_metadata_comments(text)
            if scope and re.search(r"\brootProject\.(?:ext\.)?[\w.]+\s*=", clean):
                return "", ""  # Cross-project mutations require executing Gradle.
            property_text = dependency_text = clean
            if path.endswith(('.gradle', '.gradle.kts')):
                owned = _gradle_owned_metadata(clean, inherited=scope != module)
                if owned is None:
                    return '', ''
                property_text, dependency_text = owned
            cleaned.append((path, clean, dependency_text))
            if not scope:
                root_property_text = clean
                if path.endswith(('.gradle', '.gradle.kts')):
                    root_owned = _gradle_owned_metadata(clean, inherited=False)
                    if root_owned is None:
                        return '', ''
                    root_property_text = root_owned[0]
                for key, value in _metadata_assignments(path, root_property_text).items():
                    root_properties[key] = _resolve_metadata_value(value, root_properties)
            for key, value in _metadata_assignments(path, property_text).items():
                properties[key] = _resolve_metadata_value(value, properties)
                assigned.add(key)
        if not scope:
            for key, value in root_properties.items():
                resolved = _resolve_metadata_value(value, root_properties)
                properties["rootProject." + key] = (
                    resolved if "$" not in resolved else "<unresolved-root-property>" + resolved)
        declarations = [properties[key] for key in assigned if key in _MINECRAFT_PROPERTIES or
                        (key.startswith("libs.versions.") and key[14:] in _MINECRAFT_PROPERTIES | {"minecraft", "mc"})]
        own_coordinates = []
        for path, _clean, dependency_text in cleaned:
            if not path.endswith((".gradle", ".gradle.kts")):
                continue
            for match in re.finditer(
                r'''\b(?:minecraft|modImplementation|implementation|api|compileOnly)\s*(?:\(\s*)?(["'])((?:com\.mojang:minecraft|net\.minecraftforge:forge):[^"']+)\1''', dependency_text):
                own_coordinates.append(match[2])
        if scope != module:
            shared_coordinates.extend(own_coordinates)
        coordinates = []
        for raw_coordinate in dict.fromkeys([*shared_coordinates, *(own_coordinates if scope == module else [])]):
            coordinate = _resolve_metadata_value(raw_coordinate, properties)
            group, artifact, declared = coordinate.split(":", 2)
            if (group, artifact) == ("net.minecraftforge", "forge"):
                match = re.fullmatch(r'(\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?)-[0-9][A-Za-z0-9.+-]*', declared)
                declared = match[1] if match else ''
            coordinates.append(declared)
        if coordinates or declarations:
            # An actual dependency declaration overrides inherited default
            # properties. Conflicting/unknown values never select the target.
            version = _single_minecraft_version(coordinates or [
                _resolve_metadata_value(value, properties) for value in declarations])
            has_build_version = True
        for _path, clean, _dependency_text in cleaned:
            try:
                manifest = json.loads(clean)
            except (ValueError, TypeError):
                continue
            depends = manifest.get("depends") if isinstance(manifest, Mapping) else None
            declared = depends.get("minecraft") if isinstance(depends, Mapping) else None
            values = declared if isinstance(declared, list) else [declared]
            exact = _single_minecraft_version([
                _resolve_metadata_value(value, properties) if isinstance(value, str) else ""
                for value in values])
            if exact and has_build_version and version != exact:
                version = ""
            elif exact and not has_build_version:
                version = exact
    return version, "\n".join(chunks)


def _compatibility_evidence(text: str, minecraft_version: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    folded = _without_metadata_comments(text).casefold()
    loader = adapter.loader.casefold()
    loader_aliases = {
        "fabric": ("fabric", "fabricloader", "fabric.mod.json"),
        "forge": ("forge", "net.minecraftforge", "mods.toml", "javafml"),
        "neoforge": ("neoforge", "net.neoforged", "neoforge.mods.toml"),
    }
    loader_evidenced = any(
        alias in folded for alias in loader_aliases.get(loader, (loader,))
    )
    mixin_count = len(re.findall(r"\bmixins?\b", folded))
    has_access_widener = ".accesswidener" in folded or "accesswidener" in folded

    if (
        minecraft_version == adapter.minecraft_version
    ) and loader_evidenced:
        status = "metadata_exact"
    elif loader_evidenced or minecraft_version:
        status = "metadata_adapt"
    else:
        status = "unverified"

    return CompatibilityEvidence(
        minecraft_version=minecraft_version,
        loader=adapter.loader if loader_evidenced else "",
        mixin_count=mixin_count,
        has_access_widener=has_access_widener,
        status=status,
    )


def _target_compatibility_evidence(text: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    matches = list(re.finditer(r"(?m)^# MMM_METADATA_PATH (.+)\n", text))
    metadata = {match[1]: text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
                for index, match in enumerate(matches)} if matches else {"build.gradle": text}
    version, scoped_text = _module_metadata_context(metadata, "")
    return _compatibility_evidence(scoped_text, version, adapter=adapter)


def _target_compatibility(text: str, *, adapter: PlatformAdapter) -> str:
    evidence = _target_compatibility_evidence(text, adapter=adapter)
    return (
        "exact"
        if evidence.status == "metadata_exact"
        else ("adapt" if evidence.status == "metadata_adapt" else "unverified")
    )


def _declared_dependencies(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    values: list[str] = []
    for value in _GRADLE_DEP.findall(text):
        cleaned = " ".join(value.split())[:256]
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return tuple(values)


__all__ = [
    "ArtifactEdge",
    "ArtifactNode",
    "DonorFile",
    "DonorSlice",
    "SourceTransplantError",
    "donor_closure_sha256",
    "inspect_repository_slice",
    "materialize_source_slices",
    "repository_from_candidate",
    "repository_inspection_failures",
    "repository_inspection_requires_unavailable_network",
    "repository_inspection_session",
    "validate_donor_slice_manifest",
    "validated_reuse_donor",
]
