from __future__ import annotations

"""Pinned, capability-slice source transplantation for permissive OSS donors.

A repository hit is never a reusable implementation by itself. Reuse is admitted
only after an immutable commit, SPDX license, exact target metadata (or an explicit
adaptation classification), a resource-budgeted dependency-complete source slice,
and hashes for every source blob have been recorded. Execution refetches the pinned blobs and verifies the same
hashes before exposing them to the coder.
"""

import base64
import binascii
import hashlib
import json
import os
import re
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from .platform_catalog import PlatformAdapter
from .repository_artifact_index import RepositoryArtifactIndex
from .reuse_license import is_reusable_source_license

_TYPE_DECL = re.compile(r"\b(?:class|interface|record|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
_METHOD_DECL = re.compile(
    r"\b(?:public|protected|private|static|final|synchronized|abstract|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_MINECRAFT_PROP = re.compile(r"(?m)^\s*minecraft_version\s*=\s*([^\s#]+)")
_GRADLE_DEP = re.compile(r"(?m)^\s*(?:modImplementation|implementation|api|compileOnly|runtimeOnly)\s*[\( ]\s*['\"]([^'\"]+)")
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


def _tree_request_budget() -> int:
    # Work budget only.  It never truncates a successfully enumerated tree.
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


_SNAPSHOT_LOCK = Lock()
_SNAPSHOT_CACHE: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
_SNAPSHOT_INFLIGHT: dict[str, Event] = {}
_BLOB_LOCK = Lock()
_BLOB_CACHE: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_BLOB_CACHE_BYTES = 0
_BLOB_INFLIGHT: dict[tuple[str, str], Event] = {}

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
    pass


def _donor_test_paths(blobs: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for path in blobs
            if path.endswith((".java", ".kt"))
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

    The recursive Git Trees endpoint is used as the fast path.  If GitHub marks it
    truncated (or the response exceeds the configured transport byte budget), the
    repository is walked subtree-by-subtree.  A repository is never discarded just
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
    except SourceTransplantError:
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
    """Inspect one repo and return a pinned minimal Java closure when evidence suffices."""

    snapshot = _repository_snapshot(repository, discovery_client)
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
        metadata_text = _build_metadata_text(client, repository=repository, blobs=blobs)
        compatibility = _target_compatibility(metadata_text, adapter=adapter)
        # A donor with no target evidence is not an adaptation plan; it is an
        # unverified repository hit. Do not admit it into reuse scoring.
        if compatibility not in {"exact", "adapt"}:
            return None
        required_dependencies = _declared_dependencies(metadata_text)
        donor_tests = _donor_test_paths(blobs)
        java_paths = tuple(
            path for path in blobs
            if path.endswith(".java") and "/src/" in f"/{path}"
        )
        if not java_paths:
            java_paths = tuple(path for path in blobs if path.endswith(".java"))
        if not java_paths:
            return None

        tree_items = tuple(
            {"path": path, "sha": blob_sha, "type": "blob"}
            for path, blob_sha in blobs.items()
        )
        index = RepositoryArtifactIndex.build_from_tree(
            repository,
            commit_sha,
            tree_items,
            blob_fetcher=lambda repo, blob_sha: _fetch_blob_bytes(
                client, repo, blob_sha
            ),
        )
        graph = index.build_dependency_graph()
        unreadable = tuple(index.metadata.get("unreadable_artifacts") or ())

        seed_evidence = CapabilityImplementationLocator.locate_seeds(
            capability,
            index,
        )
        seed_paths = tuple(
            evidence.node_id
            for evidence in seed_evidence
            if evidence.node_id in graph.nodes
        )
        if not seed_paths:
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
            return None

        # Build DonorFile list
        files: list[DonorFile] = []
        symbols: set[str] = set()
        for path in selected:
            raw = contents[path]
            text = raw.decode("utf-8", errors="replace")
            local_symbols = tuple(sorted(set(_TYPE_DECL.findall(text)) | set(_METHOD_DECL.findall(text))))
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
        ev = _target_compatibility_evidence(metadata_text, adapter=adapter)

        return DonorSlice(
            capability=capability,
            repository=repository,
            commit_sha=commit_sha,
            license_id=license_id,
            source_url=str(snapshot.get("source_url") or f"https://github.com/{repository}"),
            target_compatibility=compatibility if closure_complete else "unverified",
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
    except SourceTransplantError:
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
    for decision in decisions:
        if not isinstance(decision, Mapping) or decision.get("mode") not in {"source_transplant", "adapt"}:
            continue
        validated.append((decision, validated_reuse_donor(decision)))
    if not validated:
        return {"schema_version": "mmm/reuse-materialization-v1", "donors": [], "count": 0}

    # Validate every donor/proof receipt before creating any local evidence path.
    target_root = root / ".minecraft_ai" / "reuse" / "donors"
    target_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    client = _github_client(token)
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
            written: list[dict[str, Any]] = []
            for item in files:
                if not isinstance(item, Mapping):
                    raise SourceTransplantError("Malformed donor file manifest.")
                path = str(item.get("path") or "").replace("\\", "/")
                blob_sha = str(item.get("blob_sha") or "")
                expected = str(item.get("sha256") or "")
                if not path or path.startswith("/") or ".." in path.split("/"):
                    raise SourceTransplantError("Unsafe donor source path.")
                raw = _fetch_blob_bytes(client, repository, blob_sha)
                if not raw:
                    raise SourceTransplantError(
                        f"Pinned donor blob is empty for {repository}@{commit_sha}:{path}."
                    )
                actual = "sha256:" + hashlib.sha256(raw).hexdigest()
                if actual != expected:
                    raise SourceTransplantError(
                        f"Pinned donor hash mismatch for {repository}@{commit_sha}:{path}."
                    )
                expected_size = item.get("size_bytes")
                if type(expected_size) is not int or len(raw) != expected_size:
                    raise SourceTransplantError(
                        f"Pinned donor size mismatch for {repository}@{commit_sha}:{path}."
                    )
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


def _repository_snapshot(repository: str, discovery_client: Any) -> Mapping[str, Any] | None:
    """Cache immutable repository metadata/tree across target-version evaluation."""

    owner = False
    with _SNAPSHOT_LOCK:
        if repository in _SNAPSHOT_CACHE:
            cached = _SNAPSHOT_CACHE[repository]
            _SNAPSHOT_CACHE.move_to_end(repository)
            return cached
        event = _SNAPSHOT_INFLIGHT.get(repository)
        if event is None:
            event = Event()
            _SNAPSHOT_INFLIGHT[repository] = event
            owner = True
    if not owner:
        event.wait(timeout=30.0)
        with _SNAPSHOT_LOCK:
            return _SNAPSHOT_CACHE.get(repository)

    snapshot: Mapping[str, Any] | None = None
    try:
        evidence = discovery_client.inspect_github_repository(repository)
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
        }
        return snapshot
    except SourceTransplantError:
        return None
    finally:
        with _SNAPSHOT_LOCK:
            if snapshot is not None:
                _SNAPSHOT_CACHE[repository] = snapshot
                _SNAPSHOT_CACHE.move_to_end(repository)
                while len(_SNAPSHOT_CACHE) > _snapshot_cache_entries():
                    _SNAPSHOT_CACHE.popitem(last=False)
            else:
                _SNAPSHOT_CACHE.pop(repository, None)
            pending = _SNAPSHOT_INFLIGHT.pop(repository, None)
            if pending is not None:
                pending.set()


def _github_client(token: str) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mmm-source-transplant"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(timeout=15.0, headers=headers, follow_redirects=False)


def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SourceTransplantError(f"GitHub donor request failed: {url}") from exc
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


def _fetch_blob_bytes(client: httpx.Client, repository: str, blob_sha: str) -> bytes:
    global _BLOB_CACHE_BYTES

    if not re.fullmatch(r"[0-9a-f]{40,64}", blob_sha):
        raise SourceTransplantError("Donor blob is not immutable.")
    key = (repository, blob_sha)
    owner = False
    with _BLOB_LOCK:
        cached = _BLOB_CACHE.get(key)
        if cached is not None:
            _BLOB_CACHE.move_to_end(key)
            return cached
        event = _BLOB_INFLIGHT.get(key)
        if event is None:
            event = Event()
            _BLOB_INFLIGHT[key] = event
            owner = True
    if not owner:
        if not event.wait(timeout=30.0):
            raise SourceTransplantError("Timed out waiting for a concurrent donor blob download.")
        with _BLOB_LOCK:
            cached = _BLOB_CACHE.get(key)
        if cached is None:
            raise SourceTransplantError("Concurrent donor blob download failed.")
        return cached

    try:
        value = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/git/blobs/{quote(blob_sha, safe='')}",
        )
        if not isinstance(value, Mapping) or value.get("encoding") != "base64":
            raise SourceTransplantError("GitHub donor blob is not base64 encoded.")
        try:
            raw = base64.b64decode(
                str(value.get("content") or "").replace("\n", ""),
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise SourceTransplantError("GitHub donor blob contained invalid base64.") from exc
        single_blob_budget = _single_blob_byte_budget()
        if len(raw) > single_blob_budget:
            raise SourceTransplantError(
                f"Single donor blob exceeded configured byte budget ({single_blob_budget} bytes)."
            )
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
        return raw
    finally:
        with _BLOB_LOCK:
            pending = _BLOB_INFLIGHT.pop(key, None)
            if pending is not None:
                pending.set()


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
    """Fetch and verify all files in a donor slice using immutable blob SHAs.

    Validates SHA-256 integrity against donor file manifests. If any blob fails to
    fetch or hash does not match, raises SourceTransplantError (no placeholders allowed).
    """
    validate_donor_slice_manifest(donor_slice)
    token = str(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    client = getattr(discovery_client, "_client", None)
    own_client = False
    if client is None:
        client = _github_client(token)
        own_client = True

    try:
        materialized: dict[str, bytes] = {}
        for df in donor_slice.files:
            raw = _fetch_blob_bytes(client, donor_slice.repository, df.blob_sha)
            if not raw:
                raise SourceTransplantError(f"Failed to fetch blob for {df.path}")
            actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
            if actual_sha.casefold() != df.sha256.casefold():
                raise SourceTransplantError(
                    f"SHA-256 hash mismatch for {df.path}: expected {df.sha256}, got {actual_sha}"
                )
            if len(raw) != df.size_bytes:
                raise SourceTransplantError(
                    f"Pinned donor size mismatch for {df.path}: expected {df.size_bytes}, got {len(raw)}"
                )
            materialized[df.path] = raw
        return materialized
    finally:
        if own_client:
            client.close()


def _build_metadata_text(
    client: httpx.Client,
    *,
    repository: str,
    blobs: Mapping[str, str],
) -> str:
    metadata_paths = (
        "gradle.properties",
        "fabric.mod.json",
        "src/main/resources/fabric.mod.json",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
    )
    chunks: list[str] = []
    for path in metadata_paths:
        blob = blobs.get(path)
        if not blob:
            continue
        try:
            chunks.append(_fetch_blob_bytes(client, repository, blob).decode("utf-8", errors="replace"))
        except SourceTransplantError:
            continue
    return "\n".join(chunks)


def _target_compatibility_evidence(text: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    if not text:
        return CompatibilityEvidence(minecraft_version="", loader="", status="unverified")
    folded = text.casefold()
    loader = adapter.loader.casefold()
    loader_evidenced = loader in folded or (loader == "fabric" and "fabricloader" in folded)
    prop = _MINECRAFT_PROP.search(text)
    mc_ver = prop.group(1).strip() if prop else ""
    if not mc_ver:
        for v in (adapter.minecraft_version, "1.21", "1.20", "1.19"):
            if v in text:
                mc_ver = v
                break
    mixin_count = len(re.findall(r"\bmixins?\b", folded))
    has_aw = ".accesswidener" in text or "accessWidener" in text

    if (mc_ver == adapter.minecraft_version or adapter.minecraft_version in text) and loader_evidenced:
        status = "metadata_exact"
    elif loader_evidenced or mc_ver or (adapter.minecraft_version in text):
        status = "metadata_adapt"
    else:
        status = "unverified"

    return CompatibilityEvidence(
        minecraft_version=mc_ver,
        loader=adapter.loader if loader_evidenced else "",
        mixin_count=mixin_count,
        has_access_widener=has_aw,
        status=status,
    )


def _target_compatibility(text: str, *, adapter: PlatformAdapter) -> str:
    ev = _target_compatibility_evidence(text, adapter=adapter)
    return "exact" if ev.status == "metadata_exact" else ("adapt" if ev.status == "metadata_adapt" else "unverified")


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
    "validate_donor_slice_manifest",
    "validated_reuse_donor",
]
