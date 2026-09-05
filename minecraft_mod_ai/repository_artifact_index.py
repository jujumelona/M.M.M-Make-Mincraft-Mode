from __future__ import annotations

"""Repository-wide artifact index and lazy immutable blob resolver.

The external-donor path indexes source, build metadata, registries, data/assets and
cross-file relations before capability seed localization. Java and Kotlin source are
both first-class inputs; source language never decides whether a repository is visible.

Immutable repo snapshots are cached by repository/commit/tree/fetcher provenance so a
large donor is parsed into one dependency graph and then reused across capability
slicing. The cache is bounded and graph construction is synchronized.
"""

import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .artifact_dependency_graph import ArtifactDependencyGraph, ArtifactKind

_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;?")
_TYPE_RE = re.compile(
    r"\b(?:(?:data|sealed|value|annotation|enum)\s+class|class|interface|object|record|enum)"
    r"\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_JAVA_METHOD_RE = re.compile(
    r"\b(?:public|protected|private|static|final|synchronized|abstract|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_KOTLIN_FUN_RE = re.compile(
    r"\bfun\s+(?:<[^>]+>\s*)?(?:[A-Za-z_][A-Za-z0-9_?.<>]*\.)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_REGISTRY_CALL_RE = re.compile(r'Registry\.[A-Za-z0-9_]+\s*\([^,]+,\s*["\']([^"\']+)["\']')
_NAMESPACED_ID_RE = re.compile(r'["\']([a-z0-9_.-]+:[a-z0-9_/.-]+)["\']')
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_RESOURCE_PATH_RE = re.compile(r"^src/main/resources/(assets|data)/([^/]+)/(.+)$")

_INDEX_CACHE_LOCK = RLock()
_INDEX_CACHE: OrderedDict[
    tuple[str, str, str, str], "RepositoryArtifactIndex"
] = OrderedDict()


def _cache_entry_limit() -> int:
    raw = os.environ.get("MMM_REPOSITORY_ARTIFACT_INDEX_CACHE_ENTRIES", "4").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(32, value))


def _fetcher_namespace(
    blob_fetcher: Callable[[str, str], bytes] | None,
) -> str:
    if blob_fetcher is None:
        return "none"
    module = str(getattr(blob_fetcher, "__module__", "") or "")
    qualname = str(getattr(blob_fetcher, "__qualname__", "") or "")
    if module or qualname:
        return f"{module}:{qualname}"
    fetcher_type = type(blob_fetcher)
    return f"{fetcher_type.__module__}:{fetcher_type.__qualname__}"


def _tree_signature(tree_items: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    rows = sorted(
        (
            str(item.get("path") or ""),
            str(item.get("sha") or item.get("blob_sha") or ""),
            str(item.get("type") or ""),
        )
        for item in tree_items
        if isinstance(item, Mapping) and str(item.get("path") or "")
    )
    for path, sha, kind in rows:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii", errors="ignore"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_context_key(target_context: Mapping[str, Any] | None) -> str:
    encoded = json.dumps(
        dict(target_context or {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clear_repository_artifact_index_cache() -> None:
    """Drop the bounded immutable repository-index cache."""

    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def _is_repository_artifact(path: str) -> bool:
    """Return whether a repository path participates in the canonical graph."""

    kind = ArtifactDependencyGraph.kind_for_path(path)
    if kind is not ArtifactKind.OTHER:
        return True
    normalized = path.replace("\\", "/").casefold()
    return normalized.startswith(
        ("src/main/", "src/client/", "src/server/", "src/common/", "src/api/")
    )


def _declared_methods(path: str, content: str) -> tuple[str, ...]:
    methods = list(_JAVA_METHOD_RE.findall(content))
    if path.casefold().endswith(".kt"):
        methods.extend(_KOTLIN_FUN_RE.findall(content))
    return tuple(dict.fromkeys(methods))


@dataclass
class RepositoryArtifactIndex:
    repository: str
    commit_sha: str
    files_by_path: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    fqcn_to_path: dict[str, str] = field(default_factory=dict)
    symbol_to_paths: dict[str, list[str]] = field(default_factory=dict)
    registry_to_path: dict[str, str] = field(default_factory=dict)
    resource_to_path: dict[str, str] = field(default_factory=dict)
    declared_symbols_by_path: dict[str, tuple[str, ...]] = field(default_factory=dict)
    method_to_paths: dict[str, list[str]] = field(default_factory=dict)
    api_call_to_paths: dict[str, list[str]] = field(default_factory=dict)
    text_by_path: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _blob_fetcher: Callable[[str, str], bytes] | None = None
    _blob_cache: dict[str, bytes] = field(default_factory=dict)
    _dependency_graph: ArtifactDependencyGraph | None = None
    _dependency_graph_context_key: str = ""
    _graph_lock: RLock = field(default_factory=RLock, repr=False)

    @classmethod
    def build_from_tree(
        cls,
        repository: str,
        commit_sha: str,
        tree_items: Sequence[Mapping[str, Any]],
        *,
        blob_fetcher: Callable[[str, str], bytes] | None = None,
    ) -> RepositoryArtifactIndex:
        """Build or reuse an immutable full-repository artifact index."""

        items = tuple(item for item in tree_items if isinstance(item, Mapping))
        cache_key = (
            str(repository).casefold(),
            str(commit_sha).casefold(),
            _tree_signature(items),
            _fetcher_namespace(blob_fetcher),
        )
        with _INDEX_CACHE_LOCK:
            cached = _INDEX_CACHE.get(cache_key)
            if cached is not None:
                if blob_fetcher is not None:
                    # The previous HTTP client may already be closed. Refresh only
                    # the resolver; indexed identity remains commit/tree-bound.
                    cached._blob_fetcher = blob_fetcher
                cached.metadata["index_cache_hits"] = (
                    int(cached.metadata.get("index_cache_hits", 0)) + 1
                )
                _INDEX_CACHE.move_to_end(cache_key)
                return cached

        index = cls(
            repository=repository,
            commit_sha=commit_sha,
            _blob_fetcher=blob_fetcher,
        )

        for item in items:
            path = str(item.get("path") or "")
            if not path:
                continue
            index.files_by_path[path] = item

            resource_match = _RESOURCE_PATH_RE.match(path)
            if resource_match:
                _domain, namespace, rest = (
                    resource_match.group(1),
                    resource_match.group(2),
                    resource_match.group(3),
                )
                logical_id = f"{namespace}:{rest}"
                index.resource_to_path[logical_id] = path
                index.resource_to_path[path] = path

            if path.casefold().endswith((".java", ".kt")):
                filename_symbol = path.split("/")[-1].rsplit(".", 1)[0]
                index.symbol_to_paths.setdefault(filename_symbol, []).append(path)

        index.metadata["index_cache_hits"] = 0
        with _INDEX_CACHE_LOCK:
            raced = _INDEX_CACHE.get(cache_key)
            if raced is not None:
                if blob_fetcher is not None:
                    raced._blob_fetcher = blob_fetcher
                raced.metadata["index_cache_hits"] = (
                    int(raced.metadata.get("index_cache_hits", 0)) + 1
                )
                _INDEX_CACHE.move_to_end(cache_key)
                return raced
            _INDEX_CACHE[cache_key] = index
            while len(_INDEX_CACHE) > _cache_entry_limit():
                _INDEX_CACHE.popitem(last=False)
        return index

    def get_blob(self, path: str) -> bytes | None:
        """Fetch one immutable blob lazily and cache it by repository path."""
        if path in self._blob_cache:
            return self._blob_cache[path]
        item = self.files_by_path.get(path)
        if not item or not self._blob_fetcher:
            return None
        sha = str(item.get("sha") or item.get("blob_sha") or "")
        if not sha:
            return None
        try:
            data = self._blob_fetcher(self.repository, sha)
            self._blob_cache[path] = data
            return data
        except Exception:
            return None

    def populate_source_symbols(self, path: str, content: str) -> None:
        """Index Java/Kotlin packages, declared types, methods, calls and registry IDs."""
        self.text_by_path[path] = content
        package_match = _PACKAGE_RE.search(content)
        package_name = package_match.group(1) if package_match else ""
        declared: list[str] = []
        for match in _TYPE_RE.finditer(content):
            type_name = match.group(1)
            fqcn = f"{package_name}.{type_name}" if package_name else type_name
            self.fqcn_to_path[fqcn] = path
            declared.extend((fqcn, type_name))
            paths = self.symbol_to_paths.setdefault(type_name, [])
            if path not in paths:
                paths.append(path)

        self.declared_symbols_by_path[path] = tuple(dict.fromkeys(declared))

        for method in _declared_methods(path, content):
            paths = self.method_to_paths.setdefault(method, [])
            if path not in paths:
                paths.append(path)

        for call in _CALL_RE.findall(content):
            paths = self.api_call_to_paths.setdefault(call, [])
            if path not in paths:
                paths.append(path)

        for registry_match in _REGISTRY_CALL_RE.finditer(content):
            self.registry_to_path[registry_match.group(1)] = path
        for registry_id in _NAMESPACED_ID_RE.findall(content):
            self.registry_to_path.setdefault(registry_id, path)

    # Compatibility alias for callers/tests written before Kotlin became first-class.
    def populate_java_symbols(self, path: str, content: str) -> None:
        self.populate_source_symbols(path, content)

    def build_dependency_graph(
        self,
        *,
        target_context: Mapping[str, Any] | None = None,
    ) -> ArtifactDependencyGraph:
        """Materialize repository artifacts and cache one graph per target context."""

        context_key = _target_context_key(target_context)
        with self._graph_lock:
            if (
                self._dependency_graph is not None
                and self._dependency_graph_context_key == context_key
            ):
                self.metadata["graph_cache_hits"] = (
                    int(self.metadata.get("graph_cache_hits", 0)) + 1
                )
                return self._dependency_graph

            files: dict[str, bytes] = {}
            unreadable: list[str] = []
            for path in sorted(self.files_by_path):
                if not _is_repository_artifact(path):
                    continue
                raw = self.get_blob(path)
                if raw is None:
                    unreadable.append(path)
                    continue
                files[path] = raw
                kind = ArtifactDependencyGraph.kind_for_path(path)
                if kind in {ArtifactKind.JAVA_SOURCE, ArtifactKind.KOTLIN_SOURCE}:
                    self.populate_source_symbols(
                        path,
                        raw.decode("utf-8", errors="replace"),
                    )

            self.metadata["indexed_artifact_count"] = len(files)
            self.metadata["unreadable_artifacts"] = tuple(unreadable)
            graph_context = dict(target_context or {})
            graph_context.setdefault(
                "owned_packages",
                tuple(
                    sorted(
                        {
                            fqcn.rsplit(".", 1)[0]
                            for fqcn in self.fqcn_to_path
                            if "." in fqcn
                        }
                    )
                ),
            )
            graph_context.setdefault(
                "owned_namespaces",
                tuple(
                    sorted(
                        {
                            logical_id.split(":", 1)[0]
                            for logical_id in self.resource_to_path
                            if ":" in logical_id
                            and not logical_id.startswith("src/")
                        }
                    )
                ),
            )
            graph = ArtifactDependencyGraph.build_from_files(
                files,
                known_symbols=self.declared_symbols_by_path,
                target_context=graph_context,
            )
            self._dependency_graph = graph
            self._dependency_graph_context_key = context_key
            self.metadata.setdefault("graph_cache_hits", 0)
            return graph

    def artifact_bytes(self, path: str) -> bytes | None:
        """Return bytes already admitted to the repository-wide artifact index."""

        if path not in self.files_by_path or not _is_repository_artifact(path):
            return None
        return self.get_blob(path)
