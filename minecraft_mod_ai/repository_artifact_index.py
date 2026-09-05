from __future__ import annotations

"""Repository-wide artifact index and lazy immutable blob resolver.

The external-donor path indexes source, build metadata, registries, data/assets and
cross-file relations before capability seed localization. Java and Kotlin source are
both first-class inputs; source language never decides whether a repository is visible.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

    @classmethod
    def build_from_tree(
        cls,
        repository: str,
        commit_sha: str,
        tree_items: Sequence[Mapping[str, Any]],
        *,
        blob_fetcher: Callable[[str, str], bytes] | None = None,
    ) -> RepositoryArtifactIndex:
        """Build a full repository artifact index from Git tree items."""
        index = cls(
            repository=repository,
            commit_sha=commit_sha,
            _blob_fetcher=blob_fetcher,
        )

        for item in tree_items:
            path = str(item.get("path") or "")
            if not path:
                continue
            index.files_by_path[path] = item

            resource_match = _RESOURCE_PATH_RE.match(path)
            if resource_match:
                _domain, namespace, rest = resource_match.group(1), resource_match.group(2), resource_match.group(3)
                logical_id = f"{namespace}:{rest}"
                index.resource_to_path[logical_id] = path
                index.resource_to_path[path] = path

            if path.casefold().endswith((".java", ".kt")):
                filename_symbol = path.split("/")[-1].rsplit(".", 1)[0]
                index.symbol_to_paths.setdefault(filename_symbol, []).append(path)

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
        """Materialize every relevant repository artifact and build one graph.

        Seed selection is intentionally downstream of this method. This prevents a
        filename shortlist from deciding which source bodies are even inspected.
        """

        if self._dependency_graph is not None:
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
                        if ":" in logical_id and not logical_id.startswith("src/")
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
        return graph

    def artifact_bytes(self, path: str) -> bytes | None:
        """Return bytes already admitted to the repository-wide artifact index."""

        if path not in self.files_by_path or not _is_repository_artifact(path):
            return None
        return self.get_blob(path)
