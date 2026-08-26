from __future__ import annotations

"""Repository-Wide Artifact Index & Lazy Blob Resolver.

Constructs an authoritative full-repository index of FQCNs, declared symbols,
registry identifiers, data/asset resources, and mod metadata before any capability
seed localization or dependency closure is performed.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_PKG_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z0-9_.]+)\s*;")
_CLASS_RE = re.compile(r"\b(?:class|interface|record|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
_REGISTRY_CALL_RE = re.compile(r'Registry\.[A-Za-z0-9_]+\s*\([^,]+,\s*["\']([^"\']+)["\']')
_RESOURCE_PATH_RE = re.compile(r"^src/main/resources/(assets|data)/([^/]+)/(.+)$")


@dataclass
class RepositoryArtifactIndex:
    repository: str
    commit_sha: str
    files_by_path: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    fqcn_to_path: dict[str, str] = field(default_factory=dict)
    symbol_to_paths: dict[str, list[str]] = field(default_factory=dict)
    registry_to_path: dict[str, str] = field(default_factory=dict)
    resource_to_path: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _blob_fetcher: Callable[[str, str], bytes] | None = None
    _blob_cache: dict[str, bytes] = field(default_factory=dict)

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

            # Index resource paths
            m_res = _RESOURCE_PATH_RE.match(path)
            if m_res:
                _domain, ns, rest = m_res.group(1), m_res.group(2), m_res.group(3)
                logical_id = f"{ns}:{rest}"
                index.resource_to_path[logical_id] = path
                index.resource_to_path[path] = path

            # Index Java source FQCN and simple class name
            if path.endswith(".java") or path.endswith(".kt"):
                fname = path.split("/")[-1].rsplit(".", 1)[0]
                index.symbol_to_paths.setdefault(fname, []).append(path)

        return index

    def get_blob(self, path: str) -> bytes | None:
        """Fetch blob for a path using lazy resolver and cache."""
        if path in self._blob_cache:
            return self._blob_cache[path]
        item = self.files_by_path.get(path)
        if not item or not self._blob_fetcher:
            return None
        sha = str(item.get("sha") or item.get("blob_sha") or "")
        if not sha:
            return None
        try:
            bdata = self._blob_fetcher(self.repository, sha)
            self._blob_cache[path] = bdata
            return bdata
        except Exception:
            return None

    def populate_java_symbols(self, path: str, content: str) -> None:
        """Parse package and class declarations from a Java/Kotlin file."""
        pkg_match = _PKG_RE.search(content)
        pkg = pkg_match.group(1) if pkg_match else ""
        for m in _CLASS_RE.finditer(content):
            cls_name = m.group(1)
            fqcn = f"{pkg}.{cls_name}" if pkg else cls_name
            self.fqcn_to_path[fqcn] = path
            paths = self.symbol_to_paths.setdefault(cls_name, [])
            if path not in paths:
                paths.append(path)

        for m_reg in _REGISTRY_CALL_RE.finditer(content):
            reg_id = m_reg.group(1)
            self.registry_to_path[reg_id] = path
