from __future__ import annotations

"""Shared external retrieval runtime.

Pre-design RAG has one owner: ``research_grounded_rag_contract``. This module provides
process-wide retrieval/cache reuse and donor reuse, and mirrors the owner's explicit
provider gate when it must install the owner in isolation for tests or partial runtimes.
"""

import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from . import research_grounded_rag_contract as _grounded

_BASE_EXTERNAL_RETRIEVAL = _grounded._external_retrieval


def _workers() -> int:
    raw = os.environ.get("MMM_GROUNDED_RAG_WORKERS", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(2, min(16, value))


def _normalize_query(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_text(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_query(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return tuple(out)


def _external_brief_queries(value: Any) -> tuple[str, ...]:
    """Delegate public donor/source scheduling to the grounded-RAG phase owner."""

    return tuple(_grounded._external_brief_queries(value))


def _brief_versions(value: Any) -> tuple[str, ...]:
    found: list[str] = []

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                walk(child, str(child_key).casefold())
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                walk(child, key)
            return
        if (
            key in {"version", "versions", "minecraft_version"}
            and isinstance(node, str)
            and node.strip()
        ):
            found.append(node.strip())

    walk(value)
    return _dedupe_text(found)[:4]


def _repository_key(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        repo = str(metadata.get("repository") or "").strip()
        if repo:
            return repo.casefold()
    return ""


def _repo_url(repo: str) -> str:
    return f"https://github.com/{repo}" if repo else ""


def _provider_failure_payload(query: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "mmm/external-grounded-rag",
        "status": "unavailable",
        "query": query,
        "providers": ["modrinth_public", "github_public_source"],
        "credentials_required": False,
        "project_count": 0,
        "source_repository_count": 0,
        "document_count": 0,
        "actual_source_document_count": 0,
        "coverage_score": 0.0,
        "projects": [],
        "documents": [],
        "errors": [f"{type(exc).__name__}: {exc}"],
        "github_retrieval": {
            "provider_status": "error",
            "search_requests": 0,
            "source_requests": 0,
            "source_bytes": 0,
            "saturation_reason": "",
        },
    }


class GroundedRAGCoordinator:
    """One shared cache/executor around the single concrete external retriever."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=_workers(), thread_name_prefix="mmm-grounded-rag"
        )
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self._donors_by_query: dict[str, list[dict[str, Any]]] = {}

    @property
    def max_workers(self) -> int:
        return int(getattr(self.executor, "_max_workers", _workers()))

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        return self.executor.submit(fn, *args, **kwargs)

    def _closure_documents(
        self, owner: str, repo: str, query: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return _grounded._github_repo_documents(owner, repo, query)

    def _cache_key(
        self, query: str, versions: Sequence[str]
    ) -> tuple[str, tuple[str, ...]]:
        canonical = _normalize_query(query)
        versions_key = tuple(_dedupe_text(tuple(str(v) for v in versions)))
        return canonical.casefold(), versions_key

    def _record_payload(
        self,
        query: str,
        versions: Sequence[str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        canonical = _normalize_query(query)
        item = dict(payload)
        item["query"] = canonical
        item["work_graph"] = {
            "key_space": "canonical_query",
            "bounded_workers": self.max_workers,
            "nested_executor": False,
            "repository_snapshot_dedup": True,
            "query_specific_source_selection": True,
            "fixed_file_count_cutoff": False,
            "single_external_retriever": True,
        }
        donors = [
            dict(document)
            for document in item.get("documents", ())
            if isinstance(document, Mapping) and _repository_key(document)
        ]
        with self._lock:
            self._cache[self._cache_key(canonical, versions)] = dict(item)
            self._donors_by_query[canonical.casefold()] = donors
        return item

    def retrieve_one(
        self,
        query: str,
        versions: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Retrieve one canonical planned query and return its payload directly."""

        canonical = _normalize_query(query)
        if not canonical:
            raise ValueError("external retrieval requires a non-empty planned query")
        key = self._cache_key(canonical, versions)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return dict(cached)

        try:
            payload = _BASE_EXTERNAL_RETRIEVAL(canonical, key[1])
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"external retriever returned {type(payload).__name__}, expected mapping"
                )
        except Exception as exc:  # noqa: BLE001
            payload = _provider_failure_payload(canonical, exc)
        return self._record_payload(canonical, key[1], payload)

    def retrieve_many(
        self,
        queries: Sequence[str],
        versions: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Retrieve planned queries concurrently while preserving caller keys."""

        requested: list[tuple[str, str]] = []
        canonical_to_originals: dict[str, list[str]] = {}
        for raw in queries:
            original = str(raw)
            canonical = _normalize_query(original)
            if not canonical:
                continue
            requested.append((original, canonical))
            canonical_to_originals.setdefault(canonical, []).append(original)

        unique = tuple(canonical_to_originals)
        payload_by_canonical: dict[str, dict[str, Any]] = {}
        if unique:
            work = {
                self.submit(self.retrieve_one, query, versions): query
                for query in unique
            }
            for future in as_completed(work):
                query = work[future]
                payload_by_canonical[query] = dict(future.result())

        result: dict[str, dict[str, Any]] = {}
        for original, canonical in requested:
            if canonical in payload_by_canonical:
                result[original] = dict(payload_by_canonical[canonical])
        return result

    def repositories_for_capabilities(
        self,
        capabilities: Sequence[str],
        capability_graph: Mapping[str, Any] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        terms_by_capability: dict[str, list[str]] = {
            str(cap): [str(cap)] for cap in capabilities
        }
        if isinstance(capability_graph, Mapping):
            for item in capability_graph.get("search_terms", ()):
                if not isinstance(item, Mapping):
                    continue
                cap = str(item.get("capability") or "")
                if cap in terms_by_capability:
                    terms_by_capability[cap].extend(
                        str(value) for value in item.get("terms", ())
                    )

        with self._lock:
            donor_items = [
                (query, [dict(item) for item in docs])
                for query, docs in self._donors_by_query.items()
            ]

        result: dict[str, tuple[str, ...]] = {}
        for capability, terms in terms_by_capability.items():
            needles = set(_grounded._tokens(" ".join(terms)))
            scored: list[tuple[int, str]] = []
            seen: set[str] = set()
            for query, docs in donor_items:
                score = len(needles & set(_grounded._tokens(query)))
                if score <= 0:
                    continue
                for doc in docs:
                    url = _repo_url(_repository_key(doc))
                    if url and url not in seen:
                        seen.add(url)
                        scored.append((score, url))
            scored.sort(key=lambda item: (-item[0], item[1]))
            result[capability] = tuple(url for _score, url in scored[:8])
        return result


_COORDINATOR = GroundedRAGCoordinator()
_INSTALLED = False


def _augment(
    agentic_module: Any,
    payload: Mapping[str, Any],
    research_brief: Mapping[str, Any],
    *,
    versions: Sequence[str],
    local_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the owner's provider-gated public augmentation to an already-built base bundle."""

    return _grounded._augment_bundle(
        agentic_module,
        payload,
        versions=versions,
        local_index=local_index,
        external_queries=_external_brief_queries(research_brief),
    )


def _install_pre_design_owner_if_missing(agentic_module: Any) -> None:
    current = agentic_module._forced_rag_bundle
    if getattr(current, "__mmm_grounded_rag__", False):
        return

    def grounded_rag_bundle(
        router: Any, research_brief: Mapping[str, Any]
    ) -> dict[str, Any]:
        local_index = _grounded._ensure_local_index(agentic_module, router)
        payload = current(router, research_brief)
        versions = tuple(
            str(item) for item in payload.get("versions", ()) if str(item).strip()
        ) or _brief_versions(research_brief)
        return _augment(
            agentic_module,
            payload,
            research_brief,
            versions=versions,
            local_index=local_index,
        )

    grounded_rag_bundle.__name__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__qualname__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__wrapped__ = current
    grounded_rag_bundle.__mmm_grounded_rag__ = True
    agentic_module._forced_rag_bundle = grounded_rag_bundle


def install(agentic_module: Any, reuse_module: Any) -> None:
    """Attach shared retrieval without replacing the installed pre-design RAG owner."""

    global _INSTALLED
    if _INSTALLED:
        return

    _install_pre_design_owner_if_missing(agentic_module)

    original_discovery = reuse_module._parallel_donor_repository_discovery
    if not getattr(original_discovery, "__mmm_grounded_donors__", False):
        def donor_discovery(
            capabilities: Sequence[str],
            client: Any,
            *,
            capability_graph: Mapping[str, Any] | None = None,
        ) -> dict[str, tuple[str, ...]]:
            grounded = _COORDINATOR.repositories_for_capabilities(
                capabilities, capability_graph
            )
            public = (
                {}
                if client is None
                else original_discovery(
                    capabilities, client, capability_graph=capability_graph
                )
            )
            merged: dict[str, tuple[str, ...]] = {}
            for capability in capabilities:
                values = [*grounded.get(capability, ()), *public.get(capability, ())]
                merged[capability] = tuple(
                    dict.fromkeys(
                        str(value) for value in values if str(value).strip()
                    )
                )
            return merged

        donor_discovery.__wrapped__ = original_discovery
        donor_discovery.__mmm_grounded_donors__ = True
        reuse_module._parallel_donor_repository_discovery = donor_discovery

    _grounded._external_retrieval = _COORDINATOR.retrieve_one
    _INSTALLED = True


def coordinator() -> GroundedRAGCoordinator:
    return _COORDINATOR


__all__ = [
    "GroundedRAGCoordinator",
    "coordinator",
    "install",
]
