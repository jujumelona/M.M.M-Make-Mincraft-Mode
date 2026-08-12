from __future__ import annotations

import threading
from functools import wraps
from typing import Any, Callable


_THREAD_STATE = threading.local()
_INSTALL_LOCK = threading.RLock()


def _thread_index(retrieval_module: Any) -> Any:
    """Build the immutable builtin corpus index at most once per worker thread."""
    key = (
        id(retrieval_module.OfficialCorpusIndex),
        id(retrieval_module.BUILTIN_CORPUS),
    )
    entries = getattr(_THREAD_STATE, "indexes", None)
    if entries is None:
        entries = {}
        _THREAD_STATE.indexes = entries
    index = entries.get(key)
    if index is None:
        index = retrieval_module.OfficialCorpusIndex(
            documents=retrieval_module.BUILTIN_CORPUS
        )
        entries[key] = index
    return index


def _shared_retrieve_factory(retrieval_module: Any) -> Callable[..., Any]:
    original = retrieval_module.retrieve_official_evidence

    @wraps(original)
    def retrieve_official_evidence_cached_index(
        query: str,
        *,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        mappings: str = "yarn-1.20.1+build.1",
        limit: int = 6,
    ) -> Any:
        return _thread_index(retrieval_module).retrieve(
            query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            limit=limit,
        )

    retrieve_official_evidence_cached_index._mmm_thread_local_index_reuse = True  # type: ignore[attr-defined]
    return retrieve_official_evidence_cached_index


def _replace_kwonly_default(function: Any, name: str, value: Any) -> None:
    defaults = getattr(function, "__kwdefaults__", None)
    if not isinstance(defaults, dict) or name not in defaults:
        return
    updated = dict(defaults)
    updated[name] = value
    function.__kwdefaults__ = updated


def install(
    *,
    retrieval_module: Any,
    central_research_module: Any,
    platform_planning_module: Any,
) -> None:
    """Reuse only the immutable index; every query is still evaluated independently."""
    with _INSTALL_LOCK:
        current = retrieval_module.retrieve_official_evidence
        if getattr(current, "_mmm_thread_local_index_reuse", False):
            shared = current
        else:
            shared = _shared_retrieve_factory(retrieval_module)
            retrieval_module.retrieve_official_evidence = shared

        # central_research imported the function directly and also captured it in a
        # keyword-only default. Patch both references before later parallel wrappers
        # capture the legacy lane.
        central_research_module.retrieve_official_evidence = shared
        _replace_kwonly_default(
            central_research_module.retrieve_domain_evidence,
            "retrieve",
            shared,
        )

        # platform_live_rag_contract installs this exact-target helper using a fresh
        # OfficialCorpusIndex per call. Keep the target semantics but reuse the same
        # immutable thread-local index.
        target = getattr(platform_planning_module, "_target_retrieve", None)
        if callable(target) and not getattr(target, "_mmm_thread_local_index_reuse", False):

            @wraps(target)
            def target_retrieve_cached_index(
                retrieval: Any,
                query: str,
                *,
                adapter: Any,
                limit: int,
            ) -> Any:
                return _thread_index(retrieval).retrieve(
                    query,
                    minecraft_version=adapter.minecraft_version,
                    loader=adapter.loader,
                    mappings=adapter.yarn_mappings,
                    limit=limit,
                )

            target_retrieve_cached_index._mmm_live_platform_rag = True  # type: ignore[attr-defined]
            target_retrieve_cached_index._mmm_thread_local_index_reuse = True  # type: ignore[attr-defined]
            platform_planning_module._target_retrieve = target_retrieve_cached_index

        retrieval_module._mmm_retrieval_index_efficiency_installed = True


__all__ = ["install"]
