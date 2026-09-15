from __future__ import annotations

import threading
from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

_RAG_THREAD_STATE = threading.local()


class PlatformTargetContractError(ValueError):
    """Raised when production research reaches RAG without one exact platform target."""


def _thread_index(retrieval: Any) -> Any:
    """Build the immutable builtin corpus index once per retrieval worker thread."""
    key = (id(retrieval.OfficialCorpusIndex), id(retrieval.BUILTIN_CORPUS))
    indexes = getattr(_RAG_THREAD_STATE, "indexes", None)
    if indexes is None:
        indexes = {}
        _RAG_THREAD_STATE.indexes = indexes
    index = indexes.get(key)
    if index is None:
        index = retrieval.OfficialCorpusIndex(documents=retrieval.BUILTIN_CORPUS)
        indexes[key] = index
    return index


def _replace_kwonly_default(function: Any, name: str, value: Any) -> None:
    defaults = getattr(function, "__kwdefaults__", None)
    if not isinstance(defaults, dict) or name not in defaults:
        return
    updated = dict(defaults)
    updated[name] = value
    function.__kwdefaults__ = updated


def _required_target(
    retrieval: Any,
    minecraft_version: str | None,
    loader: str | None,
    mappings: str | None,
) -> tuple[str, str, str]:
    del retrieval
    version = str(minecraft_version or "").strip()
    loader_id = str(loader or "").strip().casefold()
    mapping_id = str(mappings or "").strip()
    missing = [
        name
        for name, value in (
            ("minecraft_version", version),
            ("loader", loader_id),
            ("mappings", mapping_id),
        )
        if not value
    ]
    if missing:
        raise PlatformTargetContractError(
            "Live RAG requires one resolved platform target before retrieval; "
            f"missing {', '.join(missing)}. Deferred/empty-target fallback is disabled."
        )
    return version, loader_id, mapping_id


def _with_required_target(
    retrieval: Any,
    minecraft_version: str | None,
    loader: str | None,
    mappings: str | None,
    operation: Callable[[str, str, str], Any],
) -> Any:
    return operation(
        *_required_target(retrieval, minecraft_version, loader, mappings)
    )


def _canonical_research_target(central_module: Any, research_brief: dict[str, Any]) -> Any:
    raw_target = research_brief.get("_mmm_platform_target")
    if not isinstance(raw_target, Mapping):
        raise PlatformTargetContractError(
            "Central research requires _mmm_platform_target before RAG starts. "
            "Serial/deferred research fallback is disabled."
        )

    version = str(
        raw_target.get("minecraft_version") or raw_target.get("game_version") or ""
    ).strip()
    loader = str(raw_target.get("loader") or "").strip().casefold()
    if not version or not loader:
        raise PlatformTargetContractError(
            "Central research platform target requires minecraft_version and loader. "
            "Serial/deferred research fallback is disabled."
        )

    try:
        adapter = central_module.adapter_for_target(version, loader)
    except Exception as exc:  # target resolution is part of the boundary contract
        raise PlatformTargetContractError(
            f"Central research platform target is not executable: {loader}/{version}: {exc}"
        ) from exc

    research_brief["_mmm_platform_target"] = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
    }
    return adapter


def _assert_parallel_research_ready(
    central_module: Any,
    parallel_module: Any,
    research_brief: dict[str, Any],
) -> None:
    _canonical_research_target(central_module, research_brief)

    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise PlatformTargetContractError(
            "Central research requires a non-empty domains list before RAG starts."
        )

    try:
        domains = [central_module._research_domain(raw) for raw in raw_domains]
    except Exception as exc:
        raise PlatformTargetContractError(
            f"Central research contains an invalid domain: {exc}"
        ) from exc

    _query_criteria, domain_queries, _domain_criteria = parallel_module._coverage_query_plan(
        central_module,
        domains,
    )
    official_domains = [domain for domain in domains if "official_docs" in domain.providers]
    if not official_domains:
        return

    primary_queries = list(
        dict.fromkeys(
            query
            for domain in official_domains
            for query in domain_queries[domain.domain_id]
            if str(query).strip()
        )
    )
    if not primary_queries:
        raise PlatformTargetContractError(
            "Official-doc research produced no primary queries. "
            "Serial fallback is disabled; the research graph must supply a real query."
        )


def _install_strict_research_boundary(central_module: Any) -> None:
    """Make fallback-only branches unreachable from the production research entrypoint."""
    from . import parallel_runtime_contract as parallel_module

    current_normalize = central_module.normalize_research_brief
    if not getattr(current_normalize, "_mmm_strict_platform_target", False):

        @wraps(current_normalize)
        def strict_normalize(*args: Any, **kwargs: Any) -> dict[str, Any]:
            brief = current_normalize(*args, **kwargs)
            _canonical_research_target(central_module, brief)
            brief["brief_sha256"] = central_module._sha256(
                central_module.canonical_json(brief)
            )
            return brief

        strict_normalize._mmm_strict_platform_target = True
        central_module.normalize_research_brief = strict_normalize

    current_parallel = parallel_module.retrieve_domain_evidence
    if not getattr(current_parallel, "_mmm_no_rag_fallback", False):

        @wraps(current_parallel)
        def strict_parallel_retrieve(
            research_brief: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            _assert_parallel_research_ready(
                central_module,
                parallel_module,
                research_brief,
            )
            return current_parallel(research_brief, *args, **kwargs)

        strict_parallel_retrieve._mmm_no_rag_fallback = True
        parallel_module.retrieve_domain_evidence = strict_parallel_retrieve


def install(*, retrieval_module: Any) -> None:
    """Require one exact target at every production RAG boundary.

    Target applicability, mapping identity, scoring, graph expansion and receipt creation remain
    owned by ``OfficialCorpusIndex.retrieve``. Missing or invalid targets are contract failures;
    this module never converts them into empty evidence or serial/deferred fallback execution.
    """

    cls = retrieval_module.OfficialCorpusIndex
    original = cls.retrieve
    if not getattr(original, "_mmm_live_platform_rag", False):

        @wraps(original)
        def retrieve(
            self: Any,
            query: str,
            *,
            minecraft_version: str | None = None,
            loader: str | None = None,
            mappings: str | None = None,
            limit: int = 6,
        ):
            return _with_required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
                lambda version, loader_id, mapping_id: original(
                    self,
                    query,
                    minecraft_version=version,
                    loader=loader_id,
                    mappings=mapping_id,
                    limit=limit,
                ),
            )

        retrieve._mmm_live_platform_rag = True
        cls.retrieve = retrieve

    current_public_retrieve = retrieval_module.retrieve_official_evidence
    if getattr(current_public_retrieve, "_mmm_thread_local_index_reuse", False):
        shared_retrieve = current_public_retrieve
    else:

        @wraps(current_public_retrieve)
        def shared_retrieve(
            query: str,
            *,
            minecraft_version: str | None = None,
            loader: str | None = None,
            mappings: str | None = None,
            limit: int = 6,
        ):
            return _with_required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
                lambda version, loader_id, mapping_id: _thread_index(
                    retrieval_module
                ).retrieve(
                    query,
                    minecraft_version=version,
                    loader=loader_id,
                    mappings=mapping_id,
                    limit=limit,
                ),
            )

        shared_retrieve._mmm_thread_local_index_reuse = True
        retrieval_module.retrieve_official_evidence = shared_retrieve

    from . import central_research as central_module

    central_module.retrieve_official_evidence = shared_retrieve
    _replace_kwonly_default(
        central_module.retrieve_domain_evidence,
        "retrieve",
        shared_retrieve,
    )
    _install_strict_research_boundary(central_module)


__all__ = ["PlatformTargetContractError", "install"]
