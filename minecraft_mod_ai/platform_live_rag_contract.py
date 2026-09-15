from __future__ import annotations

import threading
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


def install(*, retrieval_module: Any) -> None:
    """Require one exact target at every production RAG retrieval boundary.

    Target applicability, mapping identity, scoring, graph expansion and receipt creation remain
    owned by ``OfficialCorpusIndex.retrieve``. Missing or invalid targets are contract failures;
    this module never converts them into empty evidence or serial/deferred fallback execution.
    The central research graph contract is enforced natively by ``parallel_runtime_contract``;
    this installer does not replace central-research entrypoints at runtime.
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


__all__ = ["PlatformTargetContractError", "install"]
