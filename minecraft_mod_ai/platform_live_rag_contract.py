from __future__ import annotations

import threading
from functools import wraps
from typing import Any


_RAG_THREAD_STATE = threading.local()


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
    version = str(minecraft_version or "").strip()
    loader_id = str(loader or "").strip().casefold()
    mapping_id = str(mappings or "").strip()
    if not version or not loader_id or not mapping_id:
        raise retrieval.SpecValidationError(
            "Official RAG requires a host-selected minecraft_version, loader and mappings."
        )
    return version, loader_id, mapping_id


def install(*, retrieval_module: Any) -> None:
    """Require an explicit live target while preserving the public retrieval implementation.

    Target applicability, mapping identity, scoring, graph expansion and receipt creation all
    remain owned by ``OfficialCorpusIndex.retrieve``.  This contract only tightens the runtime
    boundary and reuses one immutable index per worker thread.  Keeping a second copy of the
    ranking algorithm here made the contract depend on deleted private methods and allowed the
    wrapper to drift from the verified retrieval implementation.
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
            version, loader_id, mapping_id = _required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
            )
            return original(
                self,
                query,
                minecraft_version=version,
                loader=loader_id,
                mappings=mapping_id,
                limit=limit,
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
            version, loader_id, mapping_id = _required_target(
                retrieval_module,
                minecraft_version,
                loader,
                mappings,
            )
            return _thread_index(retrieval_module).retrieve(
                query,
                minecraft_version=version,
                loader=loader_id,
                mappings=mapping_id,
                limit=limit,
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


__all__ = ["install"]
