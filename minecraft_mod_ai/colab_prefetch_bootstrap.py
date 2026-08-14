from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import Future
from contextvars import ContextVar
from functools import wraps
from typing import Any


_PLATFORM_LOCK = threading.RLock()
_PLATFORM_FUTURE: Future[Any] | None = None
_PLATFORM_ORIGINAL: Any = None
_PLATFORM_CATALOG_PREFETCH_STARTED = False
_WORK_GRAPH_FAST_SERIALIZE: ContextVar[bool] = ContextVar(
    "mmm_work_graph_fast_serialize",
    default=False,
)
_PROPOSAL_HASH_CACHE: ContextVar[dict[int, tuple[Any, str]] | None] = ContextVar(
    "mmm_proposal_store_hash_cache",
    default=None,
)


def _start_platform_future() -> Future[Any]:
    global _PLATFORM_FUTURE
    with _PLATFORM_LOCK:
        current = _PLATFORM_FUTURE
        if current is not None and not current.cancelled():
            return current
        future: Future[Any] = Future()
        _PLATFORM_FUTURE = future

        def worker() -> None:
            try:
                future.set_result(_PLATFORM_ORIGINAL())
            except BaseException as exc:  # pragma: no cover - network boundary
                future.set_exception(exc)

        threading.Thread(
            target=worker,
            daemon=True,
            name="mmm_platform_prefetch",
        ).start()
        return future


def _start_platform_catalog_prefetch(live: Any) -> None:
    """Warm shared official metadata without blocking package initialization."""

    global _PLATFORM_CATALOG_PREFETCH_STARTED
    with _PLATFORM_LOCK:
        if _PLATFORM_CATALOG_PREFETCH_STARTED:
            return
        _PLATFORM_CATALOG_PREFETCH_STARTED = True

    def worker() -> None:
        try:
            # Shared loader/API/Loom/Gradle/Mojang metadata first, then the small
            # version-specific Java requirement set. Both functions are process
            # cached, so the planner later consumes the exact same official data.
            live._common_platform_metadata()
            live._stable_java_versions()
        except BaseException:
            # Network failures are intentionally not sticky; lru_cache does not
            # retain exceptions, so the normal planner path can retry and then use
            # its existing compatibility fallback if the network is still down.
            return

    threading.Thread(
        target=worker,
        daemon=True,
        name="mmm_platform_catalog_prefetch",
    ).start()


def _install_platform_prefetch() -> None:
    """Overlap official Fabric/Mojang metadata lookup with earlier runtime work."""

    global _PLATFORM_ORIGINAL, _PLATFORM_FUTURE

    from . import platform_live_discovery as live

    current = live.discover_game_versions
    if getattr(current, "_mmm_platform_singleflight_prefetch", False):
        _start_platform_future()
        _start_platform_catalog_prefetch(live)
        return

    _PLATFORM_ORIGINAL = current

    @wraps(current)
    def discover_game_versions():
        global _PLATFORM_FUTURE
        future = _start_platform_future()
        try:
            return future.result()
        except BaseException:
            # The canonical caller already has an offline compatibility fallback.
            # Do not permanently memoize a transient network failure: a later call
            # may start a fresh official lookup after connectivity recovers.
            with _PLATFORM_LOCK:
                if _PLATFORM_FUTURE is future:
                    _PLATFORM_FUTURE = None
            raise

    discover_game_versions._mmm_platform_singleflight_prefetch = True
    discover_game_versions.__wrapped__ = current
    live.discover_game_versions = discover_game_versions
    _start_platform_future()
    _start_platform_catalog_prefetch(live)


def _install_request_byte_fastpath() -> None:
    """Use the C JSON encoder for whole-string byte checks in lossless paging."""

    from . import game_design

    current = game_design._json_text_bytes
    if getattr(current, "_mmm_c_json_byte_count", False):
        return

    @wraps(current)
    def json_text_bytes(value: str) -> int:
        if not isinstance(value, str):
            return current(value)
        # json.dumps(string, ensure_ascii=False) adds exactly the two surrounding
        # quote bytes. Its escaping rules are the same rules implemented by
        # _json_character_bytes, but the full scan runs in the C encoder.
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8")) - 2

    json_text_bytes._mmm_c_json_byte_count = True
    json_text_bytes.__wrapped__ = current
    game_design._json_text_bytes = json_text_bytes


def _install_work_graph_fastpath() -> None:
    """Avoid deep-copying graph payloads solely to hash a synchronous compile."""

    from . import work_graph

    node_cls = work_graph.WorkNode
    current_to_dict = node_cls.to_dict
    if not getattr(current_to_dict, "_mmm_compile_shallow_dict", False):

        @wraps(current_to_dict)
        def to_dict(self: Any) -> dict[str, Any]:
            if not _WORK_GRAPH_FAST_SERIALIZE.get():
                return current_to_dict(self)
            return {
                "node_id": self.node_id,
                "stage": self.stage,
                "input_hash": self.input_hash,
                "dependencies": self.dependencies,
                "payload": self.payload,
                "resource_class": self.resource_class,
            }

        to_dict._mmm_compile_shallow_dict = True
        to_dict.__wrapped__ = current_to_dict
        node_cls.to_dict = to_dict

    current_topological = work_graph._topological_modules
    if not getattr(current_topological, "_mmm_heap_deterministic_no_child_sort", False):

        @wraps(current_topological)
        def topological_modules(modules: Any):
            import heapq

            lookup = {module.module_id: module for module in modules}
            indegree = {
                module.module_id: len(module.depends_on)
                for module in modules
            }
            outgoing: dict[str, list[str]] = {
                module.module_id: [] for module in modules
            }
            for module in modules:
                for dependency in module.depends_on:
                    outgoing[dependency].append(module.module_id)
            ready = [
                node_id for node_id, degree in indegree.items() if degree == 0
            ]
            heapq.heapify(ready)
            ordered: list[Any] = []
            while ready:
                node_id = heapq.heappop(ready)
                ordered.append(lookup[node_id])
                # The heap already defines the globally deterministic next node.
                # Sorting each adjacency list cannot change that order.
                for dependent in outgoing[node_id]:
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        heapq.heappush(ready, dependent)
            if len(ordered) != len(modules):
                raise work_graph.WorkGraphError(
                    "Production module graph contains a cycle."
                )
            return tuple(ordered)

        topological_modules._mmm_heap_deterministic_no_child_sort = True
        topological_modules.__wrapped__ = current_topological
        work_graph._topological_modules = topological_modules

    current_build = work_graph.build_production_work_plan
    if getattr(current_build, "_mmm_compile_shallow_serialization", False):
        return

    @wraps(current_build)
    def build_production_work_plan(*args: Any, **kwargs: Any):
        token = _WORK_GRAPH_FAST_SERIALIZE.set(True)
        try:
            return current_build(*args, **kwargs)
        finally:
            _WORK_GRAPH_FAST_SERIALIZE.reset(token)

    build_production_work_plan._mmm_compile_shallow_serialization = True
    build_production_work_plan.__wrapped__ = current_build
    work_graph.build_production_work_plan = build_production_work_plan

    # Keep modules that imported the function before bootstrap on the same owner.
    for loaded in tuple(sys.modules.values()):
        if loaded is None:
            continue
        try:
            if getattr(loaded, "build_production_work_plan", None) is current_build:
                setattr(loaded, "build_production_work_plan", build_production_work_plan)
        except (AttributeError, TypeError):
            continue


def _install_proposal_store_hash_fastpath() -> None:
    """Reuse the hash already verified by CompleteProposal.from_dict during one load."""

    from . import complete_spec, proposal_store

    current_hash = complete_spec.CompleteProposal.calculate_hash
    if not getattr(current_hash, "_mmm_store_invocation_hash_cache", False):

        @wraps(current_hash)
        def calculate_hash(self: Any) -> str:
            cache = _PROPOSAL_HASH_CACHE.get()
            if cache is None:
                return current_hash(self)
            key = id(self)
            cached = cache.get(key)
            if cached is not None and cached[0] is self:
                return cached[1]
            digest = current_hash(self)
            cache[key] = (self, digest)
            return digest

        calculate_hash._mmm_store_invocation_hash_cache = True
        calculate_hash.__wrapped__ = current_hash
        complete_spec.CompleteProposal.calculate_hash = calculate_hash

    current_load = proposal_store.complete_proposal_from_index
    if getattr(current_load, "_mmm_store_hash_scope", False):
        return

    @wraps(current_load)
    def complete_proposal_from_index(*args: Any, **kwargs: Any):
        token = _PROPOSAL_HASH_CACHE.set({})
        try:
            return current_load(*args, **kwargs)
        finally:
            _PROPOSAL_HASH_CACHE.reset(token)

    complete_proposal_from_index._mmm_store_hash_scope = True
    complete_proposal_from_index.__wrapped__ = current_load
    proposal_store.complete_proposal_from_index = complete_proposal_from_index


def start(model_registry_module: Any) -> None:
    """Start non-blocking metadata prefetch and install bootstrap hot paths."""

    del model_registry_module
    _install_request_byte_fastpath()
    _install_work_graph_fastpath()
    _install_proposal_store_hash_fastpath()
    _install_platform_prefetch()
    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", "12")
    os.environ.setdefault("MMM_RESEARCH_WORKERS", "8")


__all__ = ["start"]
