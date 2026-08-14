from __future__ import annotations

import json
import os
import sys
import threading
import weakref
from concurrent.futures import Future
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any


_PLATFORM_LOCK = threading.RLock()
_PLATFORM_FUTURE: Future[Any] | None = None
_PLATFORM_ORIGINAL: Any = None
_PLATFORM_CATALOG_PREFETCH_STARTED = False
_WORK_GRAPH_FAST_SERIALIZE: ContextVar[bool] = ContextVar(
    "mmm_work_graph_fast_serialize",
    default=False,
)
_WORK_GRAPH_MODULE_VALIDATION_CACHE: ContextVar[
    dict[tuple[int, tuple[int, int, int]], Any] | None
] = ContextVar(
    "mmm_work_graph_module_validation_cache",
    default=None,
)
_PROPOSAL_HASH_CACHE: ContextVar[dict[int, tuple[Any, str]] | None] = ContextVar(
    "mmm_proposal_store_hash_cache",
    default=None,
)
_PROPOSAL_CONSTRUCTION_POLICIES: ContextVar[set[tuple[int, int, int]] | None] = ContextVar(
    "mmm_proposal_construction_policies",
    default=None,
)
_PROPOSAL_PROOF_LOCK = threading.RLock()
_PROPOSAL_VALIDATION_PROOFS: dict[
    int,
    tuple[weakref.ReferenceType[Any], dict[tuple[int, int, int], str]],
] = {}


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


def _proposal_policy_key(policy: Any) -> tuple[int, int, int]:
    """Fields that can change CompleteProposal semantic acceptance."""

    return (
        int(policy.max_single_file_bytes),
        int(policy.max_texture_dimension),
        int(policy.max_audio_seconds),
    )


def _remember_proposal_proof(
    proposal: Any,
    policy_key: tuple[int, int, int],
    digest: str,
) -> None:
    object_id = id(proposal)
    with _PROPOSAL_PROOF_LOCK:
        current = _PROPOSAL_VALIDATION_PROOFS.get(object_id)
        if current is not None and current[0]() is proposal:
            current[1][policy_key] = digest
            return

        def cleanup(reference: weakref.ReferenceType[Any], key: int = object_id) -> None:
            with _PROPOSAL_PROOF_LOCK:
                owned = _PROPOSAL_VALIDATION_PROOFS.get(key)
                if owned is not None and owned[0] is reference:
                    _PROPOSAL_VALIDATION_PROOFS.pop(key, None)

        reference = weakref.ref(proposal, cleanup)
        _PROPOSAL_VALIDATION_PROOFS[object_id] = (
            reference,
            {policy_key: digest},
        )


def _proposal_proof(
    proposal: Any,
    policy_key: tuple[int, int, int],
) -> str:
    with _PROPOSAL_PROOF_LOCK:
        current = _PROPOSAL_VALIDATION_PROOFS.get(id(proposal))
        if current is None or current[0]() is not proposal:
            return ""
        return current[1].get(policy_key, "")


def _install_proposal_validation_fastpath() -> None:
    """Reuse semantic validation only when the full current payload hash proves it."""

    from . import complete_spec
    from .scale_policy import ScalePolicy

    proposal_cls = complete_spec.CompleteProposal
    current_validate = proposal_cls.validate
    if not getattr(current_validate, "_mmm_hash_proven_semantic_validation", False):

        @wraps(current_validate)
        def validate(self: Any, *, policy: Any = None) -> None:
            effective = policy or ScalePolicy.from_environment()
            effective.validate()
            policy_key = _proposal_policy_key(effective)
            if _WORK_GRAPH_FAST_SERIALIZE.get():
                proof = _proposal_proof(self, policy_key)
                if proof and self.calculate_hash() == proof:
                    return

            current_validate(self, policy=effective)
            construction = _PROPOSAL_CONSTRUCTION_POLICIES.get()
            if construction is not None:
                construction.add(policy_key)
            # A non-empty approval hash has just been recomputed and checked by the
            # authoritative validator, so it is already the exact payload proof.
            if self.approval_hash:
                _remember_proposal_proof(self, policy_key, self.approval_hash)

        validate._mmm_hash_proven_semantic_validation = True
        validate.__wrapped__ = current_validate
        proposal_cls.validate = validate

    current_parts = complete_spec.complete_proposal_from_parts
    if getattr(current_parts, "_mmm_transfer_validation_proof", False):
        return

    @wraps(current_parts)
    def complete_proposal_from_parts(*args: Any, **kwargs: Any):
        token = _PROPOSAL_CONSTRUCTION_POLICIES.set(set())
        try:
            result = current_parts(*args, **kwargs)
            validated_policies = _PROPOSAL_CONSTRUCTION_POLICIES.get() or set()
            digest = str(result.approval_hash)
            if digest:
                for policy_key in validated_policies:
                    _remember_proposal_proof(result, policy_key, digest)
            return result
        finally:
            _PROPOSAL_CONSTRUCTION_POLICIES.reset(token)

    complete_proposal_from_parts._mmm_transfer_validation_proof = True
    complete_proposal_from_parts.__wrapped__ = current_parts
    complete_spec.complete_proposal_from_parts = complete_proposal_from_parts

    # Several planner/orchestrator modules import this constructor directly during
    # bootstrap. Retarget only M.M.M aliases that still own the exact old function.
    # Inspecting every loaded module with getattr() invokes third-party lazy-module
    # loaders (Transformers in particular), causing unrelated vision dependencies to
    # import during a CPU-only Colab bootstrap.
    _retarget_loaded_proposal_aliases(current_parts, complete_proposal_from_parts)


def _retarget_loaded_proposal_aliases(current: Any, replacement: Any) -> None:
    for module_name, loaded in tuple(sys.modules.items()):
        if not (
            module_name == "minecraft_mod_ai"
            or module_name.startswith("minecraft_mod_ai.")
        ):
            continue
        if loaded is None:
            continue
        try:
            namespace = vars(loaded)
        except TypeError:
            continue
        if namespace.get("complete_proposal_from_parts") is current:
            namespace["complete_proposal_from_parts"] = replacement


def _install_work_graph_fastpath() -> None:
    """Avoid repeated validation, hashing and deep copies during graph compilation."""

    from . import work_graph
    from .scale_policy import ScalePolicy

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

    module_cls = work_graph.ProductionModule
    current_module_validate = module_cls.validate
    if not getattr(current_module_validate, "_mmm_work_graph_validation_cache", False):

        @wraps(current_module_validate)
        def validate_module(self: Any, *, policy: Any = None) -> None:
            cache = _WORK_GRAPH_MODULE_VALIDATION_CACHE.get()
            if cache is None:
                return current_module_validate(self, policy=policy)
            effective = policy or ScalePolicy.from_environment()
            key = (id(self), _proposal_policy_key(effective))
            cached = cache.get(key)
            if cached is self:
                return
            current_module_validate(self, policy=effective)
            cache[key] = self

        validate_module._mmm_work_graph_validation_cache = True
        validate_module.__wrapped__ = current_module_validate
        module_cls.validate = validate_module

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
        serialize_token = _WORK_GRAPH_FAST_SERIALIZE.set(True)
        validation_token = _WORK_GRAPH_MODULE_VALIDATION_CACHE.set({})
        hash_token = _PROPOSAL_HASH_CACHE.set({})
        try:
            return current_build(*args, **kwargs)
        finally:
            _PROPOSAL_HASH_CACHE.reset(hash_token)
            _WORK_GRAPH_MODULE_VALIDATION_CACHE.reset(validation_token)
            _WORK_GRAPH_FAST_SERIALIZE.reset(serialize_token)

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
    _install_proposal_validation_fastpath()
    _install_work_graph_fastpath()
    _install_proposal_store_hash_fastpath()
    _install_platform_prefetch()
    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return
    discovery_workers, research_workers = _colab_worker_defaults()
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", str(discovery_workers))
    os.environ.setdefault("MMM_RESEARCH_WORKERS", str(research_workers))


def _colab_worker_defaults() -> tuple[int, int]:
    """Size network and CPU retrieval pools from the live Colab allocation.

    Discovery is mostly concurrent network I/O, while official-evidence retrieval
    also performs CPU tokenization/ranking.  Keeping separate budgets avoids the
    old fixed 12/8 pools oversubscribing a typical two-vCPU, 12.7-GiB T4 runtime.
    Explicit environment overrides remain authoritative in ``start``.
    """

    cpu_count = max(1, int(os.cpu_count() or 1))
    available_gib = 4
    try:
        for raw_line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if not raw_line.startswith("MemAvailable:"):
                continue
            available_kib = int(raw_line.split()[1])
            available_gib = max(1, available_kib // (1024 * 1024))
            break
    except (OSError, ValueError, IndexError):
        pass

    discovery = min(12, max(4, cpu_count * 4), max(4, available_gib * 2))
    research = min(8, max(2, cpu_count * 2), max(2, available_gib))
    return max(1, discovery), max(1, research)


__all__ = ["_colab_worker_defaults", "start"]
