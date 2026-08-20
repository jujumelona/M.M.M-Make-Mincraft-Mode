from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")
_PREFETCH_LOCK = threading.RLock()
_PREFETCH_FUTURES: dict[tuple[str, str], Future[str]] = {}
_PREFETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mmm_model_prefetch",
)


def _env_workers(name: str, default: int, *, maximum: int = 32) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(1, min(maximum, default))
    try:
        value = int(raw)
    except ValueError:
        return max(1, min(maximum, default))
    return max(1, min(maximum, value))


def _ordered_parallel_map(
    function: Callable[[_T], _R],
    values: Iterable[_T],
    *,
    workers: int,
) -> list[_R]:
    """Run independent work concurrently while preserving deterministic order."""
    items = list(values)
    if len(items) <= 1 or workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(workers, len(items)),
        thread_name_prefix="mmm_parallel_io",
    ) as pool:
        futures = [pool.submit(function, item) for item in items]
        return [future.result() for future in futures]


def _colab_setup_active() -> bool:
    return bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())


def _model_key(config: Any) -> tuple[str, str]:
    model_id = str(getattr(config, "model_id", "")).strip()
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", "")).strip()
        if isinstance(extra, dict)
        else ""
    )
    return model_id, filename


def _eligible_for_model_prefetch(config: Any) -> bool:
    if not _colab_setup_active():
        return False
    if str(getattr(config, "provider", "local")) != "local":
        return False
    if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
        return False
    model_id, _filename = _model_key(config)
    return bool(model_id)


def _prefetch_model_worker(config: Any, resolver: Callable[[Any], str]) -> str:
    return resolver(config)


def _ensure_model_prefetch(
    config: Any,
    resolver: Callable[[Any], str],
) -> Future[str] | None:
    """Start exactly one asynchronous GGUF resolution for a local Colab model."""
    if not _eligible_for_model_prefetch(config):
        return None
    key = _model_key(config)
    with _PREFETCH_LOCK:
        future = _PREFETCH_FUTURES.get(key)
        if future is not None:
            return future
        future = _PREFETCH_EXECUTOR.submit(_prefetch_model_worker, config, resolver)
        _PREFETCH_FUTURES[key] = future
    label = key[1] or Path(key[0]).name or key[0]
    print("GGUF prefetch: started", label, flush=True)
    return future


def resolve_model_path(config: Any, resolver: Callable[[Any], str]) -> str:
    """Reuse an already-running model download, otherwise resolve synchronously."""
    future = _ensure_model_prefetch(config, resolver)
    if future is None:
        return resolver(config)
    return future.result()


def prefetch_profile(profile: Any) -> None:
    """Start planner GGUF prefetch explicitly after a profile is resolved."""
    config = profile.roles.get("planner")
    if config is None or not _eligible_for_model_prefetch(config):
        return
    from .llama_server_autotune import _resolve_model_path_direct

    _ensure_model_prefetch(config, _resolve_model_path_direct)

def _discovery_routes(
    ecosystem_module: Any,
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any] | None,
    discovery: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve only the calls to prefetch; native discovery still owns the response."""
    query = ecosystem_module._seed_query(prompt, game_design)
    if research_brief is not None:
        return query, list(ecosystem_module.external_discovery_routes(research_brief))

    routes = [
        {
            "domain_id": "request",
            "provider": provider,
            "query": query,
            "target_profile": (
                "media" if provider == "openverse_images" else "minecraft_mod"
            ),
        }
        for provider in ("modrinth", "openverse_images")
    ]
    if getattr(discovery, "github_token", "") or os.environ.get("GITHUB_TOKEN"):
        routes.append(
            {
                "domain_id": "request",
                "provider": "github",
                "query": query,
                "target_profile": "minecraft_mod",
            }
        )
    return query, routes


def _discovery_target(
    research_brief: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    selected_target: Mapping[str, Any] = {}
    if isinstance(research_brief, Mapping):
        raw_target = research_brief.get("_mmm_platform_target")
        if isinstance(raw_target, Mapping):
            selected_target = raw_target
    minecraft_version = str(selected_target.get("minecraft_version") or "") or None
    loader = str(selected_target.get("loader") or "") or None
    return minecraft_version, loader


def _discovery_key(
    provider: str,
    query: str,
    *,
    cursor: str = "",
    limit: int = 20,
    minecraft_version: str | None = None,
    loader: str | None = None,
    target_profile: str = "minecraft_mod",
) -> tuple[Any, ...]:
    return (
        provider,
        query,
        cursor,
        limit,
        minecraft_version,
        loader,
        target_profile,
    )


class _PrefetchedDiscoveryClient:
    """Expose the native client API while reusing already-running route requests."""

    def __init__(
        self,
        base: Any,
        futures: dict[tuple[Any, ...], Future[dict[str, Any]]],
    ) -> None:
        self._base = base
        self._futures = futures
        self.transport = getattr(base, "transport", None)
        self.timeout_seconds = getattr(base, "timeout_seconds", 12.0)
        self.github_token = getattr(base, "github_token", "")
        self.openverse_token = getattr(base, "openverse_token", "")

    def search(
        self,
        provider: str,
        query: str,
        *,
        cursor: str = "",
        limit: int = 20,
        minecraft_version: str | None = None,
        loader: str | None = None,
        target_profile: str = "minecraft_mod",
    ) -> dict[str, Any]:
        key = _discovery_key(
            provider,
            query,
            cursor=cursor,
            limit=limit,
            minecraft_version=minecraft_version,
            loader=loader,
            target_profile=target_profile,
        )
        future = self._futures.get(key)
        if future is not None:
            return future.result()
        return self._base.search(
            provider,
            query,
            cursor=cursor,
            limit=limit,
            minecraft_version=minecraft_version,
            loader=loader,
            target_profile=target_profile,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _parallel_discover_seed_bundle_factory(
    ecosystem_module: Any,
    original_discover: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Prefetch independent provider calls without copying native v2 semantics."""

    @wraps(original_discover)
    def discover_seed_bundle_parallel(
        prompt: str,
        game_design: dict[str, Any],
        *,
        research_brief: dict[str, Any] | None = None,
        client: Any = None,
        route_cursor: str = "",
        route_limit: int = 12,
    ) -> dict[str, Any]:
        # Injected clients may deliberately be single-threaded test doubles or custom
        # transports. Preserve their exact call ordering and let native code own them.
        if client is not None:
            return original_discover(
                prompt,
                game_design,
                research_brief=research_brief,
                client=client,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

        mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
        if mode not in {"auto", "on"} or type(route_limit) is not int or not 1 <= route_limit <= 100:
            return original_discover(
                prompt,
                game_design,
                research_brief=research_brief,
                client=client,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

        discovery = ecosystem_module.EcosystemDiscoveryClient()
        _query, routes = _discovery_routes(
            ecosystem_module,
            prompt,
            game_design,
            research_brief,
            discovery,
        )
        minecraft_version, loader = _discovery_target(research_brief)
        route_receipt = ecosystem_module._sha256_text(
            ecosystem_module.canonical_json(
                {
                    "routes": routes,
                    "minecraft_version": minecraft_version or "unresolved",
                    "loader": loader or "unresolved",
                }
            )
        )
        route_offset = ecosystem_module._decode_seed_route_cursor(
            route_cursor,
            route_sha256=route_receipt,
            route_limit=route_limit,
        )
        selected_routes = routes[route_offset : route_offset + route_limit]
        workers = _env_workers("MMM_DISCOVERY_WORKERS", 8, maximum=32)
        if len(selected_routes) <= 1 or workers <= 1:
            return original_discover(
                prompt,
                game_design,
                research_brief=research_brief,
                client=discovery,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

        with ThreadPoolExecutor(
            max_workers=min(workers, len(selected_routes)),
            thread_name_prefix="mmm_discovery",
        ) as pool:
            futures: dict[tuple[Any, ...], Future[dict[str, Any]]] = {}
            for route in selected_routes:
                provider = route["provider"]
                provider_query = route["query"]
                if provider == "openverse_images":
                    provider_query += " visual reference texture architecture objects"
                target_profile = str(route.get("target_profile", "minecraft_mod"))
                target_version = minecraft_version if target_profile == "minecraft_mod" else None
                target_loader = loader if target_profile == "minecraft_mod" else None
                key = _discovery_key(
                    provider,
                    provider_query,
                    limit=10,
                    minecraft_version=target_version,
                    loader=target_loader,
                    target_profile=target_profile,
                )
                if key in futures:
                    continue
                futures[key] = pool.submit(
                    discovery.search,
                    provider,
                    provider_query,
                    limit=10,
                    minecraft_version=target_version,
                    loader=target_loader,
                    target_profile=target_profile,
                )

            prefetched = _PrefetchedDiscoveryClient(discovery, futures)
            return original_discover(
                prompt,
                game_design,
                research_brief=research_brief,
                client=prefetched,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

    discover_seed_bundle_parallel._mmm_parallel_routes = True
    return discover_seed_bundle_parallel


class _PrefetchedRetriever:
    """Deduplicate and pipeline official-RAG primaries and corrective hops."""

    def __init__(
        self,
        retrieve: Callable[..., Any],
        pool: ThreadPoolExecutor,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
    ) -> None:
        self._retrieve = retrieve
        self._pool = pool
        self._minecraft_version = minecraft_version
        self._loader = loader
        self._mappings = mappings
        self._lock = threading.RLock()
        self._futures: dict[tuple[str, int], Future[Any]] = {}

    def prefetch_primary(self, query: str) -> None:
        future = self._submit(query, 8)
        future.add_done_callback(self._prefetch_corrections)

    def _submit(self, query: str, limit: int) -> Future[Any]:
        key = (query, limit)
        with self._lock:
            existing = self._futures.get(key)
            if existing is not None:
                return existing
            future = self._pool.submit(
                self._retrieve,
                query,
                minecraft_version=self._minecraft_version,
                loader=self._loader,
                mappings=self._mappings,
                limit=limit,
            )
            self._futures[key] = future
            return future

    def _prefetch_corrections(self, future: Future[Any]) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        primary = future.result()
        for correction_query in getattr(primary, "correction_queries", ()):
            self._submit(str(correction_query), 4)

    def __call__(self, query: str, **kwargs: Any) -> Any:
        limit = int(kwargs.get("limit", 8))
        expected = {
            "minecraft_version": self._minecraft_version,
            "loader": self._loader,
            "mappings": self._mappings,
            "limit": limit,
        }
        if any(kwargs.get(name) != value for name, value in expected.items()):
            return self._retrieve(query, **kwargs)
        return self._submit(query, limit).result()


def _parallel_retrieve_domain_evidence_factory(
    central_module: Any,
    original_retrieve_graph: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Pipeline native official retrieval while delegating all graph semantics."""
    original_default_retrieve = central_module.retrieve_official_evidence

    @wraps(original_retrieve_graph)
    def retrieve_domain_evidence_parallel(
        research_brief: dict[str, Any],
        *,
        retrieve: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        selected_retrieve = retrieve or original_default_retrieve
        if retrieve is not None and retrieve is not original_default_retrieve:
            return original_retrieve_graph(research_brief, retrieve=retrieve)

        raw_target = research_brief.get("_mmm_platform_target")
        if not isinstance(raw_target, Mapping):
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )
        version = str(raw_target.get("minecraft_version", "")).strip()
        loader = str(raw_target.get("loader", "")).strip().casefold()
        if not version or not loader:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )
        try:
            adapter = central_module.adapter_for_target(version, loader)
        except ValueError:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        raw_domains = research_brief.get("domains")
        if not isinstance(raw_domains, list) or not raw_domains:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )
        try:
            domains = [central_module._research_domain(raw) for raw in raw_domains]
        except Exception:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )
        primary_queries = [
            query
            for domain in domains
            if "official_docs" in domain.providers
            for query in domain.queries
        ]
        workers = _env_workers("MMM_RESEARCH_WORKERS", 8, maximum=32)
        if len(primary_queries) <= 1 or workers <= 1:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        with ThreadPoolExecutor(
            max_workers=min(workers, len(primary_queries)),
            thread_name_prefix="mmm_official_rag",
        ) as pool:
            prefetched = _PrefetchedRetriever(
                selected_retrieve,
                pool,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
            )
            for query in primary_queries:
                prefetched.prefetch_primary(query)
            return original_retrieve_graph(research_brief, retrieve=prefetched)

    retrieve_domain_evidence_parallel._mmm_parallel_rag = True
    return retrieve_domain_evidence_parallel


@lru_cache(maxsize=1)
def _native_discovery_wrapper() -> Callable[..., dict[str, Any]]:
    from . import ecosystem_discovery as ecosystem_module

    return _parallel_discover_seed_bundle_factory(
        ecosystem_module,
        ecosystem_module._serial_discover_seed_bundle,
    )


def discover_seed_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _native_discovery_wrapper()(*args, **kwargs)


@lru_cache(maxsize=1)
def _native_research_wrapper() -> Callable[..., dict[str, Any]]:
    from . import central_research as central_module

    return _parallel_retrieve_domain_evidence_factory(
        central_module,
        central_module._serial_retrieve_domain_evidence,
    )


def retrieve_domain_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _native_research_wrapper()(*args, **kwargs)


def install(
    *,
    model_registry_module: Any,
    llama_server_autotune_module: Any,
) -> None:
    """Compatibility verifier; canonical owners call parallel helpers explicitly."""
    if not hasattr(llama_server_autotune_module, "_resolve_model_path_direct"):
        raise RuntimeError("llama autotune must own native prefetch delegation")
    if not hasattr(model_registry_module.ModelRegistry, "load_profile"):
        raise RuntimeError("model registry is unavailable")


__all__ = [
    "discover_seed_bundle",
    "install",
    "prefetch_profile",
    "resolve_model_path",
    "retrieve_domain_evidence",
]
