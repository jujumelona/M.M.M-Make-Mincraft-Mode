from __future__ import annotations

"""Keep planner research observable and prevent optional I/O from owning planning.

The canonical planner may use official/local evidence synchronously. Public ecosystem
search and external MCP are optional evidence lanes: pre-design records their complete
route graph without performing network I/O, and specialist stages execute those routes
only when evidence is actually needed. Later explicit discovery keeps bounded provider
I/O, connection reuse, and real parallel MCP sessions.
"""

import os
import threading
import time
import weakref
from copy import deepcopy
from functools import wraps
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False
_EXTERNAL_PROVIDERS = frozenset(
    {
        "modrinth",
        "github",
        "openverse_images",
        "openverse_audio",
        "wikipedia",
        "huggingface_models",
        "openalex_works",
        "crossref_works",
    }
)
_ALLOWED_API_HOSTS = frozenset(
    {
        "api.modrinth.com",
        "api.github.com",
        "api.openverse.org",
        "en.wikipedia.org",
        "ko.wikipedia.org",
        "huggingface.co",
        "api.openalex.org",
        "api.crossref.org",
    }
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _estimated_external_routes(research_brief: dict[str, Any]) -> tuple[int, int]:
    total = 0
    provider_slots = 0
    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list):
        return 0, 0
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, dict):
            continue
        queries = raw_domain.get("queries")
        providers = raw_domain.get("providers")
        if not isinstance(queries, list) or not isinstance(providers, list):
            continue
        external_count = len(
            {
                str(provider)
                for provider in providers
                if str(provider) in _EXTERNAL_PROVIDERS
            }
        )
        if external_count <= 0:
            continue
        provider_slots += external_count
        total += external_count * len(queries)
    return total, provider_slots


def _planning_seed_brief(research_brief: dict[str, Any]) -> dict[str, Any]:
    """Return a lossless planning projection; execution breadth is deferred, not cut."""

    projected = deepcopy(research_brief)
    estimated_routes, _provider_slots = _estimated_external_routes(projected)
    route_budget = _env_int(
        "MMM_ECOSYSTEM_SEED_ROUTE_BUDGET",
        96,
        minimum=16,
        maximum=512,
    )
    projected["_mmm_planning_seed_projection"] = {
        "schema_version": "mmm/planning-seed-projection-v3",
        "estimated_external_routes": estimated_routes,
        "route_budget": route_budget,
        "compacted": False,
        "reason": "lossless_route_graph_external_io_deferred",
        "queries_per_domain": None,
        "full_research_brief_retained": True,
        "domains_and_providers_preserved": True,
        "specialist_discovery_continues_full_brief": True,
    }
    return projected


def _brief_identity(research_brief: dict[str, Any]) -> str:
    sha = research_brief.get("brief_sha256")
    if isinstance(sha, str) and sha:
        target = research_brief.get("_mmm_platform_target")
        if isinstance(target, dict):
            target_key = tuple(
                sorted((str(key), repr(value)) for key, value in target.items())
            )
        else:
            target_key = ()
        return f"{sha}|{target_key!r}"
    return f"id:{id(research_brief)}"


def _ecosystem_key(
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any],
    page_builder: Any,
) -> tuple[str, int, str, int]:
    identity_brief = research_brief
    if not isinstance(research_brief.get("_mmm_platform_target"), dict):
        selection = game_design.get("_platform_selection")
        if isinstance(selection, dict) and isinstance(selection.get("target"), dict):
            identity_brief = {
                **research_brief,
                "_mmm_platform_target": dict(selection["target"]),
            }
    return (
        prompt,
        id(game_design),
        _brief_identity(identity_brief),
        id(page_builder),
    )


def _heartbeat(label: str, stop: threading.Event, started: float) -> None:
    interval = _env_float(
        "MMM_PLANNER_HEARTBEAT_SECONDS",
        15.0,
        minimum=5.0,
        maximum=60.0,
    )
    while not stop.wait(interval):
        print(
            f"planner research: {label} still running elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )


def _patch_pre_design_external_seed(agentic_module: Any, central_module: Any) -> None:
    """Record the full external route graph without doing public network I/O."""

    current = agentic_module.collect_ecosystem_seed_bundle
    if getattr(current, "_mmm_pre_design_external_deferred", False):
        return

    @wraps(current)
    def deferred_seed(
        prompt: str,
        game_design: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        research_brief = kwargs.get("research_brief")
        planning_seed_only = bool(kwargs.get("planning_seed_only", False))
        if not planning_seed_only or not isinstance(research_brief, dict):
            return current(prompt, game_design, *args, **kwargs)

        routes = central_module.external_discovery_routes(research_brief)
        route_receipts = [
            {
                "domain_id": str(route.get("domain_id", "")),
                "provider": str(route.get("provider", "")),
                "target_profile": str(route.get("target_profile", "")),
                "query_sha256": central_module._sha256(str(route.get("query", ""))),
            }
            for route in routes
        ]
        route_sha256 = central_module._sha256(
            central_module.canonical_json(route_receipts)
        )
        print(
            "planner research: ecosystem network deferred",
            f" routes={len(routes)}",
            flush=True,
        )
        return {
            "schema_version": "mmm/ecosystem-planning-deferred-v1",
            "status": "deferred",
            "brief_sha256": str(research_brief.get("brief_sha256", "")),
            "route_sha256": route_sha256,
            "route_count": len(routes),
            "processed_route_count": 0,
            "remaining_route_count": len(routes),
            "routes_complete": not routes,
            "candidate_count": 0,
            "pages": [],
            "errors": [],
            "route_receipts": route_receipts,
            "coverage": (
                "Complete external route graph retained; provider I/O is intentionally "
                "deferred to adaptive specialist research outside the planning critical path."
            ),
            "authorization": "none",
            "download_performed": False,
            "planning_critical_path": False,
        }

    deferred_seed._mmm_pre_design_external_deferred = True  # type: ignore[attr-defined]
    agentic_module.collect_ecosystem_seed_bundle = deferred_seed


def _patch_pre_design_observability(agentic_module: Any) -> None:
    current = agentic_module.collect_pre_design_research
    if getattr(current, "_mmm_pre_design_heartbeat", False):
        return

    @wraps(current)
    def observed(
        router: Any,
        prompt: str,
        *,
        trace_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat,
            args=("pre-design", stop, started),
            daemon=True,
            name="mmm_pre_design_heartbeat",
        )
        print("planner research: pre-design start", flush=True)
        heartbeat.start()
        try:
            result = current(router, prompt, trace_metadata=trace_metadata)
        finally:
            stop.set()
        print(
            "planner research: pre-design complete",
            f" elapsed={time.monotonic() - started:.1f}s",
            sep="",
            flush=True,
        )
        return result

    observed._mmm_pre_design_heartbeat = True  # type: ignore[attr-defined]
    observed.__wrapped__ = current  # type: ignore[attr-defined]
    agentic_module.collect_pre_design_research = observed


def _patch_external_mcp_parallel(external_module: Any) -> None:
    """Allow independent MCP sessions to overlap instead of holding one global lock."""

    cls = external_module.ExternalMCPRouter
    current = cls._call_provider
    if getattr(current, "_mmm_parallel_sessions", False):
        return

    @wraps(current)
    def call_provider_parallel(
        self: Any,
        server_name: str,
        entry: dict[str, Any],
        *,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        import asyncio
        import anyio

        async def run() -> dict[str, Any]:
            return await self._call_provider_async(
                server_name,
                entry,
                tool=tool,
                arguments=arguments,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(run)

        value: dict[str, Any] = {}
        error: list[BaseException] = []

        def worker() -> None:
            try:
                value["result"] = anyio.run(run)
            except BaseException as exc:  # pragma: no cover - thread bridge
                error.append(exc)

        thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"mmm_external_mcp_{server_name}",
        )
        thread.start()
        # This retains the provider's existing safety envelope. The structural fix is
        # that independent calls no longer serialize behind ExternalMCPRouter._lock.
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise external_module.ExternalMCPError(
                f"External MCP {server_name} exceeded the synchronous bridge timeout."
            )
        if error:
            raise external_module.ExternalMCPError(str(error[0])) from error[0]
        return value["result"]

    call_provider_parallel._mmm_parallel_sessions = True  # type: ignore[attr-defined]
    cls._call_provider = call_provider_parallel


def _patch_discovery_http_pool(ecosystem_module: Any) -> None:
    """Reuse one httpx connection pool across every route handled by one client."""

    cls = ecosystem_module.EcosystemDiscoveryClient
    current_init = cls.__init__
    current_get_json = cls._get_json
    if getattr(current_get_json, "_mmm_connection_pool", False):
        return

    @wraps(current_init)
    def init_with_pool(self: Any, *args: Any, **kwargs: Any) -> None:
        if "timeout_seconds" not in kwargs:
            kwargs["timeout_seconds"] = _env_float(
                "MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS",
                8.0,
                minimum=2.0,
                maximum=30.0,
            )
        current_init(self, *args, **kwargs)
        pooled = ecosystem_module.httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        )
        self._mmm_http_client = pooled
        self._mmm_http_finalizer = weakref.finalize(self, pooled.close)

    init_with_pool._mmm_planner_timeout_default = True  # type: ignore[attr-defined]
    init_with_pool._mmm_connection_pool = True  # type: ignore[attr-defined]

    @wraps(current_get_json)
    def get_json_pooled(
        self: Any,
        url: str,
        *,
        params: dict[str, str] | None = None,
        provider: str = "",
        include_next_url: bool = False,
    ) -> Any:
        parsed = ecosystem_module.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_API_HOSTS:
            raise ecosystem_module.SpecValidationError(
                "Discovery request escaped the API allowlist."
            )
        if parsed.hostname == "huggingface.co" and not (
            parsed.path == "/api/models" or parsed.path.startswith("/api/models/")
        ):
            raise ecosystem_module.SpecValidationError(
                "Hugging Face discovery is restricted to metadata API paths."
            )
        if parsed.hostname == "api.openalex.org" and not (
            parsed.path == "/works" or parsed.path.startswith("/works/")
        ):
            raise ecosystem_module.SpecValidationError(
                "OpenAlex discovery is restricted to works metadata paths."
            )
        if parsed.hostname == "api.crossref.org" and not (
            parsed.path == "/works" or parsed.path.startswith("/works/")
        ):
            raise ecosystem_module.SpecValidationError(
                "Crossref discovery is restricted to works metadata paths."
            )

        headers = {
            "Accept": "application/json",
            "User-Agent": ecosystem_module._USER_AGENT,
        }
        if provider == "github":
            headers["X-GitHub-Api-Version"] = "2022-11-28"
            headers["Accept"] = "application/vnd.github+json"
            if self.github_token:
                headers["Authorization"] = f"Bearer {self.github_token}"
        elif provider == "openverse" and self.openverse_token:
            headers["Authorization"] = f"Bearer {self.openverse_token}"

        client = getattr(self, "_mmm_http_client", None)
        if client is None:
            client = ecosystem_module.httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            )
            self._mmm_http_client = client
            self._mmm_http_finalizer = weakref.finalize(self, client.close)
        try:
            response = client.get(url, params=params, headers=headers)
        except ecosystem_module.httpx.HTTPError as exc:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                f"{parsed.hostname} discovery request failed: {type(exc).__name__}."
            ) from exc
        if response.status_code != 200:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                f"{parsed.hostname} discovery returned HTTP {response.status_code}."
            )
        if len(response.content) > ecosystem_module._MAX_RESPONSE_BYTES:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                f"{parsed.hostname} discovery response exceeded the byte policy."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                f"{parsed.hostname} discovery returned invalid JSON."
            ) from exc
        if include_next_url:
            next_link = response.links.get("next")
            next_url = (
                str(next_link.get("url") or "")
                if isinstance(next_link, dict)
                else ""
            )
            return payload, next_url
        return payload

    get_json_pooled._mmm_connection_pool = True  # type: ignore[attr-defined]
    cls.__init__ = init_with_pool
    cls._get_json = get_json_pooled


def _patch_worker_defaults(parallel_module: Any, agentic_module: Any) -> None:
    current_parallel = parallel_module._env_workers
    if not getattr(current_parallel, "_mmm_planning_io_tuned", False):

        @wraps(current_parallel)
        def tuned_parallel_workers(
            name: str,
            default: int,
            *,
            maximum: int = 32,
        ) -> int:
            if not os.environ.get(name, "").strip():
                if name == "MMM_DISCOVERY_WORKERS":
                    default = 24
                elif name == "MMM_RESEARCH_WORKERS":
                    default = 16
            return current_parallel(name, default, maximum=maximum)

        tuned_parallel_workers._mmm_planning_io_tuned = True  # type: ignore[attr-defined]
        parallel_module._env_workers = tuned_parallel_workers

    current_agentic = agentic_module._env_workers
    if not getattr(current_agentic, "_mmm_planning_io_tuned", False):

        @wraps(current_agentic)
        def tuned_agentic_workers(
            name: str = "MMM_RESEARCH_WORKERS",
            default: int = 8,
        ) -> int:
            if name == "MMM_RESEARCH_WORKERS" and not os.environ.get(name, "").strip():
                default = 16
            return current_agentic(name, default)

        tuned_agentic_workers._mmm_planning_io_tuned = True  # type: ignore[attr-defined]
        agentic_module._env_workers = tuned_agentic_workers


def _patch_complete_planner(complete_planner_module: Any) -> None:
    """Add observability only; never create another research future here."""

    current_impl = complete_planner_module._retrieve_implementation_evidence
    current_collect = complete_planner_module.collect_ecosystem_seed_bundle
    if getattr(current_impl, "_mmm_stall_guard", False):
        return

    @wraps(current_impl)
    def implementation_evidence_observed(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = current_impl(prompt, game_design, research_brief)
        print(
            "planner research: official RAG complete",
            f" elapsed={time.monotonic() - started:.1f}s",
            sep="",
            flush=True,
        )
        return result

    implementation_evidence_observed._mmm_stall_guard = True  # type: ignore[attr-defined]
    implementation_evidence_observed._mmm_parallel_target_rag = True  # type: ignore[attr-defined]
    implementation_evidence_observed._mmm_agentic_rag_fusion = True  # type: ignore[attr-defined]

    @wraps(current_collect)
    def ecosystem_seed_observed(
        prompt: str,
        game_design: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        started = time.monotonic()
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat,
            args=("ecosystem seed", stop, started),
            daemon=True,
            name="mmm_ecosystem_heartbeat",
        )
        print("planner research: ecosystem seed join", flush=True)
        heartbeat.start()
        try:
            result = current_collect(prompt, game_design, *args, **kwargs)
        finally:
            stop.set()
        print(
            "planner research: ecosystem seed complete",
            f" routes={result.get('route_count', 'unknown')}",
            f" processed={result.get('processed_route_count', 0)}",
            f" remaining={result.get('remaining_route_count', 0)}",
            f" candidates={result.get('candidate_count', 0)}",
            f" elapsed={time.monotonic() - started:.1f}s",
            sep="",
            flush=True,
        )
        return result

    ecosystem_seed_observed._mmm_stall_guard = True  # type: ignore[attr-defined]
    ecosystem_seed_observed._mmm_parallel_planner_overlap = True  # type: ignore[attr-defined]
    complete_planner_module._retrieve_implementation_evidence = (
        implementation_evidence_observed
    )
    complete_planner_module.collect_ecosystem_seed_bundle = ecosystem_seed_observed


def install() -> None:
    """Install single-owner planner research I/O policy after canonical composition."""

    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        from . import (
            agentic_research_fusion,
            agentic_research_game_design,
            central_research,
            complete_planner,
            ecosystem_discovery,
            external_mcp_router,
            parallel_runtime_contract,
        )

        _patch_pre_design_external_seed(agentic_research_game_design, central_research)
        _patch_pre_design_observability(agentic_research_game_design)
        _patch_external_mcp_parallel(external_mcp_router)
        _patch_discovery_http_pool(ecosystem_discovery)
        _patch_worker_defaults(parallel_runtime_contract, agentic_research_fusion)
        _patch_complete_planner(complete_planner)
        _INSTALLED = True


__all__ = ["_ecosystem_key", "_planning_seed_brief", "install"]
