from __future__ import annotations

"""Bound planner-only research latency without reducing production coverage.

The full research brief remains authoritative and is retained for later specialist
work.  Only the *planning seed* projection is compacted: every research domain and
provider is preserved, while a bounded number of representative queries per domain
is used to obtain current ecosystem hints before module planning.  Independent
provider calls remain parallel and deterministic through the existing runtime
contract.
"""

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from functools import wraps
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False
_STATE = threading.local()
_ECOSYSTEM_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mmm_planner_ecosystem",
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


def _planning_seed_brief(research_brief: dict[str, Any]) -> dict[str, Any]:
    """Keep every domain/provider but bound planning-only query fan-out."""

    per_domain = _env_int(
        "MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN",
        1,
        minimum=1,
        maximum=4,
    )
    projected = deepcopy(research_brief)
    raw_domains = projected.get("domains")
    if not isinstance(raw_domains, list):
        return projected

    for raw_domain in raw_domains:
        if not isinstance(raw_domain, dict):
            continue
        queries = raw_domain.get("queries")
        if isinstance(queries, list) and len(queries) > per_domain:
            raw_domain["queries"] = queries[:per_domain]

    projected["_mmm_planning_seed_projection"] = {
        "schema_version": "mmm/planning-seed-projection-v1",
        "queries_per_domain": per_domain,
        "full_research_brief_retained": True,
        "domains_and_providers_preserved": True,
        "specialist_discovery_continues_full_brief": True,
    }
    return projected


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


def _ecosystem_key(
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any],
) -> tuple[str, int, int]:
    return prompt, id(game_design), id(research_brief)


def _timeout_bundle(
    *,
    ecosystem_module: Any,
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    seed_brief = _planning_seed_brief(research_brief)
    routes = list(ecosystem_module.external_discovery_routes(seed_brief))
    query = ecosystem_module._seed_query(prompt, game_design)
    route_sha = ecosystem_module._sha256_text(
        ecosystem_module.canonical_json(routes)
    )
    return {
        "schema_version": "mmm/ecosystem-seed-bundle-v1",
        "aggregate_schema_version": "mmm/ecosystem-seed-aggregate-v1",
        "status": "unavailable",
        "query_sha256": ecosystem_module._sha256_text(query),
        "route_sha256": route_sha,
        "route_count": len(routes),
        "route_offset": 0,
        "processed_route_count": 0,
        "remaining_route_count": len(routes),
        "next_route_cursor": "",
        "routes_complete": False,
        "candidate_count": 0,
        "pages": [],
        "errors": [
            {
                "domain_id": "planner_seed",
                "provider": "planner_stage",
                "query_sha256": ecosystem_module._sha256_text(query),
                "error_type": "StageDeadlineExceeded",
                "message": (
                    "Planning ecosystem seed exceeded its bounded latency budget "
                    f"of {timeout_seconds:.1f}s. Full specialist discovery remains required."
                ),
            }
        ],
        "coverage": (
            "planning seed timed out; full research brief is retained and specialist "
            "dependency/asset discovery continues during production"
        ),
        "authorization": "none",
        "download_performed": False,
        "collection_receipt": {
            "schema_version": "mmm/ecosystem-route-collection-receipt-v1",
            "route_page_count": 0,
            "route_limit": _env_int(
                "MMM_ECOSYSTEM_ROUTE_PAGE",
                100,
                minimum=12,
                maximum=100,
            ),
            "bounded_timeout": True,
        },
    }


def _patch_provider_timeout(ecosystem_module: Any) -> None:
    cls = ecosystem_module.EcosystemDiscoveryClient
    current = cls.__init__
    if getattr(current, "_mmm_planner_timeout_default", False):
        return

    @wraps(current)
    def init_with_bounded_default(self: Any, *args: Any, **kwargs: Any) -> None:
        # The constructor is keyword-only. Explicit caller choices remain untouched.
        if "timeout_seconds" not in kwargs:
            kwargs["timeout_seconds"] = _env_float(
                "MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS",
                8.0,
                minimum=2.0,
                maximum=30.0,
            )
        current(self, *args, **kwargs)

    init_with_bounded_default._mmm_planner_timeout_default = True  # type: ignore[attr-defined]
    cls.__init__ = init_with_bounded_default


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


def _collect_fast_ecosystem(
    *,
    research_coordinator_module: Any,
    ecosystem_module: Any,
    prompt: str,
    game_design: dict[str, Any],
    research_brief: dict[str, Any],
) -> dict[str, Any]:
    seed_brief = _planning_seed_brief(research_brief)
    route_limit = _env_int(
        "MMM_ECOSYSTEM_ROUTE_PAGE",
        100,
        minimum=12,
        maximum=100,
    )
    routes = list(ecosystem_module.external_discovery_routes(seed_brief))
    discovery_workers = _env_int(
        "MMM_DISCOVERY_WORKERS",
        24,
        minimum=1,
        maximum=32,
    )
    provider_timeout = _env_float(
        "MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS",
        8.0,
        minimum=2.0,
        maximum=30.0,
    )
    started = time.monotonic()
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=("ecosystem seed", stop, started),
        daemon=True,
        name="mmm_ecosystem_heartbeat",
    )
    print(
        "planner research: ecosystem seed start",
        f" routes={len(routes)}",
        f" route_page={route_limit}",
        f" workers={discovery_workers}",
        f" provider_timeout={provider_timeout:.1f}s",
        sep="",
        flush=True,
    )
    heartbeat.start()
    try:
        result = research_coordinator_module.collect_ecosystem_seed_bundle(
            prompt,
            game_design,
            research_brief=seed_brief,
            route_limit=route_limit,
            page_builder=ecosystem_module.discover_seed_bundle,
            allow_legacy_terminal=True,
        )
    finally:
        stop.set()
    print(
        "planner research: ecosystem seed complete",
        f" routes={result.get('route_count', len(routes))}",
        f" candidates={result.get('candidate_count', 0)}",
        f" errors={len(result.get('errors', [])) if isinstance(result.get('errors'), list) else 0}",
        f" elapsed={time.monotonic() - started:.1f}s",
        sep="",
        flush=True,
    )
    return result


def _patch_complete_planner(
    *,
    complete_planner_module: Any,
    ecosystem_module: Any,
    research_coordinator_module: Any,
    parallel_module: Any,
    platform_rag_module: Any,
) -> None:
    current_impl = complete_planner_module._retrieve_implementation_evidence
    current_collect = complete_planner_module.collect_ecosystem_seed_bundle
    if getattr(current_impl, "_mmm_stall_guard", False):
        return

    @wraps(current_impl)
    def implementation_evidence_bounded(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or complete_planner_module.normalize_research_brief(
            prompt,
            game_design,
        )
        if platform_rag_module._adapter_from_brief(brief) is None:
            selection = game_design.get("_platform_selection")
            if isinstance(selection, dict) and isinstance(selection.get("target"), dict):
                brief = {**brief, "_mmm_platform_target": dict(selection["target"])}

        key = _ecosystem_key(prompt, game_design, brief)
        existing = getattr(_STATE, "ecosystem", None)
        if not existing or existing[0] != key:
            _STATE.ecosystem = (
                key,
                _ECOSYSTEM_EXECUTOR.submit(
                    _collect_fast_ecosystem,
                    research_coordinator_module=research_coordinator_module,
                    ecosystem_module=ecosystem_module,
                    prompt=prompt,
                    game_design=game_design,
                    research_brief=brief,
                ),
                brief,
            )

        evidence_key = parallel_module._planner_key(prompt, brief)
        existing_evidence = getattr(parallel_module._PLANNER_STATE, "evidence", None)
        rag_timeout = _env_float(
            "MMM_OFFICIAL_RAG_STAGE_TIMEOUT_SECONDS",
            60.0,
            minimum=10.0,
            maximum=300.0,
        )
        started = time.monotonic()
        if existing_evidence and existing_evidence[0] == evidence_key:
            print("planner research: official RAG join", flush=True)
            try:
                result = existing_evidence[1].result(timeout=rag_timeout)
            except FutureTimeout as exc:
                raise RuntimeError(
                    f"Official RAG exceeded planner stage budget ({rag_timeout:.1f}s)."
                ) from exc
        else:
            result = complete_planner_module.retrieve_domain_evidence(brief)
        print(
            "planner research: official RAG complete",
            f" elapsed={time.monotonic() - started:.1f}s",
            sep="",
            flush=True,
        )
        return result

    implementation_evidence_bounded._mmm_stall_guard = True  # type: ignore[attr-defined]
    implementation_evidence_bounded._mmm_parallel_target_rag = True  # type: ignore[attr-defined]
    implementation_evidence_bounded._mmm_agentic_rag_fusion = True  # type: ignore[attr-defined]

    @wraps(current_collect)
    def ecosystem_seed_bounded(
        prompt: str,
        game_design: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        brief = kwargs.get("research_brief")
        if not isinstance(brief, dict):
            return current_collect(prompt, game_design, *args, **kwargs)

        key = _ecosystem_key(prompt, game_design, brief)
        existing = getattr(_STATE, "ecosystem", None)
        if not existing or existing[0] != key:
            future: Future[dict[str, Any]] = _ECOSYSTEM_EXECUTOR.submit(
                _collect_fast_ecosystem,
                research_coordinator_module=research_coordinator_module,
                ecosystem_module=ecosystem_module,
                prompt=prompt,
                game_design=game_design,
                research_brief=brief,
            )
            existing = (key, future, brief)
            _STATE.ecosystem = existing

        timeout_seconds = _env_float(
            "MMM_ECOSYSTEM_STAGE_TIMEOUT_SECONDS",
            90.0,
            minimum=20.0,
            maximum=600.0,
        )
        future = existing[1]
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
            print(
                "planner research: ecosystem seed deadline exceeded",
                f" budget={timeout_seconds:.1f}s",
                " action=" + ("fail" if mode == "on" else "fail-soft"),
                sep="",
                flush=True,
            )
            if mode == "on":
                raise ecosystem_module.EcosystemDiscoveryUnavailable(
                    f"Required ecosystem discovery exceeded {timeout_seconds:.1f}s planner budget."
                ) from exc
            return _timeout_bundle(
                ecosystem_module=ecosystem_module,
                prompt=prompt,
                game_design=game_design,
                research_brief=brief,
                timeout_seconds=timeout_seconds,
            )
        finally:
            _STATE.ecosystem = None

    ecosystem_seed_bounded._mmm_stall_guard = True  # type: ignore[attr-defined]
    ecosystem_seed_bounded._mmm_parallel_planner_overlap = True  # type: ignore[attr-defined]

    complete_planner_module._retrieve_implementation_evidence = implementation_evidence_bounded
    complete_planner_module.collect_ecosystem_seed_bundle = ecosystem_seed_bounded


def install() -> None:
    """Install bounded, parallel planning research after the canonical bootstrap."""

    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        from . import (
            agentic_research_fusion,
            complete_planner,
            ecosystem_discovery,
            parallel_platform_rag_contract,
            parallel_runtime_contract,
            research_coordinator,
        )

        _patch_provider_timeout(ecosystem_discovery)
        _patch_worker_defaults(parallel_runtime_contract, agentic_research_fusion)
        _patch_complete_planner(
            complete_planner_module=complete_planner,
            ecosystem_module=ecosystem_discovery,
            research_coordinator_module=research_coordinator,
            parallel_module=parallel_runtime_contract,
            platform_rag_module=parallel_platform_rag_contract,
        )
        _INSTALLED = True


__all__ = ["_planning_seed_brief", "install"]
