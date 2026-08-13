from __future__ import annotations

"""Parallel reviewed-MCP evidence fusion for Minecraft technical research.

This contract is intentionally late-bound after the base runtime bootstrap. It keeps
Minecraft/Fabric documentation, mappings, source and examples as read-only evidence,
shares identical calls through a single-flight cache, and overlaps external MCP work
with the existing official RAG lane without weakening any knowledge or validation gate.
"""

import asyncio
import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import wraps
from typing import Any, Collection, Mapping

import anyio


_MARKER = "_mmm_minecraft_mcp_evidence_v1"
_MAX_RESULT_CHARS = 6000
_TECHNICAL_KINDS = frozenset(
    {
        "minecraft_api",
        "dependency",
        "source_code",
        "gameplay_reference",
        "compatibility",
        "runtime_behavior",
        "performance",
        "testing",
        "release",
    }
)
_CACHEABLE_CAPABILITIES = frozenset(
    {
        "official_mod_docs",
        "mapping_resolution",
        "mod_examples",
        "source_search",
        "version_diff",
        "registry_lookup",
        "vanilla_knowledge",
    }
)
_CAPABILITY_SCOPES: dict[str, frozenset[str]] = {
    "official_mod_docs": frozenset({"mcmodding-docs"}),
    "mapping_resolution": frozenset({"mcmodding-docs"}),
    "mod_examples": frozenset({"mcmodding-docs"}),
    "source_search": frozenset({"minecraft-dev"}),
    "vanilla_knowledge": frozenset({"minecraft-wiki"}),
}
_KIND_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "minecraft_api": ("official_mod_docs", "source_search"),
    "dependency": ("official_mod_docs", "mod_examples"),
    "source_code": ("source_search", "mod_examples"),
    "gameplay_reference": ("vanilla_knowledge",),
    "compatibility": ("official_mod_docs", "mapping_resolution"),
    "runtime_behavior": ("source_search", "mod_examples"),
    "performance": ("source_search", "mod_examples"),
    "testing": ("mod_examples", "source_search"),
    "release": ("official_mod_docs",),
}
_SHARED_ROUTER: Any | None = None
_SHARED_ROUTER_LOCK = threading.Lock()


def _workers() -> int:
    raw = os.environ.get("MMM_EXTERNAL_MCP_WORKERS", "").strip()
    try:
        value = int(raw) if raw else 6
    except ValueError:
        value = 6
    return max(1, min(16, value))


def _enabled() -> bool:
    raw = os.environ.get("MMM_EXTERNAL_MCP_PREFETCH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: Any) -> str:
    text = value if isinstance(value, str) else _stable_json(value)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _server_scope(values: Collection[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(
        value
        for raw in values
        if (value := str(raw).strip())
    )


def _target(research_brief: Mapping[str, Any]) -> dict[str, str]:
    raw = research_brief.get("_mmm_platform_target")
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "minecraft_version": str(
            raw.get(
                "minecraft_version",
                os.environ.get("MMM_MINECRAFT_VERSION", "1.20.1"),
            )
        ).strip()
        or "1.20.1",
        "loader": str(
            raw.get("loader", os.environ.get("MMM_LOADER", "fabric"))
        ).strip()
        or "fabric",
        "mappings": str(
            raw.get(
                "mappings",
                raw.get(
                    "yarn_mappings",
                    os.environ.get("MMM_MAPPINGS", "yarn-1.20.1+build.1"),
                ),
            )
        ).strip(),
    }


def _capabilities(domain: Mapping[str, Any]) -> tuple[str, ...]:
    ordered: list[str] = []
    for kind in domain.get("evidence_kinds", []):
        for capability in _KIND_CAPABILITIES.get(str(kind), ()):
            if capability not in ordered:
                ordered.append(capability)
    if not ordered:
        ordered.append("official_mod_docs")
    return tuple(ordered)


def _compact_result(value: Any) -> str:
    text = _stable_json(value)
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "…"


def _compact_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    raw_evidence = bundle.get("evidence")
    if isinstance(raw_evidence, list):
        for raw in raw_evidence[:2]:
            if not isinstance(raw, Mapping):
                continue
            evidence.append(
                {
                    "server": str(raw.get("server", "")),
                    "tool": str(raw.get("tool", "")),
                    "trust": str(raw.get("trust", "")),
                    "arguments_sha256": str(raw.get("arguments_sha256", "")),
                    "result_sha256": str(raw.get("result_sha256", "")),
                    "result_excerpt": _compact_result(raw.get("result")),
                }
            )
    attempts = bundle.get("attempts")
    return {
        "capability": str(bundle.get("capability", "")),
        "status": str(bundle.get("status", "UNAVAILABLE")),
        "bundle_sha256": str(bundle.get("bundle_sha256", "")),
        "evidence": evidence,
        "attempts": list(attempts)[:4] if isinstance(attempts, list) else [],
    }


def _shared_router() -> Any:
    global _SHARED_ROUTER
    if _SHARED_ROUTER is None:
        with _SHARED_ROUTER_LOCK:
            if _SHARED_ROUTER is None:
                from .external_mcp_router import ExternalMCPRouter

                _SHARED_ROUTER = ExternalMCPRouter()
    return _SHARED_ROUTER


def collect_external_minecraft_evidence(
    research_brief: dict[str, Any],
    *,
    router: Any | None = None,
) -> dict[str, Any]:
    """Collect reviewed Minecraft MCP evidence in one bounded parallel batch."""

    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        from .spec import SpecValidationError

        raise SpecValidationError("Central research brief has no domains.")

    target = _target(research_brief)
    requests: list[dict[str, Any]] = []
    owners: list[tuple[str, str, str]] = []
    domain_rows: list[dict[str, Any]] = []
    by_query: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            continue
        domain_id = str(raw_domain.get("domain_id", "")).strip() or "unknown"
        providers = {str(value) for value in raw_domain.get("providers", [])}
        row: dict[str, Any] = {
            "domain_id": domain_id,
            "strategy": "routed_to_other_providers",
            "queries": [],
        }
        domain_rows.append(row)
        if "external_mcp" not in providers:
            continue
        row["strategy"] = "parallel_reviewed_minecraft_mcp"
        capabilities = _capabilities(raw_domain)
        for raw_query in raw_domain.get("queries", []):
            query = str(raw_query).strip()
            if not query:
                continue
            query_sha256 = _sha256(query)
            query_row: dict[str, Any] = {
                "query_sha256": query_sha256,
                "capabilities": [],
            }
            row["queries"].append(query_row)
            by_query[(domain_id, query_sha256)] = query_row
            for capability in capabilities:
                scope = _CAPABILITY_SCOPES.get(capability)
                if not scope:
                    continue
                requests.append(
                    {
                        "capability": capability,
                        "stage": "research",
                        "arguments": {"query": query},
                        "target": target,
                        "corroborate": 1,
                        "required": False,
                        "max_access": "read",
                        "allowed_server_ids": scope,
                    }
                )
                owners.append((domain_id, query_sha256, capability))

    enabled = _enabled()
    bundles: tuple[dict[str, Any], ...] = ()
    if requests and enabled:
        selected_router = router or _shared_router()
        bundles = selected_router.invoke_many(requests)
    if bundles and len(bundles) != len(owners):
        from .spec import SpecValidationError

        raise SpecValidationError(
            "External MCP batch changed deterministic result cardinality."
        )

    pass_count = 0
    if bundles:
        for owner, bundle in zip(owners, bundles):
            domain_id, query_sha256, capability = owner
            compact = _compact_bundle(bundle)
            compact["capability"] = capability
            if compact["status"] == "PASS":
                pass_count += 1
            by_query[(domain_id, query_sha256)]["capabilities"].append(compact)

    status = "SKIPPED"
    if requests and not enabled:
        status = "DISABLED"
    elif requests:
        status = "PASS" if pass_count else "UNAVAILABLE"
    payload: dict[str, Any] = {
        "schema_version": "mmm/external-minecraft-evidence-graph-v1",
        "brief_sha256": str(research_brief.get("brief_sha256", "")),
        "target": target,
        "domains": domain_rows,
        "status": status,
        "execution": {
            "parallel": len(requests) > 1,
            "request_count": len(requests),
            "pass_count": pass_count,
            "single_flight_cache": True,
            "deterministic_merge_order": True,
            "read_only": True,
        },
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = _sha256(payload)
    return payload


def _install_router_parallel_cache() -> None:
    from .external_mcp_router import ExternalMCPError, ExternalMCPRouter, MCPRouteTarget

    if getattr(ExternalMCPRouter.invoke, _MARKER, False):
        return

    original_invoke = ExternalMCPRouter.invoke

    @wraps(original_invoke)
    def invoke_cached(
        self: Any,
        capability: str,
        *,
        stage: str,
        arguments: Mapping[str, Any] | None = None,
        target: Any = None,
        corroborate: int = 1,
        required: bool = False,
        max_access: str = "read",
        disposable_runtime: bool = False,
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        cacheable = (
            stage in {"planning", "research", "generation"}
            and max_access == "read"
            and not disposable_runtime
            and capability in _CACHEABLE_CAPABILITIES
        )
        if not cacheable:
            return original_invoke(
                self,
                capability,
                stage=stage,
                arguments=arguments,
                target=target,
                corroborate=corroborate,
                required=required,
                max_access=max_access,
                disposable_runtime=disposable_runtime,
                allowed_server_ids=allowed_server_ids,
            )

        resolved = MCPRouteTarget.from_value(target)
        scope = _server_scope(allowed_server_ids)
        key = (
            stage,
            capability,
            resolved.minecraft_version,
            resolved.loader,
            resolved.mappings,
            resolved.mapping,
            corroborate,
            None if scope is None else tuple(sorted(scope)),
            _sha256(dict(arguments or {})),
        )
        state_lock = getattr(self, "_mmm_external_cache_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._mmm_external_cache_lock = state_lock
            self._mmm_external_read_cache = {}
            self._mmm_external_inflight = {}

        with state_lock:
            cached = self._mmm_external_read_cache.get(key)
            if cached is not None:
                return deepcopy(cached)
            inflight = self._mmm_external_inflight.get(key)
            owner = inflight is None
            if owner:
                inflight = Future()
                self._mmm_external_inflight[key] = inflight
        assert inflight is not None
        if not owner:
            return deepcopy(inflight.result(timeout=self.timeout_seconds + 5.0))

        try:
            result = original_invoke(
                self,
                capability,
                stage=stage,
                arguments=arguments,
                target=target,
                corroborate=corroborate,
                required=required,
                max_access=max_access,
                disposable_runtime=disposable_runtime,
                allowed_server_ids=allowed_server_ids,
            )
        except BaseException as exc:
            with state_lock:
                self._mmm_external_inflight.pop(key, None)
                if not inflight.done():
                    inflight.set_exception(exc)
            raise
        with state_lock:
            self._mmm_external_inflight.pop(key, None)
            if result.get("status") in {"PASS", "PARTIAL"}:
                self._mmm_external_read_cache[key] = deepcopy(result)
            if not inflight.done():
                inflight.set_result(deepcopy(result))
        return result

    setattr(invoke_cached, _MARKER, True)
    ExternalMCPRouter.invoke = invoke_cached

    def invoke_many(
        self: Any,
        requests: Collection[Mapping[str, Any]],
        *,
        max_workers: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        rows = [dict(request) for request in requests]
        if not rows:
            return ()
        workers = _workers() if max_workers is None else max_workers
        if type(workers) is not int or workers < 1:
            raise ValueError("max_workers must be a positive integer.")
        workers = min(16, workers, len(rows))

        def run(index: int, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
            payload = dict(request)
            capability = str(payload.pop("capability", "")).strip()
            if not capability:
                raise ValueError("Each MCP batch request requires capability.")
            return index, self.invoke(capability, **payload)

        if len(rows) == 1:
            _, bundle = run(0, rows[0])
            return (bundle,)

        ordered: list[dict[str, Any] | None] = [None] * len(rows)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm_external_mcp",
        ) as pool:
            futures = {
                pool.submit(run, index, row): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index, bundle = future.result()
                ordered[index] = bundle
        return tuple(item for item in ordered if item is not None)

    invoke_many._mmm_minecraft_mcp_evidence_v1 = True
    ExternalMCPRouter.invoke_many = invoke_many

    async def call_async(self: Any, server_name: str, entry: Mapping[str, Any], *, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await self._call_provider_async(
            server_name,
            entry,
            tool=tool,
            arguments=arguments,
        )

    def call_provider_parallel(
        self: Any,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await call_async(
                self,
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
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                value["result"] = anyio.run(run)
            except BaseException as exc:  # pragma: no cover - thread bridge
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise ExternalMCPError(
                f"External MCP {server_name} exceeded the synchronous bridge timeout."
            )
        if errors:
            raise ExternalMCPError(str(errors[0])) from errors[0]
        return value["result"]

    call_provider_parallel._mmm_minecraft_mcp_evidence_v1 = True
    ExternalMCPRouter._call_provider = call_provider_parallel


def _install_research_routes() -> None:
    from . import central_research

    central_research._ALLOWED_PROVIDERS = frozenset(
        {*central_research._ALLOWED_PROVIDERS, "external_mcp"}
    )
    current = central_research._augment_domain_routes
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def augment(domain: Any) -> Any:
        routed = current(domain)
        if not (set(routed.evidence_kinds) & _TECHNICAL_KINDS):
            return routed
        providers = tuple(dict.fromkeys((*routed.providers, "external_mcp")))
        return central_research.ResearchDomain(
            domain_id=routed.domain_id,
            objective=routed.objective,
            requirements=routed.requirements,
            evidence_kinds=routed.evidence_kinds,
            queries=routed.queries,
            providers=providers,
            depends_on=routed.depends_on,
        )

    setattr(augment, _MARKER, True)
    central_research._augment_domain_routes = augment


def _merge_external(official: Any, external: Mapping[str, Any]) -> Any:
    if not isinstance(official, dict):
        return official
    merged = deepcopy(official)
    external_by_id = {
        str(row.get("domain_id", "")): row
        for row in external.get("domains", [])
        if isinstance(row, Mapping)
    }
    domains = merged.get("domains")
    if isinstance(domains, list):
        for row in domains:
            if not isinstance(row, dict):
                continue
            external_row = external_by_id.get(str(row.get("domain_id", "")))
            if external_row is not None:
                row["external_mcp"] = deepcopy(external_row)
    merged["external_mcp_status"] = str(external.get("status", ""))
    merged["external_mcp_evidence_sha256"] = str(
        external.get("evidence_sha256", "")
    )
    merged["external_mcp_execution"] = deepcopy(external.get("execution", {}))
    merged.pop("evidence_sha256", None)
    merged["evidence_sha256"] = _sha256(merged)
    return merged


def _install_research_fusion() -> None:
    from . import agentic_research_game_design as research_design

    current = research_design.retrieve_domain_evidence
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def retrieve_fused(research_brief: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mmm_minecraft_evidence") as pool:
            official_future = pool.submit(current, research_brief, *args, **kwargs)
            external_future = pool.submit(collect_external_minecraft_evidence, research_brief)
            official = official_future.result()
            try:
                external = external_future.result()
            except Exception as exc:
                external = {
                    "schema_version": "mmm/external-minecraft-evidence-graph-v1",
                    "status": "UNAVAILABLE",
                    "domains": [],
                    "execution": {
                        "parallel": False,
                        "request_count": 0,
                        "pass_count": 0,
                        "single_flight_cache": True,
                        "deterministic_merge_order": True,
                        "read_only": True,
                    },
                    "evidence_sha256": _sha256(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    ),
                }
        return _merge_external(official, external)

    setattr(retrieve_fused, _MARKER, True)
    research_design.retrieve_domain_evidence = retrieve_fused


def install() -> None:
    """Install the reviewed Minecraft MCP evidence layer exactly once."""

    _install_router_parallel_cache()
    _install_research_routes()
    _install_research_fusion()


__all__ = ["collect_external_minecraft_evidence", "install"]
