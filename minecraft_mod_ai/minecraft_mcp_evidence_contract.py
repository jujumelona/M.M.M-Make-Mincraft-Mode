from __future__ import annotations

"""Reviewed Minecraft MCP routing without eager planner-wide provider sweeps.

External Minecraft MCP is optional evidence. The pre-design critical path is owned by
local/official RAG plus the research agents; reviewed MCP tools remain exposed to the
ResearchAgent and later production agents so they can retrieve exact evidence when an
actual uncertainty requires it. This avoids turning every research-domain query into a
fresh external MCP process cold start before planning can continue.

Explicit callers may still request a batched external evidence graph. Those calls are
read-only, role-scoped, single-flight cached, and duplicate requests are collapsed before
provider execution.
"""

import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import wraps
from typing import Any, Collection, Mapping


_MARKER = "_mmm_minecraft_mcp_evidence_v2"
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


def _explicit_batch_enabled() -> bool:
    """Allow explicit external evidence batches; planning never calls them eagerly."""
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


def _request_identity(request: Mapping[str, Any]) -> str:
    normalized = dict(request)
    scope = normalized.get("allowed_server_ids")
    if isinstance(scope, (set, frozenset, list, tuple)):
        normalized["allowed_server_ids"] = sorted(str(value) for value in scope)
    return _sha256(normalized)


def collect_external_minecraft_evidence(
    research_brief: dict[str, Any],
    *,
    router: Any | None = None,
) -> dict[str, Any]:
    """Explicitly collect reviewed external evidence, deduplicating cold starts.

    This function is intentionally not called by the normal pre-design official-RAG
    provider. Research agents already have reviewed role-scoped external MCP tools and
    invoke them adaptively when local/official evidence leaves a real gap.
    """
    raw_domains = research_brief.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        from .spec import SpecValidationError
        raise SpecValidationError("Central research brief has no domains.")

    target = _target(research_brief)
    logical_owners: list[tuple[str, str, str, str]] = []
    unique_requests: dict[str, dict[str, Any]] = {}
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
        row["strategy"] = "adaptive_reviewed_minecraft_mcp"
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
                request = {
                    "capability": capability,
                    "stage": "research",
                    "arguments": {"query": query},
                    "target": target,
                    "corroborate": 1,
                    "required": False,
                    "max_access": "read",
                    "allowed_server_ids": scope,
                }
                identity = _request_identity(request)
                unique_requests.setdefault(identity, request)
                logical_owners.append((domain_id, query_sha256, capability, identity))

    enabled = _explicit_batch_enabled()
    bundles_by_identity: dict[str, dict[str, Any]] = {}
    if unique_requests and enabled:
        selected_router = router or _shared_router()
        identities = tuple(unique_requests)
        requests = tuple(unique_requests[key] for key in identities)
        bundles = selected_router.invoke_many(requests)
        if len(bundles) != len(identities):
            from .spec import SpecValidationError
            raise SpecValidationError(
                "External MCP batch changed deterministic result cardinality."
            )
        bundles_by_identity = dict(zip(identities, bundles))

    pass_count = 0
    for domain_id, query_sha256, capability, identity in logical_owners:
        bundle = bundles_by_identity.get(identity)
        if bundle is None:
            continue
        compact = _compact_bundle(bundle)
        compact["capability"] = capability
        if compact["status"] == "PASS":
            pass_count += 1
        by_query[(domain_id, query_sha256)]["capabilities"].append(compact)

    logical_request_count = len(logical_owners)
    unique_request_count = len(unique_requests)
    status = "SKIPPED"
    if unique_requests and not enabled:
        status = "DISABLED"
    elif unique_requests:
        status = "PASS" if pass_count else "UNAVAILABLE"
    payload: dict[str, Any] = {
        "schema_version": "mmm/external-minecraft-evidence-graph-v2",
        "brief_sha256": str(research_brief.get("brief_sha256", "")),
        "target": target,
        "domains": domain_rows,
        "status": status,
        "execution": {
            "parallel": unique_request_count > 1,
            "request_count": logical_request_count,
            "unique_request_count": unique_request_count,
            "deduplicated_request_count": max(0, logical_request_count - unique_request_count),
            "pass_count": pass_count,
            "single_flight_cache": True,
            "deterministic_merge_order": True,
            "read_only": True,
            "planning_critical_path": False,
            "retrieval_policy": "adaptive_or_explicit_only",
        },
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = _sha256(payload)
    return payload


def _install_router_parallel_cache() -> None:
    """Add shared read caching and an explicit batch API without changing MCP lifecycle."""
    from .external_mcp_router import ExternalMCPRouter, MCPRouteTarget

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
            return deepcopy(inflight.result())

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

    invoke_many._mmm_minecraft_mcp_evidence_v2 = True  # type: ignore[attr-defined]
    ExternalMCPRouter.invoke_many = invoke_many


def _install_research_routes() -> None:
    """Advertise reviewed MCP to technical research without executing it eagerly."""
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


def install() -> None:
    """Install adaptive reviewed Minecraft MCP routing exactly once.

    Deliberately do not wrap retrieve_domain_evidence. Official/project RAG owns
    deterministic pre-design evidence; external MCP remains an optional adaptive tool
    route and therefore cannot hold the planner critical path hostage.
    """
    _install_router_parallel_cache()
    _install_research_routes()


__all__ = ["collect_external_minecraft_evidence", "install"]
