from __future__ import annotations

"""Second-stage runtime hardening for ecosystem search quality and target verification.

The v1 hardening removes expensive full target resolution from candidate selection.
This module closes the remaining search-path gap: exact-target discovery must not call
the full platform adapter for every candidate/query, and target support must be proved
from machine-readable project/version metadata rather than inferred from an empty,
target-faceted search result.
"""

import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .pipeline_hardening import _replace_bound_references

_INSTALLED = False
_CACHE_LOCK = threading.Lock()
_NEUTRAL_CACHE: dict[
    tuple[int, tuple[str, ...]],
    tuple[dict[str, tuple[str, ...]], tuple[str, ...], int],
] = {}
_CACHE_LIMIT = 32

_GENERIC_QUERY_TOKENS = {
    "minecraft",
    "mod",
    "mods",
    "system",
    "semantic",
    "implementation",
    "implement",
    "task",
    "feature",
    "mechanic",
    "module",
    "generated",
    "generator",
    "interaction",
    "logic",
    "code",
}
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9.+]+|[가-힣]{2,}")


def _stable_unique(values: Sequence[str], *, limit: int | None = None) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return tuple(result)


def _search_variants(query: str) -> tuple[str, ...]:
    """Return a tiny bounded set of useful Modrinth queries.

    Generated implementation identifiers are especially bad search queries. Keep the
    original semantic phrase, but add one de-noised variant so a generated class/task
    name cannot collapse discovery to zero results.
    """

    original = " ".join(str(query or "").split()).strip()
    if not original:
        return ()

    expanded = _CAMEL_BOUNDARY.sub(" ", original)
    expanded = re.sub(r"[_/:\-]+", " ", expanded)
    tokens = _QUERY_TOKEN.findall(expanded)
    useful = [
        token
        for token in tokens
        if token.casefold() not in _GENERIC_QUERY_TOKENS
        and len(token.strip()) >= 2
    ]
    simplified = " ".join(useful[:6]).strip()
    return _stable_unique((original, simplified), limit=2)


def _cache_put(
    key: tuple[int, tuple[str, ...]],
    value: tuple[dict[str, tuple[str, ...]], tuple[str, ...], int],
) -> None:
    with _CACHE_LOCK:
        if len(_NEUTRAL_CACHE) >= _CACHE_LIMIT:
            first = next(iter(_NEUTRAL_CACHE), None)
            if first is not None:
                _NEUTRAL_CACHE.pop(first, None)
        _NEUTRAL_CACHE[key] = value


def _cache_get(
    key: tuple[int, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], int] | None:
    with _CACHE_LOCK:
        return _NEUTRAL_CACHE.get(key)


def _install_cheap_target_normalization() -> None:
    from . import ecosystem_discovery as ecosystem
    from .platform_catalog import provider_for_loader

    original = ecosystem._normalize_discovery_target
    if getattr(original, "_mmm_cheap_target_normalization", False):
        return

    def normalize(
        minecraft_version: str | None,
        loader: str | None,
        *,
        target_profile: str,
        exact_required: bool = False,
    ) -> Any:
        profile = str(target_profile or "").strip().lower()
        if profile != "minecraft_mod":
            return ecosystem._DiscoveryTarget("not_applicable", "not_applicable", False)

        version = str(minecraft_version or "").strip()
        loader_value = str(loader or "").strip().casefold()
        if bool(version) != bool(loader_value):
            raise ecosystem.SpecValidationError(
                "Minecraft ecosystem target requires both minecraft_version and loader."
            )
        if not version:
            if exact_required:
                raise ecosystem.SpecValidationError(
                    "Exact ecosystem inspection requires the host-selected Minecraft target."
                )
            return ecosystem._DiscoveryTarget("unresolved", "unresolved", False)

        # Provider existence is a cheap host-owned registry check. Do NOT resolve a
        # PlatformAdapter here: that performs remote platform/pack metadata work and
        # turned every search/inspection call into another target-resolution pass.
        try:
            provider_for_loader(loader_value)
        except ValueError as exc:
            raise ecosystem.SpecValidationError(str(exc)) from exc
        return ecosystem._DiscoveryTarget(version, loader_value, True)

    normalize._mmm_cheap_target_normalization = True  # type: ignore[attr-defined]
    ecosystem._normalize_discovery_target = normalize
    _replace_bound_references(original, normalize)


def _install_verified_mod_search() -> None:
    from . import platform_optimizer as optimizer

    original_neutral = optimizer._parallel_neutral_shallow
    original_matrix = optimizer._parallel_support_matrix
    if getattr(original_matrix, "_mmm_verified_mod_search", False):
        return

    def broad_neutral(
        queries: Sequence[str],
        client: Any,
    ) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
        query_tuple = tuple(str(query) for query in queries)
        key = (id(client), query_tuple)

        found: dict[str, tuple[str, ...]] = {}
        errors: list[str] = []
        successful_requests = 0

        def run(query: str) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
            ids: list[str] = []
            local_errors: list[str] = []
            successes = 0
            variants = _search_variants(query) or (query,)
            for variant in variants:
                try:
                    # Deliberately omit exact version/loader facets here. This is
                    # semantic candidate recall. Exact compatibility is ranked in the
                    # matrix and fully inspected only after target selection.
                    page = client.search(
                        "modrinth",
                        variant,
                        limit=30,
                        target_profile="minecraft_mod",
                    )
                    successes += 1
                    ids.extend(optimizer._candidate_ids(page))
                except Exception as exc:  # noqa: BLE001 - source state is aggregated
                    local_errors.append(
                        f"neutral:{query!r} variant={variant!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            return query, _stable_unique(ids, limit=40), tuple(local_errors), successes

        worker_count = min(optimizer._workers(), max(1, len(query_tuple)))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="mmm-platform-broad-mod-search",
        ) as pool:
            futures = {pool.submit(run, query): query for query in query_tuple}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    resolved_query, ids, local_errors, successes = future.result()
                    found[resolved_query] = ids
                    errors.extend(local_errors)
                    successful_requests += successes
                except Exception as exc:  # pragma: no cover - defensive executor boundary
                    found[query] = ()
                    errors.append(
                        f"neutral:{query!r}: {type(exc).__name__}: {exc}"
                    )

        for query in query_tuple:
            found.setdefault(query, ())

        cached = (found, tuple(sorted(set(errors))), successful_requests)
        _cache_put(key, cached)
        return cached[0], cached[1]

    def verified_matrix(
        adapters: Sequence[Any],
        queries: Sequence[str],
        client: Any,
    ) -> tuple[dict[str, dict[str, tuple[str, ...]]], tuple[str, ...]]:
        """Build a cheap exact-target support matrix from Modrinth search metadata.

        This stage intentionally does not inspect every broad candidate against every
        Minecraft version. That Cartesian product can explode into thousands of API
        requests. Exact version/loader facets rank hypotheses; the existing deep stage
        then inspects project/version metadata only for the selected target.
        """

        query_tuple = tuple(str(query) for query in queries)
        matrix_lists: dict[str, dict[str, list[str]]] = {
            adapter.adapter_id: {query: [] for query in query_tuple}
            for adapter in adapters
        }

        cached = _cache_get((id(client), query_tuple))
        if cached is not None:
            _neutral, neutral_errors, neutral_successes = cached
            if neutral_successes == 0 and neutral_errors:
                detail = "; ".join(neutral_errors[:8])
                raise ValueError(
                    "Modrinth search source unavailable; refusing to score Minecraft "
                    f"targets from empty transport evidence. Diagnostics: {detail}"
                )

        errors: list[str] = []
        successful_requests = 0

        def run(adapter: Any, query: str) -> tuple[str, str, tuple[str, ...], tuple[str, ...], int]:
            ids: list[str] = []
            local_errors: list[str] = []
            successes = 0
            for variant in (_search_variants(query) or (query,)):
                try:
                    page = client.search(
                        "modrinth",
                        variant,
                        limit=16,
                        minecraft_version=adapter.minecraft_version,
                        loader=adapter.loader,
                        target_profile="minecraft_mod",
                    )
                    successes += 1
                    ids.extend(optimizer._candidate_ids(page))
                except Exception as exc:  # noqa: BLE001 - source state is aggregated
                    local_errors.append(
                        "matrix:"
                        f"{adapter.minecraft_version}/{adapter.loader}:{query!r} "
                        f"variant={variant!r}: {type(exc).__name__}: {exc}"
                    )
            return (
                adapter.adapter_id,
                query,
                _stable_unique(ids, limit=24),
                tuple(local_errors),
                successes,
            )

        jobs = [(adapter, query) for adapter in adapters for query in query_tuple]
        with ThreadPoolExecutor(
            max_workers=min(optimizer._workers(), max(1, len(jobs))),
            thread_name_prefix="mmm-platform-exact-mod-search",
        ) as pool:
            futures = {
                pool.submit(run, adapter, query): (adapter, query)
                for adapter, query in jobs
            }
            for future in as_completed(futures):
                adapter, query = futures[future]
                try:
                    adapter_id, resolved_query, ids, local_errors, successes = future.result()
                    matrix_lists[adapter_id][resolved_query].extend(ids)
                    errors.extend(local_errors)
                    successful_requests += successes
                except Exception as exc:  # pragma: no cover - defensive executor boundary
                    errors.append(
                        "matrix:"
                        f"{adapter.minecraft_version}/{adapter.loader}:{query!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if jobs and successful_requests == 0 and errors:
            detail = "; ".join(errors[:8])
            raise ValueError(
                "Modrinth compatibility search source unavailable; refusing to "
                "interpret transport failures as unsupported mods or downgrade the "
                f"Minecraft target. Diagnostics: {detail}"
            )

        matrix = {
            adapter_id: {
                query: _stable_unique(candidate_ids)
                for query, candidate_ids in by_query.items()
            }
            for adapter_id, by_query in matrix_lists.items()
        }
        return matrix, tuple(sorted(set(errors)))

    broad_neutral._mmm_broad_mod_search = True  # type: ignore[attr-defined]
    verified_matrix._mmm_verified_mod_search = True  # type: ignore[attr-defined]
    optimizer._parallel_neutral_shallow = broad_neutral
    optimizer._parallel_support_matrix = verified_matrix
    _replace_bound_references(original_neutral, broad_neutral)
    _replace_bound_references(original_matrix, verified_matrix)


def install_pipeline_hardening_v2() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_cheap_target_normalization()
    _install_verified_mod_search()
    _INSTALLED = True
