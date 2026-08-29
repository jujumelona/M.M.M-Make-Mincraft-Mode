from __future__ import annotations

"""Canonical runtime contract composition.

This module owns the small cross-module composition points that cannot live in a
single subsystem without creating an import cycle.  It deliberately has no numbered
"hardening" generations: one production contract is installed exactly once.
"""

import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any

_INSTALLED = False
_CACHE_LOCK = threading.Lock()
_NEUTRAL_CACHE: dict[
    tuple[int, tuple[str, ...]],
    tuple[dict[str, tuple[str, ...]], tuple[str, ...], int],
] = {}
_CACHE_LIMIT = 32
_PAGE_GROUNDING_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "mmm_pre_design_page_grounding_context",
    default=None,
)
_SOURCE_EDIT_TOOL = "apply_source_edit"

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
_SEED_NOISE = frozenset(
    {
        *_GENERIC_QUERY_TOKENS,
        "make",
        "create",
        "design",
        "plan",
        "planning",
        "research",
        "fabric",
        "forge",
        "neoforge",
        "version",
        "java",
        "with",
        "that",
        "this",
        "the",
        "and",
        "for",
        "please",
    }
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9.+]+|[가-힣]{2,}")


def _replace_bound_references(original: Any, replacement: Any) -> None:
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("minecraft_mod_ai") or module is None:
            continue
        for attribute, value in tuple(vars(module).items()):
            if value is original:
                setattr(module, attribute, replacement)


def _base_evidence_ref(value: Any) -> str:
    text = str(value or "").strip()
    marker = "#synthesis-"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def _allowed_evidence_refs(group: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        ref = _base_evidence_ref(value)
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for note in group:
        add(note.get("page_ref"))
        fragment = note.get("evidence_fragment")
        if isinstance(fragment, Mapping):
            add(fragment.get("page_ref"))
        for field in ("claims", "procedures"):
            for item in note.get(field, ()):
                if not isinstance(item, Mapping):
                    continue
                for ref in item.get("evidence_refs", ()):
                    add(ref)
    return tuple(refs)


def _repair_note_provenance(
    note: Mapping[str, Any],
    *,
    allowed_refs: Sequence[str],
) -> dict[str, Any]:
    """Accept only host-issued evidence refs; an uncited claim is never grounded."""

    result = dict(note)
    allowed = tuple(
        dict.fromkeys(
            _base_evidence_ref(ref)
            for ref in allowed_refs
            if _base_evidence_ref(ref)
        )
    )
    allowed_set = set(allowed)
    claims: list[dict[str, Any]] = []
    dropped = 0
    for raw_claim in result.get("claims", ()):
        if not isinstance(raw_claim, Mapping):
            dropped += 1
            continue
        claim = dict(raw_claim)
        refs = list(
            dict.fromkeys(
                _base_evidence_ref(ref)
                for ref in claim.get("evidence_refs", ())
                if _base_evidence_ref(ref) in allowed_set
            )
        )
        if not refs:
            dropped += 1
            continue
        claim["evidence_refs"] = refs
        claims.append(claim)
    if "claims" in result:
        result["claims"] = claims

    procedures: list[dict[str, Any]] = []
    for raw_procedure in result.get("procedures", ()):
        if not isinstance(raw_procedure, Mapping):
            continue
        procedure = dict(raw_procedure)
        if "evidence_refs" in procedure:
            refs = list(
                dict.fromkeys(
                    _base_evidence_ref(ref)
                    for ref in procedure.get("evidence_refs", ())
                    if _base_evidence_ref(ref) in allowed_set
                )
            )
            if not refs:
                continue
            procedure["evidence_refs"] = refs
        procedures.append(procedure)
    if "procedures" in result:
        result["procedures"] = procedures

    if dropped:
        gaps = [str(value) for value in result.get("gaps", ()) if str(value).strip()]
        gaps.append(
            f"{dropped} synthesized claim(s) were omitted because no host-issued "
            "evidence reference survived provenance validation."
        )
        result["gaps"] = gaps
    if not claims and "claims" in result:
        result["sufficient"] = False
    return result


def _query_tokens(value: Any) -> list[str]:
    text = _CAMEL_BOUNDARY.sub(" ", str(value or ""))
    text = re.sub(r"[_/:\-]+", " ", text)
    return [
        token
        for token in _QUERY_TOKEN.findall(text)
        if len(token.strip()) >= 2 and token.casefold() not in _SEED_NOISE
    ]


def bounded_seed_query(prompt: str, game_design: Mapping[str, Any]) -> str:
    sources: list[Any] = [game_design.get("title", "")]
    modules = game_design.get("modules")
    if isinstance(modules, Sequence) and not isinstance(modules, (str, bytes, bytearray)):
        for item in modules:
            if isinstance(item, Mapping):
                sources.extend(
                    (
                        item.get("name", ""),
                        item.get("kind", ""),
                        item.get("plugin_id", ""),
                        item.get("reason", ""),
                    )
                )
    capabilities = game_design.get("capabilities")
    if isinstance(capabilities, Sequence) and not isinstance(
        capabilities, (str, bytes, bytearray)
    ):
        sources.extend(capabilities)
    sources.extend((game_design.get("pitch", ""), prompt))

    tokens: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for token in _query_tokens(source):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
            if len(tokens) >= 16:
                break
        if len(tokens) >= 16:
            break
    if not tokens:
        fallback = " ".join(str(prompt or "").split()).strip()
        return fallback[:240] or "gameplay"
    return " ".join(tokens)[:320].rstrip()


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
    original = " ".join(str(query or "").split()).strip()
    if not original:
        return ()
    expanded = _CAMEL_BOUNDARY.sub(" ", original)
    expanded = re.sub(r"[_/:\-]+", " ", expanded)
    tokens = _QUERY_TOKEN.findall(expanded)
    useful = [
        token
        for token in tokens
        if token.casefold() not in _GENERIC_QUERY_TOKENS and len(token.strip()) >= 2
    ]
    return _stable_unique((original, " ".join(useful[:6]).strip()), limit=2)


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


def _install_research_provenance() -> None:
    from . import agentic_pre_design_rag as rag

    original = rag._synthesize_group_with_recovery
    if getattr(original, "_mmm_provenance_contract", False):
        return

    @wraps(original)
    def provenance_checked(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        group = kwargs.get("group")
        if group is None and len(args) >= 6:
            group = args[5]
        allowed = _allowed_evidence_refs(group or ())
        notes = original(*args, **kwargs)
        return [
            _repair_note_provenance(note, allowed_refs=allowed)
            if isinstance(note, Mapping)
            else note
            for note in notes
        ]

    provenance_checked._mmm_provenance_contract = True  # type: ignore[attr-defined]
    rag._synthesize_group_with_recovery = provenance_checked
    _replace_bound_references(original, provenance_checked)


def _install_semantic_seed_search() -> None:
    from . import ecosystem_discovery as ecosystem

    original = ecosystem._seed_query
    if getattr(original, "_mmm_bounded_semantic_seed_query", False):
        return

    def seed_query(prompt: str, game_design: dict[str, Any]) -> str:
        return bounded_seed_query(prompt, game_design)

    seed_query._mmm_bounded_semantic_seed_query = True  # type: ignore[attr-defined]
    ecosystem._seed_query = seed_query
    _replace_bound_references(original, seed_query)


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
            for variant in (_search_variants(query) or (query,)):
                try:
                    page = client.search(
                        "modrinth",
                        variant,
                        limit=30,
                        target_profile="minecraft_mod",
                    )
                    successes += 1
                    ids.extend(optimizer._candidate_ids(page))
                except Exception as exc:  # noqa: BLE001
                    local_errors.append(
                        f"neutral:{query!r} variant={variant!r}: {type(exc).__name__}: {exc}"
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
                except Exception as exc:  # pragma: no cover
                    found[query] = ()
                    errors.append(f"neutral:{query!r}: {type(exc).__name__}: {exc}")
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
        query_tuple = tuple(str(query) for query in queries)
        matrix_lists: dict[str, dict[str, list[str]]] = {
            adapter.adapter_id: {query: [] for query in query_tuple}
            for adapter in adapters
        }
        cached = _cache_get((id(client), query_tuple))
        if cached is not None:
            _neutral, neutral_errors, neutral_successes = cached
            if neutral_successes == 0 and neutral_errors:
                raise ValueError(
                    "Modrinth search source unavailable; refusing to score Minecraft "
                    "targets from empty transport evidence. Diagnostics: "
                    + "; ".join(neutral_errors[:8])
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
                except Exception as exc:  # noqa: BLE001
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
            futures = {pool.submit(run, adapter, query): (adapter, query) for adapter, query in jobs}
            for future in as_completed(futures):
                adapter, query = futures[future]
                try:
                    adapter_id, resolved_query, ids, local_errors, successes = future.result()
                    matrix_lists[adapter_id][resolved_query].extend(ids)
                    errors.extend(local_errors)
                    successful_requests += successes
                except Exception as exc:  # pragma: no cover
                    errors.append(
                        "matrix:"
                        f"{adapter.minecraft_version}/{adapter.loader}:{query!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if jobs and successful_requests == 0 and errors:
            raise ValueError(
                "Modrinth compatibility search source unavailable; refusing to interpret "
                "transport failures as unsupported mods or downgrade the Minecraft target. "
                "Diagnostics: " + "; ".join(errors[:8])
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


class _TargetProbe:
    def __init__(self, loader: str, minecraft_version: str) -> None:
        self.loader = loader
        self.minecraft_version = minecraft_version
        self.deterministic_module_kinds = frozenset()
        self.edition = "java"
        self.yarn_mappings = "mojang"
        self.mappings_kind = "mojang"
        self.mappings_version = "mojang"

    @property
    def adapter_id(self) -> str:
        return f"probe:{self.loader}:{self.minecraft_version}"


def _version_key(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value).split("."):
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _install_two_stage_platform_optimizer() -> None:
    from . import platform_optimizer as optimizer

    original = optimizer.optimize_platform
    if getattr(original, "_mmm_two_stage", False):
        return

    def optimized(
        prompt: str,
        *,
        design: Mapping[str, Any] | None = None,
        module_kinds: Any = (),
        loader_constraint: str | None = None,
        version_constraint: str | None = None,
        top_k: int = 4,
        discovery_client: Any | None = None,
        target_research_fn: Any | None = None,
        search_fn: Any | None = None,
        version_fn: Any | None = None,
    ) -> Any:
        if search_fn is not None or version_fn is not None:
            return original(
                prompt,
                design=design,
                module_kinds=module_kinds,
                loader_constraint=loader_constraint,
                version_constraint=version_constraint,
                top_k=top_k,
                discovery_client=discovery_client,
                target_research_fn=target_research_fn,
                search_fn=search_fn,
                version_fn=version_fn,
            )
        queries = optimizer.capability_queries(prompt, design=design, module_kinds=module_kinds)
        diagnostics: list[str] = []
        target_keys = optimizer.discover_target_keys(
            loader=loader_constraint,
            minecraft_version=version_constraint,
            limit_per_loader=12,
            diagnostics=diagnostics,
        )
        if not target_keys:
            target = "/".join(
                value for value in (version_constraint, loader_constraint) if value
            ) or "automatic"
            detail = "; ".join(diagnostics) or "no provider-discovered target was returned"
            raise ValueError(
                f"No executable platform provider can satisfy target {target!r}. Diagnostics: {detail}"
            )
        discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
        if discovery_mode not in {"auto", "on", "off"}:
            raise ValueError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
        if discovery_mode == "off":
            if len(target_keys) != 1:
                raise ValueError(
                    "Ecosystem discovery is disabled and multiple executable platform targets remain. "
                    "Supply an explicit Minecraft target or enable discovery."
                )
            loader, version = target_keys[0]
            return original(
                prompt,
                design=design,
                module_kinds=module_kinds,
                loader_constraint=loader,
                version_constraint=version,
                top_k=1,
                discovery_client=discovery_client,
                target_research_fn=target_research_fn,
            )
        probes = tuple(_TargetProbe(loader, version) for loader, version in target_keys)
        client = discovery_client or optimizer.EcosystemDiscoveryClient()
        neutral, neutral_errors = optimizer._parallel_neutral_shallow(queries, client)
        shallow_count = sum(len(value) for value in neutral.values())
        matrix, matrix_errors = optimizer._parallel_support_matrix(probes, queries, client)
        selected_probe = max(
            probes,
            key=lambda probe: (
                optimizer._support_score(
                    probe,
                    queries,
                    matrix.get(probe.adapter_id, {}),
                ),
                _version_key(probe.minecraft_version),
                probe.loader,
            ),
        )
        try:
            adapter = optimizer.adapter_for_target(
                selected_probe.minecraft_version,
                selected_probe.loader,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Selected platform target metadata is unavailable; refusing implicit "
                f"version downgrade for {selected_probe.minecraft_version}/{selected_probe.loader}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        deep = optimizer._parallel_deep(
            (adapter,),
            queries=queries,
            matrix={adapter.adapter_id: dict(matrix.get(selected_probe.adapter_id, {}))},
            client=client,
            target_research_fn=target_research_fn,
            inherited_errors=(*diagnostics, *neutral_errors, *matrix_errors),
            shallow_candidate_count=shallow_count,
        )
        if not deep:
            raise ValueError("No executable platform target survived evidence verification.")
        evidence = deep[0]
        return optimizer.PlatformOptimization(
            selected=adapter,
            evidence=evidence,
            candidates=(evidence,),
            capability_queries=queries,
            discovery_mode="lightweight-support-matrix_then-single-target-full-resolution",
        )

    optimized._mmm_two_stage = True  # type: ignore[attr-defined]
    optimizer.optimize_platform = optimized
    _replace_bound_references(original, optimized)


def _target_resolution_is_deferred(domain: Mapping[str, Any]) -> bool:
    if domain.get("target_frozen") is False:
        return True
    return str(domain.get("target_state") or "").strip().casefold() in {
        "unfrozen",
        "deferred",
        "pending",
        "target-neutral",
        "target_neutral",
    }


def _page_ref(note: Mapping[str, Any]) -> str:
    direct = _base_evidence_ref(note.get("page_ref"))
    if direct:
        return direct
    fragment = note.get("evidence_fragment")
    if isinstance(fragment, Mapping):
        return _base_evidence_ref(fragment.get("page_ref"))
    return ""


def _grounded_merge(group: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not group:
        return None
    claims: list[dict[str, Any]] = []
    procedures: list[dict[str, Any]] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    seen_claims: set[tuple[str, tuple[str, ...]]] = set()
    for note in group:
        parent_ref = _page_ref(note)
        for raw_claim in note.get("claims", ()):
            if not isinstance(raw_claim, Mapping):
                continue
            claim = dict(raw_claim)
            text = str(claim.get("claim") or claim.get("claim_text") or claim.get("text") or "").strip()
            if not text:
                continue
            refs = tuple(
                dict.fromkeys(
                    ref
                    for ref in (_base_evidence_ref(value) for value in claim.get("evidence_refs", ()))
                    if ref
                )
            )
            if not refs and parent_ref:
                refs = (parent_ref,)
            if not refs:
                continue
            key = (text, refs)
            if key in seen_claims:
                continue
            seen_claims.add(key)
            claim["claim"] = text
            claim["evidence_refs"] = list(refs)
            claims.append(claim)
        for raw_procedure in note.get("procedures", ()):
            if isinstance(raw_procedure, Mapping):
                procedures.append(dict(raw_procedure))
        for value in note.get("gaps", ()):
            text = str(value or "").strip()
            if text and text not in gaps:
                gaps.append(text)
        for value in note.get("next_queries", ()):
            text = str(value or "").strip()
            if text and text not in next_queries:
                next_queries.append(text)
    if not claims:
        return None
    base = dict(next(note for note in group if isinstance(note, Mapping)))
    base.update(
        {
            "claims": claims,
            "procedures": procedures,
            "gaps": gaps,
            "next_queries": next_queries,
            "sufficient": True,
        }
    )
    if "deferred_until_target_freeze" not in base["gaps"]:
        base["gaps"].append("deferred_until_target_freeze")
    return base


def _install_target_neutral_grounding() -> None:
    from . import agentic_pre_design_rag as rag

    original_host = rag._host_page_note
    if not getattr(original_host, "_mmm_target_neutral_page_grounding", False):
        @wraps(original_host)
        def host_page_note(domain_id: str, page: Mapping[str, Any]) -> dict[str, Any]:
            base = original_host(domain_id, page)
            context = _PAGE_GROUNDING_CONTEXT.get()
            if context is None or context.get("domain_id") != domain_id:
                return base
            extracted = rag._read_page_losslessly(
                context["agentic_module"],
                context["router"],
                prompt=context["prompt"],
                domain=context["domain"],
                document=context["document"],
                page=page,
                domain_key=context["domain_key"],
                progress_label=f"domain {domain_id} target-neutral grounding",
                failures=context["failures"],
            )
            merged = _grounded_merge(
                tuple(item for item in extracted if isinstance(item, Mapping))
            )
            if merged is None:
                return base
            merged["domain_id"] = domain_id
            if "evidence_fragment" in base:
                merged["evidence_fragment"] = base["evidence_fragment"]
            page_ref = str(page.get("page_ref", "")).strip()
            if page_ref:
                merged["page_ref"] = page_ref
            return merged

        host_page_note._mmm_target_neutral_page_grounding = True  # type: ignore[attr-defined]
        rag._host_page_note = host_page_note
        _replace_bound_references(original_host, host_page_note)

    original_domain = rag._research_document_domain
    if getattr(original_domain, "_mmm_target_neutral_domain_retry", False):
        return

    @wraps(original_domain)
    def research_document_domain(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if _PAGE_GROUNDING_CONTEXT.get() is not None:
            return original_domain(*args, **kwargs)
        try:
            return original_domain(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            bounded_error = getattr(rag, "_BoundedResearchOutputError", None)
            is_bounded = (
                bounded_error is not None and isinstance(exc, bounded_error)
            ) or type(exc).__name__ == "_BoundedResearchOutputError"
            text = str(exc).casefold()
            if not is_bounded or not (
                "zero grounded claims" in text
                or "zero grounded claim" in text
                or "no evidence-backed design-relevant claim" in text
            ):
                raise
            domain = kwargs.get("domain")
            document = kwargs.get("document")
            if not isinstance(domain, Mapping) or not isinstance(document, Mapping):
                raise
            if not _target_resolution_is_deferred(domain):
                raise
            agentic_module = args[0] if len(args) >= 1 else kwargs.get("agentic_module")
            router = args[1] if len(args) >= 2 else kwargs.get("router")
            prompt = str(kwargs.get("prompt") or "")
            domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
            domain_key = rag._domain_checkpoint_key(
                router,
                prompt=prompt,
                domain=domain,
                document=document,
            )
            token = _PAGE_GROUNDING_CONTEXT.set(
                {
                    "agentic_module": agentic_module,
                    "router": router,
                    "prompt": prompt,
                    "domain": domain,
                    "document": document,
                    "domain_id": domain_id,
                    "domain_key": domain_key,
                    "failures": [],
                }
            )
            try:
                return original_domain(*args, **kwargs)
            finally:
                _PAGE_GROUNDING_CONTEXT.reset(token)

    research_document_domain._mmm_target_neutral_domain_retry = True  # type: ignore[attr-defined]
    rag._research_document_domain = research_document_domain
    _replace_bound_references(original_domain, research_document_domain)


def _install_checkpoint_policy() -> None:
    from . import agentic_pre_design_rag as rag

    original = rag._domain_checkpoint_key
    if getattr(original, "_mmm_grounded_checkpoint_policy", False):
        return

    def checkpoint_key(
        router: Any,
        *,
        prompt: str,
        domain: Mapping[str, Any],
        document: Mapping[str, Any],
    ) -> str:
        legacy = original(router, prompt=prompt, domain=domain, document=document)
        return rag._sha256(
            {
                "legacy_domain_key": legacy,
                "research_policy": "lossless-page-grounding-provenance",
            }
        ).removeprefix("sha256:")

    checkpoint_key._mmm_grounded_checkpoint_policy = True  # type: ignore[attr-defined]
    rag._domain_checkpoint_key = checkpoint_key
    _replace_bound_references(original, checkpoint_key)


def _tool_names(request: Any) -> tuple[str, ...]:
    names: list[str] = []
    for tool in getattr(request, "tools", ()) or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = (
            str(function.get("name", "")).strip()
            if isinstance(function, Mapping)
            else str(tool.get("name", "")).strip()
        )
        if name:
            names.append(name)
    return tuple(dict.fromkeys(names))


def _is_source_edit_request(request: Any) -> bool:
    return _SOURCE_EDIT_TOOL in _tool_names(request)


def _message_content(message: Any) -> str:
    if not isinstance(message, Mapping):
        return ""
    value = message.get("content", "")
    return value if isinstance(value, str) else str(value or "")


def _clip_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = "\n...[host compacted stale coder context]...\n"
    if budget <= len(marker) + 32:
        return text[-budget:]
    remaining = budget - len(marker)
    head = max(16, remaining // 3)
    return f"{text[:head]}{marker}{text[-(remaining - head):]}"


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


def _compact_source_edit_request(request: Any) -> tuple[Any, int, int]:
    messages = tuple(getattr(request, "messages", ()) or ())
    original_chars = sum(len(_message_content(message)) for message in messages)
    trigger = _env_int("MMM_CODER_TOOL_COMPACT_TRIGGER_CHARS", 50000, 32000)
    if original_chars <= trigger:
        return request, original_chars, original_chars
    target = min(trigger, _env_int("MMM_CODER_TOOL_CONTEXT_CHARS", 26000, 12000))
    first_system_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, Mapping) and str(message.get("role", "")) == "system"
        ),
        None,
    )
    system_budget = min(8000, max(2000, target // 4)) if first_system_index is not None else 0
    remaining = target - system_budget
    allocations = [0] * len(messages)
    if first_system_index is not None:
        allocations[first_system_index] = system_budget
    old_floor = 256
    for index in range(len(messages) - 1, -1, -1):
        if index == first_system_index:
            continue
        length = len(_message_content(messages[index]))
        if not length:
            continue
        if remaining <= 0:
            allocations[index] = min(old_floor, length)
            continue
        grant = (
            min(length, remaining)
            if index >= max(0, len(messages) - 5)
            else min(length, max(old_floor, remaining // max(1, index + 1)))
        )
        allocations[index] = grant
        remaining -= min(remaining, grant)
    compacted: list[Any] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            compacted.append(message)
            continue
        copied = dict(message)
        content = _message_content(message)
        if content:
            copied["content"] = _clip_text(content, allocations[index])
        compacted.append(copied)
    compact_chars = sum(len(_message_content(message)) for message in compacted)
    try:
        return replace(request, messages=tuple(compacted)), original_chars, compact_chars
    except TypeError:
        return request, original_chars, original_chars


def _install_source_edit_budget() -> None:
    from . import llama_server_hardware_policy as hardware

    original = hardware._server_payload
    if getattr(original, "_mmm_source_edit_budget", False):
        return

    @wraps(original)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        if not _is_source_edit_request(request):
            return original(adapter, request)
        compact_request, original_chars, compact_chars = _compact_source_edit_request(request)
        result = dict(original(adapter, compact_request))
        if compact_chars >= original_chars:
            return result
        configured = max(1, int(getattr(adapter.config, "max_new_tokens", 1) or 1))
        requested = _env_int("MMM_CODER_TOOL_MIN_OUTPUT_TOKENS", 4096, 512)
        result["max_tokens"] = max(
            max(1, int(result.get("max_tokens", 1) or 1)),
            min(configured, requested),
        )
        result["reasoning_effort"] = "none"
        template_kwargs = dict(result.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        result["chat_template_kwargs"] = template_kwargs
        result.pop("thinking_budget_tokens", None)
        print(
            "coder source-edit contract:"
            f" input_chars={original_chars}->{compact_chars}"
            f" max_tokens={result.get('max_tokens')} thinking=off",
            file=sys.stderr,
            flush=True,
        )
        return result

    payload._mmm_source_edit_budget = True  # type: ignore[attr-defined]
    hardware._server_payload = payload
    _replace_bound_references(original, payload)


def install_runtime_contracts() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_research_provenance()
    _install_semantic_seed_search()
    _install_cheap_target_normalization()
    _install_verified_mod_search()
    _install_two_stage_platform_optimizer()
    _install_checkpoint_policy()
    _install_target_neutral_grounding()
    _install_source_edit_budget()
    _INSTALLED = True


__all__ = [
    "_compact_source_edit_request",
    "_grounded_merge",
    "_repair_note_provenance",
    "_target_resolution_is_deferred",
    "bounded_seed_query",
    "install_runtime_contracts",
]
