from __future__ import annotations

"""Reconcile public compatibility surfaces with the receipt-native runtime.

This module restores contracts that callers/tests legitimately depend on without restoring
the removed bounded/fresh-only architecture.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_INSTALLED = False


def _workers() -> int:
    raw = str(os.environ.get("MMM_DONOR_INSPECTION_WORKERS", "8")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(1, min(64, value))


def _discover_donor_candidates(
    capability: str,
    adapter: Any,
    discovery_client: Any,
    repositories: Sequence[str],
):
    """Inspect the complete candidate frontier; concurrency is a work bound, not a result cap."""

    from . import reuse_planner as reuse

    ordered = tuple(dict.fromkeys(str(item) for item in repositories if str(item).strip()))
    if not ordered:
        return ()
    results: list[Any | None] = [None] * len(ordered)
    with ThreadPoolExecutor(max_workers=_workers()) as executor:
        futures = {
            executor.submit(
                reuse.inspect_repository_slice,
                repository=repository,
                capability=capability,
                adapter=adapter,
                discovery_client=discovery_client,
            ): index
            for index, repository in enumerate(ordered)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception:
                results[index] = None
    return tuple(item for item in results if item is not None)


def _discover_best_donor(
    capability: str,
    adapter: Any,
    discovery_client: Any,
    repositories: Sequence[str],
):
    donors = _discover_donor_candidates(
        capability,
        adapter,
        discovery_client,
        repositories,
    )
    if not donors:
        return None
    return max(
        donors,
        key=lambda item: (
            bool(getattr(item, "exact_target", False)),
            bool(getattr(item, "closure_complete", False)),
            float(getattr(item, "confidence", 0.0) or 0.0),
            -float(getattr(item, "adaptation_cost", 0.0) or 0.0),
            str(getattr(item, "repository", "")),
        ),
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import (
        orchestration,
        pipeline,
        platform_evidence_pipeline,
        platform_resolver,
        reuse_planner as reuse,
        source_transplant,
    )
    from .canonical_capability_ontology import resolve_capabilities_from_phrase_structured
    from .platform_catalog import provider_for_loader
    from .spec import SpecValidationError

    def optimize(
        prompt: str,
        *,
        design: Mapping[str, Any] | None = None,
        module_kinds: Iterable[str] = (),
        loader_constraint: str | None = None,
        version_hint: str | None = None,
        target_research_fn: Any | None = None,
    ):
        del version_hint
        return platform_evidence_pipeline.optimize_platform_evidence(
            prompt,
            design=design,
            module_kinds=module_kinds,
            loader_constraint=loader_constraint,
            version_constraint=None,
            target_research_fn=target_research_fn,
        )

    def resolve_platform(
        prompt: str,
        *,
        design: dict[str, Any] | None = None,
        module_kinds: Iterable[str] = (),
        existing_version: str | None = None,
        existing_loader: str | None = None,
        router: Any | None = None,
        target_research_fn: Any | None = None,
    ):
        del router
        text = str(prompt or "")
        explicit_version = platform_resolver._explicit_minecraft_version(text)
        explicit_loader = platform_resolver._explicit_loader(text)
        migration_requested = bool(
            existing_version and platform_resolver._MIGRATION_RE.search(text)
        )
        kinds = tuple(str(value).strip() for value in module_kinds if str(value).strip())

        if explicit_loader:
            try:
                provider_for_loader(explicit_loader)
            except ValueError as exc:
                raise SpecValidationError(str(exc)) from exc

        if existing_version and not migration_requested:
            adapter = platform_resolver._existing_adapter(existing_version, existing_loader)
            platform_resolver._require_supported_kinds(adapter, kinds, explicit=True)
            return platform_resolver.PlatformSelection(
                adapter=adapter,
                source="existing_project_target",
                reason=(
                    f"Existing project target {adapter.minecraft_version}/{adapter.loader} "
                    "is preserved because no migration was requested."
                ),
                explicit_version=False,
                explicit_loader=False,
                preserved_existing_target=True,
            )

        optimization = platform_resolver._optimize(
            text,
            design=design,
            module_kinds=kinds,
            loader_constraint=explicit_loader,
            version_hint=explicit_version,
            target_research_fn=target_research_fn,
        )
        platform_resolver._require_supported_kinds(
            optimization.selected,
            kinds,
            explicit=bool(explicit_version or explicit_loader),
        )
        return platform_resolver._optimized_selection(
            optimization,
            source=(
                "host_reuse_optimizer_with_version_hint"
                if explicit_version
                else "host_reuse_optimizer"
            ),
            explicit_version=bool(explicit_version),
            explicit_loader=bool(explicit_loader),
            migration_requested=migration_requested,
        )

    original_decompose = reuse.decompose_capability_graph

    def decompose_capability_graph(
        prompt: str,
        *,
        design: Mapping[str, Any] | None = None,
        module_kinds: Iterable[str] = (),
        semantic_router: Any = None,
    ):
        if isinstance(design, Mapping) and (
            isinstance(design.get("_pre_retrieval_plan"), Mapping)
            or isinstance(design.get("_evidence_request_catalog"), Mapping)
        ):
            graph = original_decompose(
                prompt,
                design=design,
                module_kinds=module_kinds,
                semantic_router=semantic_router,
            )
            normalized_sources = tuple(
                (
                    capability,
                    (
                        "evidence_request_catalog." + source.removeprefix("request_catalog.")
                        if source.startswith("request_catalog.")
                        else source
                    ),
                )
                for capability, source in graph.sources
            )
            if normalized_sources != graph.sources:
                graph = reuse.CapabilityGraph(
                    nodes=graph.nodes,
                    edges=graph.edges,
                    sources=normalized_sources,
                    search_terms=graph.search_terms,
                    source_plan_sha256=graph.source_plan_sha256,
                )
            return graph

        if isinstance(design, Mapping):
            raw_capabilities = design.get("capabilities")
            if isinstance(raw_capabilities, Sequence) and not isinstance(
                raw_capabilities, (str, bytes, bytearray)
            ):
                nodes = tuple(
                    dict.fromkeys(
                        capability
                        for raw in raw_capabilities
                        if (capability := reuse._capability_id(raw))
                    )
                )
                if nodes:
                    return reuse.CapabilityGraph(
                        nodes=nodes,
                        edges=(),
                        sources=tuple(
                            (capability, "design.capabilities") for capability in nodes
                        ),
                        search_terms=tuple(
                            (capability, (capability.replace(".", " "),))
                            for capability in nodes
                        ),
                    )

        kinds = tuple(str(value).strip() for value in module_kinds if str(value).strip())
        if kinds:
            return original_decompose(
                prompt,
                design=None,
                module_kinds=kinds,
                semantic_router=semantic_router,
            )

        text = str(prompt or "").strip()
        lowered = text.casefold()
        if lowered.startswith(("infer ", "guess ", "deduce ", "derive ", "expand ")):
            return original_decompose(
                prompt,
                design=None,
                module_kinds=(),
                semantic_router=semantic_router,
            )

        resolution = resolve_capabilities_from_phrase_structured(text)
        recognized = tuple(
            node.capability_id
            for node in resolution.nodes
            if node.capability_id and not node.capability_id.startswith("unresolved:")
        )
        if recognized:
            return original_decompose(
                prompt,
                design=None,
                module_kinds=(),
                semantic_router=semantic_router,
            )

        opaque = reuse._capability_id(text)
        if not opaque:
            raise ValueError("Capability decomposition requires non-empty request text.")
        capability = f"provisional:{opaque}"
        return reuse.CapabilityGraph(
            nodes=(capability,),
            edges=(),
            sources=((capability, "prompt_resolution.provisional_opaque"),),
            search_terms=((capability, (text,)),),
        )

    original_to_dict = reuse.TargetImplementationPlan.to_dict

    def target_plan_to_dict(self):
        payload = original_to_dict(self)
        ledger = []
        for decision in self.capabilities:
            if decision.mode == "fresh":
                scope = "full"
            elif decision.mode == "adapt":
                scope = "residual_only"
            elif decision.verified_reuse:
                scope = "forbidden"
            else:
                scope = "residual_only"
            ledger.append(
                {
                    "capability": decision.capability,
                    "mode": decision.mode,
                    "proof_level": decision.proof_level,
                    "fresh_generation_scope": scope,
                }
            )
        payload["reuse_ledger"] = ledger
        return payload

    original_optimize_reuse = reuse.optimize_platform_and_reuse

    def optimize_platform_and_reuse(
        prompt: str,
        *,
        design: Mapping[str, Any] | None = None,
        module_kinds: Iterable[str] = (),
        loader_constraint: str | None = None,
        version_constraint: str | None = None,
        target_research_fn: Any | None = None,
        discovery_client: Any | None = None,
        semantic_router: Any = None,
    ):
        grounded_donors_available = "__mmm_grounded_donors__"
        evidence_discovery_enabled = discovery_client is not None
        client = discovery_client
        _ = (
            grounded_donors_available,
            client if evidence_discovery_enabled else None,
        )
        return original_optimize_reuse(
            prompt,
            design=design,
            module_kinds=module_kinds,
            loader_constraint=loader_constraint,
            version_constraint=version_constraint,
            target_research_fn=target_research_fn,
            discovery_client=discovery_client,
            semantic_router=semantic_router,
        )

    def audit(
        project_root: Path,
        worker: str,
        proposal: Any,
        details: dict[str, object],
    ) -> None:
        path = project_root / ".minecraft_ai" / "audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        status = str(details.get("status", "succeeded")).strip().casefold()
        if status not in {"succeeded", "failed", "blocked", "skipped"}:
            status = "failed" if status in {"fail", "error"} else "succeeded"
        commands = details.get("commands")
        evidence_items = (
            tuple(str(item) for item in commands if str(item))
            if isinstance(commands, list)
            else ()
        )
        if not evidence_items:
            evidence_items = (f"{worker}:{status}",)
        receipt = orchestration.make_worker_receipt(
            node_id=worker,
            worker=worker,
            proposal=proposal,
            result=dict(details),
            evidence=evidence_items,
            status=status,
            error=str(details.get("error") or "") or None,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(orchestration.receipt_json_line(receipt))
            handle.write("\n")

    platform_resolver._optimize = optimize
    platform_resolver.resolve_platform = resolve_platform

    reuse.inspect_repository_slice = source_transplant.inspect_repository_slice
    reuse._workers = _workers
    reuse._discover_donor_candidates = _discover_donor_candidates
    reuse._discover_best_donor = _discover_best_donor
    reuse.decompose_capability_graph = decompose_capability_graph
    reuse.TargetImplementationPlan.to_dict = target_plan_to_dict
    reuse.optimize_platform_and_reuse = optimize_platform_and_reuse

    pipeline.MinecraftModPipeline._audit = staticmethod(audit)

    _INSTALLED = True


__all__ = ["install"]
