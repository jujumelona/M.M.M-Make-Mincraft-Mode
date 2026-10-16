from __future__ import annotations

"""Host-owned bridge from grounded repository evidence to executable code reuse.

The small coder never searches for or selects a donor. Only GitHub repositories that
have host-materialized evidence in the planning-state RAG or were explicitly
supplied by the host/user may enter this pipeline. A discovery receipt
is reference-only: candidates are pinned and license/hash checked before inspection.
Only complete slices compiled against the selected target can be installed;
partial slices remain optional source evidence for authoring on the target API.
"""

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .ecosystem_discovery import EcosystemDiscoveryClient, provider_request_failure
from .platform_catalog import adapter_for_target
from .research_reuse_candidates import planning_state_repository_cards
from .reuse_proof_executor import ReuseProofReceipt, execute_reuse_proof
from .source_transplant import (
    DonorSlice,
    SourceTransplantError,
    inspect_repository_slice,
    materialize_pinned_donor,
    repository_inspection_failures,
    repository_inspection_requires_unavailable_network,
    repository_inspection_session,
    validate_donor_slice_manifest,
    validated_reuse_donor,
)
from .target_contract import TargetContract

_SCHEMA = "mmm/grounded-repository-reuse-plan-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]{2,}", re.IGNORECASE)
_REFERENCE_SPLIT_RE = re.compile(r"[\s,;]+")
_REPOSITORY_ID_RE = re.compile(
    r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$"
)
_STOP = frozenset(
    {
        "minecraft",
        "fabric",
        "mod",
        "mods",
        "system",
        "feature",
        "implementation",
        "source",
        "code",
        "the",
        "and",
        "for",
        "with",
        "from",
    }
)
DonorInspector = Callable[..., DonorSlice | None]
ProofExecutor = Callable[..., ReuseProofReceipt]


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"[_./:+-]+", " ", str(value or "").casefold())
    return {
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token not in _STOP
    }


def _github_repository(source_id: str, source_url: str) -> str:
    identity = str(source_id or "").strip()
    if identity.casefold().startswith("github:"):
        value = identity.split(":", 1)[1].strip().removesuffix(".git")
        if _REPOSITORY_ID_RE.fullmatch(value):
            return value
    try:
        parsed = urlparse(str(source_url or "").strip())
    except ValueError:
        return ""
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    value = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return value if _REPOSITORY_ID_RE.fullmatch(value) else ""


def _normalize_reference_repository(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    direct = raw.removesuffix(".git")
    if _REPOSITORY_ID_RE.fullmatch(direct):
        return direct
    repository = _github_repository("", raw)
    if repository:
        return repository
    raise ValueError(
        "MMM_REFERENCE_MOD_URLS only accepts GitHub repository URLs or owner/repo "
        f"identifiers; unsupported reference: {raw!r}"
    )


def _host_reference_cards() -> tuple[dict[str, Any], ...]:
    raw = os.environ.get("MMM_REFERENCE_MOD_URLS", "")
    if not raw.strip():
        return ()
    repositories: list[str] = []
    seen: set[str] = set()
    for value in _REFERENCE_SPLIT_RE.split(raw.strip()):
        if not value:
            continue
        repository = _normalize_reference_repository(value)
        key = repository.casefold()
        if key in seen:
            continue
        seen.add(key)
        repositories.append(repository)
    return tuple(
        {
            "repository": repository,
            "page_refs": ["host:reference_mod"],
            "source_ids": [f"github:{repository}"],
            "source_urls": [f"https://github.com/{repository}"],
            "evidence_text": "",
            "evidence_tokens": [],
            "explicit_reference": True,
        }
        for repository in repositories
    )


def _merge_card(cards: list[dict[str, Any]], by_repository: dict[str, int], raw: Mapping[str, Any]) -> None:
    repository = str(raw.get("repository") or "").strip()
    if not repository:
        return
    key = repository.casefold()
    if key not in by_repository:
        by_repository[key] = len(cards)
        cards.append(dict(raw))
        return
    existing = cards[by_repository[key]]
    existing["explicit_reference"] = bool(existing.get("explicit_reference")) or bool(
        raw.get("explicit_reference")
    )
    for field in ("page_refs", "source_ids", "source_urls", "catalog_evidence"):
        target = existing.setdefault(field, [])
        values = raw.get(field)
        for value in values if isinstance(values, list) else ():
            if value and value not in target:
                target.append(value)
    priorities = existing.setdefault("catalog_provider_priority", {})
    for capability, priority in raw.get("catalog_provider_priority", {}).items():
        priorities[capability] = min(priorities.get(capability, priority), priority)
    merged = f"{existing.get('evidence_text', '')} {raw.get('evidence_text', '')}".strip()
    existing["evidence_text"] = merged[:1280]
    existing["evidence_tokens"] = sorted(_tokens(existing["evidence_text"]))


def _grounded_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    # Prompt-first planning writes only host-stamped, reference-only candidates here.
    # They still have zero reuse authority until inspect/proof below succeeds.
    cards: list[dict[str, Any]] = []
    by_repository: dict[str, int] = {}
    for raw in planning_state_repository_cards(design):
        _merge_card(cards, by_repository, raw)


    # Prebuilt Debug selects real repository identities, never executable donors.
    # Restrict identities, retain reference-only status, then use the SAME
    # target-bound inspection, license and compile proofs as regular production.
    preselected = design.get("_preselected_reference_repositories", ())
    if not isinstance(preselected, (list, tuple)) or len(preselected) > 8:
        raise ValueError("DEBUG_REFERENCE_CANDIDATES_INVALID")
    for raw in preselected:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("DEBUG_REFERENCE_CANDIDATE_INVALID")
        repository = _normalize_reference_repository(raw)
        _merge_card(cards, by_repository, {
            "repository": repository,
            "page_refs": ["host:prebuilt_debug_reference"],
            "source_ids": ["github:" + repository],
            "source_urls": ["https://github.com/" + repository],
            "evidence_text": "",
            "evidence_tokens": [],
            "explicit_reference": True,
            "candidate_origin": "prebuilt_debug_reference",
            "reference_only": True,
            "source_reuse_authority": "verification_required",
        })

    for host_card in _host_reference_cards():
        _merge_card(cards, by_repository, host_card)

    for card in cards:
        card["evidence_tokens"] = sorted(
            set(str(value) for value in card.get("evidence_tokens", ()))
            | _tokens(card.get("evidence_text", ""))
        )
    return tuple(cards)


def _frozen_graph(design: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    plan = design.get("_pre_retrieval_plan")
    if not isinstance(plan, Mapping):
        raise TypeError("Repository reuse requires the frozen pre-retrieval plan.")
    plan_sha256 = str(plan.get("plan_sha256") or "").strip()
    graph = plan.get("capability_graph")
    if not plan_sha256 or not isinstance(graph, Mapping):
        raise ValueError("Frozen pre-retrieval plan has no capability graph receipt.")
    payload = dict(graph)
    payload["source_plan_sha256"] = plan_sha256
    return payload, plan_sha256


def _adapter(design: Mapping[str, Any]) -> TargetContract:
    selection = design.get("_platform_selection")
    target = selection.get("target") if isinstance(selection, Mapping) else None
    if not isinstance(target, Mapping):
        raise TypeError("Repository reuse requires resolved target coordinates.")
    version = str(target.get("minecraft_version") or "").strip()
    loader = str(target.get("loader") or "").strip().casefold()
    if not version or not loader:
        raise ValueError("Repository reuse target coordinates are incomplete.")
    return adapter_for_target(version, loader)


def _search_terms(graph: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    raw = graph.get("search_terms")
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, Mapping):
            continue
        capability = str(item.get("capability") or "").strip()
        terms = item.get("terms")
        values = (
            tuple(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))
            if isinstance(terms, Sequence)
            and not isinstance(terms, (str, bytes, bytearray))
            else ()
        )
        if capability:
            result[capability] = values
    return result


def _candidate_overlap(
    capability: str,
    terms: Sequence[str],
    card: Mapping[str, Any],
) -> int:
    wanted = _tokens(" ".join((capability, *terms)))
    available = {str(item) for item in card.get("evidence_tokens", ())}
    return len(wanted & available)


def _target_context(adapter: TargetContract) -> dict[str, Any]:
    return {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "mappings_kind": adapter.mappings_kind,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
    }


def _proof_admits(donor: DonorSlice, receipt: ReuseProofReceipt) -> bool:
    decision = {
        "capability": donor.capability,
        "mode": "source_transplant",
        "donor": donor.to_dict(),
        "proof_receipt": receipt.to_dict(),
    }
    try:
        validated_reuse_donor(decision)
    except (SourceTransplantError, ValueError):
        return False
    return True


def _inspection_backend_unavailable(exc: Exception) -> bool:
    """Stop repeated impossible downloads, not semantically rejected donors."""
    failure = provider_request_failure(exc)
    return failure is not None and failure.backend_unavailable


def _record_inspection_outage(outage, capability, repository, exc=None):
    if outage is None:
        return
    failure = provider_request_failure(exc) if exc is not None else None
    if failure is None or not failure.backend_unavailable:
        failure = next(iter(repository_inspection_failures()), None)
    if failure is None or not failure.backend_unavailable:
        return
    first = not outage
    outage.setdefault("repository", repository)
    outage.setdefault("error", (f"{type(exc).__name__}: {exc}"[:1200] if exc is not None
                                else f"{failure.hostname} source request is unavailable."))
    outage.setdefault("request_failure", failure.to_dict())
    if first:
        print("SOURCE_REUSE_INSPECTION_OUTAGE: " + json.dumps(
            {"capability": capability, **outage}, ensure_ascii=False), flush=True)


def _retain_reference_source(donor, capability, client, diagnostics, references, inspections):
    """Only stop discovery for source members the Java handoff can actually read."""
    from .construction_context import java_source_units

    try:
        blobs = materialize_pinned_donor(donor, client)
        readable_member = False
        for path, raw in blobs.items():
            if not path.endswith('.java'):
                continue
            try:
                units = java_source_units(raw.decode('utf-8'), path=path)
            except (ValueError, RuntimeError, UnicodeError):
                continue
            if any(unit['kind'] in {'method', 'constructor', 'field'} for unit in units):
                readable_member = True
                break
        if not readable_member:
            raise SourceTransplantError('Pinned reference has no parseable Java member for the source handoff.')
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        inspections.append({
            'capability': capability, 'repository': donor.repository,
            'status': 'reference_unavailable', 'error': f'{type(exc).__name__}: {exc}',
        })
        return False
    references.append({
        'capability': capability, 'donor': donor.to_dict(),
        'authority': 'pinned_licensed_reference_only', 'migration_diagnostics': diagnostics,
    })
    return True


def _inspect_candidate_batch(
    cards: Sequence[Mapping[str, Any]], *, capability: str, terms: Sequence[str],
    adapter: TargetContract, client: EcosystemDiscoveryClient,
    donor_inspector: DonorInspector, inspections: list[dict[str, Any]],
    reference_sources: list[dict[str, Any]],
    outage: dict[str, Any] | None = None,
    inspection_cache: dict[tuple[str, str, str, str], tuple[DonorSlice | None, Exception | None]] | None = None,
) -> list[DonorSlice]:
    """Inspect a small ranked batch before paying for the rest of the frontier."""
    candidates: list[DonorSlice] = []
    for card in cards:
        if outage and donor_inspector is not inspect_repository_slice:
            break
        overlap = _candidate_overlap(capability, terms, card)
        explicit_reference = bool(card.get("explicit_reference"))
        repository = str(card["repository"])
        if donor_inspector is inspect_repository_slice and repository_inspection_requires_unavailable_network(
            repository, minecraft_version=adapter.minecraft_version, loader=adapter.loader,
        ):
            inspections.append({
                "capability": capability, "repository": repository,
                "page_refs": list(card.get("page_refs", ())),
                "status": "inspection_deferred", "provider_unavailable": True,
                "reason": "No cached pinned snapshot; provider network is unavailable for this plan.",
            })
            continue
        cache_key = (repository.casefold(), adapter.minecraft_version, adapter.loader, capability)
        cached = inspection_cache.get(cache_key) if inspection_cache is not None else None
        try:
            if cached is not None:
                donor, error = cached
                if error is not None:
                    raise error
            else:
                donor = donor_inspector(
                    repository=repository,
                    capability=capability,
                    adapter=adapter,
                    discovery_client=client,
                )
                if inspection_cache is not None:
                    inspection_cache[cache_key] = (donor, None)
        except Exception as exc:  # noqa: BLE001 - fail closed per external donor
            if inspection_cache is not None:
                inspection_cache[cache_key] = (None, exc)
            error = f"{type(exc).__name__}: {exc}"[:1200]
            unavailable = _inspection_backend_unavailable(exc)
            failure = provider_request_failure(exc)
            inspections.append({
                "capability": capability, "repository": repository,
                "page_refs": list(card.get("page_refs", ())),
                "explicit_reference": explicit_reference,
                "candidate_origin": str(card.get("candidate_origin") or "planning_state_or_explicit"),
                "status": "inspection_error", "error": error,
                "provider_unavailable": unavailable,
                "request_failure": failure.to_dict() if failure is not None else None,
                "inspection_cache_hit": cached is not None,
            })
            _record_inspection_outage(outage, capability, repository, exc)
            continue
        _record_inspection_outage(outage, capability, repository)
        reference_eligible = False
        rejection = "No pinned source slice was located."
        if donor is not None:
            try:
                validate_donor_slice_manifest(donor)
                if (donor.compatibility_evidence is None or
                        donor.compatibility_evidence.minecraft_version != adapter.minecraft_version):
                    raise SourceTransplantError("Pinned source Minecraft version does not match the request.")
                reference_eligible = True
            except (SourceTransplantError, ValueError) as exc:
                rejection = str(exc)
        admitted = bool(reference_eligible and donor is not None and donor.closure_complete
                        and donor.target_compatibility in {"exact", "adapt"})
        if reference_eligible and not admitted:
            diagnostics = {
                "closure_complete": donor.closure_complete,
                "target_compatibility": donor.target_compatibility,
                "truncation_reason": donor.truncation_reason,
                "unresolved_dependency_count": len(donor.unresolved_edges),
                "dependency_diagnostics_location": 'donor.unresolved_edges',
                "instruction": "Inspect available members for useful partial logic; resolve target APIs independently.",
            }
            reference_eligible = _retain_reference_source(
                donor, capability, client, json.dumps(diagnostics, ensure_ascii=False),
                reference_sources, inspections)
            _record_inspection_outage(outage, capability, repository)
            if not reference_eligible:
                rejection = 'Pinned slice has no readable Java reference for the authoring handoff.'
        inspections.append(
            {
                "capability": capability,
                "repository": repository,
                "page_refs": list(card.get("page_refs", ())),
                "explicit_reference": explicit_reference,
                "candidate_origin": str(card.get("candidate_origin") or "planning_state_or_explicit"),
                "status": ("proof_pending" if admitted else
                           "reference_only" if reference_eligible else "inspection_rejected"),
                "overlap": overlap,
                "inspection_cache_hit": cached is not None,
                "rejection_reason": "" if reference_eligible else rejection,
            }
        )
        if admitted and donor is not None:
            candidates.append(donor)

    return candidates


@repository_inspection_session()
def build_repository_reuse_plan(
    design: Mapping[str, Any],
    *,
    discovery_client: EcosystemDiscoveryClient | None = None,
    donor_inspector: DonorInspector = inspect_repository_slice,
    proof_executor: ProofExecutor = execute_reuse_proof,
    target_workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed, code-bearing reuse plan for the frozen capability graph."""

    graph, source_plan_sha256 = _frozen_graph(design)
    adapter = _adapter(design)
    adapter.validate()
    client = discovery_client or EcosystemDiscoveryClient()
    cards = list(_grounded_repository_cards(design))
    # Host-owned discovery feeds reference-only repository identities into the
    # existing immutable source, license, closure and compile-proof gates.
    # A search hit has ZERO reuse authority until the proof below passes.
    from .curseforge_catalog import curseforge_api_key
    from .reuse_discovery import discover_repositories_for_graph

    discovery_receipts: list[dict[str, Any]] = []
    query_cache: dict[tuple[str, str, str, str], Any] = {}
    discovered: dict[str, tuple[str, ...]] = {}
    by_repository = {str(card["repository"]).casefold(): i for i, card in enumerate(cards)}

    def discover_stage(capability_set: Sequence[str], provider: str) -> None:
        receipts: list[dict[str, Any]] = []
        results = discover_repositories_for_graph(
            capability_set, client, capability_graph=graph,
            minecraft_version=adapter.minecraft_version, loader=adapter.loader,
            diagnostics=receipts, providers=(provider,), query_cache=query_cache,
        )
        discovery_receipts.extend(receipts)
        for capability, repositories in results.items():
            discovered[capability] = tuple(dict.fromkeys(
                (*discovered.get(capability, ()), *repositories)))
            for repository in repositories:
                provenance = [
                    dict(item) for receipt in receipts
                    if capability in receipt.get("capabilities", ())
                    for item in receipt.get("repository_evidence", ())
                    if str(item.get("repository") or "").casefold() == repository.casefold()
                ]
                priority = min((item["provider_priority"] for item in provenance), default=2)
                source_ids = ["github:" + repository]
                source_urls = ["https://github.com/" + repository]
                for item in provenance:
                    if item.get("source_id"):
                        source_ids.append(item["source_id"])
                    for field in ("catalog_url", "source_url"):
                        if item.get(field):
                            source_urls.append(item[field])
                _merge_card(cards, by_repository, {
                    "repository": repository,
                    "page_refs": ["host:ecosystem_discovery"],
                    "source_ids": list(dict.fromkeys(source_ids)),
                    "source_urls": list(dict.fromkeys(source_urls)),
                    "evidence_text": " ".join(
                        str(item.get(field) or "")
                        for item in provenance
                        for field in ("title", "summary", "code_search_path")
                    )[:1280],
                    "evidence_tokens": sorted(_tokens(" ".join(
                        str(item.get(field) or "")
                        for item in provenance
                        for field in ("title", "summary", "code_search_path")
                    ))),
                    "explicit_reference": False,
                    "candidate_origin": "host_ecosystem_discovery",
                    "reference_only": True,
                    "source_reuse_authority": "verification_required",
                    "catalog_evidence": provenance,
                    "catalog_provider_priority": {capability: priority},
                })
    terms_by_capability = _search_terms(graph)
    capabilities = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in graph.get("nodes", ())
            if str(item).strip()
        )
    )
    if not capabilities:
        raise ValueError("Repository reuse requires at least one frozen capability.")

    provider_order = (("curseforge", "modrinth", "sourcegraph", "github") if curseforge_api_key()
                      else ("modrinth", "sourcegraph", "github"))
    print("SOURCE_REUSE_DISCOVERY_ROUTE: " + json.dumps({
        "provider_order": provider_order, "curseforge_key_present": provider_order[0] == "curseforge",
        "fallback_policy": "inspect_current_provider_before_searching_next",
        "minecraft_version": adapter.minecraft_version, "loader": adapter.loader,
    }), flush=True)
    # Combine independent catalogue and indexed-code evidence before choosing donors.
    # GitHub repository search remains the final fallback after source inspection.
    preloaded_providers = provider_order[:-1]
    for initial_provider in preloaded_providers:
        discover_stage(capabilities, initial_provider)

    def candidate_batches(capability, terms, ranked_cards):
        seen: set[str] = set()
        for stage, provider in enumerate(provider_order):
            if stage and provider in preloaded_providers:
                continue  # Already merged across the complete frozen capability graph.
            if provider not in preloaded_providers:
                discover_stage((capability,), provider)
            stage_cards = sorted(
                (card for card in cards if str(card["repository"]).casefold() not in seen),
                key=lambda card: (
                    bool(card.get("explicit_reference")),
                    -card.get("catalog_provider_priority", {}).get(capability, 2),
                    _candidate_overlap(capability, terms, card),
                ), reverse=True,
            )
            ranked_cards.extend(stage_cards)
            seen.update(str(card["repository"]).casefold() for card in stage_cards)
            for offset in range(0, len(stage_cards), 2):
                yield stage_cards[offset:offset + 2]

    decisions: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    reference_sources: list[dict[str, Any]] = []
    temporary = None
    proof_workspace = Path(target_workspace).resolve() if target_workspace else None
    if proof_workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="mmm_reuse_target_")
        proof_workspace = Path(temporary.name)
    try:
        inspection_outage: dict[str, Any] = {}
        inspection_cache: dict[tuple[str, str, str, str], tuple[DonorSlice | None, Exception | None]] = {}
        for capability in capabilities:
            terms = terms_by_capability.get(capability, (capability.replace(".", " "),))
            verified: list[tuple[DonorSlice, ReuseProofReceipt]] = []
            ranked_cards: list[dict[str, Any]] = []

            # Rank a small batch locally. Once pinned source can guide partial
            # authoring, hand it off instead of exhausting the frontier looking
            # for a donor that can be installed unchanged.
            for candidate_batch in candidate_batches(capability, terms, ranked_cards):
                if inspection_outage and donor_inspector is not inspect_repository_slice:
                    break
                candidates = _inspect_candidate_batch(
                    candidate_batch, capability=capability, terms=terms,
                    adapter=adapter, client=client, donor_inspector=donor_inspector,
                    inspections=inspections, reference_sources=reference_sources,
                    outage=inspection_outage,
                    inspection_cache=inspection_cache,
                )
                for donor in sorted(candidates, key=lambda row: (
                    row.target_compatibility == "exact", -row.adaptation_cost, row.confidence,
                ), reverse=True):
                    try:
                        receipt = proof_executor(
                            donor,
                            target_workspace=proof_workspace,
                            target_context=_target_context(adapter),
                            discovery_client=client,
                            run_tests=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - reject one failed proof only
                        proofs.append(
                            {
                                "candidate_id": f"{donor.repository}@{donor.commit_sha}",
                                "capability": capability,
                                "status": "proof_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        _retain_reference_source(
                            donor, capability, client, f"{type(exc).__name__}: {exc}"[:1200],
                            reference_sources, inspections)
                        _record_inspection_outage(inspection_outage, capability, donor.repository, exc)
                        continue
                    proof_payload = receipt.to_dict()
                    proof_payload["status"] = (
                        "verified_code_reuse" if _proof_admits(donor, receipt) else "rejected"
                    )
                    proofs.append(proof_payload)
                    if _proof_admits(donor, receipt):
                        verified.append((donor, receipt))
                        break
                    else:
                        _retain_reference_source(
                            donor, capability, client, str(receipt.failure_message or receipt.failure_code)[:1200],
                            reference_sources, inspections)
                        _record_inspection_outage(inspection_outage, capability, donor.repository)

                if verified or any(row['capability'] == capability for row in reference_sources):
                    break

            selected = verified[0] if verified else None
            if selected is None:
                failed_proofs = [r for r in proofs if r.get("capability") == capability
                                 and r.get("status") in {"rejected", "proof_error"}]
                failed_inspections = [r for r in inspections if r.get("capability") == capability
                                      and r.get("status") == "inspection_error"]
                searches = [r for r in discovery_receipts
                            if capability in r.get("capabilities", ())]
                reference_count = sum(row['capability'] == capability for row in reference_sources)
                if reference_count:
                    search_status, failure_code = 'reference_available', ''
                elif inspection_outage:
                    search_status = "provider_error"
                    failure_code = "DONOR_INSPECTION_PROVIDER_UNAVAILABLE"
                elif failed_proofs:
                    search_status = "target_compile_failed"
                    failure_code = str(failed_proofs[-1].get("failure_code") or
                                       failed_proofs[-1].get("status"))
                elif failed_inspections:
                    search_status = "download_error"
                    failure_code = "DONOR_INSPECTION_FAILED"
                elif searches and all(r["status"] == "provider_error" for r in searches):
                    search_status = "provider_error"
                    failure_code = "ALL_DISCOVERY_PROVIDERS_FAILED"
                elif searches and all(r["status"] in {"search_empty", "provider_error"} for r in searches):
                    # A successful empty page cannot certify that no source
                    # exists when another provider could not be queried.
                    incomplete = any(r["status"] == "provider_error" for r in searches)
                    search_status = "provider_error" if incomplete else "search_empty"
                    failure_code = ("INCOMPLETE_PROVIDER_SEARCH" if incomplete
                                    else "NO_SOURCE_CANDIDATES")
                else:
                    search_status, failure_code = "unresolved", "NO_VERIFIED_DONOR"
                decisions.append(
                    {
                        "capability": capability,
                        "mode": "fresh",
                        "search_status": search_status,
                        "failure_code": failure_code,
                        "source_id": "",
                        "component_refs": [],
                        "reference_source_count": reference_count,
                        "rationale": (
                            "Pinned same-version source is available for optional partial porting; "
                            "author the remaining behavior against the target API."
                            if reference_count else
                            "No grounded or explicitly referenced GitHub donor passed immutable "
                            "source, permissive license, dependency closure, authoritative compile, "
                            "and artifact coverage gates."
                        ),
                    }
                )
                print("SOURCE_REUSE_CAPABILITY: " + json.dumps({
                    "capability": capability, "mode": "fresh",
                    "discovered_repositories": len(discovered.get(capability, ())),
                    "ranked_candidates": len(ranked_cards),
                    "inspected": sum(row.get("capability") == capability
                                     and row.get("status") != "inspection_deferred" for row in inspections),
                    "deferred_unavailable": sum(row.get("capability") == capability
                                                and row.get("status") == "inspection_deferred" for row in inspections),
                    "proof_attempts": sum(row.get("capability") == capability for row in proofs),
                    "status": search_status, "failure_code": failure_code,
                    "reference_sources": sum(row.get("capability") == capability
                                             for row in reference_sources),
                    "inspection_outage": dict(inspection_outage),
                    "inspection_errors": [
                        {"repository": row.get("repository"),
                         "error": str(row.get("error", ""))[:240]}
                        for row in inspections
                        if row.get("capability") == capability and row.get("status") == "inspection_error"
                    ][:3],
                }, ensure_ascii=False), flush=True)
                continue
            donor, receipt = selected
            decisions.append(
                {
                    "capability": capability,
                    "mode": "source_transplant",
                    "source_id": f"host-donor:{donor.repository}@{donor.commit_sha}",
                    "component_refs": [],
                    "donor": donor.to_dict(),
                    "proof_receipt": receipt.to_dict(),
                    "rationale": (
                        "Host-grounded or explicitly referenced donor source was pinned, "
                        "hash-verified, adapted, and authoritatively compiled before coder generation."
                    ),
                }
            )
            print("SOURCE_REUSE_CAPABILITY: " + json.dumps({
                "capability": capability, "mode": "source_transplant",
                "selected_repository": donor.repository,
                "selected_commit": donor.commit_sha,
                "discovered_repositories": len(discovered.get(capability, ())),
                "ranked_candidates": len(ranked_cards),
                "inspected": sum(row.get("capability") == capability
                                 and row.get("status") != "inspection_deferred" for row in inspections),
                "proof_attempts": sum(row.get("capability") == capability for row in proofs),
            }, ensure_ascii=False), flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()

    planning_cards = planning_state_repository_cards(design)
    return {
        "schema_version": _SCHEMA,
        "bound_target": {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader},
        "source_plan_sha256": source_plan_sha256,
        "capability_graph": graph,
        "capabilities": decisions,
        "grounded_repository_count": len(cards),
        "host_discovered_repositories": {
            str(capability): list(repositories)
            for capability, repositories in discovered.items()
        },
        "planning_rag_candidate_repository_count": len(planning_cards),
        "explicit_reference_repository_count": sum(
            bool(card.get("explicit_reference")) for card in cards
        ),
        "discovery_receipts": discovery_receipts,
        "inspection_receipts": inspections,
        "proof_receipts": proofs,
        "reference_sources": reference_sources,
        "selection_policy": (
            "reference-only host-grounded/explicit GitHub candidate -> frozen-intent capability "
            "inspection -> immutable source + permissive-license closure -> authoritative target "
            "compile -> complete verified artifact coverage -> task-owned approved source context"
        ),
    }


__all__ = ["build_repository_reuse_plan"]
