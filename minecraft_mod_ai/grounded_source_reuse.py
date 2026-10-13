from __future__ import annotations

"""Host-owned bridge from grounded repository evidence to executable code reuse.

The small coder never searches for or selects a donor. Only GitHub repositories that
have host-materialized evidence in the planning-state RAG or were explicitly
supplied by the host/user may enter this pipeline. A discovery receipt
is reference-only: a candidate is pinned, license/closure checked, materialized, and
compiled against the selected target before its source is attached to a generation task.
"""

import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import adapter_for_target
from .research_reuse_candidates import planning_state_repository_cards
from .reuse_proof_executor import ReuseProofReceipt, execute_reuse_proof
from .source_transplant import (
    DonorSlice,
    SourceTransplantError,
    inspect_repository_slice,
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
    for field in ("page_refs", "source_ids", "source_urls"):
        target = existing.setdefault(field, [])
        values = raw.get(field)
        for value in values if isinstance(values, list) else ():
            if value and value not in target:
                target.append(value)
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


def _inspect_candidate_batch(
    cards: Sequence[Mapping[str, Any]], *, capability: str, terms: Sequence[str],
    adapter: TargetContract, client: EcosystemDiscoveryClient,
    donor_inspector: DonorInspector, inspections: list[dict[str, Any]],
) -> list[DonorSlice]:
    """Inspect a small ranked batch before paying for the rest of the frontier."""
    candidates: list[DonorSlice] = []
    for card in cards:
        overlap = _candidate_overlap(capability, terms, card)
        explicit_reference = bool(card.get("explicit_reference"))
        if overlap <= 0 and not explicit_reference:
            continue
        repository = str(card["repository"])
        try:
            donor = donor_inspector(
                repository=repository,
                capability=capability,
                adapter=adapter,
                discovery_client=client,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed per external donor
            inspections.append(
                {
                    "capability": capability,
                    "repository": repository,
                    "page_refs": list(card.get("page_refs", ())),
                    "explicit_reference": explicit_reference,
                    "candidate_origin": str(card.get("candidate_origin") or "planning_state_or_explicit"),
                    "status": "inspection_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        admitted = bool(
            donor is not None
            and donor.closure_complete
            and donor.target_compatibility in {"exact", "adapt"}
        )
        inspections.append(
            {
                "capability": capability,
                "repository": repository,
                "page_refs": list(card.get("page_refs", ())),
                "explicit_reference": explicit_reference,
                "candidate_origin": str(card.get("candidate_origin") or "planning_state_or_explicit"),
                "status": "proof_pending" if admitted else "inspection_rejected",
                "overlap": overlap,
            }
        )
        if admitted and donor is not None:
            candidates.append(donor)

    return candidates


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
    from .reuse_discovery import discover_repositories_for_graph

    discovery_receipts: list[dict[str, Any]] = []
    discovered = discover_repositories_for_graph(
        tuple(str(node) for node in graph.get("nodes", ()) if str(node).strip()),
        client,
        capability_graph=graph,
        minecraft_version=adapter.minecraft_version,
        loader=adapter.loader,
        diagnostics=discovery_receipts,
    )
    by_repository = {str(card["repository"]).casefold(): i for i, card in enumerate(cards)}
    for capability, repositories in discovered.items():
        for repository in repositories:
            _merge_card(cards, by_repository, {
                "repository": repository,
                "page_refs": ["host:ecosystem_discovery"],
                "source_ids": ["github:" + repository],
                "source_urls": ["https://github.com/" + repository],
                "evidence_text": capability,
                "evidence_tokens": sorted(_tokens(capability)),
                "explicit_reference": False,
                "candidate_origin": "host_ecosystem_discovery",
                "reference_only": True,
                "source_reuse_authority": "verification_required",
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
        for capability in capabilities:
            terms = terms_by_capability.get(capability, (capability.replace(".", " "),))
            verified: list[tuple[DonorSlice, ReuseProofReceipt]] = []
            ranked_cards = sorted(
                (card for card in cards if card.get("explicit_reference")
                 or _candidate_overlap(capability, terms, card) > 0),
                key=lambda card: (bool(card.get("explicit_reference")),
                                  _candidate_overlap(capability, terms, card)), reverse=True,
            )
            # Two inspections preserve exact-target/adaptation-cost ranking
            # locally. Later batches remain available if every proof rejects;
            # stop downloading as soon as usable code is found.
            for offset in range(0, len(ranked_cards), 2):
                candidates = _inspect_candidate_batch(
                    ranked_cards[offset:offset + 2], capability=capability, terms=terms,
                    adapter=adapter, client=client, donor_inspector=donor_inspector,
                    inspections=inspections,
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
                        reference_sources.append({
                            "capability": capability, "donor": donor.to_dict(),
                            "authority": "pinned_licensed_reference_only",
                            "migration_diagnostics": f"{type(exc).__name__}: {exc}"[:1200],
                        })
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
                        reference_sources.append({
                            "capability": capability, "donor": donor.to_dict(),
                            "authority": "pinned_licensed_reference_only",
                            "migration_diagnostics": str(receipt.failure_message or receipt.failure_code)[:1200],
                        })

                if verified:
                    break

            selected = verified[0] if verified else None
            if selected is None:
                failed_proofs = [r for r in proofs if r.get("capability") == capability
                                 and r.get("status") in {"rejected", "proof_error"}]
                failed_inspections = [r for r in inspections if r.get("capability") == capability
                                      and r.get("status") == "inspection_error"]
                searches = [r for r in discovery_receipts
                            if capability in r.get("capabilities", ())]
                if failed_proofs:
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
                    search_status = "search_empty" if any(r["status"] == "search_empty" for r in searches) else "provider_error"
                    failure_code = "NO_SOURCE_CANDIDATES" if search_status == "search_empty" else "ALL_DISCOVERY_PROVIDERS_FAILED"
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
                        "rationale": (
                            "No grounded or explicitly referenced GitHub donor passed immutable "
                            "source, permissive license, dependency closure, authoritative compile, "
                            "and artifact coverage gates."
                        ),
                    }
                )
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
    finally:
        if temporary is not None:
            temporary.cleanup()

    planning_cards = planning_state_repository_cards(design)
    return {
        "schema_version": _SCHEMA,
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
