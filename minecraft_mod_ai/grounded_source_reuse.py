from __future__ import annotations

"""Host-owned bridge from grounded repository evidence to executable code reuse.

The small coder never searches for or selects a donor.  Only GitHub repositories that
already have host-materialized evidence in pre-design RAG may enter this pipeline.  A
candidate is then pinned, license/closure checked, materialized, and compiled against
the selected target before its source is attached to a generation task.
"""

import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import PlatformAdapter, adapter_for_target
from .reuse_proof_executor import ReuseProofReceipt, execute_reuse_proof
from .source_transplant import (
    DonorSlice,
    SourceTransplantError,
    inspect_repository_slice,
    validated_reuse_donor,
)

_SCHEMA = "mmm/grounded-repository-reuse-plan-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]{2,}", re.IGNORECASE)
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
        if value.count("/") == 1:
            return value
    try:
        parsed = urlparse(str(source_url or "").strip())
    except ValueError:
        return ""
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _grounded_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    research = design.get("_pre_design_research")
    notes = research.get("domain_notes") if isinstance(research, Mapping) else None
    if not isinstance(notes, list):
        return ()
    cards: list[dict[str, Any]] = []
    by_repository: dict[str, int] = {}
    for note in notes:
        raw_cards = note.get("grounded_evidence_cards") if isinstance(note, Mapping) else None
        for raw in raw_cards if isinstance(raw_cards, list) else ():
            if not isinstance(raw, Mapping):
                continue
            repository = _github_repository(
                str(raw.get("source_id") or ""),
                str(raw.get("source_url") or ""),
            )
            if not repository:
                continue
            evidence_text = " ".join(
                str(raw.get(key) or "")
                for key in ("source_title", "exact_excerpt")
            )
            key = repository.casefold()
            if key not in by_repository:
                by_repository[key] = len(cards)
                cards.append(
                    {
                        "repository": repository,
                        "page_refs": [str(raw.get("page_ref") or "")],
                        "source_ids": [str(raw.get("source_id") or "")],
                        "source_urls": [str(raw.get("source_url") or "")],
                        "evidence_text": evidence_text,
                    }
                )
                continue
            existing = cards[by_repository[key]]
            for field, value in (
                ("page_refs", str(raw.get("page_ref") or "")),
                ("source_ids", str(raw.get("source_id") or "")),
                ("source_urls", str(raw.get("source_url") or "")),
            ):
                if value and value not in existing[field]:
                    existing[field].append(value)
            existing["evidence_text"] = (
                f"{existing['evidence_text']} {evidence_text}"
            ).strip()
    for card in cards:
        card["evidence_tokens"] = sorted(_tokens(card["evidence_text"]))
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


def _adapter(design: Mapping[str, Any]) -> PlatformAdapter:
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


def _target_context(adapter: PlatformAdapter) -> dict[str, Any]:
    return {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
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


def _donor_rank(
    donor: DonorSlice,
    receipt: ReuseProofReceipt,
) -> tuple[int, int, float, float, str]:
    return (
        1 if receipt.tests_passed else 0,
        1 if donor.target_compatibility == "exact" else 0,
        float(donor.confidence),
        -float(donor.adaptation_cost),
        donor.repository.casefold(),
    )


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
    cards = _grounded_repository_cards(design)
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
    temporary = None
    proof_workspace = Path(target_workspace).resolve() if target_workspace else None
    if proof_workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="mmm_reuse_target_")
        proof_workspace = Path(temporary.name)
    try:
        for capability in capabilities:
            terms = terms_by_capability.get(capability, (capability.replace(".", " "),))
            candidates: list[DonorSlice] = []
            for card in cards:
                overlap = _candidate_overlap(capability, terms, card)
                if overlap <= 0:
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
                        "status": "proof_pending" if admitted else "inspection_rejected",
                        "overlap": overlap,
                    }
                )
                if admitted and donor is not None:
                    candidates.append(donor)

            verified: list[tuple[DonorSlice, ReuseProofReceipt]] = []
            for donor in candidates:
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
                    continue
                proof_payload = receipt.to_dict()
                proof_payload["status"] = (
                    "verified_code_reuse" if _proof_admits(donor, receipt) else "rejected"
                )
                proofs.append(proof_payload)
                if _proof_admits(donor, receipt):
                    verified.append((donor, receipt))

            selected = max(verified, key=lambda item: _donor_rank(*item)) if verified else None
            if selected is None:
                decisions.append(
                    {
                        "capability": capability,
                        "mode": "fresh",
                        "source_id": "",
                        "component_refs": [],
                        "rationale": (
                            "No grounded GitHub donor passed immutable source, permissive "
                            "license, dependency closure, authoritative compile, and artifact coverage gates."
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
                        "Host-grounded donor source was pinned, hash-verified, adapted, "
                        "and authoritatively compiled before coder generation."
                    ),
                }
            )
    finally:
        if temporary is not None:
            temporary.cleanup()

    return {
        "schema_version": _SCHEMA,
        "source_plan_sha256": source_plan_sha256,
        "capability_graph": graph,
        "capabilities": decisions,
        "grounded_repository_count": len(cards),
        "inspection_receipts": inspections,
        "proof_receipts": proofs,
        "selection_policy": (
            "grounded GitHub evidence -> frozen-intent overlap -> immutable source inspection -> "
            "authoritative target compile -> complete verified artifact coverage -> task-owned code context"
        ),
    }


__all__ = ["build_repository_reuse_plan"]
