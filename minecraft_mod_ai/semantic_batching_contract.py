"""Bound semantic extraction without weakening host-owned requirement authority.

The requirements ledger forbids treating an arbitrarily large authored request as one
structured model call. This module does not invent an "optimal" batch size. It uses a
strict measured receipt when one is attached to the router; otherwise it falls back to
one host-owned clause per model turn and records that the value is unmeasured.

Only semantic leaf extraction is batched. Stable source records are created by the host
before any model call, all approved leaves are merged before requirement IDs are built,
and the existing global catalog/dependency passes remain authoritative.

This module is a pure helper. Production owners call ``build_bounded_requirement_catalog``
directly; installation never rewires another module at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import semantic_requirement_authority as _semantic

_INSTALLED = False
_ORIGINAL_SEMANTIC_BUILD = _semantic.build_approved_requirement_catalog

_RECEIPT_ATTRIBUTE = "semantic_extraction_batch_receipt"
_MEASURED_STATUS = "MEASURED"
_FALLBACK_BATCH_SIZE = 1


def _sha_receipt(value: Any, *, field: str) -> str:
    result = str(value or "").strip()
    if not result.startswith("sha256:") or len(result) != len("sha256:") + 64:
        raise _semantic._evidence.EvidencePlanError(
            f"REQ_SCALE_BATCH_RECEIPT: {field} must be a sha256 receipt."
        )
    return result


def _resolve_batch_contract(router: Any) -> dict[str, Any]:
    """Return an explicit measured batch contract or a conservative unmeasured fallback."""

    raw = getattr(router, _RECEIPT_ATTRIBUTE, None)
    if raw is None:
        return {
            "max_clauses_per_turn": _FALLBACK_BATCH_SIZE,
            "source": "unmeasured_conservative_single_clause",
            "measured": False,
            "model_identity_sha256": "",
            "benchmark_receipt_sha256": "",
        }
    if not isinstance(raw, Mapping):
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_RECEIPT: semantic_extraction_batch_receipt must be a mapping."
        )

    status = str(raw.get("status") or "").strip().upper()
    size = raw.get("max_clauses_per_turn")
    if status != _MEASURED_STATUS:
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_RECEIPT: supplied receipt is not in MEASURED state."
        )
    if type(size) is not int or size <= 0:
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_RECEIPT: measured max_clauses_per_turn must be a positive integer."
        )

    return {
        "max_clauses_per_turn": int(size),
        "source": "measured_model_profile_receipt",
        "measured": True,
        "model_identity_sha256": _sha_receipt(
            raw.get("model_identity_sha256"), field="model_identity_sha256"
        ),
        "benchmark_receipt_sha256": _sha_receipt(
            raw.get("benchmark_receipt_sha256"), field="benchmark_receipt_sha256"
        ),
    }


def _chunks(
    clauses: Sequence[Mapping[str, Any]],
    size: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    return tuple(
        tuple(clauses[index : index + size])
        for index in range(0, len(clauses), size)
    )


def _generate_bounded_nodes(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    nodes: list[dict[str, Any]] = []
    batches = _chunks(clauses, batch_size)

    for batch_index, batch in enumerate(batches):
        try:
            payload = _semantic._call_semantic_model(router, batch)
        except Exception as exc:
            raise _semantic._evidence.EvidencePlanError(
                "semantic requirement authority model call failed for bounded batch "
                f"{batch_index}: {type(exc).__name__}: {exc}"
            ) from exc

        batch_nodes, invalid_clauses, diagnostics = _semantic._evaluate_batch(payload, batch)
        if invalid_clauses:
            raise _semantic._evidence.EvidencePlanError(
                "semantic requirement authority rejected bounded-batch model output: "
                + _semantic._canonical(
                    {
                        "batch_index": batch_index,
                        "invalid_clause_indices": sorted(invalid_clauses),
                        "diagnostics": diagnostics,
                    }
                )
            )
        nodes.extend(batch_nodes)

    return _semantic._assign_local_ids(nodes), len(batches)


def build_bounded_requirement_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
    """Compile semantic leaves in bounded batches, then reconcile the full catalog globally."""

    if router is None:
        return _ORIGINAL_SEMANTIC_BUILD(prompt, router=None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: semantic authority requires a non-empty prompt."
        )

    clauses = _semantic._clause_records(prompt)
    contract = _resolve_batch_contract(router)
    batch_size = int(contract["max_clauses_per_turn"])
    nodes, batch_count = _generate_bounded_nodes(
        router,
        clauses,
        batch_size=batch_size,
    )

    # _build_catalog resolves prerequisite capability IDs only after every batch has
    # been merged. Cross-batch prerequisites therefore cannot be lost at a batch edge.
    catalog = _semantic._build_catalog(prompt, nodes, clauses)
    audit = dict(catalog.get("semantic_audit") or {})
    audit.update(
        {
            "normal_model_turns": batch_count,
            "semantic_model_turns": batch_count,
            "semantic_discovery_model_turns": batch_count,
            "semantic_detail_model_turns": 0,
            "max_repair_turns": 0,
            "generation_policy": "bounded_host_owned_semantic_batches",
            "semantic_generation_protocol": "bounded_host_owned_semantic_batches",
            "semantic_batch_size": batch_size,
            "semantic_batch_count": batch_count,
            "semantic_batch_size_source": contract["source"],
            "semantic_batch_size_measured": bool(contract["measured"]),
            "semantic_batch_model_identity_sha256": contract["model_identity_sha256"],
            "semantic_batch_benchmark_receipt_sha256": contract[
                "benchmark_receipt_sha256"
            ],
            "max_clauses_per_model_turn": batch_size,
            "cross_batch_prerequisite_reconciliation": (
                "global_catalog_capability_resolution"
            ),
            "source_clause_index_owner": "host",
            "source_anchor_owner": "host",
            "source_grounding_owner": "host",
        }
    )
    catalog["semantic_audit"] = audit
    catalog["catalog_sha256"] = ""
    catalog["catalog_sha256"] = _semantic._evidence._hash_without(
        catalog, "catalog_sha256"
    )
    _semantic.validate_approved_requirement_catalog(catalog, prompt=prompt)
    return catalog


build_bounded_requirement_catalog.__mmm_bounded_semantic_batching__ = True  # type: ignore[attr-defined]


def install_semantic_batching_contract() -> None:
    """Mark the pure batching helper available without runtime rebinding."""

    global _INSTALLED
    _INSTALLED = True


__all__ = [
    "build_bounded_requirement_catalog",
    "install_semantic_batching_contract",
]
