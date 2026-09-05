"""Bound semantic extraction without weakening host-owned requirement authority.

The requirements ledger forbids treating an arbitrarily large authored request as one
structured model call. This module does not invent an optimal batch size. A measured
model/runtime receipt may provide a bounded size; otherwise the conservative fallback
is one host-owned clause per model turn and is explicitly recorded as unmeasured.

Only semantic leaf extraction is batched. Stable source records are created by the host
before any model call, all approved leaves are merged before requirement IDs are built,
and host feature-model dependency resolution runs only after that global merge.

This module is a pure helper. Production owners call ``build_bounded_requirement_catalog``
directly; installation validates that static call graph and never rewires another
module's validator, request builder, or runtime routing implementation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import semantic_requirement_authority as _semantic
from .minecraft_requirement_dependencies import bind_selected_feature_dependencies

_INSTALLED = False
_RECEIPT_ATTRIBUTE = "semantic_extraction_batch_receipt"
_MEASURED_STATUS = "MEASURED"
_FALLBACK_BATCH_SIZE = 1
_SHA256_RECEIPT = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha_receipt(value: Any, *, field: str) -> str:
    result = str(value or "").strip().casefold()
    if not _SHA256_RECEIPT.fullmatch(result):
        raise _semantic._evidence.EvidencePlanError(
            f"REQ_SCALE_BATCH_RECEIPT: {field} must be an exact sha256 receipt."
        )
    return result


def _resolve_batch_contract(router: Any) -> dict[str, Any]:
    """Return a measured batch contract or an explicit conservative fallback."""

    raw = getattr(router, _RECEIPT_ATTRIBUTE, None)
    if raw is None:
        return {
            "max_clauses_per_turn": _FALLBACK_BATCH_SIZE,
            "source": "unmeasured_conservative_single_clause",
            "measured": False,
            "model_identity_sha256": "",
            "runtime_profile_sha256": "",
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
        "source": "measured_model_runtime_receipt",
        "measured": True,
        "model_identity_sha256": _sha_receipt(
            raw.get("model_identity_sha256"), field="model_identity_sha256"
        ),
        "runtime_profile_sha256": _sha_receipt(
            raw.get("runtime_profile_sha256"), field="runtime_profile_sha256"
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


def _source_batch_receipt(
    batch_index: int,
    clauses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "source_clauses": [
            {
                "source_clause_index": int(clause["clause_index"]),
                "char_start": int(clause["char_start"]),
                "char_end": int(clause["char_end"]),
                "text_sha256": str(clause["text_sha256"]),
            }
            for clause in clauses
        ],
    }


def _generate_bounded_nodes(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    nodes: list[dict[str, Any]] = []
    batches = _chunks(clauses, batch_size)
    receipts: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches):
        receipts.append(_source_batch_receipt(batch_index, batch))
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

    assigned = _semantic._assign_local_ids(nodes)
    if not assigned:
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_EMPTY: bounded semantic extraction produced no approved leaves."
        )
    return assigned, tuple(receipts)


def build_bounded_requirement_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
    """Compile semantic leaves, then let the host resolve the global feature DAG."""

    if router is None:
        return _semantic.build_approved_requirement_catalog(prompt, router=None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: semantic authority requires a non-empty prompt."
        )

    clauses = _semantic._clause_records(prompt)
    contract = _resolve_batch_contract(router)
    batch_size = int(contract["max_clauses_per_turn"])
    nodes, batch_receipts = _generate_bounded_nodes(
        router,
        clauses,
        batch_size=batch_size,
    )

    catalog = _semantic._build_catalog(prompt, nodes, clauses)
    try:
        catalog = bind_selected_feature_dependencies(catalog)
    except ValueError as exc:
        raise _semantic._evidence.EvidencePlanError(
            "host Minecraft feature dependency resolution failed: " + str(exc)
        ) from exc

    audit = dict(catalog.get("semantic_audit") or {})
    batch_count = len(batch_receipts)
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
            "semantic_batch_runtime_profile_sha256": contract["runtime_profile_sha256"],
            "semantic_batch_benchmark_receipt_sha256": contract[
                "benchmark_receipt_sha256"
            ],
            "semantic_batches": list(batch_receipts),
            "max_clauses_per_model_turn": batch_size,
            "cross_batch_prerequisite_reconciliation": (
                "host_minecraft_feature_model_after_global_merge"
            ),
            "feature_dependency_owner": "host_minecraft_feature_model",
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


def _static_owner_chain_contains_bounded_builder(target: Any) -> bool:
    current = target
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "__code__", None)
        names = set(getattr(code, "co_names", ()))
        if "build_bounded_requirement_catalog" in names:
            return True
        current = getattr(current, "__wrapped__", None)
    return False


def _assert_static_bounded_owner(target: Any, *, owner: str) -> None:
    if not _static_owner_chain_contains_bounded_builder(target):
        raise RuntimeError(
            f"{owner} is not statically wired to build_bounded_requirement_catalog"
        )
    target.__mmm_bounded_semantic_batching__ = True  # type: ignore[attr-defined]


def install_semantic_batching_contract() -> None:
    """Revalidate static bounded owners on every reconciliation pass."""

    global _INSTALLED

    from . import evidence_request_guard as guard
    from . import planning_authority as planning

    _assert_static_bounded_owner(
        guard.build_authoritative_request_catalog,
        owner="evidence_request_guard.build_authoritative_request_catalog",
    )
    _assert_static_bounded_owner(
        planning._compile_semantic_catalog,
        owner="planning_authority._compile_semantic_catalog",
    )
    _INSTALLED = True


__all__ = [
    "build_bounded_requirement_catalog",
    "install_semantic_batching_contract",
]
