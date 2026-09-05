"""Bound semantic extraction without weakening host-owned requirement authority.

The requirements ledger forbids treating an arbitrarily large authored request as one
structured model call. A measured model/runtime receipt may provide a bounded batch size;
otherwise the conservative fallback is one host-owned clause per batch and is explicitly
recorded as unmeasured.

Each batch uses two deliberately narrow small-model stages. Stage 1 segments authored
behaviors without capability authority; the host grounds and source-validates those leaves.
Stage 2 classifies only the immutable approved leaves into the host capability catalog.
All approved leaves are globally merged before host feature-model dependency resolution.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import semantic_requirement_authority as _semantic
from .minecraft_requirement_dependencies import bind_selected_feature_dependencies
from .semantic_leaf_pipeline import compile_semantic_batch

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
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "segmentation_attempts": int(metrics["segmentation_attempts"]),
        "classification_attempts": int(metrics["classification_attempts"]),
        "semantic_model_calls_total": int(metrics["semantic_model_calls_total"]),
        "semantic_repair_turns_used": int(metrics["semantic_repair_turns_used"]),
        "segmentation_repaired": bool(metrics["segmentation_repaired"]),
        "classification_repaired": bool(metrics["classification_repaired"]),
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
    receipts: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(_chunks(clauses, batch_size)):
        try:
            batch_nodes, metrics = compile_semantic_batch(router, batch)
        except Exception as exc:
            if isinstance(exc, _semantic._evidence.EvidencePlanError):
                raise
            raise _semantic._evidence.EvidencePlanError(
                "two-stage semantic compilation failed for bounded batch "
                f"{batch_index}: {type(exc).__name__}: {exc}"
            ) from exc
        nodes.extend(batch_nodes)
        receipts.append(_source_batch_receipt(batch_index, batch, metrics))

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
    model_calls_total = sum(
        int(receipt["semantic_model_calls_total"]) for receipt in batch_receipts
    )
    repair_turns = sum(
        int(receipt["semantic_repair_turns_used"]) for receipt in batch_receipts
    )
    audit.update(
        {
            "normal_model_turns": model_calls_total,
            "semantic_model_turns": model_calls_total,
            "semantic_discovery_model_turns": model_calls_total,
            "semantic_detail_model_turns": 0,
            "max_repair_turns": 2,
            "semantic_model_calls_total_observed": model_calls_total,
            "semantic_repair_turns_used": repair_turns,
            "semantic_max_repair_turns_per_batch": 2,
            "semantic_base_stage_calls_per_batch": 2,
            "generation_policy": "two_stage_bounded_host_owned_semantics",
            "semantic_generation_protocol": "segment_then_host_ground_then_classify",
            "semantic_segmentation_owner": "bounded_model_host_validated",
            "semantic_classification_owner": "host_catalog_bounded_model_choice",
            "semantic_source_fidelity_policy": "language_neutral_exact_authored_character_partition",
            "semantic_source_fidelity_owner": "host",
            "semantic_leaf_mutability_after_grounding": "immutable",
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
