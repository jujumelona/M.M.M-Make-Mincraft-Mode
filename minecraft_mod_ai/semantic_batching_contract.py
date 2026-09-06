"""Bound semantic extraction with a complete host fallback plan.

Every authored clause has a host-owned ``custom.semantic`` requirement before the model is
called. A small model gets one bounded structured call to refine those defaults into more
specific semantic leaves/capabilities. Valid model leaves replace the corresponding host
default; malformed output, missing clauses, or a model exception never erase authored work.

There is no semantic retry loop, no re-segmentation loop, no exact-character partition
requirement, and no second classification turn. Only host contract corruption is fatal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from . import semantic_requirement_authority as _semantic
from .minecraft_requirement_dependencies import bind_selected_feature_dependencies
from .minecraft_template_catalog import CUSTOM_CAPABILITY_SENTINEL
from .model_concurrency import router_native_model_parallelism
from .request_requirements import LeafAtomicityStatus, validate_leaf_atomicity

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
    clauses: Sequence[Mapping[str, Any]], size: int
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    return tuple(
        tuple(clauses[index : index + size])
        for index in range(0, len(clauses), size)
    )


def _host_default_node(clause: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    """Compile a lossless minimum requirement directly from one authored clause."""
    clause_index = int(clause["clause_index"])
    text = str(clause["text"])
    grounding = {
        "source_quote": text,
        "source_start": int(clause["char_start"]),
        "source_end": int(clause["char_end"]),
        "grounding_method": "host_full_clause_default",
        "grounding_similarity": 1.0,
        "model_anchor": "",
    }
    capability = _semantic._host_capability_id(
        CUSTOM_CAPABILITY_SENTINEL,
        grounding=grounding,
        clause_index=clause_index,
        item_index=ordinal,
    )
    return {
        "capability_id": capability,
        "model_capability_choice": CUSTOM_CAPABILITY_SENTINEL,
        "semantic_type": "gameplay_mechanic",
        "provenance_role": "explicit",
        "source_clause_index": clause_index,
        **grounding,
        "semantic_statement": text,
        "derived_from": [],
        "depends_on": [],
        "derivation_reason": "",
        "observable_behavior": {
            "given": "The authored mod context for this clause is active.",
            "when": "The behavior described by this authored clause is exercised.",
            "then": text,
        },
        "required_prerequisite_capabilities": [],
        "optional_prerequisite_capabilities": [],
    }


def _source_batch_receipt(
    batch_index: int,
    clauses: Sequence[Mapping[str, Any]],
    *,
    input_leaf_count: int,
    approved_leaf_count: int,
    compound_demotions: int,
    non_executable_drops: int,
    host_fallback_count: int,
    model_error: str,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "semantic_compile_calls": 1,
        "semantic_model_calls_total": 1,
        "semantic_repair_turns_used": 0,
        "input_leaf_count": input_leaf_count,
        "approved_leaf_count": approved_leaf_count,
        "compound_demotions": compound_demotions,
        "non_executable_drops": non_executable_drops,
        "host_fallback_count": host_fallback_count,
        "model_error": model_error,
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


def _demote_compound_node(
    node: Mapping[str, Any], *, item_index: int
) -> dict[str, Any]:
    result = dict(node)
    result["model_capability_choice"] = CUSTOM_CAPABILITY_SENTINEL
    result["capability_id"] = _semantic._host_capability_id(
        CUSTOM_CAPABILITY_SENTINEL,
        grounding=result,
        clause_index=int(result["source_clause_index"]),
        item_index=item_index,
    )
    return result


def _compile_bounded_batch(
    router: Any,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine host defaults with one model call; never fail because of model output."""
    defaults = {
        int(clause["clause_index"]): _host_default_node(clause, ordinal=index)
        for index, clause in enumerate(batch)
    }
    model_error = ""
    nodes: list[dict[str, Any]] = []
    invalid_clauses: set[int] = set(defaults)
    diagnostics: list[dict[str, Any]] = []
    try:
        payload = _semantic._call_semantic_model(router, batch)
        nodes, invalid_clauses, diagnostics = _semantic._evaluate_batch(payload, batch)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

    approved: list[dict[str, Any]] = []
    compound_demotions = 0
    non_executable_drops = 0
    valid_clause_indices: set[int] = set()
    for item_index, raw in enumerate(nodes):
        node = dict(raw)
        clause_index = int(node["source_clause_index"])
        status, _ = validate_leaf_atomicity(
            {
                "source_anchor": str(node.get("source_quote") or ""),
                "semantic_statement": str(node.get("semantic_statement") or ""),
            }
        )
        if status in {LeafAtomicityStatus.CONTEXT, LeafAtomicityStatus.CATCH_ALL}:
            non_executable_drops += 1
            continue
        if status == LeafAtomicityStatus.COMPOUND:
            node = _demote_compound_node(node, item_index=item_index)
            compound_demotions += 1
        approved.append(node)
        valid_clause_indices.add(clause_index)

    fallback_indices = set(defaults) - valid_clause_indices
    fallback_indices.update(index for index in invalid_clauses if index in defaults)
    for clause_index in sorted(fallback_indices):
        approved.append(defaults[clause_index])

    # Deduplicate a fallback that was added because another leaf from the same clause was
    # invalid while a valid leaf also survived. Valid authored leaves take precedence.
    if valid_clause_indices:
        approved = [
            node
            for node in approved
            if not (
                node.get("grounding_method") == "host_full_clause_default"
                and int(node["source_clause_index"]) in valid_clause_indices
            )
        ]

    if not approved:
        # This can only happen if the host itself supplied an empty batch, which callers
        # must never do.
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_EMPTY: host semantic batch contained no authored clause."
        )

    return approved, _source_batch_receipt(
        batch_index,
        batch,
        input_leaf_count=len(nodes),
        approved_leaf_count=len(approved),
        compound_demotions=compound_demotions,
        non_executable_drops=non_executable_drops,
        host_fallback_count=sum(
            1
            for node in approved
            if node.get("grounding_method") == "host_full_clause_default"
        ),
        model_error=model_error or (_semantic._canonical(diagnostics) if diagnostics else ""),
    )


def _batch_worker_count(router: Any, batch_count: int) -> int:
    if batch_count <= 1:
        return 1
    return max(1, min(batch_count, router_native_model_parallelism(router)))


def _generate_bounded_nodes(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    batches = _chunks(clauses, batch_size)
    workers = _batch_worker_count(router, len(batches))
    if workers == 1:
        compiled = tuple(
            _compile_bounded_batch(router, batch_index, batch)
            for batch_index, batch in enumerate(batches)
        )
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm-semantic",
        ) as executor:
            futures = [
                executor.submit(
                    copy_context().run,
                    _compile_bounded_batch,
                    router,
                    batch_index,
                    batch,
                )
                for batch_index, batch in enumerate(batches)
            ]
            compiled = tuple(future.result() for future in futures)

    nodes: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for batch_nodes, receipt in compiled:
        nodes.extend(batch_nodes)
        receipts.append(receipt)
    assigned = _semantic._assign_local_ids(nodes)
    if not assigned:
        raise _semantic._evidence.EvidencePlanError(
            "REQ_SCALE_BATCH_EMPTY: host semantic defaults produced no requirement."
        )
    return assigned, tuple(receipts)


def build_bounded_requirement_catalog(
    prompt: str,
    router: Any | None = None,
) -> dict[str, Any]:
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
        router, clauses, batch_size=batch_size
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
    parallel_workers = _batch_worker_count(router, batch_count)
    model_calls_total = sum(
        int(receipt["semantic_model_calls_total"]) for receipt in batch_receipts
    )
    compound_demotions = sum(int(receipt["compound_demotions"]) for receipt in batch_receipts)
    non_executable_drops = sum(
        int(receipt["non_executable_drops"]) for receipt in batch_receipts
    )
    host_fallbacks = sum(int(receipt["host_fallback_count"]) for receipt in batch_receipts)
    audit.update(
        {
            "normal_model_turns": model_calls_total,
            "semantic_model_turns": model_calls_total,
            "semantic_discovery_model_turns": model_calls_total,
            "semantic_detail_model_turns": 0,
            "max_repair_turns": 0,
            "semantic_model_calls_total_observed": model_calls_total,
            "semantic_repair_turns_used": 0,
            "semantic_max_repair_turns_per_batch": 0,
            "semantic_base_stage_calls_per_batch": 1,
            "generation_policy": "host_default_single_pass_model_refinement",
            "semantic_generation_protocol": "host_clause_defaults_then_one_structured_refinement",
            "semantic_segmentation_owner": "host_defaults_model_optional_refinement",
            "semantic_classification_owner": "host_catalog_model_optional_choice",
            "semantic_source_fidelity_policy": "host_exact_clause_fallback_or_host_grounded_model_anchor",
            "semantic_source_fidelity_owner": "host",
            "semantic_leaf_mutability_after_grounding": "immutable",
            "semantic_compound_policy": "demote_to_custom_without_resegmentation",
            "semantic_compound_demotions": compound_demotions,
            "semantic_non_executable_drops": non_executable_drops,
            "semantic_host_fallback_count": host_fallbacks,
            "semantic_batch_size": batch_size,
            "semantic_batch_count": batch_count,
            "semantic_batch_parallel_workers": parallel_workers,
            "semantic_batch_parallelism_policy": (
                "measured_native_model_slots"
                if parallel_workers > 1
                else "serial_unproven_or_single_batch"
            ),
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
            "source_anchor_owner": "host_default_or_host_grounded_model_locator",
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


__all__ = ["build_bounded_requirement_catalog", "install_semantic_batching_contract"]
