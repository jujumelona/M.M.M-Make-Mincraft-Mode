from __future__ import annotations

"""Keep internal quality readers aligned with the proposal's public acceptance surface.

The production validator intentionally requires external ``acceptance_tests`` to match only
public catalog entries. Internal quality checks remain code-owned evidence criteria. Some
read-only/internal quality helpers historically passed the entire acceptance catalog back into
the strict validator, which became invalid once that boundary was enforced. This contract
adapts those internal callers without weakening the validator or allowing a proposal to submit
internal quality statements as public acceptance.
"""

import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False


def _public_acceptance(contract: Mapping[str, Any]) -> list[str]:
    values = contract.get("acceptance_catalog")
    if not isinstance(values, list):
        return []
    return [
        str(item["statement"])
        for item in values
        if isinstance(item, Mapping)
        and item.get("visibility") == "public"
        and isinstance(item.get("statement"), str)
    ]


def install(production_module: Any, quality_evidence_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    strict_validate = production_module.validate_production_contract
    original_summary = production_module.quality_contract_summary
    original_evaluate = production_module.evaluate_quality_contract

    @wraps(original_summary)
    def quality_contract_summary(contract: Mapping[str, Any]) -> str:
        module_ids = [
            item["implementation_id"]
            for item in contract.get("implementation_catalog", [])
            if isinstance(item, Mapping) and item.get("source_kind") == "module"
        ]
        strict_validate(contract, module_ids, _public_acceptance(contract))
        stats = contract["catalog_stats"]
        dimensions = ", ".join(
            item["title"] for item in contract["quality_dimension_catalog"]
        )
        return (
            f"Tracks {stats['requirements']} request-derived requirements across "
            f"{stats['implementations']} implementation entries and "
            f"{stats['acceptance_tests']} observable checks. Required quality: "
            f"{dimensions}. Completion requires fresh proposal-bound evidence from "
            "an independent verifier for every dimension."
        )

    quality_contract_summary._mmm_public_acceptance_view = True

    @wraps(original_evaluate)
    def evaluate_quality_contract(
        contract: Mapping[str, Any],
        evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        proposal_hash: str,
        previous: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        module_ids = [
            item["implementation_id"]
            for item in contract.get("implementation_catalog", [])
            if isinstance(item, Mapping) and item.get("source_kind") == "module"
        ]
        strict_validate(contract, module_ids, _public_acceptance(contract))
        if not isinstance(proposal_hash, str) or not production_module._SHA256.fullmatch(proposal_hash):
            raise production_module.ProductionContractError(
                "proposal_hash must be a canonical SHA-256"
            )
        previous_by_dimension: dict[str, Mapping[str, Any]] = {}
        iteration = 1
        if previous is not None:
            production_module._validate_quality_report(previous)
            if previous["proposal_hash"] != proposal_hash:
                raise production_module.ProductionContractError(
                    "previous report belongs to another proposal"
                )
            if previous["contract_sha256"] != contract["contract_sha256"]:
                raise production_module.ProductionContractError(
                    "previous report belongs to another contract"
                )
            iteration = production_module._strict_positive_int(
                previous["iteration"], "previous iteration"
            ) + 1
            previous_by_dimension = {
                item["dimension_id"]: item for item in previous["dimensions"]
            }
        receipts = production_module._normalize_evidence(evidence)
        active_ids = {
            item["dimension_id"] for item in contract["quality_dimension_catalog"]
        }
        unknown = sorted(set(receipts) - active_ids)
        if unknown:
            raise production_module.ProductionContractError(
                f"evidence targets unknown dimensions: {unknown}"
            )
        dimension_results: list[dict[str, Any]] = []
        plateau_dimensions: list[str] = []
        for dimension in contract["quality_dimension_catalog"]:
            dimension_id = dimension["dimension_id"]
            result = production_module._evaluate_dimension_receipt(
                dimension_id=dimension_id,
                route_ref=dimension["evidence_route_ref"],
                receipts=receipts.get(dimension_id, []),
                proposal_hash=proposal_hash,
                previous=previous_by_dimension.get(dimension_id),
            )
            if result["plateau"]:
                plateau_dimensions.append(dimension_id)
            dimension_results.append(result)
        unresolved = [
            item["dimension_id"]
            for item in dimension_results
            if item["status"] != "PASS"
        ]
        overall_status = (
            "PASS"
            if not unresolved
            else "FAIL"
            if any(item["status"] == "FAIL" for item in dimension_results)
            else "MISSING"
        )
        report: dict[str, Any] = {
            "schema_version": production_module.REPORT_SCHEMA,
            "proposal_hash": proposal_hash,
            "contract_sha256": contract["contract_sha256"],
            "iteration": iteration,
            "overall_status": overall_status,
            "dimensions": dimension_results,
            "unresolved_dimension_ids": unresolved,
            "plateau": {
                "detected": bool(plateau_dimensions),
                "dimension_ids": plateau_dimensions,
                "identical_failure_threshold": production_module._PLATEAU_THRESHOLD,
            },
            "report_sha256": "",
        }
        report["report_sha256"] = production_module._hash_without_field(
            report, "report_sha256"
        )
        production_module._validate_quality_report(report)
        return report

    evaluate_quality_contract._mmm_public_acceptance_view = True

    def validate_internal_quality_view(
        contract: Mapping[str, Any],
        modules: Any,
        _acceptance_tests: Any,
        assets: Any = None,
        evidence_plan: Mapping[str, Any] | None = None,
    ) -> None:
        strict_validate(
            contract,
            modules,
            _public_acceptance(contract),
            assets,
            evidence_plan,
        )

    validate_internal_quality_view._mmm_public_acceptance_view = True

    production_module.quality_contract_summary = quality_contract_summary
    production_module.evaluate_quality_contract = evaluate_quality_contract
    quality_evidence_module.validate_production_contract = validate_internal_quality_view

    # Repair direct imports that may have been bound before late finalization. Keep the
    # replacement exact: only aliases that still point to the wrapped originals move.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("minecraft_mod_ai.") or module is None:
            continue
        if getattr(module, "quality_contract_summary", None) is original_summary:
            setattr(module, "quality_contract_summary", quality_contract_summary)
        if getattr(module, "evaluate_quality_contract", None) is original_evaluate:
            setattr(module, "evaluate_quality_contract", evaluate_quality_contract)

    _INSTALLED = True


__all__ = ["_public_acceptance", "install"]
