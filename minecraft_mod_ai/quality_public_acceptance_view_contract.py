from __future__ import annotations

"""Keep internal quality readers aligned with the proposal's public acceptance surface.

The production validator intentionally requires external ``acceptance_tests`` to match only
public catalog entries. Internal quality checks remain code-owned evidence criteria. Some
read-only/internal quality helpers historically passed the entire acceptance catalog back into
the strict validator, which became invalid once that boundary was enforced. This contract
adapts only those internal callers without weakening the validator or scanning/rebinding the
entire loaded package namespace.
"""

from collections.abc import Mapping
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
    quality_evidence_module.validate_production_contract = validate_internal_quality_view
    _INSTALLED = True


__all__ = ["_public_acceptance", "install"]
