from __future__ import annotations

from functools import wraps
from typing import Any, Mapping, Sequence


_EVIDENCE_FIELDS = (
    "module_ids",
    "asset_ids",
    "audio_ids",
    "acceptance_tests",
)
_PRODUCTION_KEYS = frozenset(
    {
        "modules",
        "assets",
        "audio",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _recent_catalog_ids(request: Any, field: str) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    receipt = request.get(field)
    if not isinstance(receipt, Mapping):
        return set()
    return _string_set(receipt.get("recent_ids", ()))


def _dependency_export_ids(request: Any) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    exports = request.get("dependency_exports")
    if not isinstance(exports, Mapping):
        return set()
    result: set[str] = set()
    for values in exports.values():
        result.update(_string_set(values))
    return result


def _produced_ids(page: Mapping[str, Any], field: str, id_field: str) -> set[str]:
    result: set[str] = set()
    values = page.get(field)
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get(id_field, "")).strip()
        if value:
            result.add(value)
    return result


def _evidence_has_declared_reference(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(_string_set(value.get(field, ())) for field in _EVIDENCE_FIELDS)


def _evidence_is_grounded(
    value: Any,
    *,
    module_ids: set[str],
    asset_ids: set[str],
    audio_ids: set[str],
    acceptance_tests: set[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        (_string_set(value.get("module_ids", ())) & module_ids)
        or (_string_set(value.get("asset_ids", ())) & asset_ids)
        or (_string_set(value.get("audio_ids", ())) & audio_ids)
        or (_string_set(value.get("acceptance_tests", ())) & acceptance_tests)
    )


def _sanitize_production_page(
    page: Mapping[str, Any],
    request: dict[str, Any] | str,
) -> dict[str, Any]:
    """Validate explicit evidence without weakening legacy host-owned bookkeeping."""
    result = dict(page)
    evidence_value = result.get("deliverable_evidence")
    if not isinstance(evidence_value, Mapping):
        return result

    completed = result.get("completed_deliverables")
    if not isinstance(completed, list):
        completed = []

    evidence = {
        str(key): dict(value)
        for key, value in evidence_value.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    result["deliverable_evidence"] = evidence

    current_modules = _produced_ids(result, "modules", "module_id")
    current_assets = _produced_ids(result, "assets", "asset_id")
    current_audio = _produced_ids(result, "audio", "sound_id")
    current_tests = _string_set(result.get("acceptance_tests", ()))

    valid_modules = (
        current_modules
        | _recent_catalog_ids(request, "known_module_catalog")
        | _dependency_export_ids(request)
    )
    valid_assets = current_assets | _recent_catalog_ids(
        request, "known_asset_catalog"
    )
    valid_audio = current_audio | _recent_catalog_ids(
        request, "known_audio_catalog"
    )

    supported: list[str] = []
    for raw in completed:
        deliverable = str(raw).strip()
        if not deliverable:
            continue
        if _evidence_is_grounded(
            evidence.get(deliverable),
            module_ids=valid_modules,
            asset_ids=valid_assets,
            audio_ids=valid_audio,
            acceptance_tests=current_tests,
        ):
            supported.append(deliverable)
    result["completed_deliverables"] = supported
    return result


def _is_production_decode(
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> bool:
    if not isinstance(request, Mapping) or "remaining_deliverables" not in request:
        return False
    return any(set(contract) == set(_PRODUCTION_KEYS) for contract in expected_contracts)


def _install_evidence_contract(complete_planner_module: Any) -> None:
    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_small_model_evidence_guard", False):
        return

    @wraps(current)
    def generate_with_evidence_guard(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        # Production evidence is optional input to the host-owned bookkeeping layer.
        # Never widen the strict top-level contract here: doing so turns an optional
        # quality signal into a required model field and can create pointless repair
        # decodes for otherwise valid pages.
        page = current(
            router,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )
        if not _is_production_decode(request, expected_contracts):
            return page
        if "deliverable_evidence" not in page:
            return page
        sanitized = _sanitize_production_page(page, request)
        sanitized.pop("deliverable_evidence", None)
        return sanitized

    generate_with_evidence_guard._mmm_small_model_evidence_guard = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_evidence_guard


def _install_evidence_aware_scoring(agentic_module: Any) -> None:
    current = agentic_module._score_plan_page
    if getattr(current, "_mmm_evidence_aware_plan_score", False):
        return

    @wraps(current)
    def score_with_evidence(page: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        base_score, verifier = current(page)
        if "deliverable_evidence" not in page:
            return base_score, dict(verifier)

        completed = page.get("completed_deliverables")
        completed_values = (
            [str(item).strip() for item in completed if str(item).strip()]
            if isinstance(completed, list)
            else []
        )
        evidence = page.get("deliverable_evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}

        current_modules = _produced_ids(page, "modules", "module_id")
        current_assets = _produced_ids(page, "assets", "asset_id")
        current_audio = _produced_ids(page, "audio", "sound_id")
        current_tests = _string_set(page.get("acceptance_tests", ()))

        declared = 0
        grounded = 0
        unsupported = 0
        for deliverable in completed_values:
            item = evidence.get(deliverable)
            if _evidence_has_declared_reference(item):
                declared += 1
            else:
                unsupported += 1
            if _evidence_is_grounded(
                item,
                module_ids=current_modules,
                asset_ids=current_assets,
                audio_ids=current_audio,
                acceptance_tests=current_tests,
            ):
                grounded += 1

        score = base_score + 24.0 * grounded - 36.0 * unsupported
        details = {
            **dict(verifier),
            "declared_completion_evidence": declared,
            "grounded_completion_evidence": grounded,
            "unsupported_completion_claims": unsupported,
        }
        return score, details

    score_with_evidence._mmm_evidence_aware_plan_score = True  # type: ignore[attr-defined]
    agentic_module._score_plan_page = score_with_evidence


def install() -> None:
    """Bind research-derived small-model amplification to the live planner."""
    from . import agentic_optimization_contract, complete_planner
    from .small_model_agent_policy import enhance_planner

    _install_evidence_aware_scoring(agentic_optimization_contract)
    _install_evidence_contract(complete_planner)
    enhance_planner(complete_planner)


__all__ = ["install"]
